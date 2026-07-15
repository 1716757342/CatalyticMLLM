#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: IRFT (Iterative Reward Fine-Tuning) trainer
GP-GRPO: iterative reinforcement fine-tuning combining structure-quality reward (R_step2) and energy reward (R_energy)

Reward formula:
    R_step3 = ω₁ · R_step2 + ω₂ · R_energy
    R_energy = exp(-λ · |E_pred - E_target|)

Workflow:
    Round 1: no exemplar; directly generate K CIF candidates
    Later rounds: randomly take 1 exemplar from ExemplarPool and extract its 3D structure
            Use it as additional multimodal input together with the original prompt to generate K candidates
    At the end of each round, add the best candidate to ExemplarPool
"""

import os
import sys
import gc
import re
import json
import math
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import transformers
from transformers import (
    Trainer,
    HfArgumentParser,
    AutoTokenizer,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model

# Add project root path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from qwenvl.train.train_qwen import Qwen2_5_VLForMolecule, set_model
    from qwenvl.train.reward_model_cif import CIFRewardModel
    from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
    from qwenvl.train.exemplar_pool import (
        ExemplarPool,
        extract_3d_from_cif,
        extract_target_energy,
    )
    from qwenvl.data.grpo_dataset import GRPODataset, GRPODataCollator
except ImportError as e:
    print(f"[Stage3] Failed to import modules: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument definitions
# ---------------------------------------------------------------------------

@dataclass
class IRFTArguments(TrainingArguments):
    """Stage 3 IRFT training arguments (inherits TrainingArguments and already includes Stage 3 fields)"""
    # GRPO sampling parameters (kept consistent with Stage 2)
    grpo_beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient"})
    num_samples_per_prompt: int = field(default=3, metadata={"help": "number of candidates K sampled per prompt"})
    temperature: float = field(default=0.7, metadata={"help": "sampling temperature"})
    top_p: float = field(default=0.9, metadata={"help": "nucleus sampling"})
    max_new_tokens: int = field(default=2048, metadata={"help": "maximum token count for CIF generation"})
    model_max_length: int = field(default=4096, metadata={"help": "maximum sequence length"})
    mm_projector_lr: Optional[float] = field(default=None)
    vision_tower_lr: Optional[float] = field(default=None)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def rank0_print(*args, **kwargs):
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def parse_energy_from_text(text: str) -> Optional[float]:
    """Parse energy value from model-generated text (eV/atom)."""
    patterns = [
        r"(-?\d+\.?\d*(?:e[+-]?\d+)?)\s*eV/atom",
        r"energy[:\s=]+(-?\d+\.?\d*(?:e[+-]?\d+)?)",
        r"(-?\d+\.\d+)",   # loosest: any floating-point number
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None



# ---------------------------------------------------------------------------
# IRFTTrainer
# ---------------------------------------------------------------------------

class IRFTTrainer(Trainer):
    """
    Stage 3 IRFT trainer.

    Adds to Stage 2 OnlineGRPOLoRATrainer:
    1. ExemplarPool management
    2. energy prediction (full multimodal inference, no_grad)
    3. composite reward R_step3 = ω₁·R_step2 + ω₂·R_energy
    4. multimodal generation with exemplar (from round 2 onward)
    """

    def __init__(
        self,
        model=None,
        irft_args: IRFTArguments = None,
        reward_model: CIFRewardModel = None,
        exemplar_pool: ExemplarPool = None,
        tokenizer=None,
        processor=None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)

        self.irft_args    = irft_args
        self.reward_model = reward_model or CIFRewardModel()
        self.pool         = exemplar_pool

        # sampling hyperparameters
        self.num_samples    = irft_args.num_samples_per_prompt
        self.temperature    = irft_args.temperature
        self.top_p          = irft_args.top_p
        self.max_new_tokens = irft_args.max_new_tokens
        self.beta           = irft_args.grpo_beta

        # Stage 3 reward weights
        self.w_struct  = irft_args.structure_reward_weight   # ω₁
        self.w_energy  = irft_args.energy_reward_weight      # ω₂
        self.lam       = irft_args.energy_lambda             # λ
        self.max_e_tok = irft_args.max_energy_pred_tokens

        # tokenizer / processor (for energy-prediction inference)
        self._tokenizer  = tokenizer or self.processing_class
        self._processor  = processor

        # log directory
        self.samples_dir = Path(irft_args.output_dir) / "irft_samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.global_step_counter = 0

        logger.info(
            f"IRFTTrainer initialized: K={self.num_samples}, "
            f"ω₁={self.w_struct}, ω₂={self.w_energy}, λ={self.lam}"
        )

    # ------------------------------------------------------------------
    # Helper: manually build inputs_embeds (fuse molecular 3D features)
    # ------------------------------------------------------------------

    def _build_inputs_embeds(
        self,
        raw_model,
        input_ids: torch.Tensor,
        mol_data: Optional[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Reproduce Qwen2_5_VLForMolecule.forward() feature-fusion logic in:
          1. embed_tokens(input_ids) -> text_embeds
          2. if present mol_data and input_ids contains IMAGE_TOKEN_INDEX placeholders:
             EquiformerV2 -> molecule_projector -> masked_scatter replace placeholders
          3. return fused inputs_embeds

        This lets generate() receive only inputs_embeds without needing to know molecule parameters.
        """
        from qwenvl.data.data_qwen import IMAGE_TOKEN_INDEX

        input_ids = input_ids.to(device)
        inputs_embeds = raw_model.model.embed_tokens(input_ids)

        if mol_data is not None and torch.sum(input_ids == IMAGE_TOKEN_INDEX) > 0:
            # mol_data keys: z (atomic_numbers), pos (coordinates), cell
            # Need to build molecule_mask (all True because single sample has no padding)
            z    = mol_data["z"].to(device)    # [1, N] or [N]
            pos  = mol_data["pos"].to(device)  # [1, N, 3] or [N, 3]
            cell = mol_data.get("cell")

            # Ensure batch dimension exists
            if z.dim() == 1:
                z   = z.unsqueeze(0)
                pos = pos.unsqueeze(0)
                if cell is not None:
                    cell = cell.unsqueeze(0)

            batch_size, n_atoms = z.shape
            molecule_mask = torch.ones(batch_size, n_atoms, dtype=torch.bool, device=device)

            if cell is not None:
                cell = cell.to(device)

            try:
                vision_outputs = raw_model.visual(
                    atomic_numbers=z,
                    coordinates=pos,
                    molecule_mask=molecule_mask,
                    cell=cell,
                )
                mol_embeds_raw        = vision_outputs[0]
                mol_embeds_projected  = raw_model.molecule_projector(mol_embeds_raw)

                source_features       = mol_embeds_projected[molecule_mask]
                image_mask            = (input_ids == IMAGE_TOKEN_INDEX)
                image_mask_expanded   = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds         = inputs_embeds.masked_scatter(
                    image_mask_expanded, source_features
                )
            except Exception as e:
                logger.warning(f"_build_inputs_embeds: molecular feature fusion failed; using text-only embeddings: {e}")

        return inputs_embeds

    # ------------------------------------------------------------------
    # Energy prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_energy(
        self,
        cif_text: str,
        prompt_text: str,
        device: torch.device,
    ) -> Optional[float]:
        """
        Run full multimodal inference on generated CIF to predict energy (eV/atom).

        Workflow:
          1. Extract 3D structure from CIF (z, pos, cell)
          2. Build energy-prediction prompt (text + 3D structure)
          3. Manually fuse molecule features into inputs_embeds
          4. Call generate(inputs_embeds=...) without passing molecule_data
          5. Parse floating-point number from output text with regex

        Returns:
            predicted energy (float), or None on failure
        """
        mol_data = extract_3d_from_cif(cif_text)
        if mol_data is None:
            logger.debug("_predict_energy: CIF parsing failed; skipping energy prediction")
            return None

        # Build energy-prediction prompt
        energy_prompt = (
            "Given the following crystal structure, predict the formation energy "
            "in eV/atom. Output only the numerical value.\n\n"
            f"Crystal structure (CIF):\n{cif_text[:500]}\n\n"
            "Formation energy (eV/atom):"
        )

        try:
            enc = self._tokenizer(
                energy_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.irft_args.model_max_length - self.max_e_tok,
                padding=False,
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Get bare model (remove DDP/PEFT wrappers)
            raw_model = self.model
            if hasattr(raw_model, "module"):
                raw_model = raw_model.module
            # unwrap PEFT wrapper one more layer
            base_model = getattr(raw_model, "base_model", raw_model)
            base_model = getattr(base_model, "model", base_model)

            raw_model.eval()

            # Manually build inputs_embeds (fuse molecule features)
            # mol_data: {z:[N], pos:[N,3], cell:[3,3]}, needs batch dimension
            mol_batch = {
                "z":   mol_data["z"].unsqueeze(0).to(device),
                "pos": mol_data["pos"].unsqueeze(0).to(device),
            }
            if "cell" in mol_data:
                mol_batch["cell"] = mol_data["cell"].unsqueeze(0).to(device)

            inputs_embeds = self._build_inputs_embeds(
                base_model, input_ids, mol_batch, device
            )

            output_ids = raw_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.max_e_tok,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

            raw_model.train()

            # generate() when passing inputs_embeds, output_ids contains only newly generated tokens
            pred_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            energy = parse_energy_from_text(pred_text)

            logger.debug(f"_predict_energy: output='{pred_text}' -> parse={energy}")
            return energy

        except Exception as e:
            logger.warning(f"_predict_energy failed: {e}")
            return None

    # ------------------------------------------------------------------
    # energy reward
    # ------------------------------------------------------------------

    def _compute_energy_reward(
        self,
        e_pred: Optional[float],
        e_target: Optional[float],
    ) -> float:
        """
        R_energy = exp(-λ · |E_pred - E_target|)

        If predicted or target energy is missing, return 0.0.
        """
        if e_pred is None or e_target is None:
            return 0.0
        diff = abs(e_pred - e_target)
        return math.exp(-self.lam * diff)

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        exemplar: Optional[Dict[str, Any]],
        device: torch.device,
    ) -> List[str]:
        """
        Generate K CIF candidates.

        If exemplar is not None, fuse its 3D structure into inputs_embeds and pass it to generate().
        note:Generate K candidates regardless of whether an exemplar exists.

        Returns:
            List[str], lengthas K (failed positions are empty strings)
        """
        raw_model = self.model
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
        # unwrap PEFT wrapper one more layer, for _build_inputs_embeds
        base_model = getattr(raw_model, "base_model", raw_model)
        base_model = getattr(base_model, "model", base_model)
        raw_model.eval()

        # Prepare exemplar mol_data (if any)
        exemplar_mol = None
        if exemplar is not None:
            mol_data = extract_3d_from_cif(exemplar.get("cif_text", ""))
            if mol_data is not None:
                exemplar_mol = {
                    "z":   mol_data["z"].unsqueeze(0).to(device),
                    "pos": mol_data["pos"].unsqueeze(0).to(device),
                }
                if "cell" in mol_data:
                    exemplar_mol["cell"] = mol_data["cell"].unsqueeze(0).to(device)

        try:
            candidates = []
            # Generate one by one (build independent inputs_embeds each time) to avoid inconsistent batch dimensions
            for _ in range(self.num_samples):
                # Build inputs_embeds (fuse exemplar molecule features if any)
                inputs_embeds = self._build_inputs_embeds(
                    base_model,
                    prompt_input_ids,   # [1, seq_len]
                    exemplar_mol,
                    device,
                )
                # attention_mask align with inputs_embeds
                attn_mask = prompt_attention_mask  # [1, seq_len]

                output_ids = raw_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
                # when passing inputs_embeds output_ids contains only newly generated tokens
                text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
                candidates.append(text)

                del inputs_embeds, output_ids
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"_generate_candidates failed: {e}")
            candidates = [""] * self.num_samples
        finally:
            raw_model.train()
            torch.cuda.empty_cache()
            gc.collect()

        return candidates

    # ------------------------------------------------------------------
    # Candidate scoring
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: List[str],
        expected_atoms: Dict,
        prompt_text: str,
        device: torch.device,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute R_step3 for each candidate.

        Returns:
            r_step3_list  : List[float]  composite reward
            r_struct_list : List[float]  structure reward (R_step2)
            r_energy_list : List[float]  energy reward
        """
        e_target = extract_target_energy(prompt_text)

        r_step3_list, r_struct_list, r_energy_list = [], [], []

        for cif_text in candidates:
            if not cif_text.strip():
                r_step3_list.append(-10.0)
                r_struct_list.append(-10.0)
                r_energy_list.append(0.0)
                continue

            # structure reward (reuse Stage 2 reward model)
            try:
                result   = self.reward_model.compute_single_reward(
                    cif_text, expected_atoms, return_details=False
                )
                r_struct = result if isinstance(result, float) else result.get("total_reward", 0.0)
            except Exception:
                r_struct = 0.0

            # energy reward (full multimodal inference)
            e_pred   = self._predict_energy(cif_text, prompt_text, device)
            r_energy = self._compute_energy_reward(e_pred, e_target)

            r_step3 = self.w_struct * r_struct + self.w_energy * r_energy

            r_step3_list.append(r_step3)
            r_struct_list.append(r_struct)
            r_energy_list.append(r_energy)

        return r_step3_list, r_struct_list, r_energy_list


    # ------------------------------------------------------------------
    # training_step
    # ------------------------------------------------------------------

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Stage 3 IRFT training step.

        1. Randomly take 1 exemplar from ExemplarPool (empty at step 1, skip)
        2. Generate K CIF candidates (with or without exemplar)
        3. compute R_step3 = ω₁·R_step2 + ω₂·R_energy
        4. Compute GRPO policy-gradient loss
        5. Add best candidate to ExemplarPool
        """
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        input_ids          = inputs.get("input_ids")
        attention_mask     = inputs.get("attention_mask")
        prompts            = inputs.get("prompts", [])
        expected_atoms_list = inputs.get("expected_atoms_list", [])
        device             = input_ids.device

        if rank == 0:
            logger.info(f"\n{'='*70}")
            logger.info(f"[Stage3] Step {self.global_step_counter} | batch={len(prompts)} | K={self.num_samples}")
            pool_stats = self.pool.get_stats() if self.pool else {}
            logger.info(f"[Stage3] ExemplarPool: {pool_stats}")
            logger.info(f"{'='*70}")

        # Get bare model
        actual_model = model.module if hasattr(model, "module") else model

        total_loss       = torch.tensor(0.0, device=device, requires_grad=True)
        num_valid        = 0
        all_r_step3      = []

        for batch_idx in range(len(prompts)):
            prompt          = prompts[batch_idx]
            expected_atoms  = expected_atoms_list[batch_idx]
            pid_single      = input_ids[batch_idx:batch_idx+1]
            pmask_single    = attention_mask[batch_idx:batch_idx+1]

            # ── 1. from ExemplarPool randomly take exemplar ──────────────────────
            exemplar = None
            if self.pool and not self.pool.is_empty():
                exemplar = self.pool.sample_random()
                if rank == 0:
                    logger.info(
                        f"  [Exemplar] reward={exemplar['reward']:.4f}, "
                        f"step={exemplar.get('step', '?')}"
                    )
            else:
                if rank == 0:
                    logger.info("  [Exemplar] pool is empty, no exemplar in first round")

            # ── 2. generate K Candidate ─────────────────────────────────────────
            t0 = time.time()
            candidates = self._generate_candidates(
                pid_single, pmask_single, exemplar, device
            )
            gen_time = time.time() - t0
            if rank == 0:
                logger.info(f"  generate {len(candidates)} candidates, elapsed {gen_time:.1f}s")

            # ── 3. Scoring ──────────────────────────────────────────────────
            r_step3_list, r_struct_list, r_energy_list = self._score_candidates(
                candidates, expected_atoms, prompt, device
            )
            all_r_step3.extend(r_step3_list)

            if rank == 0:
                for i, (r3, rs, re) in enumerate(
                    zip(r_step3_list, r_struct_list, r_energy_list)
                ):
                    logger.info(
                        f"  Candidate {i+1}: R_step3={r3:+.4f} "
                        f"(struct={rs:+.4f}, energy={re:.4f})"
                    )

            # ── 4. GRPO Policy-gradient loss ──────────────────────────────────────
            mean_r = sum(r_step3_list) / max(len(r_step3_list), 1)
            if len(r_step3_list) > 1:
                std_r = (
                    sum((r - mean_r) ** 2 for r in r_step3_list) / len(r_step3_list)
                ) ** 0.5
                std_r = max(std_r, 1e-8)
            else:
                std_r = 1.0

            for cif_text, r_step3 in zip(candidates, r_step3_list):
                if not cif_text.strip():
                    continue
                try:
                    advantage = (r_step3 - mean_r) / std_r
                    full_text = prompt + " " + cif_text
                    safe_len  = min(
                        self.max_new_tokens + 1024,
                        self.irft_args.model_max_length,
                    )
                    enc = self._tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=safe_len,
                    )
                    ids_full  = enc["input_ids"].to(device)
                    mask_full = enc["attention_mask"].to(device)

                    with torch.set_grad_enabled(True):
                        out    = actual_model(
                            input_ids=ids_full,
                            attention_mask=mask_full,
                        )
                        logits = out.logits
                        del out

                    # prompt length
                    p_enc    = self._tokenizer(
                        prompt,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    p_len    = p_enc["input_ids"].shape[1]

                    resp_logits = logits[0, p_len - 1:-1, :]
                    resp_ids    = ids_full[0, p_len:]

                    # Compute log-prob in chunks
                    chunk = 512
                    lp_chunks = []
                    for ci in range(0, resp_logits.shape[0], chunk):
                        ce = min(ci + chunk, resp_logits.shape[0])
                        cl = resp_logits[ci:ce]
                        ct = resp_ids[ci:ce]
                        lp = torch.nn.functional.log_softmax(cl, dim=-1)
                        lp = lp.gather(-1, ct.unsqueeze(-1)).squeeze(-1)
                        lp_chunks.append(lp)
                        del cl, lp
                    token_lp  = torch.cat(lp_chunks, dim=0)
                    seq_lp    = token_lp.sum()
                    seq_len   = max(resp_ids.shape[0], 1)

                    del logits, resp_logits, resp_ids, token_lp
                    del ids_full, mask_full
                    torch.cuda.empty_cache()
                    gc.collect()

                    norm_lp   = seq_lp / seq_len
                    loss_i    = -advantage * norm_lp
                    total_loss = total_loss + loss_i
                    num_valid += 1

                except Exception as e:
                    logger.warning(f"  Policy-gradient computation failed: {e}")
                    continue
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            # ── 5. Update ExemplarPool ──────────────────────────────────────
            if self.pool is not None and candidates:
                best_idx = int(
                    max(range(len(r_step3_list)), key=lambda i: r_step3_list[i])
                )
                best_cif    = candidates[best_idx]
                best_reward = r_step3_list[best_idx]
                if best_cif.strip():
                    added = self.pool.add(
                        cif_text=best_cif,
                        reward=best_reward,
                        prompt=prompt,
                        step=self.global_step_counter,
                    )
                    if rank == 0 and added:
                        logger.info(
                            f"  [Pool] added best candidate reward={best_reward:.4f}, "
                            f"pool size={len(self.pool)}"
                        )

        # ── Summary ─────────────────────────────────────────────────────────
        if num_valid > 0:
            loss = total_loss / num_valid
        else:
            logger.warning("[Stage3] No valid samples, useplaceholderloss")
            loss = torch.tensor(0.01, device=device, requires_grad=True)

        if rank == 0 and all_r_step3:
            avg_r = sum(all_r_step3) / len(all_r_step3)
            logger.info(
                f"[Stage3] Loss={loss.item():.4f} | "
                f"avg_R_step3={avg_r:+.4f} | valid={num_valid}"
            )

        # Periodically save ExemplarPool
        if (
            self.pool is not None
            and rank == 0
            and self.global_step_counter % 50 == 0
        ):
            self.pool.save()

        del all_r_step3
        torch.cuda.empty_cache()
        gc.collect()

        self.global_step_counter += 1
        return loss


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_irft_stage3():
    """Stage 3 IRFT main training entry point."""

    parser = HfArgumentParser((ModelArguments, DataArguments, IRFTArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ── Loading model ──────────────────────────────────────────────────────────
    rank0_print("=" * 80)
    rank0_print(f"[Stage3] Loading model: {model_args.model_name_or_path}")
    rank0_print("=" * 80)

    model = Qwen2_5_VLForMolecule.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )
    model.config.use_cache = False
    set_model(model_args, model)

    # ── LoRA ──────────────────────────────────────────────────────────────
    if model_args.use_lora:
        rank0_print("[Stage3] Enable LoRA ...")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        rank0_print("[Stage3] LoRA parameter statistics:")
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # ── Tokenizer / Processor ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        use_fast=False,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    rank0_print(f"[Stage3] loaddata: {data_args.data_path}")
    train_dataset = GRPODataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
    )
    data_collator = GRPODataCollator(
        tokenizer=tokenizer,
        processor=processor,
        max_length=training_args.model_max_length,
    )

    # ── ExemplarPool ──────────────────────────────────────────────────────
    pool = ExemplarPool(
        max_size=training_args.exemplar_pool_size,
        pool_path=training_args.exemplar_pool_path,
    )

    # ── rewardmodel ──────────────────────────────────────────────────────────
    reward_model = CIFRewardModel()

    # ── trainer ────────────────────────────────────────────────────────────
    trainer = IRFTTrainer(
        model=model,
        irft_args=training_args,
        reward_model=reward_model,
        exemplar_pool=pool,
        tokenizer=tokenizer,
        processor=processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ── training ──────────────────────────────────────────────────────────────
    rank0_print("[Stage3] Start IRFT training ...")
    trainer.train()

    # ── save ──────────────────────────────────────────────────────────────
    rank0_print("[Stage3] Training complete, saving model ...")
    trainer.save_model()
    trainer.save_state()

    # Save final ExemplarPool
    if training_args.exemplar_pool_path:
        pool.save(training_args.exemplar_pool_path)
    else:
        pool.save(os.path.join(training_args.output_dir, "exemplar_pool_final.json"))

    rank0_print(f"[Stage3] LoRA adapter saved to: {training_args.output_dir}")
    rank0_print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    print("=" * 80)
    print("Stage 3: IRFT (Iterative Reward Fine-Tuning)")
    print("GP-GRPO: R_step3 = ω₁·R_step2 + ω₂·R_energy")
    print("=" * 80)
    train_irft_stage3()
