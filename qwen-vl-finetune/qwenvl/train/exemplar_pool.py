#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExemplarPool — Stage 3 IRFT exemplar pool
Maintain a fixed-size pool of high-quality CIF samples for iterative reinforcement fine-tuning.

Design principles:
- Fixed capacity; randomly replace when exceeded (non-FIFO, preserves diversity)
- Random sampling (not similarity retrieval) to avoid retrieval bias
- JSON persistence supports resuming training
- Extract 3D structures from CIF text (z, pos, cell), for use by the multimodal encoder
"""

import os
import json
import random
import logging
import re
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_3d_from_cif(cif_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract 3D structure information from CIF text and return the format required by the multimodal encoder.

    Returns:
        dict with keys:
            z    : torch.LongTensor  [N]        atomic numbers
            pos  : torch.FloatTensor [N, 3]     fractional coordinates -> Cartesian coordinates
            cell : torch.FloatTensor [3, 3]     cell matrix (row vectors)
        or None (parsing failed)
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifParser
        import io

        parser = CifParser(io.StringIO(cif_text))
        structures = parser.get_structures(primitive=False)
        if not structures:
            return None
        structure = structures[0]

        # atomic numbers
        z = torch.tensor(
            [site.specie.Z for site in structure],
            dtype=torch.long
        )

        # Cartesian coordinates (Å)
        pos = torch.tensor(
            [site.coords.tolist() for site in structure],
            dtype=torch.float32
        )

        # cell matrix (row vector, shape [3,3])
        cell = torch.tensor(
            structure.lattice.matrix.tolist(),
            dtype=torch.float32
        )

        return {"z": z, "pos": pos, "cell": cell}

    except Exception as e:
        logger.debug(f"pymatgen failed to parse CIF; trying manual parsing: {e}")
        return _extract_3d_manually(cif_text)


def _extract_3d_manually(cif_text: str) -> Optional[Dict[str, Any]]:
    """
    Manual CIF parsing fallback (does not depend on pymatgen).
    Only extracts lattice parameters and atomic coordinates; accuracy is limited.
    """
    try:
        # Parse lattice parameters
        a = _parse_cif_float(cif_text, r"_cell_length_a\s+([\d.]+)")
        b = _parse_cif_float(cif_text, r"_cell_length_b\s+([\d.]+)")
        c = _parse_cif_float(cif_text, r"_cell_length_c\s+([\d.]+)")
        alpha = _parse_cif_float(cif_text, r"_cell_angle_alpha\s+([\d.]+)", default=90.0)
        beta  = _parse_cif_float(cif_text, r"_cell_angle_beta\s+([\d.]+)",  default=90.0)
        gamma = _parse_cif_float(cif_text, r"_cell_angle_gamma\s+([\d.]+)", default=90.0)

        if a is None or b is None or c is None:
            return None

        cell = _lattice_to_matrix(a, b, c, alpha, beta, gamma)

        # Parse atomic coordinates (fractional coordinates)
        atom_pattern = re.compile(
            r"^\s*(\w+)\s+\w+\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)",
            re.MULTILINE
        )
        matches = atom_pattern.findall(cif_text)
        if not matches:
            return None

        element_to_z = _get_element_z_map()
        z_list, pos_list = [], []
        for symbol, fx, fy, fz in matches:
            # Remove trailing digits (e.g. Fe1 -> Fe)
            elem = re.sub(r"\d+$", "", symbol).capitalize()
            z_val = element_to_z.get(elem)
            if z_val is None:
                continue
            frac = np.array([float(fx), float(fy), float(fz)])
            cart = frac @ cell  # fractional -> Cartesian
            z_list.append(z_val)
            pos_list.append(cart.tolist())

        if not z_list:
            return None

        return {
            "z":    torch.tensor(z_list, dtype=torch.long),
            "pos":  torch.tensor(pos_list, dtype=torch.float32),
            "cell": torch.tensor(cell.tolist(), dtype=torch.float32),
        }

    except Exception as e:
        logger.debug(f"Manual CIF parsing failed: {e}")
        return None


def _parse_cif_float(text: str, pattern: str, default=None) -> Optional[float]:
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


def _lattice_to_matrix(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Convert lattice parameters to a 3x3 matrix (row vectors)."""
    import math
    alpha_r = math.radians(alpha)
    beta_r  = math.radians(beta)
    gamma_r = math.radians(gamma)

    cos_a, cos_b, cos_g = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
    sin_g = math.sin(gamma_r)

    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = math.sqrt(max(c**2 - cx**2 - cy**2, 0.0))

    return np.array([
        [a,          0.0,  0.0],
        [b * cos_g,  b * sin_g, 0.0],
        [cx,         cy,   cz],
    ], dtype=np.float64)


def _get_element_z_map() -> Dict[str, int]:
    """Return a mapping from common element symbols to atomic numbers."""
    elements = [
        "H","He","Li","Be","B","C","N","O","F","Ne",
        "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
        "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
        "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
        "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
        "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
        "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
        "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
        "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
        "Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    ]
    return {sym: idx + 1 for idx, sym in enumerate(elements)}


def extract_target_energy(prompt_text: str) -> Optional[float]:
    """
    Extract target energy value from prompt text (eV/atom).
    Supports multiple formats:
      - "energy: -3.45 eV/atom"
      - "target energy = -3.45"
      - "E = -3.45 eV"
    """
    patterns = [
        r"energy[:\s=]+(-?\d+\.?\d*(?:e[+-]?\d+)?)\s*eV",
        r"E[_\s]*=\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)",
        r"(-?\d+\.?\d+)\s*eV/atom",
        r"formation[_\s]*energy[:\s=]+(-?\d+\.?\d*(?:e[+-]?\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, prompt_text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# ExemplarPool
# ---------------------------------------------------------------------------

class ExemplarPool:
    """
    Fixed-size high-quality CIF exemplar pool.

    Each exemplar stores:
        cif_text   : str    — generateof CIF text
        reward     : float  — corresponding R_step3 reward
        prompt     : str    — original prompt (optional, for debugging)
        step       : int    — training step when added

    Usage:
        pool = ExemplarPool(max_size=50, pool_path="pool.json")
        pool.add(cif_text, reward, prompt, step)
        exemplar = pool.sample_random()   # returns one exemplar dict or None
        pool.save()
    """

    def __init__(
        self,
        max_size: int = 50,
        pool_path: Optional[str] = None,
        min_reward_threshold: float = 0.0,
    ):
        """
        Args:
            max_size: maximum pool capacity
            pool_path: JSON persistence path; automatically loaded if the file exists
            min_reward_threshold: samples below this reward are not added to the pool
        """
        self.max_size = max_size
        self.pool_path = pool_path
        self.min_reward_threshold = min_reward_threshold
        self._pool: List[Dict[str, Any]] = []

        if pool_path and os.path.exists(pool_path):
            self.load(pool_path)
            logger.info(f"ExemplarPool: from {pool_path} load {len(self._pool)} exemplars")
        else:
            logger.info(f"ExemplarPool: initialized empty pool (max_size={max_size})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pool)

    def is_empty(self) -> bool:
        return len(self._pool) == 0

    def add(
        self,
        cif_text: str,
        reward: float,
        prompt: str = "",
        step: int = 0,
    ) -> bool:
        """
        Try to add a CIF sample to the pool.

        Policy:
        - If reward is below threshold, reject directly
        - If the pool is not full, add directly
        - If the pool is full, randomly replace an existing sample (preserve diversity)

        Returns:
            True means added successfully, False means rejected
        """
        if reward < self.min_reward_threshold:
            return False

        entry = {
            "cif_text": cif_text,
            "reward":   reward,
            "prompt":   prompt,
            "step":     step,
        }

        if len(self._pool) < self.max_size:
            self._pool.append(entry)
            logger.debug(f"ExemplarPool: added new sample reward={reward:.4f}, pool size={len(self._pool)}")
            return True
        else:
            # Random replacement
            idx = random.randint(0, len(self._pool) - 1)
            old_reward = self._pool[idx]["reward"]
            self._pool[idx] = entry
            logger.debug(
                f"ExemplarPool: Random replacement idx={idx} (old reward={old_reward:.4f} -> new reward={reward:.4f})"
            )
            return True

    def sample_random(self) -> Optional[Dict[str, Any]]:
        """
        Randomly sample one exemplar from the pool.

        Returns:
            exemplar dict (contains cif_text, reward, prompt, step)or None (pool is empty)
        """
        if self.is_empty():
            return None
        return random.choice(self._pool)

    def sample_random_batch(self, k: int) -> List[Dict[str, Any]]:
        """
        Randomly sample k exemplars from the pool (with replacement).
        If pool size < k, sample repeatedly.
        """
        if self.is_empty():
            return []
        return random.choices(self._pool, k=k)

    def get_stats(self) -> Dict[str, Any]:
        """Return pool statistics."""
        if self.is_empty():
            return {"size": 0, "max_size": self.max_size}
        rewards = [e["reward"] for e in self._pool]
        return {
            "size":       len(self._pool),
            "max_size":   self.max_size,
            "avg_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "std_reward": float(np.std(rewards)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Save the pool as a JSON file."""
        save_path = path or self.pool_path
        if save_path is None:
            logger.warning("ExemplarPool.save(): No path specified; skipping save")
            return

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # torch.Tensor is not JSON-serializable; the pool only stores strings/numbers, so no special handling is needed
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "max_size":              self.max_size,
                    "min_reward_threshold":  self.min_reward_threshold,
                    "pool":                  self._pool,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"ExemplarPool: saved {len(self._pool)} exemplarsto {save_path}")

    def load(self, path: str) -> None:
        """Load pool from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.max_size             = data.get("max_size", self.max_size)
        self.min_reward_threshold = data.get("min_reward_threshold", self.min_reward_threshold)
        self._pool                = data.get("pool", [])
        logger.info(f"ExemplarPool: from {path} load complete, total {len(self._pool)} exemplars")
