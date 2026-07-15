#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward model for CIF file generation tasks
Core function: evaluate whether the atomic composition in generated CIF files exactly matches expectations

Key checks:
1. Exact atomic-composition match (60%weight)- the count of each element must be exactly correct
2. CIF file parseability (20%weight)
3. structure validity (10%weight)
4. physical plausibility (10%weight)
"""

import re
import numpy as np
from typing import Dict, Optional, List
import torch
import torch.nn as nn

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.core import Structure, Element
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    Structure = None  # define placeholder to avoid type-annotation errors
    Element = None
    print("Warning: pymatgen not installed. Advanced validation will be limited.")


class CIFRewardModel(nn.Module):
    """
    Reward model for CIF file generation tasks
    
    Core goal: ensure the atomic composition in generated CIF files exactly matches the input
    For example:inputexpected Ca8Ga16H4C2O1, then the generated CIF must contain:
    - 8Ca atoms (not 7 or 9)
    - 16Ga atoms
    - 4H atoms
    - 2C atoms
    - 1O atoms
    """
    
    def __init__(self):
        super().__init__()
        
        # Reward weight configuration
        self.weights = {
            "atom_composition": 0.60,    # atomic composition match - most important!
            "parseability": 0.20,        # parseability
            "structure_validity": 0.10,  # structure validity
            "physical": 0.10,            # physical plausibility
        }
        
        # atomic radius database (for physical-plausibility checks)
        # Use pymatgen to obtain atomic radii for all elements (covering the first 118 elements of the periodic table)
        self.atomic_radii = self._build_atomic_radii_database()
    
    def _build_atomic_radii_database(self) -> Dict[str, float]:
        """
        Build a complete atomic radius database
        Use pymatgen Element class to get atomic radii for all elements
        
        Returns:
            Dict[str, float]: mapping from element symbol to atomic radius (Å)
        """
        radii = {}
        
        if PYMATGEN_AVAILABLE and Element is not None:
            # Use pymatgen to getatomic radii of the first 118 elements in the periodic table
            for z in range(1, 119):  # atomic numbers 1-118
                try:
                    element = Element.from_Z(z)
                    symbol = element.symbol
                    
                    # Priority: atomic radius > van der Waals radius > default value
                    if element.atomic_radius is not None:
                        radii[symbol] = element.atomic_radius
                    elif element.van_der_waals_radius is not None:
                        radii[symbol] = element.van_der_waals_radius
                    else:
                        # Use empirical estimates (based on periodic trends)
                        radii[symbol] = 1.5
                except Exception:
                    continue
            
            print(f"✓ Successfully loaded from pymatgen {len(radii)} element atomic-radius entries")
        else:
            # Fallback: manually define atomic radii for common elements
            print("⚠ pymatgen unavailable; using fallback atomic-radius database (covers about 70 common elements)")
            radii = {
                # Period 1
                'H': 0.31, 'He': 0.28,
                
                # Period 2
                'Li': 1.45, 'Be': 1.05, 'B': 0.85, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
                
                # Period 3
                'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
                
                # Period 4 (3d transition metals)
                'K': 2.03, 'Ca': 1.76, 'Sc': 1.62, 'Ti': 1.40, 'V': 1.35, 'Cr': 1.40, 
                'Mn': 1.40, 'Fe': 1.40, 'Co': 1.35, 'Ni': 1.35, 'Cu': 1.32, 'Zn': 1.22,
                'Ga': 1.22, 'Ge': 1.25, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
                
                # Period 5 (4d transition metals)
                'Rb': 2.20, 'Sr': 1.91, 'Y': 1.62, 'Zr': 1.55, 'Nb': 1.45, 'Mo': 1.45,
                'Tc': 1.35, 'Ru': 1.30, 'Rh': 1.35, 'Pd': 1.40, 'Ag': 1.45, 'Cd': 1.44,
                'In': 1.42, 'Sn': 1.45, 'Sb': 1.45, 'Te': 1.40, 'I': 1.39, 'Xe': 1.40,
                
                # Period 6 (5d transition metals + lanthanides)
                'Cs': 2.44, 'Ba': 1.98, 'La': 1.69, 'Ce': 1.65, 'Pr': 1.65, 'Nd': 1.64,
                'Sm': 1.62, 'Eu': 1.85, 'Gd': 1.61, 'Tb': 1.59, 'Dy': 1.59, 'Ho': 1.58,
                'Er': 1.57, 'Tm': 1.56, 'Yb': 1.74, 'Lu': 1.56,
                'Hf': 1.55, 'Ta': 1.45, 'W': 1.35, 'Re': 1.35, 'Os': 1.30, 'Ir': 1.35,
                'Pt': 1.35, 'Au': 1.35, 'Hg': 1.49, 'Tl': 1.48, 'Pb': 1.54, 'Bi': 1.55,
                
                # actinides (nuclear-material related)
                'Th': 1.65, 'U': 1.42, 'Np': 1.50, 'Pu': 1.50,
            }
        
        return radii
    
    def compute_single_reward(self, 
                             generated_cif: str,
                             expected_atoms: Dict[str, int] = None,
                             ground_truth_cif: str = None,
                             return_details: bool = True) -> Dict:
        """
        Compute the reward score for one CIF file
        
        Args:
            generated_cif: generated CIF file text
            expected_atoms: expected atomic composition, For example {"Ca": 8, "Ga": 16, "H": 4, "C": 2, "O": 1}
                          if None, ground_truth_cif must be provided
            ground_truth_cif: ground-truth CIF text (optional)
                            if provided, expected_atoms will be extracted automatically
            return_details: whether to return details
        
        Returns:
            {
                "total_reward": 6.5,           # total reward score (-20.0 to +10.0)
                "category": "good",            # category
                "sub_scores": {                # scores for each dimension
                    "atom_composition": 8.0,
                    "parseability": 10.0,
                    "structure_validity": 5.0,
                    "physical": 7.0
                },
                "details": {                   # details (optional)
                    "actual_atoms": {...},
                    "composition": {...},
                    ...
                }
            }
        
        Note:
            Using ground_truth_cif is recommended because it can automatically and accurately extract the expected atomic composition.
        """
        # Extract expected atoms from ground_truth_cif (if provided)
        if ground_truth_cif is not None:
            expected_atoms = extract_expected_atoms_from_cif(ground_truth_cif)
        
        # Check whether expected values exist
        if expected_atoms is None or len(expected_atoms) == 0:
            return {
                "total_reward": -20.0,
                "category": "very_poor",
                "sub_scores": {
                    "atom_composition": -20.0,
                    "parseability": -20.0,
                    "structure_validity": -20.0,
                    "physical": -20.0
                },
                "details": {"error": "No expected_atoms provided and no ground_truth_cif"} if return_details else None
            }
        
        result = {
            "total_reward": -20.0,
            "category": "very_poor",
            "sub_scores": {},
            "details": {} if return_details else None
        }
        
        # ========== Step1:try parsing CIF file ==========
        if PYMATGEN_AVAILABLE:
            structure, parse_success = self._parse_cif_with_pymatgen(generated_cif)
        else:
            structure, parse_success = self._parse_cif_manually(generated_cif)
        
        if not parse_success:
            # unparseable -> minimum score for all dimensions
            result["sub_scores"]["parseability"] = -20.0
            result["sub_scores"]["atom_composition"] = -15.0
            result["sub_scores"]["structure_validity"] = -10.0
            result["sub_scores"]["physical"] = -10.0
            result["total_reward"] = -20.0
            result["category"] = "unparseable"
            if return_details:
                result["details"]["parseable"] = False
                result["details"]["error"] = "CIF parsing failed"
            return result
        
        result["sub_scores"]["parseability"] = 10.0
        if return_details:
            result["details"]["parseable"] = True
        
        # ========== Step2:extract actual atomic composition ==========
        if PYMATGEN_AVAILABLE and structure is not None:
            actual_atoms = self._extract_atoms_from_structure(structure)
        else:
            actual_atoms = self._extract_atoms_manually(generated_cif)
        
        if return_details:
            result["details"]["actual_atoms"] = actual_atoms
            result["details"]["expected_atoms"] = expected_atoms
        
        # ========== Step3:exact atomic-composition match (core)==========
        composition_result = self._evaluate_atom_composition(
            actual_atoms, expected_atoms
        )
        result["sub_scores"]["atom_composition"] = composition_result["score"]
        if return_details:
            result["details"]["composition"] = composition_result
        
        # ========== Step4:structure validity check ==========
        if PYMATGEN_AVAILABLE and structure is not None:
            validity_result = self._evaluate_structure_validity(structure)
        else:
            validity_result = self._evaluate_structure_validity_manually(generated_cif, actual_atoms)
        result["sub_scores"]["structure_validity"] = validity_result["score"]
        if return_details:
            result["details"]["validity"] = validity_result
        
        # ========== Step5:physical plausibility check ==========
        if PYMATGEN_AVAILABLE and structure is not None:
            physics_result = self._evaluate_physical(structure)
        else:
            physics_result = self._evaluate_physical_manually(generated_cif, actual_atoms)
        result["sub_scores"]["physical"] = physics_result["score"]
        if return_details:
            result["details"]["physics"] = physics_result
        
        # ========== Step6:compute weighted total score ==========
        total_reward = sum(
            self.weights[key] * result["sub_scores"][key]
            for key in self.weights.keys()
        )
        
        result["total_reward"] = total_reward
        result["category"] = self._categorize(total_reward)
        
        return result
    
    # ==================== Parsing-related functions ====================
    
    def _parse_cif_with_pymatgen(self, cif_text: str):
        """Parse CIF file with pymatgen"""
        try:
            parser = CifParser.from_string(cif_text)
            structure = parser.get_structures()[0]
            return structure, True
        except Exception as e:
            return None, False
    
    def _parse_cif_manually(self, cif_text: str):
        """Manually parse CIF file (fallback)"""
        try:
            # Check basic CIF format
            if '_cell_length_a' in cif_text and '_atom_site_' in cif_text:
                return cif_text, True
            else:
                return None, False
        except:
            return None, False
    
    # ==================== Atomic-composition extraction functions ====================
    
    def _extract_atoms_from_structure(self, structure) -> Dict[str, int]:
        """Extract atomic composition from pymatgen Structure object"""
        atoms = {}
        for site in structure:
            element = str(site.specie.symbol)
            atoms[element] = atoms.get(element, 0) + 1
        return atoms
    
    def _extract_atoms_manually(self, cif_text: str) -> Dict[str, int]:
        """
        Extract actual atom counts from CIF text
        
        Only count atoms listed in the _atom_site section (atoms with 3D coordinates)
        Keep consistent with extract_atoms_from_cif in clean_cif_data.py
        
        return:{'P': 36, 'Rh': 54, 'C': 2, 'O': 1, ...}
        """
        atoms = {}
        lines = cif_text.split('\n')
        in_atom_block = False
        
        for line in lines:
            line = line.strip()
            
            # Detect start of atom_site block
            if '_atom_site_type_symbol' in line or '_atom_site_label' in line:
                in_atom_block = True
                continue
            
            if in_atom_block and line:
                # Detect end of atom_site block
                if line.startswith('_') or line.startswith('loop_'):
                    if '_atom_site_' not in line:
                        break
                    continue
                
                # Skip comments
                if line.startswith('#'):
                    continue
                
                # Parse atom row
                parts = line.split()
                if len(parts) >= 3:
                    # Extract element symbol (from first column)
                    match = re.match(r'([A-Z][a-z]?)', parts[0])
                    if match:
                        element = match.group(1)
                        atoms[element] = atoms.get(element, 0) + 1
        
        return atoms
    
    # ==================== Atomic-composition evaluation (core function)====================
    
    def _evaluate_atom_composition(self, 
                                   actual: Dict[str, int],
                                   expected: Dict[str, int]) -> Dict:
        """
        Evaluate atomic-composition match (core scoring function)
        
        Scoring logic:
        1. missing required elements -> -8.0 to -15.0  (serious error)
        2. extra elements present -> -5.0 to -15.0  (error)
        3. element types are correct but counts have errors -> score according to error magnitude:
           - perfect match (error=0%)-> +10.0
           - excellent (error<5%)-> +8.0
           - good (error<10%)-> +5.0
           - acceptable (error<20%)-> +2.0
           - poor (error<30%)-> -2.0
           - poor (error < 50%)-> -5.0
           - very poor (error>=50%)-> -10.0
        """
        expected_elements = set(expected.keys())
        actual_elements = set(actual.keys())
        
        # Check missing and extra elements
        missing = expected_elements - actual_elements
        extra = actual_elements - expected_elements
        
        # Case 1:missing required elements -> serious error
        if len(missing) > 0:
            score = -8.0 - len(missing) * 2.0
            return {
                "score": max(-15.0, score),
                "match_type": "missing_elements",
                "missing": list(missing),
                "extra": list(extra),
                "expected": expected,
                "actual": actual,
                "avg_error": 1.0
            }
        
        # Case 2:extra elements present -> error
        if len(extra) > 0:
            score = -5.0 - len(extra) * 2.0
            return {
                "score": max(-15.0, score),
                "match_type": "extra_elements",
                "missing": [],
                "extra": list(extra),
                "expected": expected,
                "actual": actual,
                "avg_error": 0.5
            }
        
        # Case 3:element types are fully correct; check counts for each element
        total_error = 0.0
        element_errors = {}
        
        for element, expected_count in expected.items():
            actual_count = actual.get(element, 0)
            
            if expected_count > 0:
                relative_error = abs(actual_count - expected_count) / expected_count
                total_error += relative_error
                element_errors[element] = {
                    "expected": expected_count,
                    "actual": actual_count,
                    "error": relative_error
                }
        
        # Compute average relative error
        avg_error = total_error / len(expected) if len(expected) > 0 else 0.0
        
        # Score according to average error
        if avg_error == 0.0:
            score = 10.0
            match_type = "perfect"
        elif avg_error < 0.05:  # < 5%
            score = 8.0
            match_type = "excellent"
        elif avg_error < 0.10:  # < 10%
            score = 5.0
            match_type = "good"
        elif avg_error < 0.20:  # < 20%
            score = 2.0
            match_type = "acceptable"
        elif avg_error < 0.30:  # < 30%
            score = -2.0
            match_type = "poor"
        elif avg_error < 0.50:  # < 50%
            score = -5.0
            match_type = "very_poor"
        else:  # >= 50%
            score = -10.0
            match_type = "disaster"
        
        return {
            "score": score,
            "match_type": match_type,
            "avg_error": avg_error,
            "element_errors": element_errors,
            "missing": [],
            "extra": [],
            "expected": expected,
            "actual": actual
        }
    
    # ==================== structure validity check ====================
    
    def _evaluate_structure_validity(self, structure) -> Dict:
        """Evaluate structure validity with pymatgen"""
        score = 10.0
        issues = []
        
        # 1. Check lattice parameters
        lattice = structure.lattice
        if lattice.volume < 1.0 or lattice.volume > 100000.0:
            score -= 3.0
            issues.append(f"abnormal cell volume: {lattice.volume:.2f} Å³")
        
        # 2. Check atom count
        num_atoms = len(structure)
        if num_atoms < 1:
            score -= 10.0
            issues.append("no atoms")
        elif num_atoms > 1000:
            score -= 2.0
            issues.append(f"too many atoms: {num_atoms}")
        
        # 3. Check fractional coordinate range
        invalid_coords = 0
        for site in structure:
            frac_coords = site.frac_coords
            if not all(-0.1 <= c <= 1.1 for c in frac_coords):  # allow small out-of-range values
                invalid_coords += 1
        
        if invalid_coords > 0:
            score -= min(3.0, invalid_coords * 0.5)
            issues.append(f"{invalid_coords} atoms have abnormal coordinates")
        
        # 4. Check density
        density = structure.density
        if density < 0.1 or density > 30.0:
            score -= 2.0
            issues.append(f"abnormal density: {density:.2f} g/cm³")
        
        return {
            "score": max(-10.0, score),
            "issues": issues,
            "num_atoms": num_atoms,
            "volume": lattice.volume,
            "density": density
        }
    
    def _evaluate_structure_validity_manually(self, cif_text: str, atoms: Dict[str, int]) -> Dict:
        """Manually evaluate structure validity (when pymatgen is unavailable)"""
        score = 10.0
        issues = []
        
        # Check whether required fields are present
        required_fields = ['_cell_length_a', '_cell_length_b', '_cell_length_c']
        for field in required_fields:
            if field not in cif_text:
                score -= 2.0
                issues.append(f"missing field: {field}")
        
        # Check atom count
        num_atoms = sum(atoms.values()) if atoms else 0
        if num_atoms < 1:
            score -= 10.0
            issues.append("no atoms")
        elif num_atoms > 1000:
            score -= 2.0
            issues.append(f"too many atoms: {num_atoms}")
        
        return {
            "score": max(-10.0, score),
            "issues": issues
        }
    
    # ==================== physical plausibility check ====================
    
    def _evaluate_physical(self, structure) -> Dict:
        """Evaluate physical plausibility with pymatgen"""
        score = 10.0
        issues = []
        
        # 1. Check minimum interatomic distance
        if len(structure) > 1:
            min_dist = float('inf')
            for i in range(min(len(structure), 50)):  # limit check range for speed
                for j in range(i+1, min(len(structure), 50)):
                    try:
                        dist = structure.get_distance(i, j)
                        min_dist = min(min_dist, dist)
                    except:
                        continue
            
            if min_dist < float('inf'):
                if min_dist < 0.5:  # atoms severely overlap
                    score -= 8.0
                    issues.append(f"atoms too close: {min_dist:.3f} Å")
                elif min_dist < 1.0:  # atoms close
                    score -= 3.0
                    issues.append(f"atoms close: {min_dist:.3f} Å")
        
        # 2. Check volume per atom
        volume_per_atom = structure.volume / len(structure) if len(structure) > 0 else 0
        if volume_per_atom < 3.0:
            score -= 4.0
            issues.append(f"volume per atom too small: {volume_per_atom:.2f} Å³")
        elif volume_per_atom > 150.0:
            score -= 2.0
            issues.append(f"volume per atom too large: {volume_per_atom:.2f} Å³")
        
        return {
            "score": max(-10.0, score),
            "issues": issues,
            "min_distance": min_dist if min_dist != float('inf') else None,
            "volume_per_atom": volume_per_atom
        }
    
    def _evaluate_physical_manually(self, cif_text: str, atoms: Dict[str, int]) -> Dict:
        """Manually evaluate physical plausibility (when pymatgen unavailable)"""
        score = 10.0
        issues = []
        
        # Basic check: atom-count plausibility
        num_atoms = sum(atoms.values()) if atoms else 0
        if num_atoms < 1:
            score -= 5.0
            issues.append("no atoms")
        
        return {
            "score": max(-10.0, score),
            "issues": issues
        }
    
    # ==================== Helper functions ====================
    
    def _categorize(self, total_reward: float) -> str:
        """Classify by total reward score"""
        if total_reward >= 9.0:
            return "excellent"
        elif total_reward >= 7.0:
            return "good"
        elif total_reward >= 4.0:
            return "acceptable"
        elif total_reward >= 0.0:
            return "poor"
        else:
            return "very_poor"
    
    # ==================== Batch processing ====================
    
    def compute_batch_rewards(self, 
                             generated_cifs: List[str],
                             expected_atoms_list: List[Dict[str, int]],
                             return_details: bool = False) -> Dict:
        """
        Compute reward scores in batch
        
        Args:
            generated_cifs: list of generated CIF files
            expected_atoms_list: list of expected atomic compositions
            return_details: whether to return details
        
        Returns:
            {
                "rewards": [6.5, 8.2, -5.0, ...],
                "mean_reward": 3.2,
                "categories": ["good", "excellent", "poor", ...],
                "details": [...] (optional)
            }
        """
        rewards = []
        categories = []
        details_list = [] if return_details else None
        
        for cif, expected in zip(generated_cifs, expected_atoms_list):
            result = self.compute_single_reward(cif, expected, return_details)
            rewards.append(result["total_reward"])
            categories.append(result["category"])
            if return_details:
                details_list.append(result["details"])
        
        return {
            "rewards": rewards,
            "mean_reward": np.mean(rewards) if rewards else 0.0,
            "categories": categories,
            "details": details_list
        }
    
    def forward(self, 
                generated_cifs: List[str],
                expected_atoms_list: List[Dict[str, int]]) -> torch.Tensor:
        """
        Forward pass, Return reward tensor (for training)
        
        Args:
            generated_cifs: list of generated CIF files
            expected_atoms_list: list of expected atomic compositions
        
        Returns:
            rewards_tensor: shape=(batch_size,)
        """
        batch_results = self.compute_batch_rewards(generated_cifs, expected_atoms_list, return_details=False)
        rewards = torch.tensor(batch_results["rewards"], dtype=torch.float32)
        return rewards


# ==================== Helper utility functions ====================

def extract_expected_atoms_from_cif(ground_truth_cif: str) -> Dict[str, int]:
    """
    Extract expected atomic composition from ground-truth CIF (recommended)
    
    This is the most accurate method; it extracts actual atom counts directly from the ground-truth CIF.
    and use the same logic as extract_atoms_from_cif in clean_cif_data.py.
    
    Args:
        ground_truth_cif: ground-truth CIF text
    
    Returns:
        {"Ca": 8, "Ga": 16, "H": 4, "C": 2, "O": 1}
    
    Note:
        During GRPO training, prefer this function to extract expected values from ground-truth CIF, 
        This avoids uncertainty from parsing input text, especially organic formulas.
    """
    atoms = {}
    lines = ground_truth_cif.split('\n')
    in_atom_block = False
    
    for line in lines:
        line = line.strip()
        
        # Detect start of atom_site block
        if '_atom_site_type_symbol' in line or '_atom_site_label' in line:
            in_atom_block = True
            continue
        
        if in_atom_block and line:
            # Detect end of atom_site block
            if line.startswith('_') or line.startswith('loop_'):
                if '_atom_site_' not in line:
                    break
                continue
            
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Parse atom row
            parts = line.split()
            if len(parts) >= 3:
                # Extract element symbol (from first column)
                match = re.match(r'([A-Z][a-z]?)', parts[0])
                if match:
                    element = match.group(1)
                    atoms[element] = atoms.get(element, 0) + 1
    
    return atoms


def extract_metals_from_input(input_text: str) -> Dict[str, int]:
    """
    Extract metal-element counts from input text (consistent with clean_cif_data.py)
    
    Format: organic part</s>metal part (Miller indices)
    For example:CCO</s>P36Rh54 (0 0 1)
    
    Only extract the metal part after </s> and before parentheses.
    
    Args:
        input_text: inputtiptext
    
    Returns:
        {'P': 36, 'Rh': 54}
    
    Note:
        This function only extracts the metal part and does not include organic molecules.
        If complete atomic composition is needed, use extract_expected_atoms_from_cif 
        to extract it from ground-truth CIF.
    """
    metals = {}
    # Extract the part after </s> and before parentheses (metal formula)
    match = re.search(r'</s>([A-Z][A-Za-z0-9]+)\s+\(', input_text)
    if match:
        formula = match.group(1)
        # Extract elements and counts
        pattern = r'([A-Z][a-z]?)(\d+)'
        for element, count in re.findall(pattern, formula):
            if len(element) <= 2 and element.isalpha():
                metals[element] = int(count)
    return metals


def extract_expected_atoms_from_input(input_text: str) -> Dict[str, int]:
    """
    Extract expected atomic composition from input text (fallback)
    
    ⚠️ Note: this function cannot correctly handle complex organic formulas, such as parenthesized formulas.
    Recommend using extract_expected_atoms_from_cif to extract from ground-truth CIF.
    
    Input format examples:
    "ONOH</s>Ta18V18Si72 (1 1 2) is the basic identification and chemical formula..."
    
    Extraction logic:
    1. Only extract "is the basic" or "Please" chemical formula part before
    2. Remove Miller indices "(1 0 0)" etc.
    3. Use a simple regular expression to parse element symbols and counts
    
    Args:
        input_text: inputtiptext
    
    Returns:
        {"O": 2, "N": 1, "H": 1, "Ta": 18, "V": 18, "Si": 72}
    
    Limitations:
        - Cannot correctly handle parenthesized formulas, e.g. NH2N(CH3)2
        - May extract pseudo-elements from English descriptions
    """
    atoms = {}
    
    # remove <image> tag
    text = input_text.replace('<image>', '').strip()
    
    # Only extract formula part before English description
    if "is the basic" in text:
        formula_part = text.split("is the basic")[0].strip()
    elif "Please generate" in text:
        formula_part = text.split("Please generate")[0].strip()
    elif "Please" in text:
        formula_part = text.split("Please")[0].strip()
    else:
        formula_part = text
    
    # Remove Miller indicespart "(1 1 2)" etc.
    formula_part = re.sub(r'\([0-9\s]+\)', '', formula_part).strip()
    
    # Remove </s> separator and replace it with a space
    formula_part = formula_part.replace('</s>', ' ')
    
    # extractelement symbols and counts
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula_part)
    
    for element, count in matches:
        if element and element.isalpha() and len(element) <= 2:
            count = int(count) if count else 1
            atoms[element] = atoms.get(element, 0) + count
    
    return atoms


def test_cif_reward_model():
    """
    Test CIF reward model functionality
    """
    print("=" * 80)
    print("CIF reward model test")
    print("=" * 80)
    
    reward_model = CIFRewardModel()
    
    # test case1:perfect match
    print("\n[test1]perfect match")
    expected = {"Ca": 8, "Ga": 16, "H": 4, "C": 2, "O": 1}
    actual = {"Ca": 8, "Ga": 16, "H": 4, "C": 2, "O": 1}
    result = reward_model._evaluate_atom_composition(actual, expected)
    print(f"expected: {expected}")
    print(f"actual: {actual}")
    print(f"Score: {result['score']:.2f}, type: {result['match_type']}, error: {result['avg_error']:.1%}")
    
    # test case2:small error
    print("\n[test2]small error (Gaone fewer)")
    actual = {"Ca": 8, "Ga": 15, "H": 4, "C": 2, "O": 1}
    result = reward_model._evaluate_atom_composition(actual, expected)
    print(f"expected: {expected}")
    print(f"actual: {actual}")
    print(f"Score: {result['score']:.2f}, type: {result['match_type']}, error: {result['avg_error']:.1%}")
    
    # test case3:medium error
    print("\n[test3]medium error (Catwo more, Gafour fewer)")
    actual = {"Ca": 10, "Ga": 12, "H": 4, "C": 2, "O": 1}
    result = reward_model._evaluate_atom_composition(actual, expected)
    print(f"expected: {expected}")
    print(f"actual: {actual}")
    print(f"Score: {result['score']:.2f}, type: {result['match_type']}, error: {result['avg_error']:.1%}")
    
    # test case4:missing element
    print("\n[test4]missing element (missing O)")
    actual = {"Ca": 8, "Ga": 16, "H": 4, "C": 2}
    result = reward_model._evaluate_atom_composition(actual, expected)
    print(f"expected: {expected}")
    print(f"actual: {actual}")
    print(f"Score: {result['score']:.2f}, type: {result['match_type']}")
    print(f"missing: {result['missing']}")
    
    # test case5:Extra elements
    print("\n[test5]Extra elements (extra Cu)")
    actual = {"Ca": 8, "Ga": 16, "H": 4, "C": 2, "O": 1, "Cu": 2}
    result = reward_model._evaluate_atom_composition(actual, expected)
    print(f"expected: {expected}")
    print(f"actual: {actual}")
    print(f"Score: {result['score']:.2f}, type: {result['match_type']}")
    print(f"extra: {result['extra']}")
    
    # test case6:extract expected atoms from input
    print("\n[test6]extract expected atomic composition from input")
    test_inputs = [
        "CHCH2OH</s>Ca8Ga16 (1 0 0)",
        "Ca8Ga16H4C2O",
        "Na10Cl10",
    ]
    for inp in test_inputs:
        atoms = extract_expected_atoms_from_input(inp)
        print(f"input: {inp}")
        print(f"extract: {atoms}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_cif_reward_model()

