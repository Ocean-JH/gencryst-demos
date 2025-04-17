#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def check_wyckoff_validity(structure, target_sgnum, tol=0.01):
    try:
        sga = SpacegroupAnalyzer(structure, symprec=tol)
        actual_sgnum = sga.get_space_group_number()
        wyckoffs = sga.get_symmetry_dataset()["wyckoffs"]  # e.g. ['4c', '4c', '8f']

        if actual_sgnum != target_sgnum:
            return False, f"SG mismatch: expected {target_sgnum}, found {actual_sgnum}"

        return True, f"Valid Wyckoffs: {wyckoffs}"
    except Exception as e:
        return False, f"Error: {e}"


def check_atomic_clashes(structure, min_dist=1.5):
    coords = structure.cart_coords
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < min_dist:
                return True, f"Clash: atoms {i}-{j} distance = {dist:.2f}"
    return False, "No clashes"
