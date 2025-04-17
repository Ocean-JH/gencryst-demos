#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import json
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

def structure_to_tokens(filepath):
    try:
        tokens = ["<bos>"]
        structure = Structure.from_file(filepath)
        # SPG Number
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-3)
        sg_number = analyzer.get_space_group_number()
        tokens.append(f"SG#{sg_number}")

        # Lattice parameters: a, b, c, alpha, beta, gamma
        lattice = structure.lattice
        constants = [lattice.a, lattice.b, lattice.c,
                     lattice.alpha, lattice.beta, lattice.gamma]
        tokens += [f"{x:.4f}" for x in constants]

        # Wyckoff positions
        sym_struct = analyzer.get_symmetrized_structure()
        wyckoff_list = analyzer.get_symmetry_dataset()["wyckoffs"]

        for site_group in sym_struct.equivalent_sites:
            rep_site = site_group[0]
            idx = structure.index(rep_site)

            element = rep_site.species_string
            wyckoff = wyckoff_list[idx]
            coords = rep_site.frac_coords

            tokens.append(element)
            tokens.append(wyckoff)
            tokens += [f"{v:.4f}" for v in coords]  # x, y, z

        tokens.append("<eos>")
        return tokens
    except Exception as e:
        print(f"[Error] Skipped {filepath}: {e}")
        return None


def tokens_to_structure(tokens):
    # 1. 去除 <bos> 和 <eos>
    tokens = [t for t in tokens if t not in ["<bos>", "<eos>"]]

    # 2. 解析空间群
    if not tokens[0].startswith("SG_"):
        raise ValueError("Token sequence must start with space group (SG#xx)")
    sg_number = int(tokens[0][3:])
    space_group = SpaceGroup.from_int_number(sg_number)

    # 3. 解析晶格常数
    lattice_params = list(map(float, tokens[1:7]))
    a, b, c, alpha, beta, gamma = lattice_params
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    # 4. 原子部分
    species = []
    coords = []

    i = 7
    while i < len(tokens):
        element = tokens[i]  # 元素符号
        wyckoff = tokens[i + 1]  # Wyckoff 字母（目前暂不使用还原，只记录）
        x = float(tokens[i + 2])
        y = float(tokens[i + 3])
        z = float(tokens[i + 4])
        species.append(element)
        coords.append([x, y, z])
        i += 5

    structure = Structure(
        lattice=lattice,
        species=species,
        coords=coords,
        to_unit_cell=True,
        coords_are_cartesian=False
    )

    return structure


def dataloader(folder_path, output_file="tokens.jsonl"):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if not f.startswith(".")]
    with open(output_file, "w") as fout:
        for struct in tqdm(files):
            tokens = structure_to_tokens(struct)
            if tokens:
                fout.write(json.dumps(tokens) + "\n")


if __name__ == "__main__":
    structure_dir = "data/"  # <-- 修改为你的目录
    dataloader(structure_dir)
