import numpy as np
import uproot
from pathlib import Path
import os
import fast_vertex_quality_inference.processing.processing_tools as pts


def create_inferable_file(input_path: os.PathLike, output_path: os.PathLike = ""):
    input_path = Path(input_path)
    if not output_path:
        output_path = Path(input_path.parent, input_path.stem + "_inferable" + input_path.suffix)

    # suffixes that all have fooX, fooY, fooZ versions
    mother_xyz_suffixes = [
        "P",
        "TRUEP_",
        "TRUEORIGINVERTEX_",
        "OWNPV_",
        "ENDVERTEX_",
        "TRUEENDVERTEX_"
    ]

    daughter_xyz_suffixes = [
        "P",
        "TRUEP_",
        "TRUEORIGINVERTEX_",
        "TRUEENDVERTEX_"
    ]

    # uses f string format to allow X/Y/Z not just at the end
    suffix_map = {
        "P": "P{}",
        "TRUEP_": "P{}_TRUE",
        "TRUEORIGINVERTEX_": "orig{}_TRUE",
        "OWNPV_": "orig{}",
        "ENDVERTEX_": "vtx{}",
        "TRUEENDVERTEX_": "vtx{}_TRUE"
    }

    columns_to_get = []
    output_column_names = []
    for dim in ("X", "Y", "Z"):
        for suff in mother_xyz_suffixes:
            columns_to_get.append("MOTHER_" + suff + dim)
            output_column_names.append("MOTHER_" + suffix_map[suff].format(dim))

        for i in range(1, 4):
            for suff in daughter_xyz_suffixes:
                columns_to_get.append(f"DAUGHTER{i}_" + suff + dim)
                output_column_names.append(f"DAUGHTER{i}_" + suffix_map[suff].format(dim))


    branches = []
    with uproot.open(input_path) as input_file:
        input_tree = input_file["DecayTree"]
        for col in columns_to_get:
            branches.append(input_tree[col].array(library="np"))

    # TTrees to write to file
    output_dict = {}
    for branch, branch_name in zip(branches, output_column_names):
        output_dict[branch_name] = branch

    output_dict["MOTHER_P_TRUE"] = np.sqrt(output_dict["MOTHER_PX_TRUE"]**2 + output_dict["MOTHER_PY_TRUE"]**2 + output_dict["MOTHER_PZ_TRUE"]**2) * 1e-3
    output_dict["MOTHER_PT_TRUE"] = np.sqrt(output_dict["MOTHER_PX_TRUE"]**2 + output_dict["MOTHER_PY_TRUE"]**2)

    for i in range(1, 4):
        output_dict[f"DAUGHTER{i}_PT"] = np.sqrt(output_dict[f"DAUGHTER{i}_PX"]**2 + output_dict[f"DAUGHTER{i}_PY"]**2)
        output_dict[f"DAUGHTER{i}_P"] = np.sqrt(output_dict[f"DAUGHTER{i}_PX"]**2 + output_dict[f"DAUGHTER{i}_PY"]**2 + output_dict[f"DAUGHTER{i}_PZ"]**2)

    with uproot.recreate(output_path) as output_file:
        output_file["DecayTree"] = output_dict


if __name__ == "__main__":
    create_inferable_file("combinatorial_select_Kuu.root")