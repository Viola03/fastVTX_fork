#source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

import uproot
import awkward as ak
import numpy as np
import os
from tqdm import tqdm
import glob
import uproot3 
import pandas as pd

def write_df_to_root(df, output_name):
	branch_dict = {}
	data_dict = {}
	dtypes = df.dtypes
	used_columns = [] # stop repeat columns, kpipi_correction was getting repeated
	for dtype, branch in enumerate(df.keys()):
		if branch not in used_columns:
			if dtypes[dtype] == 'uint32': dtypes[dtype] = 'int32'
			if dtypes[dtype] == 'uint64': dtypes[dtype] = 'int64'
			branch_dict[branch] = dtypes[dtype]
			# stop repeat columns, kpipi_correction was getting repeated
			if np.shape(df[branch].shape)[0] > 1:
				data_dict[branch] = df[branch].iloc[:, 0]
			else:
				data_dict[branch] = df[branch]
		used_columns.append(branch)
	with uproot3.recreate(output_name) as f:
		f["DecayTree"] = uproot3.newtree(branch_dict)
		f["DecayTree"].extend(data_dict)

# List of branches to keep
branches_to_keep = [
    'MOTHER_DIRA_OWNPV', 'MOTHER_ENDVERTEX_CHI2', 'MOTHER_ENDVERTEX_NDOF', 'MOTHER_ENDVERTEX_X', 'MOTHER_ENDVERTEX_Y', 'MOTHER_ENDVERTEX_Z',
    'MOTHER_FDCHI2_OWNPV', 'MOTHER_IPCHI2_OWNPV', 'MOTHER_OWNPV_X', 'MOTHER_OWNPV_Y', 'MOTHER_OWNPV_Z', 'MOTHER_PX', 'MOTHER_PY', 
    'MOTHER_PZ', 'MOTHER_TRUEP_X', 'MOTHER_TRUEP_Y', 'MOTHER_TRUEP_Z', 'MOTHER_TRUEID', 'MOTHER_BKGCAT',
    'DAUGHTER1_ID', 'DAUGHTER1_IPCHI2_OWNPV', 'DAUGHTER1_PX', 'DAUGHTER1_PY', 'DAUGHTER1_PZ', 'DAUGHTER1_TRACK_CHI2NDOF', 'DAUGHTER1_TRUEID', 'DAUGHTER1_TRUEP_X', 
    'DAUGHTER1_TRUEP_Y', 'DAUGHTER1_TRUEP_Z', 'DAUGHTER2_ID', 'DAUGHTER2_IPCHI2_OWNPV', 'DAUGHTER2_PX', 'DAUGHTER2_PY', 'DAUGHTER2_PZ', 'DAUGHTER2_TRACK_CHI2NDOF', 
    'DAUGHTER2_TRUEID', 'DAUGHTER2_TRUEP_X', 'DAUGHTER2_TRUEP_Y', 'DAUGHTER2_TRUEP_Z', 'DAUGHTER3_ID', 'DAUGHTER3_IPCHI2_OWNPV', 'DAUGHTER3_PX', 'DAUGHTER3_PY', 
    'DAUGHTER3_PZ', 'DAUGHTER3_TRACK_CHI2NDOF', 'DAUGHTER3_TRUEID', 'DAUGHTER3_TRUEP_X', 'DAUGHTER3_TRUEP_Y', 'DAUGHTER3_TRUEP_Z', 'nSPDHits', 'nTracks',
    'INTERMEDIATE_TRUEID',
    'INTERMEDIATE_DIRA_OWNPV', 'INTERMEDIATE_ENDVERTEX_CHI2', 'INTERMEDIATE_ENDVERTEX_NDOF',
    'INTERMEDIATE_FDCHI2_OWNPV', 'INTERMEDIATE_IPCHI2_OWNPV',
    'MOTHER_TRUEENDVERTEX_X',
    'MOTHER_TRUEENDVERTEX_Y',
    'MOTHER_TRUEENDVERTEX_Z',

    'MOTHER_TRUEORIGINVERTEX_X',
    'MOTHER_TRUEORIGINVERTEX_Y',
    'MOTHER_TRUEORIGINVERTEX_Z',

    'INTERMEDIATE_TRUEENDVERTEX_X',
    'INTERMEDIATE_TRUEENDVERTEX_Y',
    'INTERMEDIATE_TRUEENDVERTEX_Z',
    'INTERMEDIATE_TRUEORIGINVERTEX_X',
    'INTERMEDIATE_TRUEORIGINVERTEX_Y',
    'INTERMEDIATE_TRUEORIGINVERTEX_Z',
    'EVT_GenEvent',


    "MOTHER_MC_MOTHER_ID",
    "MOTHER_MC_GD_MOTHER_ID",
    "MOTHER_MC_GD_GD_MOTHER_ID",
    "INTERMEDIATE_MC_MOTHER_ID",
    "INTERMEDIATE_MC_GD_MOTHER_ID",
    "INTERMEDIATE_MC_GD_GD_MOTHER_ID",
    "DAUGHTER1_MC_MOTHER_ID",
    "DAUGHTER1_MC_GD_MOTHER_ID",
    "DAUGHTER1_MC_GD_GD_MOTHER_ID",
    "DAUGHTER2_MC_MOTHER_ID",
    "DAUGHTER2_MC_GD_MOTHER_ID",
    "DAUGHTER2_MC_GD_GD_MOTHER_ID",
    "DAUGHTER3_MC_MOTHER_ID",
    "DAUGHTER3_MC_GD_MOTHER_ID",
    "DAUGHTER3_MC_GD_GD_MOTHER_ID",

]

# Define the cut condition
cut_condition = "(MOTHER_TRUEID != 0) & (MOTHER_BKGCAT < 60)"

with uproot.open('/users/am13743/fast_vertexing_variables/datasets/cocktail_hierarchy.root') as ur_file:
	
    tree = ur_file["B2Kee_Tuple/DecayTree"]

    data = tree.arrays(branches_to_keep, library="pd")

    filtered_data = data.query(cut_condition)

write_df_to_root(filtered_data, '/users/am13743/fast_vertexing_variables/datasets/cocktail_hierarchy_cut.root')