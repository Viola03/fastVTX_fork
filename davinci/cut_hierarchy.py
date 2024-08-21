#source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

import uproot
import awkward as ak
import numpy as np
import os
from tqdm import tqdm
import glob
import uproot3 
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def write_df_to_root(df, output_name):

	print(f'Writing to {output_name}..')
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
	print("Written.")

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

	'DAUGHTER1_TRUEENDVERTEX_X',
	'DAUGHTER1_TRUEENDVERTEX_Y',
	'DAUGHTER1_TRUEENDVERTEX_Z',
	'DAUGHTER1_TRUEORIGINVERTEX_X',
	'DAUGHTER1_TRUEORIGINVERTEX_Y',
	'DAUGHTER1_TRUEORIGINVERTEX_Z',

	'DAUGHTER2_TRUEENDVERTEX_X',
	'DAUGHTER2_TRUEENDVERTEX_Y',
	'DAUGHTER2_TRUEENDVERTEX_Z',
	'DAUGHTER2_TRUEORIGINVERTEX_X',
	'DAUGHTER2_TRUEORIGINVERTEX_Y',
	'DAUGHTER2_TRUEORIGINVERTEX_Z',

	'DAUGHTER3_TRUEENDVERTEX_X',
	'DAUGHTER3_TRUEENDVERTEX_Y',
	'DAUGHTER3_TRUEENDVERTEX_Z',
	'DAUGHTER3_TRUEORIGINVERTEX_X',
	'DAUGHTER3_TRUEORIGINVERTEX_Y',
	'DAUGHTER3_TRUEORIGINVERTEX_Z',
	
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

    "INTERMEDIATE_ENDVERTEX_CHI2",
    "INTERMEDIATE_DIRA_OWNPV",
    "MOTHER_VTXISOBDTHARDFIRSTVALUE",
    "MOTHER_VTXISOBDTHARDSECONDVALUE",
    "MOTHER_VTXISOBDTHARDTHIRDVALUE",
    "MOTHER_SmallestDeltaChi2OneTrack",
    "MOTHER_SmallestDeltaChi2TwoTracks",
    "MOTHER_cp_0.70",
    "MOTHER_cpt_0.70",
    "MOTHER_cmult_0.70",
    "DAUGHTER1_TRACK_GhostProb",
    "DAUGHTER2_TRACK_GhostProb",
    "DAUGHTER3_TRACK_GhostProb",
    "MOTHER_OWNPV_X",
    "MOTHER_ENDVERTEX_X",
    "INTERMEDIATE_ENDVERTEX_X",
    "MOTHER_OWNPV_Y",
    "MOTHER_ENDVERTEX_Y",
    "INTERMEDIATE_ENDVERTEX_Y",
    "MOTHER_OWNPV_Z",
    "MOTHER_ENDVERTEX_Z",
    "INTERMEDIATE_ENDVERTEX_Z",
    "MOTHER_OWNPV_XERR",
    "MOTHER_ENDVERTEX_XERR",
    "INTERMEDIATE_ENDVERTEX_XERR",
    "MOTHER_OWNPV_YERR",
    "MOTHER_ENDVERTEX_YERR",
    "INTERMEDIATE_ENDVERTEX_YERR",
    "MOTHER_OWNPV_ZERR",
    "MOTHER_ENDVERTEX_ZERR",
    "INTERMEDIATE_ENDVERTEX_ZERR",
    "MOTHER_OWNPV_COV_", # these are 9 values each!
    "MOTHER_ENDVERTEX_COV_",
    "INTERMEDIATE_ENDVERTEX_COV_", # Also need XERR?
]

particle_masses_dict_mothers = {}

particle_masses_dict_mothers[411] = 1.86962
particle_masses_dict_mothers[421] = 1.86484
particle_masses_dict_mothers[431] = 1.96847

particle_masses_dict_mothers[511] = 5.27965
particle_masses_dict_mothers[521] = 5.27934
particle_masses_dict_mothers[531] = 5.36688
particle_masses_dict_mothers[541] = 6.2749

particle_masses_dict_daughters = {}

particle_masses_dict_daughters[321] = 493.677
particle_masses_dict_daughters[211] = 139.57039
particle_masses_dict_daughters[13] = 105.66
particle_masses_dict_daughters[11] = 0.51099895000 * 1e-3

def compute_mass_3(df, i, j, k):

	PE = np.sqrt(
		df[f"{i}_mass"]**2
		+ df[f"{i}_TRUEP_X"] ** 2
		+ df[f"{i}_TRUEP_Y"] ** 2
		+ df[f"{i}_TRUEP_Z"] ** 2
	) + np.sqrt(
		df[f"{j}_mass"]**2
		+ df[f"{j}_TRUEP_X"] ** 2
		+ df[f"{j}_TRUEP_Y"] ** 2
		+ df[f"{j}_TRUEP_Z"] ** 2
	) + np.sqrt(
		df[f"{k}_mass"]**2
		+ df[f"{k}_TRUEP_X"] ** 2
		+ df[f"{k}_TRUEP_Y"] ** 2
		+ df[f"{k}_TRUEP_Z"] ** 2
	)
	PX = df[f"{i}_TRUEP_X"] + df[f"{j}_TRUEP_X"] + df[f"{k}_TRUEP_X"]
	PY = df[f"{i}_TRUEP_Y"] + df[f"{j}_TRUEP_Y"] + df[f"{k}_TRUEP_Y"]
	PZ = df[f"{i}_TRUEP_Z"] + df[f"{j}_TRUEP_Z"] + df[f"{k}_TRUEP_Z"]

	mass = np.sqrt((PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6)

	return mass


def cut(loc, out_loc, file, throw_away_partreco_frac=0.):

	if '*' in file:
		files = glob.glob(f'{loc}/{file}')
	else:
		files = [f'{loc}/{file}']
	
	# Define the cut condition
	cut_condition = "(MOTHER_TRUEID != 0) & (MOTHER_BKGCAT < 60)"

	for file in files:

		print(file)

		with uproot.open(file) as ur_file:
			
			for key in ur_file.keys():
				if 'DecayTree' in key:
					key = key.split(';')[0]
					break
			
			tree = ur_file[key]
			
			data = tree.arrays(list(np.unique(branches_to_keep)), library="pd")

			filtered_data = data.query(cut_condition)

			if "dedicated_Kee_MC" in file:
				filtered_data = filtered_data.query("MOTHER_BKGCAT<11")

			mother_masses = -1.*np.ones(filtered_data.shape[0])
			mother_TRUEID = filtered_data["MOTHER_TRUEID"]
			for PID in list(particle_masses_dict_mothers.keys()):
				mother_masses[np.where(np.abs(mother_TRUEID.astype(int))==PID)] = particle_masses_dict_mothers[PID]

			particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
			for particle in particles:
				mass = np.asarray(filtered_data[f'{particle}_TRUEID']).astype('float32')
				for pid in list(particle_masses_dict_daughters.keys()):
					mass[np.where(np.abs(mass)==pid)] = particle_masses_dict_daughters[pid]
				filtered_data[f'{particle}_mass'] = mass

			filtered_data[f'MOTHER_M'] = compute_mass_3(filtered_data, "DAUGHTER1", "DAUGHTER2", "DAUGHTER3")

			filtered_data['abs_mass_diff'] = np.abs(np.asarray(filtered_data["MOTHER_M"])-mother_masses)
			
			fully_reco = np.zeros(filtered_data.shape[0])
			fully_reco[np.where(filtered_data['abs_mass_diff']<0.05)] = 1
			filtered_data['fully_reco'] = fully_reco

			if throw_away_partreco_frac > 0.:
				partially_reco_rows = filtered_data[filtered_data['fully_reco'] == 0]
				sampled_zero_reco = partially_reco_rows.sample(frac=0.5)
				filtered_data = filtered_data.drop(sampled_zero_reco.index)

		write_df_to_root(filtered_data, f'{out_loc}/{file[:-5]}_cut.root')

cut(loc='.', out_loc='.', file='MergeTest_*', throw_away_partreco_frac=0.5)
