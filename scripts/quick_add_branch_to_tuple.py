from fast_vertex_quality.tools.config import read_definition, rd

import pandas as pd
import numpy as np
import uproot
import uproot3
import shutil
import os

def write_df_to_root(df, output_name):
	 
	print(f'Writing {output_name}...')
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




path = '/users/am13743/fast_vertexing_variables/rapidsim/Kee/Signal_tree_NNvertex_more_vars.root'
# path = '/users/am13743/fast_vertexing_variables/rapidsim/Kee/Signal_tree_LARGE_NNvertex_more_vars.root'
# path = '/users/am13743/fast_vertexing_variables/rapidsim/Kstree/Partreco_tree_LARGE_NNvertex_more_vars.root'
# path = '/users/am13743/fast_vertexing_variables/rapidsim/BuD0piKenu/BuD0piKenu_tree_NNvertex_more_vars.root'
# path = '/users/am13743/fast_vertexing_variables/rapidsim/Kmumu/Kmumu_tree_NNvertex_more_vars.root'
new_path = f"{path}_temp.root"

file = uproot.open(f"{path}:DecayTree")
branches = file.keys()
print(branches)

events = file.arrays(branches, library='pd')

events['MOTHER_nPositive_missing'] = np.asarray(np.ones(np.shape(events['MOTHER_M']))*0,dtype=np.int32)
events['MOTHER_nNegative_missing'] = np.asarray(np.ones(np.shape(events['MOTHER_M']))*0,dtype=np.int32)

events['fully_reco'] = np.asarray(np.ones(np.shape(events['MOTHER_M']))*1,dtype=np.int32)

events['B_plus_TRUEID'] = np.asarray(np.ones(np.shape(events['MOTHER_M']))*521,dtype=np.int32)
# events['B_plus_TRUEID'] = np.asarray(np.ones(np.shape(events['MOTHER_M']))*511,dtype=np.int32)

write_df_to_root(events, new_path)
print("overwriting...")
shutil.copyfile(new_path,path)
os.remove(new_path)