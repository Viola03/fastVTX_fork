import uproot
import uproot3
import numpy as np

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
		
reduce_factor = 10
file_name = "general_sample_chargeCounters_cut_more_vars.root"

file = uproot.open(file_name)['DecayTree']

print(file.num_entries)

N = int(file.num_entries/reduce_factor)

print(N)

events = file.arrays(library='pd', entry_stop=N)

print(events)

out_name = f"{file_name[:-5]}_HEADfactor{reduce_factor}.root"
write_df_to_root(events, out_name)