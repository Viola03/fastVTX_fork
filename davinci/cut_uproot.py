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



# mode = 'cocktail_three_body'
# job_ID = 1344
#mode = 'B2KEE_three_body'
#job_ID = 719
# job_ID = 'temp'

mode = 'cocktail_mix'
job_ID = 1405

add_cut = True
# add_cut = False

localDir = f'/eos/lhcb/user/m/marshall/gangaDownload/{job_ID}/'

# Define the original and new file paths
original_file_path = f"{localDir}{mode}.root"
new_file_path = f"{localDir}{mode}_cut.root"

# Remove the existing file if it exists
if os.path.exists(new_file_path):
    print(f"File {new_file_path} already exists. Deleting it.")
    os.remove(new_file_path)

# Define the cut condition
cut_condition = "(MOTHER_TRUEID != 0) & (MOTHER_BKGCAT < 60)"

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
# branches_to_keep = [
# 'MOTHER_TRUEID', 'MOTHER_BKGCAT'
# ]


####
files = glob.glob(f'{localDir}/DTT_2018_Reco18Strip34_Down_ALLSTREAMS.DST_*.root')

sucesses = 0
# for file_idx, file in enumerate(files):
for file_idx, file in tqdm(enumerate(files), total=len(files), desc="Processing files"):

    print(f'{file_idx}/{len(files)}')

    if '_cut' in file:
        continue
    try:
        with uproot.open(file) as ur_file:
            tree = ur_file["B2Kee_Tuple/DecayTree"]

            if add_cut:
                data = tree.arrays(branches_to_keep, library="pd")
            else:
                data = tree.arrays(list(tree.keys()), library="pd")
                print(data)

            try:
                filtered_data = data.query(cut_condition)
            except:
                filtered_data = data[0].query(cut_condition)
                

            if file_idx == 0:
                filtered_data_total = filtered_data
            else:
                filtered_data_total = pd.concat((filtered_data_total, filtered_data), axis=0)
            sucesses += 1
            # with uproot.recreate(f'{file[:-5]}_cut.root') as temp_file:
                # temp_file["DecayTree"] = {branch: filtered_data[branch] for branch in branches_to_keep if branch in filtered_data.columns}

            # write_df_to_root(data, f'{file[:-5]}_cut.root')
    except:
         pass
    # if sucesses > 50:
    #       break
write_df_to_root(filtered_data_total, new_file_path)


# import glob
# temp_files = glob.glob(f'{localDir}*_cut.root')

# entries = 0
# for idx, file in enumerate(files):
#     uproot_file = uproot.open(file)['DecayTree']
#     entries += uproot_file.num_entries

# print(entries)

# os.system(f'hadd -f {new_file_path} {" ".join(str(x) for x in temp_files)}')
# print(new_file_path)
print("Opening file...")

file = uproot.open(f"{new_file_path}:DecayTree")
branches = file.keys()
print(file.num_entries)

for branch in branches:
    # print('\n',branch)
    # # try:
    events = file.arrays([branch],library='pd')
    # print(events.shape)


print(new_file_path)

quit()




# Open the original ROOT file
print("Open the original ROOT file")
with uproot.open(original_file_path) as file:
    print("Get the TTree from the file")
    tree = file["B2Kee_Tuple/DecayTree"]

    # Set up a list to store temporary file paths
    temp_files = []

    # Process entries in chunks
    chunk_size = 50000  # Adjust this based on memory capacity
    n_chunks = (tree.num_entries + chunk_size - 1) // chunk_size  # Calculate the total number of chunks
    with tqdm(total=n_chunks, desc="Processing chunks") as pbar:
        for i, start in enumerate(range(0, tree.num_entries, chunk_size)):
            stop = min(start + chunk_size, tree.num_entries)

            # Read a chunk of data
            data = tree.arrays(branches_to_keep, entry_start=start, entry_stop=stop, library="ak")

            # Apply the cut
            mask = (data["MOTHER_TRUEID"] != 0) & (data["MOTHER_BKGCAT"] < 60)
            filtered_data = data[mask]

            # Write filtered data to a temporary file
            temp_file_path = f"{new_file_path}_{i}.root"
            temp_files.append(temp_file_path)
            with uproot.recreate(temp_file_path) as temp_file:
                temp_file["DecayTree"] = {branch: filtered_data[branch] for branch in branches_to_keep if branch in filtered_data.fields}

            pbar.update(1)
            # break

# import glob
# temp_files = glob.glob('/eos/lhcb/user/m/marshall/gangaDownload/716/*root_*')

entries = 0
for idx, file in enumerate(temp_files):
    uproot_file = uproot.open(file)['DecayTree']
    entries += uproot_file.num_entries

os.system(f'hadd -fk {new_file_path} {" ".join(str(x) for x in temp_files)}')

print(entries)

# print("Removing temp files...")
# for file in temp_files:
#     os.remove(file)


print("Process completed successfully")


