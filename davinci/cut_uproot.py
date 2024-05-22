import uproot
import awkward as ak
import numpy as np
import os
from tqdm import tqdm

mode = 'B2KEE_three_body'
job_ID = 716

localDir = f'/eos/lhcb/user/m/marshall/gangaDownload/{job_ID}/'

# Define the original and new file paths
original_file_path = f"{localDir}{mode}.root"
new_file_path = f"{localDir}{mode}_cut.root"

# Remove the existing file if it exists
if os.path.exists(new_file_path):
    print(f"File {new_file_path} already exists. Deleting it.")
    os.remove(new_file_path)

# Define the cut condition
cut_condition = "(M_TRUEID != 0) & (M_BKGCAT < 60)"

# List of branches to keep
branches_to_keep = [
    'M_DIRA_OWNPV', 'M_ENDVERTEX_CHI2', 'M_ENDVERTEX_X', 'M_ENDVERTEX_Y', 'M_ENDVERTEX_Z',
    'M_FDCHI2_OWNPV', 'M_IPCHI2_OWNPV', 'M_OWNPV_X', 'M_OWNPV_Y', 'M_OWNPV_Z', 'M_PX', 'M_PY', 
    'M_PZ', 'M_TRUEP_X', 'M_TRUEP_Y', 'M_TRUEP_Z', 'M_TRUEID', 'M_BKGCAT',
    'A_ID', 'A_IPCHI2_OWNPV', 'A_PX', 'A_PY', 'A_PZ', 'A_TRACK_CHI2NDOF', 'A_TRUEID', 'A_TRUEP_X', 
    'A_TRUEP_Y', 'A_TRUEP_Z', 'C_ID', 'C_IPCHI2_OWNPV', 'C_PX', 'C_PY', 'C_PZ', 'C_TRACK_CHI2NDOF', 
    'C_TRUEID', 'C_TRUEP_X', 'C_TRUEP_Y', 'C_TRUEP_Z', 'B_ID', 'B_IPCHI2_OWNPV', 'B_PX', 'B_PY', 
    'B_PZ', 'B_TRACK_CHI2NDOF', 'B_TRUEID', 'B_TRUEP_X', 'B_TRUEP_Y', 'B_TRUEP_Z', 'nSPDHits', 'nTracks'
]

# Open the original ROOT file
print("Open the original ROOT file")
with uproot.open(original_file_path) as file:
    print("Get the TTree from the file")
    tree = file["B2Kee_Tuple/DecayTree"]

    # Set up a list to store temporary file paths
    temp_files = []

    # Process entries in chunks
    chunk_size = 250000  # Adjust this based on memory capacity
    n_chunks = (tree.num_entries + chunk_size - 1) // chunk_size  # Calculate the total number of chunks
    with tqdm(total=n_chunks, desc="Processing chunks") as pbar:
        for i, start in enumerate(range(0, tree.num_entries, chunk_size)):
            stop = min(start + chunk_size, tree.num_entries)

            # Read a chunk of data
            data = tree.arrays(branches_to_keep, entry_start=start, entry_stop=stop, library="ak")

            # Apply the cut
            mask = (data["M_TRUEID"] != 0) & (data["M_BKGCAT"] < 60)
            filtered_data = data[mask]

            # Write filtered data to a temporary file
            temp_file_path = f"{new_file_path}_{i}.root"
            temp_files.append(temp_file_path)
            with uproot.recreate(temp_file_path) as temp_file:
                temp_file["DecayTree"] = {branch: filtered_data[branch] for branch in branches_to_keep if branch in filtered_data.fields}

            pbar.update(1)
            break

entries = 0
for idx, file in enumerate(temp_files):
    uproot_file = uproot.open(file)['DecayTree']
    entries += uproot_file.num_entries

os.system(f'hadd -fk {mode}_cut.root {" ".join(str(x) for x in temp_files)}')

print(entries)

print("Process completed successfully")


