import uproot
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import matplotlib.pyplot as plt

m_K = 493.677   # K+ mass in MeV
m_e = 0.511     # e- mass in MeV

def compute_Bmass(df, m_k, m_e):
    PE = (
        np.sqrt(df["K_plus_PX_TRUE"]**2 + df["K_plus_PY_TRUE"]**2 + df["K_plus_PZ_TRUE"]**2 + m_k**2) 
        + np.sqrt(df["e_plus_PX_TRUE"]**2 + df["e_plus_PY_TRUE"]**2 + df["e_plus_PZ_TRUE"]**2 + m_e**2)
        + np.sqrt(df["e_minus_PX_TRUE"]**2 + df["e_minus_PY_TRUE"]**2 + df["e_minus_PZ_TRUE"]**2 + m_e**2)
    )
    
    PZ = df["K_plus_PZ_TRUE"] + df["e_plus_PZ_TRUE"] + df["e_minus_PZ_TRUE"]
    PX = df["K_plus_PX_TRUE"] + df["e_plus_PX_TRUE"] + df["e_minus_PX_TRUE"]
    PY = df["K_plus_PY_TRUE"] + df["e_plus_PY_TRUE"] + df["e_minus_PY_TRUE"]
    
    return np.sqrt(PE**2 - PX**2 - PY**2 - PZ**2) #* 1e-2 # ?

def compute_Bmomentum(df):
    return np.sqrt(df["B_plus_PX_TRUE"]**2 + df["B_plus_PY_TRUE"]**2 + df["B_plus_PZ_TRUE"]**2) #* 1e-3 # ?

def compute_impactParameter(df, particle):
    origX = df[f"{particle}_origX_TRUE"]
    origY = df[f"{particle}_origY_TRUE"]
    origZ = df[f"{particle}_origZ_TRUE"]
    
    PX = df[f"{particle}_PX_TRUE"]
    PY = df[f"{particle}_PY_TRUE"]
    PZ = df[f"{particle}_PZ_TRUE"]
    
    # Calculate the impact parameter
    impact_parameter = np.sqrt(origX**2 + origY**2 + origZ**2) / np.sqrt(PX**2 + PY**2 + PZ**2)
    
    return impact_parameter
    

def rot_matrix(source, target):
    """ 
    Returns a rotation matrix that aligns vec1 to vec2.
    """
    # Normalize both vectors
    a, b = (source / np.linalg.norm(source)).reshape(3), (target / np.linalg.norm(target)).reshape(3)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s == 0:
        return np.eye(3)  # No rotation needed, return identity matrix
    
    # Compute the cross-product matrix for v
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    # Compute the rotation matrix using Rodrigues' rotation formula
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

filename = "/home/violam/RapidSimConfigs/Kee/kee_tree.root"
file = uproot.open(filename)
tree = file["DecayTree"] 

K_branches = [
    
    "B_plus_PX_TRUE",
    "B_plus_PY_TRUE",
    "B_plus_PZ_TRUE",
    
    "K_plus_origX_TRUE",
    "K_plus_origY_TRUE",
    "K_plus_origZ_TRUE",
    
    "K_plus_PX_TRUE",
    "K_plus_PY_TRUE",
    "K_plus_PZ_TRUE",
]

e_branches = [
    
    "B_plus_vtxX_TRUE",
    "B_plus_vtxY_TRUE",
    "B_plus_vtxZ_TRUE",
    
    "B_plus_origX_TRUE",
    "B_plus_origY_TRUE",
    "B_plus_origZ_TRUE",
    
    "B_plus_PX_TRUE",
    "B_plus_PY_TRUE",
    "B_plus_PZ_TRUE",
    
    "e_minus_PX_TRUE",
    "e_minus_PY_TRUE",
    "e_minus_PZ_TRUE",

    "e_plus_PX_TRUE",
    "e_plus_PY_TRUE",
    "e_plus_PZ_TRUE",
    
    "e_plus_origX_TRUE",
    "e_plus_origY_TRUE",
    "e_plus_origZ_TRUE",
    
    "e_minus_origX_TRUE",
    "e_minus_origY_TRUE",
    "e_minus_origZ_TRUE",
    
    
    
]

K_events = tree.arrays(K_branches, library="pd").astype(float)
e_events = tree.arrays(e_branches, library="pd").astype(float)

K_events_sampled = K_events.sample(n=310, replace=True).reset_index(drop=True)
e_events_sampled = e_events.sample(n=310, replace=True).reset_index(drop=True)

# Initialize lists to store the rotated K momenta
rotated_K_PX = []
rotated_K_PY = []
rotated_K_PZ = []

for idx, (k_row, e_row) in enumerate(zip(K_events_sampled.iterrows(), e_events_sampled.iterrows())):
    # Extract the B+ momentum vector from e_events
    B_vector_e = np.array([e_row[1]["B_plus_PX_TRUE"], e_row[1]["B_plus_PY_TRUE"], e_row[1]["B_plus_PZ_TRUE"]])
    
    # Extract the K+ momentum vector from K_events
    B_vector_k = np.array([k_row[1]["B_plus_PX_TRUE"], k_row[1]["B_plus_PY_TRUE"], k_row[1]["B_plus_PZ_TRUE"]])
    
    K_vector = np.array([k_row[1]["K_plus_PX_TRUE"], k_row[1]["K_plus_PY_TRUE"], k_row[1]["K_plus_PZ_TRUE"]])
    
    # Calculate the rotation matrix to align B_vector_K to B_vector_e
    rotation_mat = rot_matrix(B_vector_k, B_vector_e)
    #print(rotation_mat)
    
    # Rotate the K+ vector using the rotation matrix
    rotated_K_vector = rotation_mat.dot(K_vector)
    
    # Store the rotated values
    rotated_K_PX.append(rotated_K_vector[0])
    rotated_K_PY.append(rotated_K_vector[1])
    rotated_K_PZ.append(rotated_K_vector[2])
    

#Mixed sample 

rot_K_events = pd.DataFrame({
    # K data
    "K_plus_origX_TRUE": K_events_sampled["K_plus_origX_TRUE"],
    "K_plus_origY_TRUE": K_events_sampled["K_plus_origY_TRUE"],
    "K_plus_origZ_TRUE": K_events_sampled["K_plus_origZ_TRUE"],
    
    "K_plus_PX_TRUE": rotated_K_PX,
    "K_plus_PY_TRUE": rotated_K_PY,
    "K_plus_PZ_TRUE": rotated_K_PZ,
})

mixed_events = pd.concat(
    (e_events_sampled, rot_K_events), axis=1
)

mixed_events["B_plus_P_TRUE"] = compute_Bmomentum(mixed_events)
mixed_events["B_plus_PT_TRUE"] = np.sqrt(mixed_events["B_plus_PX_TRUE"]**2 + mixed_events["B_plus_PY_TRUE"]**2)


output_file_path = "fast_vtx/datasets_mixed/mixed_Kee.root"

data_dict = {col: mixed_events[col].values for col in mixed_events.columns}

# Create a new ROOT file and write the DataFrame to it
with uproot.recreate(output_file_path) as f:
    f["DecayTree"] = data_dict