import uproot
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import matplotlib.pyplot as plt

m_K = 493.677   # K+ mass in MeV
m_e = 0.511     # e- mass in MeV

def compute_momentum(df, particle, suffix="_TRUE"):
    return np.sqrt(df[f"{particle}_PX{suffix}"]**2 + df[f"{particle}_PY{suffix}"]**2 + df[f"{particle}_PZ{suffix}"]**2)

def compute_PT(df, particle, suffix="_TRUE"):
    return np.sqrt(df[f"{particle}_PX{suffix}"]**2 + df[f"{particle}_PY{suffix}"]**2)

def compute_impactParameter(df, particle, suffix="_TRUE"):
    origX = df[f"{particle}_origX{suffix}"]
    origY = df[f"{particle}_origY{suffix}"]
    origZ = df[f"{particle}_origZ{suffix}"]
    
    PX = df[f"{particle}_PX{suffix}"]
    PY = df[f"{particle}_PY{suffix}"]
    PZ = df[f"{particle}_PZ{suffix}"]
    
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


def process_events(tree, suffix="_TRUE"):
        
    K_branches = [
        
        f"B_plus_PX{suffix}",
        f"B_plus_PY{suffix}",
        f"B_plus_PZ{suffix}",
        
        f"K_plus_origX{suffix}",
        f"K_plus_origY{suffix}",
        f"K_plus_origZ{suffix}",
        
        f"K_plus_PX{suffix}",
        f"K_plus_PY{suffix}",
        f"K_plus_PZ{suffix}",
    ]

    e_branches = [
        
        f"B_plus_vtxX{suffix}",
        f"B_plus_vtxY{suffix}",
        f"B_plus_vtxZ{suffix}",
        
        f"B_plus_origX{suffix}",
        f"B_plus_origY{suffix}",
        f"B_plus_origZ{suffix}",
        
        f"B_plus_PX{suffix}",
        f"B_plus_PY{suffix}",
        f"B_plus_PZ{suffix}",
        
        f"e_minus_PX{suffix}",
        f"e_minus_PY{suffix}",
        f"e_minus_PZ{suffix}",

        f"e_plus_PX{suffix}",
        f"e_plus_PY{suffix}",
        f"e_plus_PZ{suffix}",
        
        f"e_plus_origX{suffix}",
        f"e_plus_origY{suffix}",
        f"e_plus_origZ{suffix}",
        
        f"e_minus_origX{suffix}",
        f"e_minus_origY{suffix}",
        f"e_minus_origZ{suffix}",
    ]
    
    
    K_events = tree.arrays(K_branches, library="pd").astype(float)
    e_events = tree.arrays(e_branches, library="pd").astype(float)

    K_events_sampled = K_events.sample(n=30000, replace=True).reset_index(drop=True)
    e_events_sampled = e_events.sample(n=30000, replace=True).reset_index(drop=True)

    # Initialize lists to store the rotated K momenta
    rotated_K_PX = []
    rotated_K_PY = []
    rotated_K_PZ = []

    for idx, (k_row, e_row) in enumerate(zip(K_events_sampled.iterrows(), e_events_sampled.iterrows())):
        # Extract the B+ momentum vector from e_events
        B_vector_e = np.array([e_row[1][f"B_plus_PX{suffix}"], e_row[1][f"B_plus_PY{suffix}"], e_row[1][f"B_plus_PZ{suffix}"]])
        
        # Extract the K+ momentum vector from K_events
        B_vector_k = np.array([k_row[1][f"B_plus_PX{suffix}"], k_row[1][f"B_plus_PY{suffix}"], k_row[1][f"B_plus_PZ{suffix}"]])
        
        K_vector = np.array([k_row[1][f"K_plus_PX{suffix}"], k_row[1][f"K_plus_PY{suffix}"], k_row[1][f"K_plus_PZ{suffix}"]])
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
        f"K_plus_origX{suffix}": K_events_sampled[f"K_plus_origX{suffix}"],
        f"K_plus_origY{suffix}": K_events_sampled[f"K_plus_origY{suffix}"],
        f"K_plus_origZ{suffix}": K_events_sampled[f"K_plus_origZ{suffix}"],
        
        f"K_plus_PX{suffix}": rotated_K_PX,
        f"K_plus_PY{suffix}": rotated_K_PY,
        f"K_plus_PZ{suffix}": rotated_K_PZ,
    })
    
    mixed_events = pd.concat(
    (e_events_sampled, rot_K_events), axis=1
    )
      
    particles = ["B_plus", "K_plus", "e_plus", "e_minus"]
    
    for particle in particles:
        mixed_events[f"{particle}_PT{suffix}"] = compute_PT(mixed_events, particle, suffix)
        mixed_events[f"{particle}_P{suffix}"] = compute_momentum(mixed_events, particle, suffix)  
    
    return mixed_events

# Process true and non-true events
filename = "/home/violam/RapidSimConfigs/Kmumu/Kmumu_tree.root"
file = uproot.open(filename)
tree = file["DecayTree"] 

mixed_events_true = process_events(tree)
mixed_events = process_events(tree,suffix="")
# Concatenate the true and non-true mixed events
mixed_events = pd.concat([mixed_events_true, mixed_events], axis=1) #, ignore_index=True
# Save the mixed events to a ROOT file
output_file_path = "fast_vtx/datasets_mixed/mixed_kmumu.root"

data_dict = {col: mixed_events[col].values for col in mixed_events.columns}

with uproot.recreate(output_file_path) as f:
    f["DecayTree"] = data_dict
