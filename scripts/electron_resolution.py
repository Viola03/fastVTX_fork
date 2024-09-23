import uproot
import uproot3
import numpy as np
import matplotlib.pyplot as plt


# file_new_res = uproot.open(f"/users/am13743/fast_vertexing_variables/rapidsim/Signal_tree_new_res.root:DecayTree")
# file_old_res = uproot.open(f"/users/am13743/fast_vertexing_variables/rapidsim/Signal_tree_old_res.root:DecayTree")

# new_res = file_new_res.arrays(['B_plus_M'], library="pd")['B_plus_M']
# old_res = file_old_res.arrays(['B_plus_M'], library="pd")['B_plus_M']

# plt.hist([new_res,old_res],bins=100,label=['new','old'],histtype='step')
# plt.legend()
# plt.savefig("compare.png")

# quit()


hist_file = uproot.open("/users/am13743/fast_vertexing_variables/rapidsim/electronSmearingHistogram.root")


# Access the histogram (replace 'histogram_name' with the actual name of your histogram)
histogram = hist_file["histE__x"]

# Extract bin contents and edges
bin_values = histogram.values()
bin_edges = histogram.axes[0].edges()

# Plot the histogram using matplotlib
plt.figure(figsize=(8,6))
plt.hist(bin_edges[:-1], bins=bin_edges, weights=bin_values, histtype="step", label="Histogram")
plt.xlabel("Bin")
plt.ylabel("Counts")
plt.title("Histogram from ROOT file")
plt.legend()
plt.savefig('resolution_hist.png')
plt.close('all')

file = uproot.open(f"/users/am13743/fast_vertexing_variables/datasets/general_sample_chargeCounters_cut_more_vars.root:DecayTree")
file_Kee = uproot.open(f"/users/am13743/fast_vertexing_variables/datasets/Kee_Merge_cut_chargeCounters_more_vars.root:DecayTree")
# file = uproot.open(f"/users/am13743/fast_vertexing_variables/datasets/Kee_Merge_cut_chargeCounters_more_vars.root:DecayTree")
# file_Kee = uproot.open(f"/users/am13743/fast_vertexing_variables/datasets/Kee_cut_more_vars.root:DecayTree")


electron_mass = 0.51099895000 #* 1e-3

particles = ["DAUGHTER1","DAUGHTER2","DAUGHTER3"]
alt_particles = ["DAUGHTER1","DAUGHTER2","DAUGHTER3"]
# alt_particles = ["K_Kst","e_minus","e_plus"]

def compute_E(data, particle, true_vars):

    if true_vars:
        E_sq = (data[f'{particle}_TRUEP_X']*1E-3)**2 + (data[f'{particle}_TRUEP_Y']*1E-3)**2 + (data[f'{particle}_TRUEP_Z']*1E-3)**2 + electron_mass**2
    else:
        E_sq = (data[f'{particle}_PX']*1E-3)**2 + (data[f'{particle}_PY']*1E-3)**2 + (data[f'{particle}_PZ']*1E-3)**2 + electron_mass**2

    return np.sqrt(E_sq)#*1e-3 # convert to GeV

residuals = np.empty(0)

residuals_alt = np.empty(0)

for particle in particles:

    branches = [ # MeV
        f'{particle}_TRUEID',
        f'{particle}_TRUEP_X',
        f'{particle}_PX',
        f'{particle}_TRUEP_Y',
        f'{particle}_PY',
        f'{particle}_TRUEP_Z',
        f'{particle}_PZ',

        f'{particle}_TRACK_CHI2NDOF',
        f'{particle}_IPCHI2_OWNPV',
    ]

    data = file.arrays(branches, library="pd")

    shape_pre = data.shape
    data = data.query(f"abs({particle}_TRUEID)==11")
    print(f'Eff: {data.shape[0]/shape_pre[0]}, shape: {data.shape[0]}')

    true_E = compute_E(data, particle, true_vars=True)
    reco_E = compute_E(data, particle, true_vars=False)
    residual = (reco_E-true_E)/true_E
    residuals = np.concatenate((residuals,residual))


for particle in alt_particles:

    branches = [ # MeV
        f'{particle}_TRUEID',
        f'{particle}_TRUEP_X',
        f'{particle}_PX',
        f'{particle}_TRUEP_Y',
        f'{particle}_PY',
        f'{particle}_TRUEP_Z',
        f'{particle}_PZ',

        f'{particle}_TRACK_CHI2NDOF',
        f'{particle}_IPCHI2_OWNPV',
    ]

    data = file_Kee.arrays(branches, library="pd")
    shape_pre = data.shape
    data = data.query(f"abs({particle}_TRUEID)==11")
    print(f'Eff: {data.shape[0]/shape_pre[0]}, shape: {data.shape[0]}')
    true_E = compute_E(data, particle, true_vars=True)
    reco_E = compute_E(data, particle, true_vars=False)
    residual = (reco_E-true_E)/true_E
    residuals_alt = np.concatenate((residuals_alt,residual))


plt.figure(figsize=(12,8))
hist = plt.hist(residuals, bins=bin_edges)
hist_alt = plt.hist(residuals_alt, bins=bin_edges)
plt.close('all')
plt.figure(figsize=(12,8))
plt.hist(bin_edges[:-1], bins=bin_edges, weights=bin_values/np.sum(bin_values), histtype="step", label='RapidSim')
bin_edges_me = hist[1]
bin_values_me = hist[0]
plt.hist(bin_edges_me[:-1], bins=bin_edges_me, weights=bin_values_me/np.sum(bin_values_me), histtype="step", label='Cockail')
bin_edges_me = hist_alt[1]
bin_values_me = hist_alt[0]
plt.hist(bin_edges_me[:-1], bins=bin_edges_me, weights=bin_values_me/np.sum(bin_values_me), histtype="step", label='Kee tuple')
plt.savefig('resolution.png')
plt.close('all')

# # Create a ROOT file to save the histogram
# with uproot.recreate("output_histogram.root") as root_file:
#     # Write the histogram to the ROOT file
#     root_file["histE__x"] = np.histogram(bin_edges_me[:-1], bins=bin_edges_me, weights=bin_values_me)

# print("Histogram written successfully to output_histogram.root")

