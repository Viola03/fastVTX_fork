from fast_vertex_quality.tools.config import read_definition, rd

import tensorflow as tf

# tf.config.run_functions_eagerly(True)

from fast_vertex_quality.training_schemes.track_chi2 import trackchi2_trainer
from fast_vertex_quality.training_schemes.vertex_quality import vertex_quality_trainer
from fast_vertex_quality.testing_schemes.BDT import BDT_tester
import fast_vertex_quality.tools.data_loader as data_loader

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

transformers = pickle.load(open("networks/vertex_job_WGANcocktail_transfomers.pkl", "rb"))

rd.latent = 50 # noise dims

rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'
rd.intermediate_particle = 'J_psi_1S'

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        "datasets/cocktail_three_body_cut.root",
    ],
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
)
transformers = training_data_loader.get_transformers()

training_data_loader.print_branches()

df = training_data_loader.get_branches(['B_plus_TRUEENDVERTEX_Z', 'B_plus_TRUEORIGINVERTEX_Z', 'J_psi_1S_TRUEORIGINVERTEX_Z', 'J_psi_1S_TRUEENDVERTEX_Z','B_plus_TRUEENDVERTEX_X', 'B_plus_TRUEORIGINVERTEX_X', 'J_psi_1S_TRUEORIGINVERTEX_X', 'J_psi_1S_TRUEENDVERTEX_X','B_plus_TRUEP_Z','pass_stripping','B_plus_TRUEID','J_psi_1S_TRUEID'], processed=False)
print(df)
df = df.query('B_plus_TRUEENDVERTEX_Z!=J_psi_1S_TRUEORIGINVERTEX_Z')
df = df.query('B_plus_TRUEORIGINVERTEX_Z!=J_psi_1S_TRUEORIGINVERTEX_Z')
print(df)
df = df.query('B_plus_TRUEP_Z>100 & pass_stripping')
print(df)
# df = df.query('abs(J_psi_1S_TRUEID)==421')
# print(df)
df_full = df.head(n=25)
#print(df_full)
# plt.scatter(df['B_plus_TRUEORIGINVERTEX_X'], df['B_plus_TRUEORIGINVERTEX_Z'])
# plt.scatter(df['B_plus_TRUEENDVERTEX_X'], df['B_plus_TRUEENDVERTEX_Z'])

# Assuming df is your DataFrame and you have the necessary columns

for i in range(25):

    df = df_full.iloc[i]

    print(i, df['B_plus_TRUEID'], df['J_psi_1S_TRUEID'])

    color = 'red'
    x_start = np.asarray(df['B_plus_TRUEORIGINVERTEX_X'])
    z_start = np.asarray(df['B_plus_TRUEORIGINVERTEX_Z'])
    x_end = np.asarray(df['B_plus_TRUEENDVERTEX_X'])
    z_end = np.asarray(df['B_plus_TRUEENDVERTEX_Z'])
    AB = plt.scatter(x_start, z_start, c = color, marker = 'o', s = 10, zorder = 3,alpha=0.5)
    CD = plt.scatter(x_end, z_end, c = color, marker = 'o', s = 10, zorder = 2,alpha=0.5)
    plt.quiver(x_start, z_start, (x_end-x_start), (z_end-z_start), angles='xy', scale_units='xy', scale=1, color=color,alpha=0.5)

    color = 'blue'
    x_start = np.asarray(df['B_plus_TRUEENDVERTEX_X'])
    z_start = np.asarray(df['B_plus_TRUEENDVERTEX_Z'])
    x_end = np.asarray(df['J_psi_1S_TRUEORIGINVERTEX_X'])
    z_end = np.asarray(df['J_psi_1S_TRUEORIGINVERTEX_Z'])
    AB = plt.scatter(x_start, z_start, c = color, marker = 'o', s = 10, zorder = 3,alpha=0.5)
    CD = plt.scatter(x_end, z_end, c = color, marker = 'o', s = 10, zorder = 2,alpha=0.5)
    plt.quiver(x_start, z_start, (x_end-x_start), (z_end-z_start), angles='xy', scale_units='xy', scale=1, color=color,alpha=0.5)


    color = 'green'
    x_start = np.asarray(df['J_psi_1S_TRUEORIGINVERTEX_X'])
    z_start = np.asarray(df['J_psi_1S_TRUEORIGINVERTEX_Z'])
    x_end = np.asarray(df['J_psi_1S_TRUEENDVERTEX_X'])
    z_end = np.asarray(df['J_psi_1S_TRUEENDVERTEX_Z'])
    AB = plt.scatter(x_start, z_start, c = color, marker = 'o', s = 10, zorder = 3,alpha=0.5)
    CD = plt.scatter(x_end, z_end, c = color, marker = 'o', s = 10, zorder = 2,alpha=0.5)
    plt.quiver(x_start, z_start, (x_end-x_start), (z_end-z_start), angles='xy', scale_units='xy', scale=1, color=color,alpha=0.5)

    plt.xlabel('B_plus_TRUEORIGINVERTEX_X')
    plt.ylabel('B_plus_TRUEORIGINVERTEX_Z')
    # plt.title('Arrows from TRUEORIGINVERTEX to TRUEENDVERTEX')
    plt.savefig(f'test_{i}')
    plt.close('all')

# 'MOTHER_TRUEENDVERTEX_X',
# 'MOTHER_TRUEENDVERTEX_Y',
# 'MOTHER_TRUEENDVERTEX_Z',

# 'MOTHER_TRUEORIGINVERTEX_X',
# 'MOTHER_TRUEORIGINVERTEX_Y',
# 'MOTHER_TRUEORIGINVERTEX_Z',

# 'INTERMEDIATE_TRUEENDVERTEX_X',
# 'INTERMEDIATE_TRUEENDVERTEX_Y',
# 'INTERMEDIATE_TRUEENDVERTEX_Z',
# 'INTERMEDIATE_TRUEORIGINVERTEX_X',
# 'INTERMEDIATE_TRUEORIGINVERTEX_Y',
# 'INTERMEDIATE_TRUEORIGINVERTEX_Z',



quit()


print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        "datasets/cocktail_three_body_cut_more_vars.root",
    ],
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
)
transformers = training_data_loader.get_transformers()

df = training_data_loader.get_branches(['J_psi_1S_TRUEID', 'B_plus_TRUEID', 'pass_stripping'], processed=False)
# df = df.query('abs(B_plus_TRUEID)==521 & pass_stripping')
trueIDs = np.asarray(df['B_plus_TRUEID'])
IDs = np.unique(trueIDs)
for ID in IDs:
    shape = np.shape(np.where(trueIDs==ID))[1]
    if ID > 0:
        print(ID, 'n:',shape)
quit()


df = training_data_loader.get_branches(['J_psi_1S_TRUEID', 'B_plus_TRUEID', 'pass_stripping'], processed=False)
df = df.query('abs(B_plus_TRUEID)==521 & pass_stripping')
trueIDs = np.asarray(df['J_psi_1S_TRUEID'])
IDs = np.unique(trueIDs)
intermediate_IDs = []
for ID in IDs:
    shape = np.shape(np.where(trueIDs==ID))[1]
    if shape > 500 and ID > 0:
        print(ID, 'n:',shape)
        intermediate_IDs.append(ID)

conditions = [
    "B_plus_P",
    "B_plus_PT",
    "angle_K_Kst",
    "angle_e_plus",
    "angle_e_minus",
    "K_Kst_eta",
    "e_plus_eta",
    "e_minus_eta",
    "IP_B_plus",
    "IP_K_Kst",
    "IP_e_plus",
    "IP_e_minus",
    "FD_B_plus",
    "DIRA_B_plus",
    "J_psi_1S_FLIGHT",
    "missing_B_plus_P",
    "missing_B_plus_PT",
    "missing_J_psi_1S_P",
    "missing_J_psi_1S_PT",
    "m_01",
    "m_02",
    "m_12",
    ]

targets = [
    "B_plus_ENDVERTEX_CHI2",
    "B_plus_IPCHI2_OWNPV",
    "B_plus_FDCHI2_OWNPV",
    "B_plus_DIRA_OWNPV",
    "K_Kst_IPCHI2_OWNPV",
    "K_Kst_TRACK_CHI2NDOF",
    "e_minus_IPCHI2_OWNPV",
    "e_minus_TRACK_CHI2NDOF",
    "e_plus_IPCHI2_OWNPV",
    "e_plus_TRACK_CHI2NDOF",
    "J_psi_1S_FDCHI2_OWNPV",
]

vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader,
    None,
    conditions=conditions,
    targets=targets,
    beta=float(rd.beta),
    latent_dim=rd.latent,
    batch_size=64,
    D_architecture=[1000,2000,1000],
    G_architecture=[1000,2000,1000],
    network_option='WGAN',
)
vertex_quality_trainer_obj.load_state(tag=f"networks/vertex_job_WGANcocktail")

print(f"Initialising BDT tester...")
BDT_tester_obj = BDT_tester(
    transformers=transformers,
    tag="networks/BDT_sig_comb_WGANcocktail",
    train=False,
    BDT_vars=targets[:-1],
    signal="datasets/Kee_2018_truthed_more_vars.csv",
    background="datasets/B2Kee_2018_CommonPresel.csv",
    signal_label="Train - sig",
    background_label="Train - comb",
    gen_track_chi2=False
)
# scores = BDT_tester_obj.make_BDT_plot(
#     vertex_quality_trainer_obj, f"BDT_job_WGANcocktail.pdf", include_combinatorial=False, include_jpsiX=False
# )

scores = BDT_tester_obj.make_BDT_plot_intermediates(vertex_quality_trainer_obj, f"BDT_intermediates.pdf", include_combinatorial=False, include_jpsiX=False,intermediate_IDs=intermediate_IDs
)

