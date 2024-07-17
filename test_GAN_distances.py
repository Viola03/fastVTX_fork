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


from particle import Particle


transformers = pickle.load(open("networks/vertex_job_WGANcocktail_distances_newconditions_transfomers.pkl", "rb"))

rd.latent = 50 # noise dims

rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'
rd.intermediate_particle = 'J_psi_1S'

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        "datasets/cocktail_hierarchy_cut_more_vars.root",
    ],
    transformers=transformers,
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
)
transformers = training_data_loader.get_transformers()


print(f"Creating vertex_quality_trainer...")

trackchi2_trainer_obj = None

conditions = [
    "B_plus_P",
    "B_plus_PT",
    "angle_K_Kst",
    "angle_e_plus",
    "angle_e_minus",
    "K_Kst_eta",
    "e_plus_eta",
    "e_minus_eta",
    # "IP_B_plus",
    # "IP_K_Kst",
    # "IP_e_plus",
    # "IP_e_minus",
    # "FD_B_plus",
    # "DIRA_B_plus",
    "IP_B_plus_true_vertex",
    "IP_K_Kst_true_vertex",
    "IP_e_plus_true_vertex",
    "IP_e_minus_true_vertex",
    "FD_B_plus_true_vertex",
    "DIRA_B_plus_true_vertex",
    "missing_B_plus_P",
    "missing_B_plus_PT",
    "missing_J_psi_1S_P",
    "missing_J_psi_1S_PT",
    "m_01",
    "m_02",
    "m_12",

    # "B_plus_FLIGHT",
    "K_Kst_FLIGHT",
    "e_plus_FLIGHT",
    "e_minus_FLIGHT",

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

# network_option = 'WGAN'
network_option = 'VAE'

vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader,
    trackchi2_trainer_obj,
    conditions=conditions,
    targets=targets,
    beta=float(rd.beta),
    latent_dim=rd.latent,
    batch_size=64,
    D_architecture=[1000,2000,2000,1000],
    G_architecture=[1000,2000,2000,1000],
    network_option=network_option,
)

vertex_quality_trainer_obj.load_state(tag=f"networks/vertex_job_{network_option}cocktail_distances_newconditions")
# vertex_quality_trainer_obj.gen_data(f'saved_output_WGANcocktail_hierarchy_{network_option}.root')
# quit()

print(f"Initialising BDT tester...")
# BDT_tester_obj = BDT_tester(
#     transformers=transformers,
#     tag="networks/BDT_sig_comb_WGANcocktail_newconditions",
#     train=False,
#     BDT_vars=targets[:-1],
#     signal="datasets/Kee_2018_truthed_more_vars.csv",
#     background="datasets/B2Kee_2018_CommonPresel.csv",
#     signal_label="Train - sig",
#     background_label="Train - comb",
#     gen_track_chi2=False
# )

print(f"Initialising BDT tester...")
BDT_tester_obj = BDT_tester(
    transformers=transformers,
    tag="networks/BDT_sig_comb_WGANcocktail_newconditions",
    train=False,
    BDT_vars=targets,
    signal="datasets/Kee_2018_truthed_more_vars.csv",
    background="datasets/B2Kee_2018_CommonPresel.csv",
    signal_label="Train - sig",
    background_label="Train - comb",
    gen_track_chi2=False
)

# BDT_tester_obj.get_vars_of_samples_that_pass_a_cut(
#     vertex_quality_trainer_obj, conditions, save=True, filename="BuD0enuKenu_passing_BDT.root")
# quit()

scores = BDT_tester_obj.make_BDT_plot_hierarchy(
    vertex_quality_trainer_obj, f"BDT_job_WGANcocktail_newconditions{network_option}.pdf", include_combinatorial=False, include_jpsiX=False
)
