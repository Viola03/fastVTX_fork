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

transformers = pickle.load(open("networks/chi2_ROOT_transfomers.pkl", "rb"))

# rd.daughter_particles = ['A','B','C'] # K e e
# rd.mother_particle = 'M' #'B_plus'
rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        "datasets/B2KEE_three_body_cut_more_vars.root",
    ],
    # N=50000,
    transformers=transformers,
    convert_to_RK_branch_names=True,
    conversions={'M_':'B_plus_', 'A_':'K_Kst_', 'B_':'e_plus_', 'C_':'e_minus_', '_A':'_K_Kst', '_B':'_e_plus', '_C':'_e_minus'}
)
transformers = training_data_loader.get_transformers()


################################################################
train_chi2 = False
print(f"Creating track chi2 network trainer...")
trackchi2_trainer_obj = trackchi2_trainer(training_data_loader)
if train_chi2:
    for particle in ["K_Kst", "e_minus", "e_plus"]:
        print(f"Training chi2 network {particle}...")
        trackchi2_trainer_obj.train(particle, steps=7500)
        trackchi2_trainer_obj.make_plots(particle)

    trackchi2_trainer_obj.save_state(tag="networks/chi2_ROOT")
else:
    trackchi2_trainer_obj.load_state(tag="networks/chi2_ROOT")
################################################################
# quit()

################################################################
training_data_loader.fill_chi2_gen(trackchi2_trainer_obj)
print(f"Creating vertex_quality_trainer...")

conditions = [
    "B_P",
    "B_PT",
    "angle_K_Kst",
    "angle_e_plus",
    "angle_e_minus",
    "K_Kst_eta",
    "e_plus_eta",
    "e_minus_eta",
    "IP_B",
    "IP_K_Kst",
    "IP_e_plus",
    "IP_e_minus",
    "FD_B",
    "DIRA_B",
    "K_Kst_TRACK_CHI2NDOF_gen",
    "e_minus_TRACK_CHI2NDOF_gen",
    "e_plus_TRACK_CHI2NDOF_gen",
]

print(f"Initialising BDT tester...")
BDT_tester_obj = BDT_tester(
    transformers=transformers,
    tag="networks/BDT_sig_comb",
    train=False,
    signal="datasets/Kee_2018_truthed_more_vars.csv",
    background="datasets/B2Kee_2018_CommonPresel.csv",
    signal_label="Train - sig",
    background_label="Train - comb",
)

# training_data_loader.print_branches()


vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader,
    trackchi2_trainer_obj,
    conditions=conditions,
    beta=float(rd.beta),
    latent_dim=rd.latent,
    E_architecture=[125, 125, 250, 125],
    D_architecture=[125, 250, 125, 125],
)
# vertex_quality_trainer_obj.train(steps=25000)
# vertex_quality_trainer_obj.save_state(tag=f"networks/vertex_job_ROOT")
vertex_quality_trainer_obj.load_state(tag="networks/vertex_job_ROOT")
################################################################

scores = BDT_tester_obj.make_BDT_plot(
    vertex_quality_trainer_obj, f"BDT_job_ROOT.pdf", include_combinatorial=False, include_jpsiX=True
)

# print(float(rd.beta), scores)

# f = open("logging.txt", "a")
# f.write(f"{float(rd.beta)}, {rd.latent}, {scores[0]}, {scores[1]}, {scores[2]}\n")
# f.close()


# print(f"Initialising BDT tester...")
# BDT_tester_obj_prc = BDT_tester(
#     transformers=transformers,
#     tag="networks/BDT_sig_prc",
#     train=True,
#     signal="datasets/Kee_2018_truthed_more_vars.csv",
#     background="datasets/Kstee_2018_truthed_more_vars.csv",
#     signal_label="Train - sig",
#     background_label="Train - prc",
# )

# BDT_tester_obj_prc.make_BDT_plot(vertex_quality_trainer_obj, "BDT_prc.pdf")
