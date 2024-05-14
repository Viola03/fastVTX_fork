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

transformers = pickle.load(open("networks/vertex_all_transfomers.pkl", "rb"))

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        "datasets/Kee_2018_truthed_more_vars.csv",
        "datasets/Kstee_2018_truthed_more_vars.csv",
        "datasets/B2Kee_2018_CommonPresel_more_vars.csv",
    ],
    N=150000,
    transformers=transformers,
)
transformers = training_data_loader.get_transformers()

################################################################
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
]

print(f"Initialising BDT tester...")
BDT_tester_obj = BDT_tester(
    transformers=transformers,
    tag="networks/BDT_sig_comb_all",
    train=False,
    signal="datasets/Kee_2018_truthed_more_vars.csv",
    background="datasets/B2Kee_2018_CommonPresel.csv",
    signal_label="Train - sig",
    background_label="Train - comb",
    gen_track_chi2=False,
)

vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader,
    conditions=conditions,
    targets=targets,
    beta=1000.0,
    latent_dim=2,
    E_architecture=[125, 250, 125],
    D_architecture=[125, 250, 125],
)

# vertex_quality_trainer_obj.train(steps=2500)
# vertex_quality_trainer_obj.save_state(tag="networks/vertex_all")
# vertex_quality_trainer_obj.make_plots()

vertex_quality_trainer_obj.load_state(tag="networks/vertex_all")
scores = BDT_tester_obj.make_BDT_plot(
    vertex_quality_trainer_obj, "BDT_0.pdf", include_combinatorial=True
)
print(scores)
quit()


for i in range(25):
    vertex_quality_trainer_obj.train_more_steps(steps=10000)
    BDT_tester_obj.make_BDT_plot(
        vertex_quality_trainer_obj, f"BDT_{i+1}.pdf", include_combinatorial=True
    )
    vertex_quality_trainer_obj.save_state(tag="networks/vertex_all")

# vertex_quality_trainer_obj.load_state(tag="networks/vertex_all")
################################################################

BDT_tester_obj.make_BDT_plot(
    vertex_quality_trainer_obj, "BDT_all.pdf", include_combinatorial=True
)


# print(f"Initialising BDT tester...")
# BDT_tester_obj_prc = BDT_tester(
#     transformers=transformers,
#     tag="networks/BDT_sig_prc_all",
#     train=True,
#     signal="datasets/Kee_2018_truthed_more_vars.csv",
#     background="datasets/Kstee_2018_truthed_more_vars.csv",
#     signal_label="Train - sig",
#     background_label="Train - prc",
# )

# BDT_tester_obj_prc.make_BDT_plot(vertex_quality_trainer_obj, "BDT_prc.pdf")
