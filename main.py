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

transformers = pickle.load(open("networks/vertex_transfomers.pkl", "rb"))

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        "datasets/Kee_2018_truthed_more_vars.csv",
        "datasets/Kstee_2018_truthed_more_vars.csv",
    ],
    N=150000,
    transformers=transformers,
)
transformers = training_data_loader.get_transformers()

################################################################
train_chi2 = False
print(f"Creating track chi2 network trainer...")
trackchi2_trainer_obj = trackchi2_trainer(training_data_loader)
if train_chi2:
    for particle in ["K_Kst", "e_minus", "e_plus"]:
        print(f"Training chi2 network {particle}...")
        trackchi2_trainer_obj.train(particle, steps=10000)
        trackchi2_trainer_obj.make_plots(particle)

    trackchi2_trainer_obj.save_state(tag="networks/chi2")
else:
    trackchi2_trainer_obj.load_state(tag="networks/chi2")
################################################################

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
    # "delta_0_P",
    # "delta_0_PT",
    # "delta_1_P",
    # "delta_1_PT",
    # "delta_2_P",
    # "delta_2_PT",
    "K_Kst_TRACK_CHI2NDOF_gen",
    "e_minus_TRACK_CHI2NDOF_gen",
    "e_plus_TRACK_CHI2NDOF_gen",
]

# branches = training_data_loader.get_branches(conditions + ["file"], processed=False)
# branches_processed = training_data_loader.get_branches(
#     conditions + ["file"], processed=True
# )

# with PdfPages(f"conditions_processed.pdf") as pdf:
#     for condition in conditions:
#         print(condition)
#         plt.figure(figsize=(8, 8))
#         plt.subplot(2, 2, 1)
#         plt.hist(
#             [
#                 branches.query("file==0")[condition],
#                 branches.query("file==1")[condition],
#             ],
#             label=["sig", "prc"],
#             bins=50,
#             histtype="step",
#         )
#         plt.xlabel(condition)
#         plt.subplot(2, 2, 3)
#         plt.hist(
#             [
#                 branches_processed.query("file==0")[condition],
#                 branches_processed.query("file==1")[condition],
#             ],
#             label=["sig", "prc"],
#             bins=50,
#             histtype="step",
#         )
#         plt.xlabel(f"{condition} processed")

#         if condition == "DIRA_B":
#             plt.subplot(2, 2, 4)
#             plt.hist(
#                 [
#                     np.log(1.0 - branches.query("file==0")[condition]),
#                     np.log(1.0 - branches.query("file==1")[condition]),
#                 ],
#                 label=["sig", "prc"],
#                 bins=50,
#                 histtype="step",
#             )
#             plt.xlabel(condition)

#         plt.subplot(2, 2, 2)
#         plt.hist(
#             [
#                 np.log(branches.query("file==0")[condition]),
#                 np.log(branches.query("file==1")[condition]),
#             ],
#             label=["sig", "prc"],
#             bins=50,
#             histtype="step",
#         )
#         plt.xlabel(condition)
#         plt.legend()
#         pdf.savefig(bbox_inches="tight")
#         plt.close()
# quit()


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


vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader, trackchi2_trainer_obj, conditions=conditions
)
vertex_quality_trainer_obj.train(steps=75000)
# BDT_tester_obj.make_BDT_plot(vertex_quality_trainer_obj, "BDT_0.pdf")

# for i in range(25):
#     vertex_quality_trainer_obj.train_more_steps(steps=5000)
#     BDT_tester_obj.make_BDT_plot(vertex_quality_trainer_obj, f"BDT_{i+1}.pdf")


# vertex_quality_trainer_obj.make_plots()
vertex_quality_trainer_obj.save_state(tag="networks/vertex")
# vertex_quality_trainer_obj.load_state(tag="networks/vertex")
################################################################

BDT_tester_obj.make_BDT_plot(vertex_quality_trainer_obj, "BDT.pdf")


print(f"Initialising BDT tester...")
BDT_tester_obj_prc = BDT_tester(
    transformers=transformers,
    tag="networks/BDT_sig_prc",
    train=False,
    signal="datasets/Kee_2018_truthed_more_vars.csv",
    background="datasets/Kstee_2018_truthed_more_vars.csv",
    signal_label="Train - sig",
    background_label="Train - prc",
)

BDT_tester_obj_prc.make_BDT_plot(vertex_quality_trainer_obj, "BDT_prc.pdf")
