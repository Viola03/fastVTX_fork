from fast_vertex_quality.tools.config import read_definition, rd

import tensorflow as tf

# tf.config.run_functions_eagerly(True)

from fast_vertex_quality.training_schemes.track_chi2 import trackchi2_trainer
from fast_vertex_quality.training_schemes.vertex_quality import vertex_quality_trainer
from fast_vertex_quality.testing_schemes.BDT import BDT_tester
import fast_vertex_quality.tools.new_data_loader as data_loader

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# transformers = pickle.load(
#     open("save_state/track_chi2_QuantileTransformers_e_minus.pkl", "rb")
# )

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        "datasets/Kee_2018_truthed_more_vars.csv",
    ],
    N=50000,
    # transformers=transformers,
)
transformers = training_data_loader.get_transformers()

print(f"Creating track chi2 network trainer...")
trackchi2_trainer_obj = trackchi2_trainer(training_data_loader)

# for particle in ["K_Kst", "e_minus", "e_plus"]:

#     print(f"Training chi2 network {particle}...")
#     trackchi2_trainer_obj.train(particle, steps=10000)
#     trackchi2_trainer_obj.make_plots(particle)

# trackchi2_trainer_obj.save_state(tag="networks/chi2")
trackchi2_trainer_obj.load_state(tag="networks/chi2")
# quit()

training_data_loader = data_loader.load_data(
    [
        "datasets/Kee_2018_truthed_more_vars.csv",
        "datasets/Kstee_2018_truthed_more_vars.csv",
    ],
    N=50000,
    transformers=transformers,
)
training_data_loader.fill_chi2_gen(trackchi2_trainer_obj)

print(f"Creating vertex_quality_trainer...")
vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader, trackchi2_trainer_obj
)
# vertex_quality_trainer_obj.train(steps=25000)
# vertex_quality_trainer_obj.make_plots()
# vertex_quality_trainer_obj.save_state(tag="networks/vertex")
vertex_quality_trainer_obj.load_state(tag="networks/vertex")

print(f"Initialising BDT tester...")
BDT_tester_obj = BDT_tester(
    transformers=transformers,
    tag="networks/BDT",
    train=False,
    signal="datasets/Kee_2018_truthed_more_vars.csv",
    background="datasets/B2Kee_2018_CommonPresel.csv",
)

signal_gen = BDT_tester_obj.get_sample(
    "datasets/Kee_2018_truthed_more_vars.csv",
    vertex_quality_trainer_obj,
    generate=True,
    N=10000,
)

prc_MC = BDT_tester_obj.get_sample(
    "datasets/Kstee_2018_truthed_more_vars.csv",
    None,
    generate=False,
    N=10000,
)

prc_gen = BDT_tester_obj.get_sample(
    "datasets/Kstee_2018_truthed_more_vars.csv",
    vertex_quality_trainer_obj,
    generate=True,
    N=10000,
)

BDT_tester_obj.query_and_plot_samples(
    [signal_gen, prc_MC, prc_gen], ["sig - gen", "prc - MC", "prc - gen"]
)
