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


transformers = pickle.load(open("networks/chi2_ROOT2_transfomers.pkl", "rb"))

rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'
rd.intermediate_particle = 'J_psi_1S'

rd.beta = 1000
rd.latent = 3
rd.batch_size = 1

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        "datasets/B2KEE_three_body_cut_more_vars.root",
    ],
    N=150000,
    transformers=transformers,
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
)
transformers = training_data_loader.get_transformers()


################################################################
train_chi2 = True
print(f"Creating track chi2 network trainer...")
trackchi2_trainer_obj = trackchi2_trainer(training_data_loader,E_architecture=[250,250,250],
            D_architecture=[250,250,250])
if train_chi2:
    for particle in ["K_Kst", "e_minus", "e_plus"]:
        print(f"Training chi2 network {particle}...")
        trackchi2_trainer_obj.train(particle, steps=7500)
        # trackchi2_trainer_obj.train(particle, steps=75)
        trackchi2_trainer_obj.make_plots(particle)

    trackchi2_trainer_obj.save_state(tag="networks/chi2_ROOT2")
else:
    trackchi2_trainer_obj.load_state(tag="networks/chi2_ROOT2")
################################################################
# quit()

training_data_loader.fill_chi2_gen(trackchi2_trainer_obj)
print(f"Creating vertex_quality_trainer...")

targets = [
    "B_plus_ENDVERTEX_CHI2",
    "B_plus_IPCHI2_OWNPV",
    "B_plus_FDCHI2_OWNPV",
    "B_plus_DIRA_OWNPV",
    "K_Kst_IPCHI2_OWNPV",
    # "K_Kst_TRACK_CHI2NDOF",
    "e_minus_IPCHI2_OWNPV",
    # "e_minus_TRACK_CHI2NDOF",
    "e_plus_IPCHI2_OWNPV",
    # "e_plus_TRACK_CHI2NDOF",
    "J_psi_1S_FDCHI2_OWNPV",
]

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
    "K_Kst_TRACK_CHI2NDOF_gen",
    "e_minus_TRACK_CHI2NDOF_gen",
    "e_plus_TRACK_CHI2NDOF_gen",
]

# training_data_loader.print_branches()

vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader,
    trackchi2_trainer_obj,
    conditions=conditions,
    targets=targets,
    beta=float(rd.beta),
    latent_dim=rd.latent,
    batch_size=rd.batch_size,
    # E_architecture=[125, 125, 250, 125],
    # D_architecture=[125, 250, 125, 125],
    # E_architecture=[1024, 1024, 1024, 1024, 1024],
    # D_architecture=[1024, 1024, 1024, 1024, 1024],
    E_architecture=[1024, 1024, 1024],
    D_architecture=[1024, 1024, 1024],
    # E_architecture=[2024, 2024],
    # D_architecture=[2024, 2024],
)
vertex_quality_trainer_obj.train(steps=10000)
vertex_quality_trainer_obj.save_state(tag=f"networks/vertex_job_ROOT2")
# vertex_quality_trainer_obj.load_state(tag=f"networks/vertex_job_ROOT2")

vertex_quality_trainer_obj.make_plots()

vertex_quality_trainer_obj.gen_data('saved_output.root')