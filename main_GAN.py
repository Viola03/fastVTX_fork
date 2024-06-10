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
        # "datasets/Kee_2018_truthed_more_vars.csv",
        # "datasets/Kstee_2018_truthed_more_vars.csv",
        # "datasets/B2Kee_2018_CommonPresel_more_vars.csv",
        # "datasets/B2KEE_three_body_cut_more_vars.root",
        "datasets/cocktail_three_body_cut_more_vars.root",
    ],
    transformers=transformers,
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
)
transformers = training_data_loader.get_transformers()
print(training_data_loader.shape())
# training_data_loader.cut('abs(B_plus_TRUEID)==521')
# training_data_loader.get_branches(['J_psi_1S_FLIGHT'])
# plt.hist(np.log10(training_data_loader.get_branches(['J_psi_1S_FLIGHT'], processed=False)), bins=50)
# plt.savefig('test')
# plt.close('all')
# training_data_loader.cut('J_psi_1S_FLIGHT<1E-3')
# print(training_data_loader.shape())

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
    "IP_B_plus",
    "IP_K_Kst",
    "IP_e_plus",
    "IP_e_minus",
    "FD_B_plus",
    "DIRA_B_plus",
    "J_psi_1S_FLIGHT"
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

# print(training_data_loader.shape())
# training_data_loader.cut('abs(J_psi_1S_TRUEID)==313')
# print(training_data_loader.shape())
# training_data_loader.plot('dataset_313.pdf',variables=conditions+targets)
# quit()

vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader,
    trackchi2_trainer_obj,
    conditions=conditions,
    targets=targets,
    beta=float(rd.beta),
    latent_dim=rd.latent,
    batch_size=64,
    D_architecture=[1000,2000,1000],
    G_architecture=[1000,2000,1000],
    network_option='WGAN',
)
steps_for_plot = 5000
# steps_for_plot = 50
vertex_quality_trainer_obj.train(steps=steps_for_plot)
vertex_quality_trainer_obj.save_state(tag=f"networks/vertex_job_WGANcocktail_521")
# vertex_quality_trainer_obj.load_state(tag="networks/vertex_job_ROOT2")
vertex_quality_trainer_obj.make_plots(filename=f'plots_0.pdf',testing_file=training_data_loader.get_file_names())

for i in range(100):
    vertex_quality_trainer_obj.train_more_steps(steps=steps_for_plot)
    vertex_quality_trainer_obj.save_state(tag=f"networks/vertex_job_WGANcocktail_521")
    vertex_quality_trainer_obj.make_plots(filename=f'plots_{i+1}.pdf',testing_file=training_data_loader.get_file_names())

# vertex_quality_trainer_obj.load_state(tag=f"networks/vertex_job_WGANcocktail")
# vertex_quality_trainer_obj.gen_data('saved_output_WGANcocktail.root')

print(f"Initialising BDT tester...")
BDT_tester_obj = BDT_tester(
    transformers=transformers,
    tag="networks/BDT_sig_comb_WGANcocktail",
    train=True,
    BDT_vars=targets[:-1],
    signal="datasets/Kee_2018_truthed_more_vars.csv",
    background="datasets/B2Kee_2018_CommonPresel.csv",
    signal_label="Train - sig",
    background_label="Train - comb",
    gen_track_chi2=False
)
scores = BDT_tester_obj.make_BDT_plot(
    vertex_quality_trainer_obj, f"BDT_job_WGANcocktail.pdf", include_combinatorial=False, include_jpsiX=False
)

