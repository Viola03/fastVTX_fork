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

rd.use_QuantileTransformer = False

use_intermediate = False
rd.include_dropout = True

# load_state = f"networks/vertex_job_9thSept_C"
# load_state = f"networks/vertex_job_9thSept_D"
# load_state = f"networks/vertex_job_9thSept_E_best"
# load_state = f"networks/vertex_job_9thSept_E"
# load_state = f"networks/vertex_job_9thSept_RAIN"
# load_state = f"networks/vertex_job_9thSept_RAIN_best"
# load_state = f"networks/vertex_job_9thSept_RAIN_best"
# load_state = f"networks/testing_50"
# load_state = f"test_runs/20th_VAE_beta_2000_dropout/networks/20th_VAE_beta_2000_dropout"
# load_state = f"test_runs/20th_long_2000_lower_LR/networks/20th_long_2000_lower_LR"
load_state = f"test_runs/22nf_nomissmass_deeper/networks/22nf_nomissmass_deeper"


####################################################################################################################################

transformers = pickle.load(open(f"{load_state}_transfomers.pkl", "rb"))


rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'
rd.intermediate_particle = 'J_psi_1S'

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        "datasets/general_sample_chargeCounters_cut_more_vars_HEADfactor20.root",
    ],
    transformers=transformers,
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
)
training_data_loader.add_missing_mass_frac_branch()

transformers = training_data_loader.get_transformers()




print(f"Creating vertex_quality_trainer...")

trackchi2_trainer_obj = None

vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader,
    trackchi2_trainer_obj,
    # conditions=conditions,
    # targets=targets,
    # beta=float(rd.beta),
    # latent_dim=rd.latent,
    batch_size=64,
    # D_architecture=D_architecture,
    # G_architecture=G_architecture,
    # network_option=network_option,
    load_config=load_state
)

vertex_quality_trainer_obj.load_state(tag=load_state)
# vertex_quality_trainer_obj.make_plots(filename=f'example_training_plots_general',testing_file=training_data_loader.get_file_names(),offline=True)
# quit()


# print("plot conditions...")
# training_data_loader.plot('conditions.pdf',rd.conditions)
# print("plot targets...")
# training_data_loader.plot('targets.pdf',rd.targets)
# quit()

# BDT_targets = rd.targets
BDT_targets = [
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
"J_psi_1S_IPCHI2_OWNPV"
]

print(f"Initialising BDT tester...")
BDT_tester_obj = BDT_tester(
    transformers=transformers,
    tag="networks/BDT_sig_comb_WGANcocktail_newconditions",
    # tag="networks/BDT_sig_comb_WGANcocktail_general",
    train=False,
    BDT_vars=BDT_targets,
    signal="datasets/dedicated_Kee_MC_hierachy_cut_more_vars.root",
    background="datasets/B2Kee_2018_CommonPresel.csv",
    signal_label=r"Signal $B^+\to K^+e^+e^-$ MC",
    background_label=r"UMSB Combinatorial",
    gen_track_chi2=False,
    signal_convert_branches=True,
    use_intermediate=use_intermediate
)

scores = BDT_tester_obj.plot_detailed_metrics(
    rd.conditions,
    rd.targets,
    vertex_quality_trainer_obj, f"metrics_{rd.network_option}.pdf",
    # only_signal=True,
    # avoid_rapidsim=True,
)

scores = BDT_tester_obj.plot_differential_metrics(
    rd.conditions,
    rd.targets,
    vertex_quality_trainer_obj, f"differential_metrics_{rd.network_option}.pdf",
    # only_signal=True,
    BDT_cut=0.9,
    # avoid_rapidsim=True,
)
quit()

print(f"Initialising BDT tester...")
BDT_tester_obj = BDT_tester(
    transformers=transformers,
    tag="networks/BDT_sig_prc_WGANcocktail_newconditions",
    train=False,
    BDT_vars=BDT_targets,
    # signal="datasets/Kee_2018_truthed_more_vars.csv",
    signal="datasets/dedicated_Kee_MC_hierachy_cut_more_vars.root",
    background="datasets/dedicated_Kstee_MC_hierachy_cut_more_vars.root",
    signal_label=r"Signal $B^+\to K^+e^+e^-$ MC",
    background_label=r"Part. Reco. $B^0\to K^{*0}e^+e^-$ MC",
    gen_track_chi2=False,
    signal_convert_branches=True,
    background_convert_branches=True,
)

scores = BDT_tester_obj.plot_detailed_metrics(
    rd.conditions,
    rd.targets,
    vertex_quality_trainer_obj, f"metrics_{rd.network_option}_prcBDT.pdf",
    # only_signal=True,
)

scores = BDT_tester_obj.plot_differential_metrics(
    rd.conditions,
    rd.targets,
    vertex_quality_trainer_obj, f"differential_metrics_{rd.network_option}_prcBDT.pdf",
    # only_signal=True,
    BDT_cut=0.55
)




