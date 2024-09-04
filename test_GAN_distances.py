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

use_intermediate = False

network_option = 'VAE'
load_state = f"networks/vertex_job_{network_option}general_9"
# rd.latent = 7 # VAE latent dims
# D_architecture=[1600,2600,2600,1600]
# G_architecture=[1600,2600,2600,1600]
rd.latent = 10 # VAE latent dims
D_architecture=[1600,2600,2600,1600]
G_architecture=[1600,2600,2600,1600]
rd.beta = 7500.


####################################################################################################################################

transformers = pickle.load(open(f"{load_state}_transfomers.pkl", "rb"))


rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'
rd.intermediate_particle = 'J_psi_1S'

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        # "datasets/general_sample_intermediate_more_vars.root",
        "datasets/general_sample_chargeCounters_cut_more_vars_HEADfactor20.root",
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
    # "m_01",
    # "m_02",
    # "m_12",
    "K_Kst_FLIGHT",
    "e_plus_FLIGHT",
    "e_minus_FLIGHT",
    "delta_0_P",
    "delta_0_PT",
    "delta_1_P",
    "delta_1_PT",
    "delta_2_P",
    "delta_2_PT",
    "K_Kst_TRUEID",
    "e_plus_TRUEID",
    "e_minus_TRUEID",
    # "B_plus_nPositive_missing",
    # "B_plus_nNegative_missing",
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
    "J_psi_1S_IPCHI2_OWNPV",
    # # new targets
    "J_psi_1S_ENDVERTEX_CHI2",
    "J_psi_1S_DIRA_OWNPV",
    # # VertexIsoBDTInfo:
    "B_plus_VTXISOBDTHARDFIRSTVALUE",
    "B_plus_VTXISOBDTHARDSECONDVALUE",
    "B_plus_VTXISOBDTHARDTHIRDVALUE",
    # # TupleToolVtxIsoln:
    # "B_plus_SmallestDeltaChi2OneTrack",
    # "B_plus_SmallestDeltaChi2TwoTracks",
    # # TupleToolTrackIsolation:
    # # "B_plus_cp_0.70",
    # # "B_plus_cpt_0.70",
    # # "B_plus_cmult_0.70",
    # # Ghost:
    "e_plus_TRACK_GhostProb",
    "e_minus_TRACK_GhostProb",
    "K_Kst_TRACK_GhostProb",
]




vertex_quality_trainer_obj = vertex_quality_trainer(
    training_data_loader,
    trackchi2_trainer_obj,
    conditions=conditions,
    targets=targets,
    beta=float(rd.beta),
    latent_dim=rd.latent,
    batch_size=64,
    D_architecture=D_architecture,
    G_architecture=G_architecture,
    network_option=network_option,
)

vertex_quality_trainer_obj.load_state(tag=load_state)
vertex_quality_trainer_obj.make_plots(filename=f'example_training_plots_general',testing_file=training_data_loader.get_file_names(),offline=True)
quit()

# print(f"Initialising BDT tester...")
# BDT_tester_obj = BDT_tester(
#     transformers=transformers,
#     tag="networks/BDT_sig_comb_WGANcocktail_newconditions",
#     # tag="networks/BDT_sig_comb_WGANcocktail_general",
#     train=False,
#     BDT_vars=targets,
#     signal="datasets/dedicated_Kee_MC_hierachy_cut_more_vars.root",
#     background="datasets/B2Kee_2018_CommonPresel.csv",
#     signal_label=r"Signal $B^+\to K^+e^+e^-$ MC",
#     background_label=r"UMSB Combinatorial",
#     gen_track_chi2=False,
#     signal_convert_branches=True,
#     use_intermediate=use_intermediate
# )

# scores = BDT_tester_obj.plot_detailed_metrics(
#     conditions,
#     targets,
#     vertex_quality_trainer_obj, f"metrics_{network_option}.pdf",
#     # only_signal=True,
#     # avoid_rapidsim=True,
# )


# scores = BDT_tester_obj.plot_differential_metrics(
#     conditions,
#     targets,
#     vertex_quality_trainer_obj, f"differential_metrics_{network_option}.pdf",
#     # only_signal=True,
#     BDT_cut=0.9,
#     # avoid_rapidsim=True,
# )
# quit()

print(f"Initialising BDT tester...")
BDT_tester_obj = BDT_tester(
    transformers=transformers,
    tag="networks/BDT_sig_prc_WGANcocktail_newconditions",
    train=False,
    BDT_vars=targets,
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
    conditions,
    targets,
    vertex_quality_trainer_obj, f"metrics_{network_option}_prcBDT.pdf",
    only_signal=False
)

scores = BDT_tester_obj.plot_differential_metrics(
    conditions,
    targets,
    vertex_quality_trainer_obj, f"differential_metrics_{network_option}_prcBDT.pdf",
    # only_signal=True,
    only_signal=False,
    BDT_cut=0.55
)




