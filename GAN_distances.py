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

rd.use_QuantileTransformer = False # doesnt work well

use_intermediate = False

# network_option = 'WGAN'
# rd.latent = 50 # noise dims

rd.network_option = 'VAE'
# load_state = f"networks/vertex_job_9thSept_C" # looks good
# load_state = f"networks/vertex_job_9thSept_D" decent
load_state = f"networks/vertex_job_9thSept_RAIN"
# rd.latent = 6 # VAE latent dims
# rd.latent = 4 # VAE latent dims
rd.latent = 5 # VAE latent dims
rd.D_architecture=[1600,2600,2600,1600]
rd.G_architecture=[1600,2600,2600,1600]
# rd.D_architecture=[1600/2,2600/2,2600/2,1600/2]
# rd.G_architecture=[1600/2,2600/2,2600/2,1600/2]
rd.beta = 750.
# rd.batch_size = 64
rd.batch_size = 100


# network_option = 'VAE'
# load_state = f"networks/vertex_job_testing"
# rd.latent = 7 # VAE latent dims
# rd.D_architecture=[1600,2600,2600,1600]
# rd.G_architecture=[1600,2600,2600,1600]
# rd.beta = 750.


rd.conditions = [
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
	"B_plus_nPositive_missing",
	"B_plus_nNegative_missing",
]

rd.targets = [
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


####################################################################################################################################


rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'
rd.intermediate_particle = 'J_psi_1S'

print(f"Loading data...")
training_data_loader = data_loader.load_data(
	[
		# "datasets/general_sample_chargeCounters_cut_more_vars.root",
		"datasets/general_sample_chargeCounters_cut_more_vars_HEADfactor20.root",
	],
	convert_to_RK_branch_names=True,
	conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'},
)
transformers = training_data_loader.get_transformers()
print(training_data_loader.shape())

# # temporary function!
training_data_loader.reweight_for_training("B_plus_M", weight_value=100.)
# training_data_loader.reweight_for_training("B_plus_M", weight_value=50.)

# from matplotlib.backends.backend_pdf import PdfPages

# with PdfPages('example.pdf') as pdf:

# 	for col in rd.targets:
# 		processed=training_data_loader.get_branches([col],processed=True)
# 		unprocessed=training_data_loader.get_branches([col],processed=False)

# 		try:
# 			plt.subplot(2,2,1)
# 			plt.title(col)
# 			plt.hist(processed[col], bins=50)
# 			plt.subplot(2,2,2)
# 			plt.hist(np.log10(unprocessed[col]), bins=50)
# 			unprocessed_2 = training_data_loader.Transformers[col].unprocess(np.asarray(processed[col]).copy())
# 			plt.subplot(2,2,3)
# 			plt.hist(np.log10(unprocessed_2), bins=50)
# 			plt.subplot(2,2,4)
# 			plt.hist([np.log10(unprocessed[col]),np.log10(unprocessed_2)], histtype='step', bins=50)
# 			pdf.savefig(bbox_inches="tight")
# 			plt.close()
# 		except:
# 			pass
# 		# plt.close()
# 		# plt.subplot(2,2,1)
# 		# plt.title(col)
# 		# plt.hist(processed[col], bins=50)
# 		# plt.subplot(2,2,2)
# 		# plt.hist(unprocessed[col], bins=50)
# 		# unprocessed_2 = training_data_loader.Transformers[col].unprocess(np.asarray(processed[col]).copy())
# 		# plt.subplot(2,2,3)
# 		# plt.hist(unprocessed_2, bins=50)
# 		# plt.subplot(2,2,4)
# 		# plt.hist([unprocessed[col],unprocessed_2], histtype='step', bins=50)
# 		# pdf.savefig(bbox_inches="tight")
# 		# plt.close()

# quit()




print(f"Creating vertex_quality_trainer...")

trackchi2_trainer_obj = None


# training_data_loader.print_branches()

# print("plot conditions...")
# training_data_loader.plot('conditions.pdf',rd.conditions)
# print("plot targets...")
# training_data_loader.plot('targets.pdf',rd.targets)
# quit()


vertex_quality_trainer_obj = vertex_quality_trainer(
	training_data_loader,
	trackchi2_trainer_obj,
	conditions=rd.conditions,
	targets=rd.targets,
	beta=float(rd.beta),
	latent_dim=rd.latent,
	batch_size=rd.batch_size,
	D_architecture=rd.D_architecture,
	G_architecture=rd.G_architecture,
	network_option=rd.network_option,
)

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

steps_for_plot = 5000
# steps_for_plot = 50

# vertex_quality_trainer_obj.load_state(tag=load_state)

chi2_collect = np.empty((0,3))
chi2_collect_best = np.empty((0,3))

vertex_quality_trainer_obj.train(steps=steps_for_plot)
vertex_quality_trainer_obj.save_state(tag=load_state)
# vertex_quality_trainer_obj.make_plots(filename=f'plots_0.pdf',testing_file=["datasets/general_sample_chargeCounters_cut_more_vars_HEADfactor20.root"])
# vertex_quality_trainer_obj.make_plots(filename=f'example_training_plots_general.pdf',testing_file=training_data_loader.get_file_names(),offline=True)
vertex_quality_trainer_obj.initalise_BDT_test(BDT_tester_obj, BDT_cut=0.9)
chi2 = vertex_quality_trainer_obj.run_BDT_test(filename=f'plots_BDT_0.pdf')
chi2_collect = np.append(chi2_collect, [chi2], axis=0)
chi2_collect_best = np.append(chi2_collect_best, [chi2], axis=0)
min_mean_chi2 = np.mean(chi2)
best_chi2 = chi2

for i in range(70):
	vertex_quality_trainer_obj.train_more_steps(steps=steps_for_plot)
	# vertex_quality_trainer_obj.make_plots(filename=f'plots_{i+1}.pdf',testing_file=["datasets/general_sample_chargeCounters_cut_more_vars_HEADfactor20.root"])
	chi2 = vertex_quality_trainer_obj.run_BDT_test(filename=f'plots_BDT_{i+1}.pdf')
	chi2_collect = np.append(chi2_collect, [chi2], axis=0)

	vertex_quality_trainer_obj.save_state(tag=load_state)

	mean_chi2 = np.mean(chi2)
	if mean_chi2 < min_mean_chi2:
		print("NEW BEST MEAN_CHI2")
		min_mean_chi2 = mean_chi2
		vertex_quality_trainer_obj.save_state(tag=load_state+"_best")
		best_chi2 = chi2

	chi2_collect_best = np.append(chi2_collect_best, [best_chi2], axis=0)


	plt.plot(chi2_collect[:,0],label=r'$q^2$')
	plt.plot(chi2_collect[:,1],label=r'$m(Ke)$')
	plt.plot(chi2_collect[:,2],label=r'$m(Kee)$')
	plt.legend()
	plt.savefig('Progress')
	plt.close('all')

	plt.plot(chi2_collect_best[:,0],label=r'$q^2$')
	plt.plot(chi2_collect_best[:,1],label=r'$m(Ke)$')
	plt.plot(chi2_collect_best[:,2],label=r'$m(Kee)$')
	plt.legend()
	plt.savefig('Progress_best')
	plt.close('all')


