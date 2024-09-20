
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
import pandas as pd

from matplotlib.colors import LogNorm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import os

rd.use_QuantileTransformer = False # doesnt work well
# rd.use_QuantileTransformer = True # doesnt work well

use_intermediate = False

rd.current_mse_raw = tf.convert_to_tensor(1.0)

# fix DIRA_B_plus_true_vertex
# need to stabalise training - 


# rd.network_option = 'VAE'
# # load_state = f"networks/vertex_job_9thSept_C" # looks good
# # load_state = f"networks/vertex_job_9thSept_D" decent
# load_state = f"networks/vertex_job_9thSept_RAIN"
# rd.latent = 5 # VAE latent dims
# rd.D_architecture=[1600,2600,2600,1600]
# rd.G_architecture=[1600,2600,2600,1600]

# rd.beta = 750.
# rd.batch_size = 64

# test_tag = 'VAE_latent_10'
# test_tag = 'VAE_latent_10_missing_mass_in_conditions'
# test_tag = 'VAE_beta_2500'
# test_tag = 'VAE_balanced'
# test_tag = 'VAE_apply_grads'
# test_tag = 'VAE_apply_clipped_grads_1'
# test_tag = 'VAE_apply_clipped_grads_05'
# test_tag = 'VAE_lower_LR_extra_layer'
# test_tag = '20th_VAE_beta_2000_nodropout'
# test_tag = '20th_dont_use_schedule'
test_tag = '20th_new_PV_smear'

#/software/am13743/env_may_2024/bin/python3.9 GAN_distances.py -processID 1

test_loc = f'test_runs/{test_tag}/'
try:
	os.mkdir(f'{test_loc}')
	os.mkdir(f'{test_loc}/networks')
except:
	print(f"{test_loc} will be overwritten")
rd.test_loc = test_loc

# 0.92 - didnt improve much yet 
rd.network_option = 'VAE'
load_state = f"{test_loc}/networks/{test_tag}"
# rd.latent = 5 # VAE latent dims
rd.latent = 10 # VAE latent dims
# rd.D_architecture=[1600,2600,1600]
# rd.G_architecture=[1600,2600,1600]
# rd.D_architecture=[512,1024,512]
# rd.G_architecture=[512,1024,512]
# rd.D_architecture=[512,1024,1024,512] # couldnt get to work
# rd.G_architecture=[512,1024,1024,512]
rd.D_architecture=[int(512*1.5),int(1024*1.5),int(1024*1.5),int(512*1.5)] # couldnt get to work
rd.G_architecture=[int(512*1.5),int(1024*1.5),int(1024*1.5),int(512*1.5)]
rd.include_dropout = True
rd.use_beta_schedule = False

# rd.D_architecture=[128,256,512,1024] # decent
# rd.G_architecture=[1024,512,256,128]

# rd.beta = 750. # this is the minimum value of beta, the annealing means it starts from higher

rd.beta = 2000.

# rd.beta = 2500. # maybe good
# rd.batch_size = 1024
rd.batch_size = 256
# rd.batch_size = 64

# # 0.94
# rd.network_option = 'VAE'
# load_state = f"networks/testing"
# rd.latent = 7 # VAE latent dims
# rd.D_architecture=[500,1000,500]
# rd.G_architecture=[500,1000,500]
# rd.beta = 2500.
# rd.batch_size = 64

# rd.network_option = 'VAE'
# load_state = f"networks/testing"
# rd.latent = 7 # VAE latent dims
# rd.D_architecture=[1600,2600,1600]
# rd.G_architecture=[1600,2600,1600]
# rd.beta = 10000.
# rd.batch_size = 64



# rd.network_option = 'WGAN'
# rd.latent = 50 # noise dims
# # rd.latent = 100 # noise dims



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
	"fully_reco",
	"missing_mass_frac",
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

rd.conditional_targets = []
# rd.conditional_targets = ['missing_mass_frac']


####################################################################################################################################


# rd.conditions = [
# 	"B_plus_P",
# 	"B_plus_PT",
# ]

# rd.targets = [
# 	"B_plus_ENDVERTEX_CHI2",
# 	"B_plus_IPCHI2_OWNPV",
# 	"B_plus_FDCHI2_OWNPV",
# 	"B_plus_DIRA_OWNPV",
# ]

####
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
	testing_frac=0.1
	# testing_frac=0.1/20.
)

training_data_loader.add_missing_mass_frac_branch()
# training_data_loader.add_branch_and_process('missing_mass_frac','(B_plus_M-B_plus_M_reco)/B_plus_M')

transformers = training_data_loader.get_transformers()


# data = training_data_loader.get_branches(rd.targets+['B_plus_M','B_plus_M_reco'],processed=False)
# data['missing_mass'] = data.eval('B_plus_M-B_plus_M_reco')
# with PdfPages('2D.pdf') as pdf:
# 	for tar in rd.targets:
# 		plt.hist2d(data['missing_mass'], data[tar], bins=50, norm=LogNorm())
# 		pdf.savefig(bbox_inches="tight")
# 		plt.close()
# quit()


# print(f"Loading Kee...")
# data_loader_Kee = data_loader.load_data(
# 	[
# 		"datasets/Kee_Merge_cut_chargeCounters_more_vars.root",
# 	],
# 	convert_to_RK_branch_names=True,
# 	conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'},
# 	testing_frac=0.08,
# 	transformers=transformers,
# )
# data_loader_Kee.add_missing_mass_frac_branch()


print(training_data_loader.shape())

# training_data_loader.reweight_for_training("fully_reco", weight_value=1., plot_variable='B_plus_M')
# training_data_loader.reweight_for_training("fully_reco", weight_value=50., plot_variable='B_plus_M')
training_data_loader.reweight_for_training("fully_reco", weight_value=100., plot_variable='B_plus_M')

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

# steps_for_plot = 10000
# steps_for_plot = 500
steps_for_plot = 5000
# steps_for_plot = 1550
# steps_for_plot = 50

# vertex_quality_trainer_obj.load_state(tag=load_state)
# vertex_quality_trainer_obj.load_state(tag=f"networks/vertex_job_9thSept_RAIN_best")



# vertex_quality_trainer_obj.train(steps=5000)
# vertex_quality_trainer_obj.save_state(tag='ROC')

# vertex_quality_trainer_obj.load_state(tag='ROC')
# TESTING CODE


def test_with_ROC(training_data_loader_roc, vertex_quality_trainer_obj, it, last_BDT_distributions=None, tag='', weight=True):

	ROC_vars = [tar for tar in rd.targets if tar not in rd.conditional_targets]


	resample = False

	if weight and resample:
		X_test_data_all_pp = training_data_loader_roc.get_branches(
							rd.targets + rd.conditions + ['training_weight'], processed=True, option='testing'
						)
		X_test_data_all_pp = X_test_data_all_pp.sample(frac=1, weights=X_test_data_all_pp['training_weight'],replace=True)
		X_test_data_all_pp = X_test_data_all_pp.drop(columns=['training_weight'])
	else:
		X_test_data_all_pp = training_data_loader_roc.get_branches(
							rd.targets + rd.conditions, processed=True, option='testing'
						)

	print(f'test_with_ROC shape: {X_test_data_all_pp.shape}')

	images_true = np.asarray(X_test_data_all_pp[rd.targets])

	X_test_conditions = X_test_data_all_pp[rd.conditions]
	X_test_conditions = np.asarray(X_test_conditions)
	X_test_targets = X_test_data_all_pp[rd.targets]
	X_test_targets = np.asarray(X_test_targets)

	if rd.network_option == 'VAE':

		z, z_mean, z_log_var = np.asarray(vertex_quality_trainer_obj.encoder([X_test_targets, X_test_conditions]))
		images_cheating = np.asarray(vertex_quality_trainer_obj.decoder([z, X_test_conditions]))

		gen_noise = np.random.normal(0, 1, (np.shape(X_test_conditions)[0], rd.latent))
		images = np.asarray(vertex_quality_trainer_obj.decoder([gen_noise, X_test_conditions]))
	elif rd.network_option == 'WGAN':
		gen_noise = np.random.normal(0, 1, (np.shape(X_test_conditions)[0], rd.latent))
		images = np.asarray(vertex_quality_trainer_obj.generator([gen_noise, X_test_conditions]))
	
	# post-processing didnt help.
	images_dict = {}
	for i in range(len(ROC_vars)):
		# if rd.targets[i] in BDT_targets and "DIRA" not in rd.targets[i]:
		# if rd.targets[i] in BDT_targets:
		images_dict[ROC_vars[i]] = images[:,i]
	images = training_data_loader_roc.post_process(pd.DataFrame(images_dict))

	images_true_dict = {}
	for i in range(len(ROC_vars)):
		# if rd.targets[i] in BDT_targets and "DIRA" not in rd.targets[i]:
		# if rd.targets[i] in BDT_targets:
		images_true_dict[ROC_vars[i]] = images_true[:,i]
	images_true = training_data_loader_roc.post_process(pd.DataFrame(images_true_dict))

	if rd.network_option == 'VAE':
		images_cheating_dict = {}
		for i in range(len(ROC_vars)):
			images_cheating_dict[ROC_vars[i]] = images_cheating[:,i]
		images_cheating = np.squeeze(training_data_loader_roc.post_process(pd.DataFrame(images_cheating_dict)))


	# with PdfPages('testing.pdf') as pdf:
	# 	for i in range(np.shape(images)[1]):
	# 		plt.subplot(1,2,1)
	# 		plt.title(rd.targets[i])
	# 		plt.hist2d(images_true[:,i],images[:,i],bins=75,range=[[-1,1],[-1,1]], norm=LogNorm())
	# 		plt.ylabel('images')
	# 		plt.xlabel('images_true')
	# 		plt.subplot(1,2,2)
	# 		plt.hist([images_true[:,i],images[:,i]],bins=75,range=[-1,1], label=['true','gen'],histtype='step')
	# 		plt.legend()
	# 		plt.xlabel(rd.targets[i])
	# 		pdf.savefig(bbox_inches="tight")
	# 		plt.close()

			

	# take conditions, pass through network

	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

	bdt_train_size = int(np.shape(images)[0]/2)

	real_training_data = np.squeeze(images_true[:bdt_train_size])

	real_test_data = np.squeeze(images_true[bdt_train_size:])

	fake_training_data = np.squeeze(images[:bdt_train_size])

	fake_test_data = np.squeeze(images[bdt_train_size:])

	real_training_labels = np.ones(bdt_train_size)

	fake_training_labels = np.zeros(bdt_train_size)

	total_training_data = np.concatenate((real_training_data, fake_training_data))

	total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

	clf.fit(total_training_data, total_training_labels)

	out_real = clf.predict_proba(real_test_data)

	out_fake = clf.predict_proba(fake_test_data)
	
	if rd.network_option == 'VAE':
		out_cheat = clf.predict_proba(images_cheating)
		plt.hist([out_real[:,1],out_fake[:,1], out_cheat[:,1]], bins = 100,label=['real','gen','gen - cheat'], histtype='step', color=['tab:red','tab:blue','tab:green'], density=True)
	else:
		plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step', color=['tab:red','tab:blue'], density=True)
	if last_BDT_distributions:
		plt.hist(last_BDT_distributions, bins = 100,alpha=0.5, histtype='step', color=['tab:red','tab:blue'], density=True)
	last_BDT_distributions = [out_real[:,1],out_fake[:,1]]
	plt.xlabel('Output of BDT')
	plt.legend(loc='upper right')
	plt.savefig(f'{test_loc}BDT_out_{it}_{rd.network_option}{tag}.png', bbox_inches='tight')
	plt.close('all')

	importance = clf.feature_importances_
	for idx, target in enumerate(ROC_vars):
		print(f'{target}:\t {importance[idx]/np.amax(importance):.2f}')

	ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))


	# Labels: 1 for real, 0 for fake
	y_true = np.append(np.ones(np.shape(out_real[:, 1])), np.zeros(np.shape(out_fake[:, 1])))
	y_scores = np.append(out_real[:, 1], out_fake[:, 1])
	fpr, tpr, thresholds = roc_curve(y_true, y_scores)

	# Plot the ROC curve
	# plt.figure()
	# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_scores):.2f})')
	# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random guessing
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Receiver Operating Characteristic (ROC) Curve')
	# plt.legend(loc="lower right")
	# plt.savefig(f'{test_loc}ROC_out_{it}_{rd.network_option}{tag}.png', bbox_inches='tight')
	# plt.close('all')

	print(ROC_AUC_SCORE_curr)
	return ROC_AUC_SCORE_curr, last_BDT_distributions


ROC_collect = np.empty((0,2))
ROC_collect_Kee = np.empty((0,2))
ROC_collect = np.append(ROC_collect, [[0, 1.]], axis=0)
ROC_collect_Kee = np.append(ROC_collect_Kee, [[0, 1.]], axis=0)

chi2_collect = np.empty((0,3))
chi2_collect_best = np.empty((0,3))

vertex_quality_trainer_obj.train(steps=steps_for_plot)
vertex_quality_trainer_obj.save_state(tag=load_state)
# vertex_quality_trainer_obj.make_plots(filename=f'plots_0.pdf',testing_file=["datasets/general_sample_chargeCounters_cut_more_vars_HEADfactor20.root"])
# vertex_quality_trainer_obj.make_plots(filename=f'example_training_plots_general.pdf',testing_file=training_data_loader.get_file_names(),offline=True)


ROC_AUC_SCORE_curr, last_BDT_distributions = test_with_ROC(training_data_loader, vertex_quality_trainer_obj, 0)
ROC_collect = np.append(ROC_collect, [[0, ROC_AUC_SCORE_curr]], axis=0)
# ROC_AUC_SCORE_curr_Kee, last_BDT_distributions_Kee = test_with_ROC(data_loader_Kee, vertex_quality_trainer_obj, 0, tag='Kee', weight=False)
# ROC_collect_Kee = np.append(ROC_collect_Kee, [[0, ROC_AUC_SCORE_curr_Kee]], axis=0)


vertex_quality_trainer_obj.initalise_BDT_test(BDT_tester_obj, BDT_cut=0.9)
chi2 = vertex_quality_trainer_obj.run_BDT_test(filename=f'{test_loc}plots_BDT_0_{rd.network_option}.pdf')
chi2_collect = np.append(chi2_collect, [chi2], axis=0)
chi2_collect_best = np.append(chi2_collect_best, [chi2], axis=0)
min_mean_chi2 = np.mean(chi2)
best_chi2 = chi2



# for i in range(int(1E30)):
for i in range(250):
	vertex_quality_trainer_obj.train_more_steps(steps=steps_for_plot)
	# vertex_quality_trainer_obj.train_more_steps(steps=steps_for_plot, reset_optimizer_state=True)

	# vertex_quality_trainer_obj.make_plots(filename=f'plots_{i+1}.pdf',testing_file=["datasets/general_sample_chargeCounters_cut_more_vars_HEADfactor20.root"])

	print("test with ROC")
	ROC_AUC_SCORE_curr, last_BDT_distributions = test_with_ROC(training_data_loader, vertex_quality_trainer_obj, i+1, last_BDT_distributions=last_BDT_distributions)
	ROC_collect = np.append(ROC_collect, [[i+1, ROC_AUC_SCORE_curr]], axis=0)
	# ROC_AUC_SCORE_curr_Kee, last_BDT_distributions_Kee = test_with_ROC(data_loader_Kee, vertex_quality_trainer_obj, i+1, tag='Kee', weight=False, last_BDT_distributions=last_BDT_distributions_Kee)
	# ROC_collect_Kee = np.append(ROC_collect_Kee, [[i+1, ROC_AUC_SCORE_curr_Kee]], axis=0)


	print("run BDT test")
	chi2 = vertex_quality_trainer_obj.run_BDT_test(filename=f'{test_loc}plots_BDT_{i+1}_{rd.network_option}.pdf')
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
	plt.savefig(f'{test_loc}Progress_{rd.network_option}')
	plt.close('all')

	plt.plot(chi2_collect_best[:,0],label=r'$q^2$')
	plt.plot(chi2_collect_best[:,1],label=r'$m(Ke)$')
	plt.plot(chi2_collect_best[:,2],label=r'$m(Kee)$')
	plt.legend()
	plt.savefig(f'{test_loc}Progress_best_{rd.network_option}')
	plt.close('all')


	plt.plot(ROC_collect[:,1])
	plt.savefig(f'{test_loc}Progress_ROC_{rd.network_option}')
	plt.close('all')

	# plt.plot(ROC_collect_Kee[:,1])
	# plt.savefig(f'{test_loc}Progress_ROC_{rd.network_option}_Kee')
	# plt.close('all')


