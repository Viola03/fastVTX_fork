from fast_vertex_quality_inference.processing.data_manager import tuple_manager
from fast_vertex_quality_inference.processing.network_manager import network_manager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from fast_vertex_quality_inference.example.Kee_selection import analyser
import fast_vertex_quality_inference.example.plotter as plotter

# Ntests = 50
Ntests = 50

#### 
# Kee
###

tag = 'Kee'
latex_tag = r"$B^+\to K^+e^+e^-$"

def load_up_a_rapidsim_tuple():
	return tuple_manager(
						# tuple_location="/users/am13743/fast_vertexing_variables/rapidsim/Kee/Signal_tree_LARGE.root",
						tuple_location="/users/am13743/fast_vertexing_variables/understanding_rapidsim/new_momenta.root",
						particles_TRUEID=[321, 11, 11],
						fully_reco=1,
						nPositive_missing_particles=0,
						nNegative_missing_particles=0,
						mother_particle_name="B_plus",
						intermediate_particle_name="J_psi",
						daughter_particle_names=["K_plus","e_plus","e_minus"],
						entry_stop=50000,
	)

data_tuple = load_up_a_rapidsim_tuple()

MC_data_tuple = tuple_manager(
					tuple_location="/users/am13743/fast_vertexing_variables/datasets/Kee_Merge_cut_chargeCounters_more_vars.root",
					particles_TRUEID=[321, 11, 11],
					fully_reco=1,
					nPositive_missing_particles=0,
					nNegative_missing_particles=0,
					mother_particle_name="B_plus",
					intermediate_particle_name="J_psi_1S",
					daughter_particle_names=["K_Kst","e_plus","e_minus"],
					entry_stop=50000,
					)
MC_data_tuple.tuple['MOTHER_M'] = MC_data_tuple.tuple['MOTHER_M_reco']
Kee_analyser_MC = analyser(MC_data_tuple.tuple, "pass_stripping", "BDT")


# ### 
# # Kstree
# ##

# tag = 'Kstree'
# latex_tag = r"$B^0\to K^{*0}e^+e^-$"


# def load_up_a_rapidsim_tuple():
# 	return tuple_manager(
# 					tuple_location="/users/am13743/fast_vertexing_variables/rapidsim/Kstree/Partreco_tree_LARGE.root",
# 					particles_TRUEID=[321, 11, 11],
# 					fully_reco=0,
# 					nPositive_missing_particles=1,
# 					nNegative_missing_particles=0,
# 					mother_particle_name="B_plus",
# 					intermediate_particle_name="J_psi",
# 					daughter_particle_names=["K_plus","e_plus","e_minus"],
# 					entry_stop=50000,
# 					)
# data_tuple = load_up_a_rapidsim_tuple()

# MC_data_tuple = tuple_manager(
# 					tuple_location="/users/am13743/fast_vertexing_variables/datasets/Kstzeroee_Merge_chargeCounters_cut_more_vars.root",
# 					particles_TRUEID=[321, 11, 11],
# 					fully_reco=0,
# 					nPositive_missing_particles=1,
# 					nNegative_missing_particles=0,
# 					mother_particle_name="B_plus",
# 					intermediate_particle_name="J_psi_1S",
# 					daughter_particle_names=["K_Kst","e_plus","e_minus"],
# 					entry_stop=50000,
# 					)
# MC_data_tuple.tuple['MOTHER_M'] = MC_data_tuple.tuple['MOTHER_M_reco']
# Kee_analyser_MC = analyser(MC_data_tuple.tuple, "pass_stripping", "BDT")




# #### 
# # BuD0piKenu
# ###

# tag = 'BuD0piKenu'
# latex_tag = r"$B^+\to \bar{D}^{0}(\to K^+e^-\bar{\nu}_e)\pi^+$"

# def load_up_a_rapidsim_tuple():
# 	return tuple_manager(
# 					tuple_location="/users/am13743/fast_vertexing_variables/rapidsim/BuD0piKenu/BuD0piKenu_tree.root",
# 					particles_TRUEID=[321, 11, 211],
# 					fully_reco=0,
# 					nPositive_missing_particles=0,
# 					nNegative_missing_particles=0,
# 					mother_particle_name="B_plus",
# 					intermediate_particle_name="J_psi",
# 					daughter_particle_names=["K_plus","e_plus","e_minus"],
# 					entry_stop=50000,
# 					)
# data_tuple = load_up_a_rapidsim_tuple()

# MC_data_tuple = tuple_manager(
# 					tuple_location="/users/am13743/fast_vertexing_variables/datasets/BuD0piKenu_Merge_chargeCounters_cut_more_vars.root",
# 					particles_TRUEID=[321, 11, 211],
# 					fully_reco=0,
# 					nPositive_missing_particles=0,
# 					nNegative_missing_particles=0,
# 					mother_particle_name="B_plus",
# 					intermediate_particle_name="J_psi_1S",
# 					daughter_particle_names=["K_Kst","e_plus","e_minus"],
# 					entry_stop=50000,
# 					)
# MC_data_tuple.tuple['MOTHER_M'] = MC_data_tuple.tuple['MOTHER_M_reco']
# Kee_analyser_MC = analyser(MC_data_tuple.tuple, "pass_stripping", "BDT")






#### 
# INITIALISE NETWORKS
###

rapidsim_PV_smearing_network = network_manager(
					network="inference/example/models/smearing_decoder_model.onnx", 
					config="inference/example/models/smearing_configs.pkl", 
					transformers="inference/example/models/smearing_transfomers.pkl", 
					)

vertexing_network = network_manager(
					network="inference/example/models/vertexing_decoder_model.onnx", 
					config="inference/example/models/vertexing_configs.pkl", 
					transformers="inference/example/models/vertexing_transfomers.pkl", 
					)

vertexing_encoder = network_manager(
					network="inference/example/models/vertexing_encoder_model.onnx", 
					config="inference/example/models/vertexing_configs.pkl", 
					transformers="inference/example/models/vertexing_transfomers.pkl", 
					)


#### 
# SMEAR PV
###

smearing_conditions = data_tuple.get_branches(
					rapidsim_PV_smearing_network.conditions, 
					rapidsim_PV_smearing_network.Transformers, 
					numpy=True,
					)
smeared_PV_output = rapidsim_PV_smearing_network.query_network(
					['noise',smearing_conditions],
					)
data_tuple.smearPV(smeared_PV_output)



#### 
# COMPUTE CONDITIONS AND RUN VERTEXING NETWORK
###

data_tuple.append_conditional_information()
vertexing_conditions = data_tuple.get_branches(
					vertexing_network.conditions, 
					vertexing_network.Transformers, 
					numpy=True,
					)
noise = np.random.normal(0, 1, (np.shape(vertexing_conditions)[0], vertexing_network.latent_dim))
vertexing_output = vertexing_network.query_network(
					[noise,vertexing_conditions],
					)
data_tuple.add_branches(
					vertexing_output
					)




#### 
# PLAY WITH SELECTION
###


Kee_analyser = analyser(data_tuple.tuple, "pass_stripping", "BDT")

print(Kee_analyser)
print(Kee_analyser_MC)

with PdfPages(f"systematics_plots/{tag}_Vars.pdf") as pdf:

	plotter.variable_plotter(pdf, 'BDT', Kee_analyser_MC, Kee_analyser, 'pass_stripping>0.5 and MOTHER_M>4.2 and MOTHER_M<5.8', 'MC', 'Gen', title='BDT after stripping')
	plotter.variable_plotter(pdf, 'BDT', Kee_analyser_MC, Kee_analyser, 'pass_stripping>0.5 and MOTHER_M>4.2 and MOTHER_M<5.8', 'MC', 'Gen', log=True, title='BDT after stripping')

	# for var in vertexing_network.targets:
	# 	plotter.variable_plotter(pdf, f'{var}', Kee_analyser_MC, Kee_analyser, 'MOTHER_M>4.2 and MOTHER_M<5.8', 'MC', 'Gen', title=f'{var} before stripping')
	# 	plotter.variable_plotter(pdf, f'{var}', Kee_analyser_MC, Kee_analyser, 'pass_stripping>0.5 and MOTHER_M>4.2 and MOTHER_M<5.8', 'MC', 'Gen', title=f'{var} after stripping')

# quit()

with PdfPages(f"systematics_plots/{tag}_diff.pdf") as pdf:

	plotter.plot_efficiency_as_a_function_of_variable(pdf,
	tuple_A=Kee_analyser_MC.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
	tuple_B=Kee_analyser.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
	label_A='MC',
	label_B='Gen',
	variable="MOTHER_M",cut="pass_stripping>0.5",range_array=[4,5.7],title=f"{latex_tag} Stripping", xlabel=r'$m(Kee)$ (GeV)', signal=True
	)
	
	plotter.plot_efficiency_as_a_function_of_variable(pdf,
	tuple_A=Kee_analyser_MC.data.query("pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7"),
	tuple_B=Kee_analyser.data.query("pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7"),
	label_A='MC',
	label_B='Gen',
	variable="MOTHER_M",cut="BDT>0.9",range_array=[4,5.7],title=f"{latex_tag} BDT (given Stripping)", xlabel=r'$m(Kee)$ (GeV)', signal=True
	)

	plotter.plot_efficiency_as_a_function_of_variable(pdf,
	tuple_A=Kee_analyser_MC.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
	tuple_B=Kee_analyser.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
	label_A='MC',
	label_B='Gen',
	variable="MOTHER_M",cut="BDT>0.9 and pass_stripping>0.5",range_array=[4,5.7],title=f"{latex_tag} Stripping + BDT", xlabel=r'$m(Kee)$ (GeV)', signal=True
	)
quit()


def repass_network(tuple_in):
	repass_vertexing_conditions = tuple_in.get_branches(
						vertexing_network.conditions, 
						vertexing_network.Transformers, 
						numpy=True,
						)
	repass_vertexing_targets = tuple_in.get_branches(
						vertexing_network.targets, 
						vertexing_network.Transformers, 
						numpy=True,
						)
	encoder_output_z = vertexing_encoder.query_network(
						[repass_vertexing_targets,repass_vertexing_conditions],
						process=False,
						numpy=True,
						ignore_targets=True,
						)
	repass_vertexing_output = vertexing_network.query_network(
						[encoder_output_z,repass_vertexing_conditions],
						)

	data_tuple_alt = load_up_a_rapidsim_tuple()

	data_tuple_alt.add_branches(
						repass_vertexing_output
						)

	return data_tuple_alt


def renoise_network(tuple_in):
	repass_vertexing_conditions = tuple_in.get_branches(
						vertexing_network.conditions, 
						vertexing_network.Transformers, 
						numpy=True,
						)
	repass_vertexing_output = vertexing_network.query_network(
						['noise',repass_vertexing_conditions],
						)

	data_tuple_alt = load_up_a_rapidsim_tuple()

	data_tuple_alt.add_branches(
						repass_vertexing_output
						)

	return data_tuple_alt


def re_query_drop_condition(tuple_in, noise_in, condition_idx):
	
	repass_vertexing_conditions = tuple_in.get_branches(
						vertexing_network.conditions, 
						vertexing_network.Transformers, 
						numpy=True,
						)
	
	if not isinstance(condition_idx, list):
		condition_idx = [condition_idx]

	if len(condition_idx) == 1:
		np.random.shuffle(repass_vertexing_conditions[:,condition_idx[0]])
	else:
		indexes = np.arange(0,np.shape(repass_vertexing_conditions)[0])
		indexes_shuffle = np.arange(0,np.shape(repass_vertexing_conditions)[0])
		np.random.shuffle(indexes_shuffle)

		for i in range(np.shape(repass_vertexing_conditions)[1]):
			if i in condition_idx:
				repass_vertexing_conditions[:,i] = repass_vertexing_conditions[indexes_shuffle,i]
			else:
				repass_vertexing_conditions[:,i] = repass_vertexing_conditions[indexes,i]



	repass_vertexing_output = vertexing_network.query_network(
						[noise_in,repass_vertexing_conditions],
						)

	data_tuple_alt = load_up_a_rapidsim_tuple()

	data_tuple_alt.add_branches(
						repass_vertexing_output
						)

	return data_tuple_alt

def get_surviving_events(tuple_i, variable, cut):
	return tuple_i.data.query(cut).eval(variable)


####################################################################################################
# SYSTEMATIC SHUFFLING
####################################################################################################

with PdfPages(f"systematics_plots/{tag}_shuffling.pdf") as pdf:

	events = get_surviving_events(Kee_analyser, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')

	pairs = [
		["MOTHER_P", "MOTHER_PT"],
		["missing_MOTHER_P", "missing_MOTHER_PT"],
		["missing_INTERMEDIATE_P", "missing_INTERMEDIATE_PT"],
		["delta_0_P", "delta_0_PT"],
		["delta_1_P", "delta_1_PT"],
		["delta_2_P", "delta_2_PT"],
		]
	
	nBins = 20

	ratio_results = np.empty((0,4,nBins))

	for pair in pairs:
		pair_idx = []
		for pair_i in pair:
			idx = np.where(np.asarray(vertexing_network.conditions)==pair_i)[0][0]
			pair_idx.append(idx)
		
		print(pair[0], pair[1])

		data_tuple_alt = re_query_drop_condition(data_tuple, noise, pair_idx)
		Kee_analyser_alt = analyser(data_tuple_alt.tuple, "pass_stripping", "BDT")

		events_alt = get_surviving_events(Kee_analyser_alt, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')

		ratio_i = plotter.make_shuffle_impact_plots(pdf, events, events_alt, 'Shuffled', bins=nBins, title=f'{pair[0]} and {pair[1]}')

		ratio_results = np.append(ratio_results, [ratio_i], axis=0)

	for condition_idx, condition in enumerate(vertexing_network.conditions):

		if condition not in list(np.asarray(pairs).flatten()):
			
			print(condition)

			data_tuple_alt = re_query_drop_condition(data_tuple, noise, condition_idx)
			Kee_analyser_alt = analyser(data_tuple_alt.tuple, "pass_stripping", "BDT")

			events_alt = get_surviving_events(Kee_analyser_alt, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')

			ratio_i = plotter.make_shuffle_impact_plots(pdf, events, events_alt, 'Shuffled', bins=nBins, title=condition)

			ratio_results = np.append(ratio_results, [ratio_i], axis=0)
	

	plt.axhline(y=1,c='k')
	for i in range(np.shape(ratio_results)[0]):
		ratio_results_i = ratio_results[i]
		bin_midpoints = ratio_results_i[0]
		ratio = ratio_results_i[1]
		ratio_err = ratio_results_i[2]
		bin_widths = ratio_results_i[3]
		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2)

	plt.ylabel('Ratio (Shuffled/Default)')
	plt.xlabel('MOTHER_M')

	pdf.savefig(bbox_inches="tight")
	plt.close()


	fig, ax1 = plt.subplots()
	plt.axhline(y=1,c='k')
	for i in range(np.shape(ratio_results)[0]):
		ratio_results_i = ratio_results[i]
		bin_midpoints = ratio_results_i[0]
		ratio = ratio_results_i[1]
		ratio_err = ratio_results_i[2]
		bin_widths = ratio_results_i[3]
		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2,alpha=0.25)
	
	ax2 = ax1.twinx()
	ax2.plot(bin_midpoints, np.std(ratio_results[:,1,:],axis=0), c='k', marker='o')
	ax2.set_ylabel('Std')

	ax1.set_ylabel('Ratio (Shuffled/Default)')
	ax1.set_xlabel('MOTHER_M')

	pdf.savefig(bbox_inches="tight")
	plt.close()


	ratio_results[:,1,:][np.where(ratio_results[:,1,:]<1.)] = 1./ratio_results[:,1,:][np.where(ratio_results[:,1,:]<1.)]


	for i in range(np.shape(ratio_results)[0]):
		ratio_results_i = ratio_results[i]
		bin_midpoints = ratio_results_i[0]
		ratio = ratio_results_i[1]-1.
		ratio_err = ratio_results_i[2]
		bin_widths = ratio_results_i[3]

		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2, alpha=0.25)

		if i == 0:
			ratio_max = ratio
			ratio_max_err = ratio_err
		else:
			where = np.where(ratio>ratio_max)
			ratio_max[where] = ratio[where]
			ratio_max_err[where] = ratio_err[where]
	
	plt.errorbar(bin_midpoints, ratio_max, yerr=ratio_max_err, xerr=bin_widths / 2, color='k', fmt='o', capsize=2)

	plt.ylabel('Ratio (symmetric) - 1.')
	plt.xlabel('MOTHER_M')
	plt.ylim(ymin=0)
	pdf.savefig(bbox_inches="tight")
	plt.close()





####################################################################################################
# SYSTEMATIC REPASS
####################################################################################################

data_tuple_alt = repass_network(data_tuple)
Kee_analyser_alt = analyser(data_tuple_alt.tuple, "pass_stripping", "BDT")


with PdfPages(f"systematics_plots/{tag}_repass.pdf") as pdf:

	plotter.plot_efficiency_as_a_function_of_variable(pdf,
	tuple_A=Kee_analyser.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
	tuple_B=Kee_analyser_alt.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
	label_A='Default',
	label_B='Re-passed',
	variable="MOTHER_M",cut="BDT>0.9 and pass_stripping>0.5",range_array=[4,5.7],title=r"$B^+\to K^+e^+e^-$ Stripping + BDT", xlabel=r'$m(Kee)$ (GeV)', signal=True
	)


nBins = 20
ratio_results = np.empty((0,4,nBins))

with PdfPages(f"systematics_plots/{tag}_repass_syst.pdf") as pdf:

	events = get_surviving_events(Kee_analyser, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')

	for repass in range(Ntests):
		print(repass)
		makeplot = True
		if repass > 9: makeplot = False

		data_tuple_alt = repass_network(data_tuple)
		Kee_analyser_alt = analyser(data_tuple_alt.tuple, "pass_stripping", "BDT")
		events_alt = get_surviving_events(Kee_analyser_alt, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')
		# events

		ratio_i = plotter.make_shuffle_impact_plots(pdf, events, events_alt, 'Re-passed', bins=nBins, title=f'Re-pass',makeplot=makeplot)

		ratio_results = np.append(ratio_results, [ratio_i], axis=0)

		####################################################################################################

	plt.axhline(y=1,c='k')
	for i in range(np.shape(ratio_results)[0]):
		ratio_results_i = ratio_results[i]
		bin_midpoints = ratio_results_i[0]
		ratio = ratio_results_i[1]
		ratio_err = ratio_results_i[2]
		bin_widths = ratio_results_i[3]
		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2)

	plt.ylabel('Ratio (Re-pass/Default)')
	plt.xlabel('MOTHER_M')

	pdf.savefig(bbox_inches="tight")
	plt.close()

	mean_std = np.empty((0,nBins))
	mean_std = np.append(mean_std, [np.std(ratio_results[:,1,:],axis=0)], axis=0)

	fig, ax1 = plt.subplots()
	plt.axhline(y=1,c='k')
	for i in range(np.shape(ratio_results)[0]):
		ratio_results_i = ratio_results[i]
		bin_midpoints = ratio_results_i[0]
		ratio = ratio_results_i[1]
		ratio_err = ratio_results_i[2]
		bin_widths = ratio_results_i[3]
		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2,alpha=0.25)
	
	ax2 = ax1.twinx()
	ax2.plot(bin_midpoints, np.std(ratio_results[:,1,:],axis=0), c='k', marker='o')
	ax2.set_ylabel('Std')

	ax1.set_ylabel('Ratio (Re-pass/Default)')
	ax1.set_xlabel('MOTHER_M')

	pdf.savefig(bbox_inches="tight")
	plt.close()

	ratio_results[:,1,:][np.where(ratio_results[:,1,:]<1.)] = 1./ratio_results[:,1,:][np.where(ratio_results[:,1,:]<1.)]
	ratio_results[:,1,:] += -1.

	fig, ax1 = plt.subplots()

	for i in range(np.shape(ratio_results)[0]):
		ratio_results_i = ratio_results[i]
		bin_midpoints = ratio_results_i[0]
		ratio = ratio_results_i[1]
		ratio_err = ratio_results_i[2]
		bin_widths = ratio_results_i[3]
		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2,alpha=0.25)
	
	plt.ylabel('Ratio (symmetric) - 1.')
	plt.ylim(ymin=0.)

	ax2 = ax1.twinx()
	ax2.plot(bin_midpoints, np.mean(ratio_results[:,1,:],axis=0), c='k', marker='o')
	ax2.set_ylabel('Mean')
	ax2.set_ylim(ymin=0.)


	ax1.set_xlabel('MOTHER_M')

	pdf.savefig(bbox_inches="tight")
	plt.close()


	mean_std = np.append(mean_std, [np.mean(ratio_results[:,1,:],axis=0)], axis=0)

	plt.plot(bin_midpoints, np.amax(mean_std,axis=0), c='k', marker='o')
	plt.ylabel('Max(mean,std)')
	plt.ylim(ymin=0.)

	pdf.savefig(bbox_inches="tight")
	plt.close()


####################################################################################################
# SYSTEMATIC RENOISE
####################################################################################################

data_tuple_alt = renoise_network(data_tuple)
Kee_analyser_alt = analyser(data_tuple_alt.tuple, "pass_stripping", "BDT")

with PdfPages(f"systematics_plots/{tag}_renoise.pdf") as pdf:

	plotter.plot_efficiency_as_a_function_of_variable(pdf,
	tuple_A=Kee_analyser.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
	tuple_B=Kee_analyser_alt.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
	label_A='Default',
	label_B='Re-noised',
	variable="MOTHER_M",cut="BDT>0.9 and pass_stripping>0.5",range_array=[4,5.7],title=r"$B^+\to K^+e^+e^-$ Stripping + BDT", xlabel=r'$m(Kee)$ (GeV)', signal=True
	)

nBins = 20
ratio_results = np.empty((0,4,nBins))

with PdfPages(f"systematics_plots/{tag}_renoise_syst.pdf") as pdf:

	events = get_surviving_events(Kee_analyser, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')

	for repass in range(Ntests):
		print(repass)
		makeplot = True
		if repass > 9: makeplot = False

		data_tuple_alt = renoise_network(data_tuple)
		Kee_analyser_alt = analyser(data_tuple_alt.tuple, "pass_stripping", "BDT")
		events_alt = get_surviving_events(Kee_analyser_alt, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')
		# events

		ratio_i = plotter.make_shuffle_impact_plots(pdf, events, events_alt, 'Re-noised', bins=nBins, title=f'Re-noise', makeplot=makeplot)

		ratio_results = np.append(ratio_results, [ratio_i], axis=0)

		####################################################################################################

	plt.axhline(y=1,c='k')
	for i in range(np.shape(ratio_results)[0]):
		ratio_results_i = ratio_results[i]
		bin_midpoints = ratio_results_i[0]
		ratio = ratio_results_i[1]
		ratio_err = ratio_results_i[2]
		bin_widths = ratio_results_i[3]
		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2)

	plt.ylabel('Ratio (Re-noise/Default)')
	plt.xlabel('MOTHER_M')

	pdf.savefig(bbox_inches="tight")
	plt.close()

	mean_std = np.empty((0,nBins))
	mean_std = np.append(mean_std, [np.std(ratio_results[:,1,:],axis=0)], axis=0)

	fig, ax1 = plt.subplots()
	plt.axhline(y=1,c='k')
	for i in range(np.shape(ratio_results)[0]):
		ratio_results_i = ratio_results[i]
		bin_midpoints = ratio_results_i[0]
		ratio = ratio_results_i[1]
		ratio_err = ratio_results_i[2]
		bin_widths = ratio_results_i[3]
		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2,alpha=0.25)
	
	ax2 = ax1.twinx()
	ax2.plot(bin_midpoints, np.std(ratio_results[:,1,:],axis=0), c='k', marker='o')
	ax2.set_ylabel('Std')

	ax1.set_ylabel('Ratio (Re-noise/Default)')
	ax1.set_xlabel('MOTHER_M')

	pdf.savefig(bbox_inches="tight")
	plt.close()

	ratio_results[:,1,:][np.where(ratio_results[:,1,:]<1.)] = 1./ratio_results[:,1,:][np.where(ratio_results[:,1,:]<1.)]
	ratio_results[:,1,:] += -1.

	fig, ax1 = plt.subplots()

	for i in range(np.shape(ratio_results)[0]):
		ratio_results_i = ratio_results[i]
		bin_midpoints = ratio_results_i[0]
		ratio = ratio_results_i[1]
		ratio_err = ratio_results_i[2]
		bin_widths = ratio_results_i[3]
		plt.errorbar(bin_midpoints, ratio, yerr=ratio_err, xerr=bin_widths / 2, fmt='o', capsize=2,alpha=0.25)
	
	plt.ylabel('Ratio (symmetric) - 1.')
	plt.ylim(ymin=0.)

	ax2 = ax1.twinx()
	ax2.plot(bin_midpoints, np.mean(ratio_results[:,1,:],axis=0), c='k', marker='o')
	ax2.set_ylabel('Mean')
	ax2.set_ylim(ymin=0.)


	ax1.set_xlabel('MOTHER_M')

	pdf.savefig(bbox_inches="tight")
	plt.close()


	mean_std = np.append(mean_std, [np.mean(ratio_results[:,1,:],axis=0)], axis=0)

	plt.plot(bin_midpoints, np.amax(mean_std,axis=0), c='k', marker='o')
	plt.ylabel('Max(mean,std)')
	plt.ylim(ymin=0.)

	pdf.savefig(bbox_inches="tight")
	plt.close()