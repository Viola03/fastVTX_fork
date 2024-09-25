from fast_vertex_quality_inference.processing.data_manager import tuple_manager
from fast_vertex_quality_inference.processing.network_manager import network_manager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from fast_vertex_quality_inference.example.Kee_selection import analyser
import fast_vertex_quality_inference.example.plotter as plotter


data_tuple = tuple_manager(
					tuple_location="/users/am13743/fast_vertexing_variables/datasets/Kee_Merge_cut_chargeCounters_more_vars.root",
					particles_TRUEID=[321, 11, 11],
					fully_reco=1,
					nPositive_missing_particles=0,
					nNegative_missing_particles=0,
					mother_particle_name="B_plus",
					intermediate_particle_name="J_psi_1S",
					daughter_particle_names=["K_Kst","e_plus","e_minus"],
					)
data_tuple.tuple['MOTHER_M'] = data_tuple.tuple['MOTHER_M_reco']
Kee_analyser_MC = analyser(data_tuple.tuple, "pass_stripping", "BDT")


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
# LOAD RAPIDSIM TUPLE
###


data_tuple = tuple_manager(
					tuple_location="/users/am13743/fast_vertexing_variables/inference/example/example.root",
					particles_TRUEID=[321, 11, 11],
					fully_reco=1,
					nPositive_missing_particles=0,
					nNegative_missing_particles=0,
					mother_particle_name="B_plus",
					intermediate_particle_name="J_psi",
					daughter_particle_names=["K_plus","e_plus","e_minus"],
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



# #### 
# # WRITE TUPLE
# ###

# data_tuple.write(new_branches_to_keep=vertexing_network.targets)




#### 
# PLAY WITH SELECTION
###


Kee_analyser = analyser(data_tuple.tuple, "pass_stripping", "BDT")

print(Kee_analyser)
print(Kee_analyser_MC)

# with PdfPages("systematics_plots/Vars.pdf") as pdf:

# 	plotter.variable_plotter(pdf, 'BDT', Kee_analyser_MC, Kee_analyser, 'pass_stripping>0.5 and MOTHER_M>4.2 and MOTHER_M<5.8', 'MC', 'Gen', title='BDT after stripping')
# 	plotter.variable_plotter(pdf, 'BDT', Kee_analyser_MC, Kee_analyser, 'pass_stripping>0.5 and MOTHER_M>4.2 and MOTHER_M<5.8', 'MC', 'Gen', log=True, title='BDT after stripping')

# 	# for var in vertexing_network.targets:
# 	# 	plotter.variable_plotter(pdf, f'{var}', Kee_analyser_MC, Kee_analyser, 'MOTHER_M>4.2 and MOTHER_M<5.8', 'MC', 'Gen', title=f'{var} before stripping')
# 	# 	plotter.variable_plotter(pdf, f'{var}', Kee_analyser_MC, Kee_analyser, 'pass_stripping>0.5 and MOTHER_M>4.2 and MOTHER_M<5.8', 'MC', 'Gen', title=f'{var} after stripping')


# with PdfPages("systematics_plots/diff.pdf") as pdf:

# 	plotter.plot_efficiency_as_a_function_of_variable(pdf,
# 	tuple_A=Kee_analyser_MC.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
# 	tuple_B=Kee_analyser.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
# 	label_A='MC',
# 	label_B='Gen',
# 	variable="MOTHER_M",cut="pass_stripping>0.5",range_array=[4,5.7],title=r"$B^+\to K^+e^+e^-$ Stripping", xlabel=r'$m(Kee)$ (GeV)', signal=True
# 	)
	
# 	plotter.plot_efficiency_as_a_function_of_variable(pdf,
# 	tuple_A=Kee_analyser_MC.data.query("pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7"),
# 	tuple_B=Kee_analyser.data.query("pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7"),
# 	label_A='MC',
# 	label_B='Gen',
# 	variable="MOTHER_M",cut="BDT>0.9",range_array=[4,5.7],title=r"$B^+\to K^+e^+e^-$ BDT (given Stripping)", xlabel=r'$m(Kee)$ (GeV)', signal=True
# 	)

# 	plotter.plot_efficiency_as_a_function_of_variable(pdf,
# 	tuple_A=Kee_analyser_MC.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
# 	tuple_B=Kee_analyser.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
# 	label_A='MC',
# 	label_B='Gen',
# 	variable="MOTHER_M",cut="BDT>0.9 and pass_stripping>0.5",range_array=[4,5.7],title=r"$B^+\to K^+e^+e^-$ Stripping + BDT", xlabel=r'$m(Kee)$ (GeV)', signal=True
# 	)




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

	data_tuple_alt = tuple_manager(
						tuple_location="/users/am13743/fast_vertexing_variables/inference/example/example.root",
						particles_TRUEID=[321, 11, 11],
						fully_reco=1,
						nPositive_missing_particles=0,
						nNegative_missing_particles=0,
						mother_particle_name="B_plus",
						intermediate_particle_name="J_psi",
						daughter_particle_names=["K_plus","e_plus","e_minus"],
						)
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

	data_tuple_alt = tuple_manager(
						tuple_location="/users/am13743/fast_vertexing_variables/inference/example/example.root",
						particles_TRUEID=[321, 11, 11],
						fully_reco=1,
						nPositive_missing_particles=0,
						nNegative_missing_particles=0,
						mother_particle_name="B_plus",
						intermediate_particle_name="J_psi",
						daughter_particle_names=["K_plus","e_plus","e_minus"],
						)
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
	np.random.shuffle(repass_vertexing_conditions[:,condition_idx])

	repass_vertexing_output = vertexing_network.query_network(
						[noise_in,repass_vertexing_conditions],
						)

	data_tuple_alt = tuple_manager(
						tuple_location="/users/am13743/fast_vertexing_variables/inference/example/example.root",
						particles_TRUEID=[321, 11, 11],
						fully_reco=1,
						nPositive_missing_particles=0,
						nNegative_missing_particles=0,
						mother_particle_name="B_plus",
						intermediate_particle_name="J_psi",
						daughter_particle_names=["K_plus","e_plus","e_minus"],
						)
	data_tuple_alt.add_branches(
						repass_vertexing_output
						)

	return data_tuple_alt

def get_surviving_events(tuple_i, variable, cut):
	return tuple_i.data.query(cut).eval(variable)




####################################################################################################
data_tuple_alt = renoise_network(data_tuple)
Kee_analyser_alt = analyser(data_tuple_alt.tuple, "pass_stripping", "BDT")

events = get_surviving_events(Kee_analyser, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')
events_alt = get_surviving_events(Kee_analyser_alt, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')

plt.hist([events,events_alt], bins=50, histtype='step')
plt.savefig("test_renoise.png")
plt.close('all')
####################################################################################################


####################################################################################################
data_tuple_alt = repass_network(data_tuple)
Kee_analyser_alt = analyser(data_tuple_alt.tuple, "pass_stripping", "BDT")

events = get_surviving_events(Kee_analyser, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')
events_alt = get_surviving_events(Kee_analyser_alt, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')

plt.hist([events,events_alt], bins=50, histtype='step')
plt.savefig("test_repass.png")
plt.close('all')
####################################################################################################


####################################################################################################
with PdfPages("systematics_plots/shuffling.pdf") as pdf:

	events = get_surviving_events(Kee_analyser, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')

	for condition_idx, condition in enumerate(vertexing_network.conditions):

		print(condition_idx, condition, len(vertexing_network.conditions))

		data_tuple_alt = re_query_drop_condition(data_tuple, noise, condition_idx)
		Kee_analyser_alt = analyser(data_tuple_alt.tuple, "pass_stripping", "BDT")

		events_alt = get_surviving_events(Kee_analyser_alt, 'MOTHER_M', 'BDT>0.9 and pass_stripping>0.5 and MOTHER_M>4 and MOTHER_M<5.7')

		# plt.title(condition)
		# plt.hist([events,events_alt], bins=50, histtype='step')
		# pdf.savefig(bbox_inches="tight")
		# plt.close()

		plt.title(condition)
		plt.hist([events,events_alt], bins=50, histtype='step', density=True)
		pdf.savefig(bbox_inches="tight")
		plt.close()
		
		# plt.savefig("test_shuffle.png")
		# plt.close('all')
####################################################################################################





# with PdfPages("systematics_plots/repass.pdf") as pdf:

# 	plotter.plot_efficiency_as_a_function_of_variable(pdf,
# 	tuple_A=Kee_analyser.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
# 	tuple_B=Kee_analyser_alt.data.query("MOTHER_M>4 and MOTHER_M<5.7"),
# 	label_A='Default',
# 	label_B='Repassed',
# 	variable="MOTHER_M",cut="BDT>0.9 and pass_stripping>0.5",range_array=[4,5.7],title=r"$B^+\to K^+e^+e^-$ Stripping + BDT", xlabel=r'$m(Kee)$ (GeV)', signal=True
# 	)
