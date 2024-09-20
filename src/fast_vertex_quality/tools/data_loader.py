import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from fast_vertex_quality.tools.config import rd, read_definition
import tensorflow as tf
import uproot

import uproot3 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pickle
from particle import Particle
from hep_ml.reweight import BinsReweighter, GBReweighter, FoldingReweighter
from termcolor import colored

# from fast_vertex_quality.tools.transformers import OriginalTransformer as Transformer
from fast_vertex_quality.tools.transformers import UpdatedTransformer as Transformer


def write_df_to_root(df, output_name):
	branch_dict = {}
	data_dict = {}
	dtypes = df.dtypes
	used_columns = [] # stop repeat columns, kpipi_correction was getting repeated
	for dtype, branch in enumerate(df.keys()):
		if branch not in used_columns:
			if dtypes[dtype] == 'uint32': dtypes[dtype] = 'int32'
			if dtypes[dtype] == 'uint64': dtypes[dtype] = 'int64'
			branch_dict[branch] = dtypes[dtype]
			# stop repeat columns, kpipi_correction was getting repeated
			if np.shape(df[branch].shape)[0] > 1:
				data_dict[branch] = df[branch].iloc[:, 0]
			else:
				data_dict[branch] = df[branch]
		used_columns.append(branch)
	with uproot3.recreate(output_name) as f:
		f["DecayTree"] = uproot3.newtree(branch_dict)
		f["DecayTree"].extend(data_dict)


def produce_physics_variables(data):

	physics_variables = {}

	for particle_i in rd.daughter_particles:

		physics_variables[f"{particle_i}_P"] = np.sqrt(
			data[f"{particle_i}_PX"] ** 2
			+ data[f"{particle_i}_PY"] ** 2
			+ data[f"{particle_i}_PZ"] ** 2
		)

		physics_variables[f"{particle_i}_PT"] = np.sqrt(
			data[f"{particle_i}_PX"] ** 2 + data[f"{particle_i}_PY"] ** 2
		)

		physics_variables[f"{particle_i}_eta"] = -np.log(
			np.tan(
				np.arcsin(
					physics_variables[f"{particle_i}_PT"]
					/ physics_variables[f"{particle_i}_P"]
				)
				/ 2.0
			)
		)

	physics_variables["kFold"] = np.random.randint(
		low=0,
		high=9,
		size=np.shape(data[f"{rd.daughter_particles[0]}_PX"])[0],
	)

	electron_mass = 0.51099895000 * 1e-3

	PE = np.sqrt(
		electron_mass**2
		+ data[f"{rd.daughter_particles[1]}_PX"] ** 2
		+ data[f"{rd.daughter_particles[1]}_PY"] ** 2
		+ data[f"{rd.daughter_particles[1]}_PZ"] ** 2
	) + np.sqrt(
		electron_mass**2
		+ data[f"{rd.daughter_particles[2]}_PX"] ** 2
		+ data[f"{rd.daughter_particles[2]}_PY"] ** 2
		+ data[f"{rd.daughter_particles[2]}_PZ"] ** 2
	)
	PX = data[f"{rd.daughter_particles[1]}_PX"] + data[f"{rd.daughter_particles[2]}_PX"]
	PY = data[f"{rd.daughter_particles[1]}_PY"] + data[f"{rd.daughter_particles[2]}_PY"]
	PZ = data[f"{rd.daughter_particles[1]}_PZ"] + data[f"{rd.daughter_particles[2]}_PZ"]

	physics_variables["q2"] = (PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6

	df = pd.DataFrame.from_dict(physics_variables)

	return df


class NoneError(Exception):
	pass


class dataset:

	def __init__(self, filenames, transformers=None, name=''):

		self.Transformers = transformers
		self.all_data = {"processed": None, "physical": None}
		self.filenames = filenames

		self.reweight_for_training_bool = False
		self.name = name

	def fill_stripping_bool(self):

		self.all_data["physical"]["pass_stripping"] = np.zeros(self.all_data["physical"].shape[0])

		# print(self.all_data["physical"]["pass_stripping"])

		if 'B_plus_ENDVERTEX_NDOF' not in list(self.all_data["physical"].keys()):
			self.all_data["physical"]["B_plus_ENDVERTEX_NDOF"] = np.ones(self.all_data["physical"].shape[0])*3.

		cuts = {}
		cuts['B_plus_FDCHI2_OWNPV'] = ">100."
		cuts['B_plus_DIRA_OWNPV'] = ">0.9995"
		cuts['B_plus_IPCHI2_OWNPV'] = "<25"
		cuts['(B_plus_ENDVERTEX_CHI2/B_plus_ENDVERTEX_NDOF)'] = "<9"
		# cuts['J_psi_1S_PT'] = ">0"
		cuts['J_psi_1S_FDCHI2_OWNPV'] = ">16"
		cuts['J_psi_1S_IPCHI2_OWNPV'] = ">0"
		for lepton in ['e_minus', 'e_plus']:
			cuts[f'{lepton}_IPCHI2_OWNPV'] = ">9"
			# cuts[f'{lepton}_PT'] = ">300"
		for hadron in ['K_Kst']:
			cuts[f'{hadron}_IPCHI2_OWNPV'] = ">9"
			# cuts[f'{hadron}_PT'] = ">400"
		# cuts['m_12'] = "<5500"
		# cuts['B_plus_M_Kee_reco'] = ">(5279.34-1500)"
		# cuts['B_plus_M_Kee_reco'] = "<(5279.34+1500)"

		if isinstance(cuts, dict):
			cut_string = ''
			for cut_idx, cut_i in enumerate(list(cuts.keys())):
				if cut_idx > 0:
					cut_string += ' & '
				if cut_i == 'extra_cut':
					cut_string += f'{cuts[cut_i]}'
				else:
					cut_string += f'{cut_i}{cuts[cut_i]}'
			cuts = cut_string   
		
		# gen_tot_val = self.all_data['physical'].shape[0]
		try:
			cut_array = self.all_data['physical'].query(cuts)
			self.all_data["physical"].loc[cut_array.index,"pass_stripping"] = 1.
		except Exception as e:
			# for key in list(self.all_data['physical'].keys()): print(key)
			print(f"\n\nAn error occurred: {e}")
			print("continuing with pass_stripping = 1\n")
			self.all_data["physical"]["pass_stripping"] = np.ones(self.all_data["physical"].shape[0])

		# print('\n pass_stripping',self.all_data["physical"]["pass_stripping"])

	def print_branches(self):
		for key in list(self.all_data["physical"].keys()):
			print(key)

	def sample_with_replacement_with_reweight(self, target_loader, reweight_vars):

		original = []
		for var in reweight_vars:
			original.append(self.all_data['processed'][var])
		
		target_branches = target_loader.get_branches(reweight_vars, processed=True)

		target = []
		for var in reweight_vars:
			target.append(target_branches[var])

		original = np.swapaxes(np.asarray(original),0,1)
		target = np.swapaxes(np.asarray(target),0,1)

		print("Using GBReweighter to reweight then re-select data...")
		reweighter_base = GBReweighter(max_depth=2, gb_args={'subsample': 0.5})
		reweighter = FoldingReweighter(reweighter_base, n_folds=3)
		reweighter.fit(original=original, target=target)
		MC_weights = reweighter.predict_weights(original)
		MC_weights = np.clip(MC_weights, a_min=0, a_max=5.)
		
		N = self.all_data['processed'].shape[0]
		indexes = np.random.choice(np.arange(N), size=N, replace=True, p=MC_weights/np.sum(MC_weights))

		self.all_data['processed'] = self.all_data['processed'].iloc[indexes]
		self.all_data['physical'] = self.all_data['physical'].iloc[indexes]

		self.all_data['processed'].reset_index(drop=True, inplace=True)
		self.all_data['physical'].reset_index(drop=True, inplace=True)


	def fill(self, data, turn_off_processing=False, avoid_physics_variables=False, testing_frac=0.1):

		self.turn_off_processing = turn_off_processing

		if not isinstance(data, pd.DataFrame):
			raise NoneError("Dataset must be a pd.dataframe.")

		in_training = np.ones(data.shape[0])
		# in_training[np.random.choice(np.arange(data.shape[0]),size=int(testing_frac*data.shape[0]))] = 0
		in_training[-int(testing_frac*data.shape[0]):] = 0
		data['in_training'] = in_training

		self.all_data["physical"] = data
		if self.turn_off_processing:
			return
		
		if not avoid_physics_variables:
			self.physics_variables = produce_physics_variables(self.all_data["physical"])
			shared = list(
				set(list(self.physics_variables.keys())).intersection(
					set(list(self.all_data["physical"].keys()))
				)
			)
			difference = list(
				set(list(self.physics_variables.keys())).difference(
					set(list(self.all_data["physical"].keys()))
				)
			)
			if len(shared) > 0:
				for key in shared:
					self.all_data["physical"][key] = self.physics_variables[key]
			if len(difference) > 0:
				self.physics_variables = self.physics_variables[difference]
				self.all_data["physical"] = pd.concat(
					(self.all_data["physical"], self.physics_variables), axis=1
				)

			self.all_data["physical"] = self.all_data["physical"].loc[
				:, ~self.all_data["physical"].columns.str.contains("^Unnamed")
			]

		self.fill_stripping_bool()

		self.all_data["processed"] = self.pre_process(self.all_data["physical"])

	def fill_chi2_gen(self, trackchi2_trainer_obj):

		for particle_i in rd.daughter_particles:

			# decoder_chi2 = tf.keras.models.load_model(
			#     f"save_state/track_chi2_decoder_{particle_i}.h5"
			# )
			latent_dim_chi2 = 1

			conditions_i = [
				f"{particle_i}_PX",
				f"{particle_i}_PY",
				f"{particle_i}_PZ",
				f"{particle_i}_P",
				f"{particle_i}_PT",
				f"{particle_i}_eta",
			]

			X_test_conditions = self.get_branches(conditions_i, processed=True)
			X_test_conditions = X_test_conditions[conditions_i]
			X_test_conditions = np.asarray(X_test_conditions)

			gen_noise = np.random.normal(
				0, 1, (np.shape(X_test_conditions)[0], latent_dim_chi2)
			)

			images = np.squeeze(
				trackchi2_trainer_obj.predict(
					particle_i, [gen_noise, X_test_conditions]
				)
			)

			self.fill_new_column(
				images,
				f"{particle_i}_TRACK_CHI2NDOF_gen",
				f"{particle_i}_TRACK_CHI2NDOF",
				processed=True,
			)

	def post_process(self, processed_data):

		df = {}

		for column in list(processed_data.keys()):
			if column == "file" or column == "training_weight":
				df[column] = processed_data[column]
			else:
				df[column] = self.Transformers[column].unprocess(
					np.asarray(processed_data[column]).copy()
				)

		return pd.DataFrame.from_dict(df)

	def update_transformer(self, variable, new_transformer):
		self.Transformers[variable] = new_transformer
		self.all_data["processed"][variable] = self.Transformers[variable].process(
			np.asarray(self.all_data["physical"][variable]).copy()
		)

	def fill_new_column(
		self, data, new_column_name, transformer_variable, processed=True
	):

		if processed:

			self.all_data["processed"][new_column_name] = data

			data_physical = self.Transformers[transformer_variable].unprocess(
				np.asarray(data).copy()
			)

			self.all_data["physical"][new_column_name] = data_physical

			self.Transformers[new_column_name] = self.Transformers[transformer_variable]
		else:
			print("fill_new_column, processed False not implemented quitting...")
			quit()

	def fill_new_condition(self, conditon_dict):

		for condition in list(conditon_dict.keys()):
			self.all_data["physical"][condition] = np.ones(self.all_data["physical"].shape[0])*conditon_dict[condition]
			self.all_data["processed"][condition] = self.Transformers[condition].process(
						np.asarray(self.all_data["physical"][condition]).copy()
					)


	
	def fill_target(self, processed_data, targets=None):

		if targets == None:
			targets = rd.targets

		df_processed = pd.DataFrame(processed_data, columns=targets)

		df_physical = self.post_process(df_processed)

		for column in targets:
			self.all_data["processed"][column] = np.asarray(df_processed[column])
			self.all_data["physical"][column] = np.asarray(df_physical[column])

		self.fill_stripping_bool()
		self.all_data["processed"]["pass_stripping"] = self.all_data["physical"]["pass_stripping"]

	def select_randomly(self, Nevents):

		idx = np.random.choice(
			self.all_data["processed"].shape[0], replace=False, size=Nevents
		)

		self.all_data["processed"] = self.all_data["processed"].iloc[idx]
		self.all_data["physical"] = self.all_data["physical"].iloc[idx]

	def get_physical(self):
		return self.all_data["physical"]

	def get_branches(self, branches, processed=True, option=''):

		if not isinstance(branches, list):
			branches = [branches]

		if processed:
			missing = list(
				set(branches).difference(set(list(self.all_data["processed"].keys())))
			)
			branches = list(
				set(branches).intersection(set(list(self.all_data["processed"].keys())))
			)

			if len(missing) > 0:
				print(f"missing branches: {missing}\n {self.filenames} \n")

			if option != '':
				output = self.all_data["processed"][branches+['in_training']]
			else:
				output = self.all_data["processed"][branches]

		else:
			missing = list(
				set(branches).difference(set(list(self.all_data["physical"].keys())))
			)
			branches = list(
				set(branches).intersection(set(list(self.all_data["physical"].keys())))
			)

			if len(missing) > 0:
				print(f"missing branches: {missing}\n {self.filenames} \n")

			if option != '':
				output = self.all_data["physical"][branches+['in_training']]
			else:
				output = self.all_data["physical"][branches]

		if option == 'training':
			output = output.query('in_training==1.0')
			output = output[branches]
		elif option == 'testing':
			output = output.query('in_training==0.0')
			output = output[branches]

		return output

	def virtual_get_branches(self, branches, processed=True):

		if not isinstance(branches, list):
			branches = [branches]

		if processed:
			missing = list(
				set(branches).difference(set(list(self.all_data["processed_virtual"].keys())))
			)
			branches = list(
				set(branches).intersection(set(list(self.all_data["processed_virtual"].keys())))
			)

			if len(missing) > 0:
				print(f"missing branches: {missing}\n {self.filenames} \n")

			output = self.all_data["processed_virtual"][branches]

		else:
			missing = list(
				set(branches).difference(set(list(self.all_data["physical_virtual"].keys())))
			)
			branches = list(
				set(branches).intersection(set(list(self.all_data["physical_virtual"].keys())))
			)

			if len(missing) > 0:
				print(f"missing branches: {missing}\n {self.filenames} \n")

			output = self.all_data["physical_virtual"][branches]

		return output
	
	def compute_dalitz_mass(self, df, i, j, mass_i, mass_j, true_vars=True):

		if true_vars:
			PE = np.sqrt(
				mass_i**2
				+ df[f"{i}_TRUEP_X"] ** 2
				+ df[f"{i}_TRUEP_Y"] ** 2
				+ df[f"{i}_TRUEP_Z"] ** 2
			) + np.sqrt(
				mass_j**2
				+ df[f"{j}_TRUEP_X"] ** 2
				+ df[f"{j}_TRUEP_Y"] ** 2
				+ df[f"{j}_TRUEP_Z"] ** 2
			)
			PX = df[f"{i}_TRUEP_X"] + df[f"{j}_TRUEP_X"]
			PY = df[f"{i}_TRUEP_Y"] + df[f"{j}_TRUEP_Y"]
			PZ = df[f"{i}_TRUEP_Z"] + df[f"{j}_TRUEP_Z"]
		else:
			PE = np.sqrt(
				mass_i**2 + df[f"{i}_PX"] ** 2 + df[f"{i}_PY"] ** 2 + df[f"{i}_PZ"] ** 2
			) + np.sqrt(
				mass_j**2 + df[f"{j}_PX"] ** 2 + df[f"{j}_PY"] ** 2 + df[f"{j}_PZ"] ** 2
			)
			PX = df[f"{i}_PX"] + df[f"{j}_PX"]
			PY = df[f"{i}_PY"] + df[f"{j}_PY"]
			PZ = df[f"{i}_PZ"] + df[f"{j}_PZ"]


		mass = np.sqrt((PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6)

		return mass
	
	# def add_dalitz_masses(self, pair_1 = ["K_Kst", "e_minus"], pair_2 = ["e_plus", "e_minus"], true_vars=True):

	#     if true_vars:
	#         branches = ["e_plus_TRUEP_X",
	#                     "e_plus_TRUEP_Y",
	#                     "e_plus_TRUEP_Z", 
	#                     "e_minus_TRUEP_X",
	#                     "e_minus_TRUEP_Y",
	#                     "e_minus_TRUEP_Z", 
	#                     "K_Kst_TRUEP_X",
	#                     "K_Kst_TRUEP_Y",
	#                     "K_Kst_TRUEP_Z",
	#                     "K_Kst_TRUEID",
	#                     "e_minus_TRUEID",
	#                     "e_plus_TRUEID",
	#                     ]
	#     else:
	#         branches = ["e_plus_PX",
	#                     "e_plus_PY",
	#                     "e_plus_PZ", 
	#                     "e_minus_PX",
	#                     "e_minus_PY",
	#                     "e_minus_PZ", 
	#                     "K_Kst_PX",
	#                     "K_Kst_PY",
	#                     "K_Kst_PZ",
	#                     "K_Kst_TRUEID",
	#                     "e_minus_TRUEID",
	#                     "e_plus_TRUEID",
	#                     ]

	#     compute_variables = self.get_branches(branches, processed=False)
	#     # e_minus and K_Kst
	#     # e_minus and e_plus

	#     dalitz_mass_mkl = self.compute_dalitz_mass(compute_variables, pair_1[0], pair_1[1], 493.677, 0.51099895000 * 1e-3, true_vars=true_vars)
	#     dalitz_mass_mee = self.compute_dalitz_mass(compute_variables, pair_2[0], pair_2[1], 0.51099895000 * 1e-3, 0.51099895000 * 1e-3, true_vars=true_vars)
		
	#     self.add_branch_to_physical("dalitz_mass_mee", np.asarray(dalitz_mass_mee))
	#     self.add_branch_to_physical("dalitz_mass_mkl", np.asarray(dalitz_mass_mkl))

	def add_dalitz_masses(self, pair_1 = ["K_Kst", "e_minus"], pair_2 = ["e_plus", "e_minus"], true_vars=True):

		if true_vars:
			branches = ["e_plus_TRUEP_X",
						"e_plus_TRUEP_Y",
						"e_plus_TRUEP_Z", 
						"e_minus_TRUEP_X",
						"e_minus_TRUEP_Y",
						"e_minus_TRUEP_Z", 
						"K_Kst_TRUEP_X",
						"K_Kst_TRUEP_Y",
						"K_Kst_TRUEP_Z",
						"K_Kst_TRUEID",
						"e_minus_TRUEID",
						"e_plus_TRUEID",
						]
		else:
			branches = ["e_plus_PX",
						"e_plus_PY",
						"e_plus_PZ", 
						"e_minus_PX",
						"e_minus_PY",
						"e_minus_PZ", 
						"K_Kst_PX",
						"K_Kst_PY",
						"K_Kst_PZ",
						"K_Kst_TRUEID",
						"e_minus_TRUEID",
						"e_plus_TRUEID",
						]

		events_i = self.get_branches(branches, processed=False)

		events = events_i.copy()

		particles = ["K_Kst", "e_plus", "e_minus"]

		masses = {}
		masses[321] = 493.677
		masses[211] = 139.57039
		masses[13] = 105.66
		masses[11] = 0.51099895000 * 1e-3


		pid_list = [11,13,211,321]

		PID_charges = {11:-1, -11:1, 13:-1, -13:1, 211:1, -211:-1, 321:1, -321:-1}

		for particle in particles:
			mass = np.asarray(events[f'{particle}_TRUEID']).astype('float32')
			for pid in pid_list:
				mass[np.where(np.abs(mass)==pid)] = masses[pid]
			events[f'{particle}_mass'] = mass

		for particle in particles:
			charge_ = np.asarray(events[f'{particle}_TRUEID']).copy()
			for key in list(PID_charges.keys()):
				charge_[np.where(charge_==key)] = PID_charges[key]
			events[f'{particle}_charge'] = charge_
	
		where_keep = np.where(events[f'{particles[0]}_charge']!=events[f'{particles[1]}_charge'])
		where_swap = np.where(events[f'{particles[0]}_charge']==events[f'{particles[1]}_charge'])
		print(np.shape(events[f'{particles[0]}_charge'])[0])
		print(np.shape(where_keep)[1])
		print(np.shape(where_swap)[1])
		print(np.shape(where_keep)[1]+np.shape(where_swap)[1])
		if np.shape(where_swap)[1] == 0:
			print("probably RapidSim")
			where_swap = where_keep

		# Retrieve momenta and masses
		px1, py1, pz1 = np.asarray(events[f'{particles[0]}_TRUEP_X']).copy(), np.asarray(events[f'{particles[0]}_TRUEP_Y']).copy(), np.asarray(events[f'{particles[0]}_TRUEP_Z']).copy()
		px2, py2, pz2 = np.asarray(events[f'{particles[1]}_TRUEP_X']).copy(), np.asarray(events[f'{particles[1]}_TRUEP_Y']).copy(), np.asarray(events[f'{particles[1]}_TRUEP_Z']).copy()
		px3, py3, pz3 = np.asarray(events[f'{particles[2]}_TRUEP_X']).copy(), np.asarray(events[f'{particles[2]}_TRUEP_Y']).copy(), np.asarray(events[f'{particles[2]}_TRUEP_Z']).copy()

		px2_buffer, py2_buffer, pz2_buffer = px2.copy(), py2.copy(), pz2.copy()
		px3_buffer, py3_buffer, pz3_buffer = px3.copy(), py3.copy(), pz3.copy()

		px2[where_swap], py2[where_swap], pz2[where_swap] = px3_buffer[where_swap], py3_buffer[where_swap], pz3_buffer[where_swap]
		px3[where_swap], py3[where_swap], pz3[where_swap] = px2_buffer[where_swap], py2_buffer[where_swap], pz2_buffer[where_swap]


		mass1, mass2, mass3 = np.asarray(events[f'{particles[0]}_mass']).copy(), np.asarray(events[f'{particles[1]}_mass']).copy(), np.asarray(events[f'{particles[2]}_mass']).copy()

		mass2_buffer, mass3_buffer = mass2.copy(), mass3.copy()
		mass2_buffer[where_swap], mass3_buffer[where_swap] = mass3_buffer[where_swap], mass2_buffer[where_swap]

		# Calculate energies
		E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + mass1**2)
		E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + mass2**2)
		E3 = np.sqrt(px3**2 + py3**2 + pz3**2 + mass3**2)


		# Compute invariant mass squared for each pair
		def invariant_mass_squared(E1, E2, px1, px2, py1, py2, pz1, pz2):
			return (E1 + E2)**2 - ((px1 + px2)**2 + (py1 + py2)**2 + (pz1 + pz2)**2)

		# m12^2, m13^2, and m23^2
		m12_squared = invariant_mass_squared(E1, E2, px1, px2, py1, py2, pz1, pz2)*1E-6
		m13_squared = invariant_mass_squared(E1, E3, px1, px3, py1, py3, pz1, pz3)*1E-6
		m23_squared = invariant_mass_squared(E2, E3, px2, px3, py2, py3, pz2, pz3)*1E-6


		self.add_branch_to_physical("dalitz_mass_m12", np.asarray(m12_squared))
		self.add_branch_to_physical("dalitz_mass_m13", np.asarray(m13_squared))

		self.add_branch_to_physical("sqrt_dalitz_mass_mee", np.sqrt(np.asarray(m23_squared)))
		self.add_branch_to_physical("sqrt_dalitz_mass_mkl", np.sqrt(np.asarray(m12_squared)))

	
	def add_eta_phi(self):

		branches = ["B_plus_TRUEP_X",
					"B_plus_TRUEP_Y",
					"B_plus_TRUEP_Z", 
					]


		compute_variables = self.get_branches(branches, processed=False)

		phi = np.arctan2(compute_variables["B_plus_TRUEP_Y"], compute_variables["B_plus_TRUEP_X"])
		eta = np.arcsinh(compute_variables["B_plus_TRUEP_Z"] / np.sqrt(compute_variables["B_plus_TRUEP_X"]**2 + compute_variables["B_plus_TRUEP_Y"]**2))

		self.add_branch_to_physical("phi_B", np.asarray(phi))
		self.add_branch_to_physical("eta_B", np.asarray(eta))
		

	def pre_process(self, physical_data):

		df = {}

		if self.Transformers == None:
			fresh_transformers = True
			self.Transformers = {}
		else:
			print("using loaded transformers")
			fresh_transformers = False

		for column in list(physical_data.keys()):

			if column == "file" or column == "pass_stripping" or column == "training_weight" or column == "in_training":
				df[column] = physical_data[column]
			else:
				# if "TRUEID" in column:
				#     print("MUST DELETE THIS")
				#     if fresh_transformers:
				#         data_array = np.asarray(physical_data[column]).copy()
				#         transformer_i = Transformer()
				#         transformer_i.fit(data_array, column)
				#         self.Transformers[column] = transformer_i

				#     df[column] = self.Transformers[column].process(
				#         np.asarray(physical_data[column]).copy()
				#     )
				# else:
					try:
						if fresh_transformers:
							data_array = np.asarray(physical_data[column]).copy()
							transformer_i = Transformer()
							transformer_i.fit(data_array, column)
							self.Transformers[column] = transformer_i

						df[column] = self.Transformers[column].process(
							np.asarray(physical_data[column]).copy()
						)
					# except Exception as e:
					    # print(f"\n\n pre_process: An error occurred: {e}")
					except:
						pass
			# print(np.shape(df[column]), column)

		return pd.DataFrame.from_dict(df)

	def get_transformers(self):
		return self.Transformers

	def save_transformers(self, file):
		pickle.dump(
			self.Transformers,
			open(
				file,
				"wb",
			),
		)
	# def load_transformers(self, file):
	#     self.Transformers = pickle.load(open(file, "rb"))

	def reweight_for_training(self, variable, weight_value, plot_variable=''):
		
		if plot_variable != '':
			plt.hist(self.all_data['physical'][plot_variable], bins=75)
			plt.savefig('reweight_before.pdf')
			plt.close('all')

		weight = np.ones(np.shape(self.all_data['physical'][variable]))

		if variable == 'fully_reco':

			frac_fully_reco_pre = np.sum(self.all_data['physical']['fully_reco'])/np.shape(self.all_data['physical']['fully_reco'])[0]
			print(f"full reco frac: {frac_fully_reco_pre}")

			weight[np.where(self.all_data['physical'][variable])] = weight_value
		else:
			print('data_loader.py, reweight_for_training(), other variables not set')
			quit()

		# weight[np.where((self.all_data['physical'][variable]>5.27934-0.05)&(self.all_data['physical'][variable]<5.27934+0.05))] = weight_value
		
		if plot_variable != '':
			plt.hist(self.all_data['physical'][plot_variable], bins=75, weights=weight)
			plt.savefig('reweight_after.pdf')
			plt.close('all')

		self.all_data['physical']['training_weight'] = weight
		self.all_data['processed']['training_weight'] = weight

		if variable == 'fully_reco':
			frac_fully_reco_pre = np.sum(self.all_data['physical']['fully_reco']*self.all_data['physical']['training_weight'])/np.sum(self.all_data['physical']['training_weight'])
			print(f"full reco frac: {frac_fully_reco_pre}")


		self.reweight_for_training_bool = True

	def shape(self):
		return self.all_data['physical'].shape
	
	def getBinomialEff(self, pass_sum, tot_sum, pass_sumErr, tot_sumErr):
		'''
		Function for computing efficiency (and uncertainty).
		'''
		eff = pass_sum/tot_sum # Easy part

		# Compute uncertainty taken from Eqs. (13) from LHCb-PUB-2016-021
		x = (1 - 2*eff)*(pass_sumErr*pass_sumErr)
		y = (eff*eff)*(tot_sumErr*tot_sumErr)

		effErr = np.sqrt(abs(x + y)/(tot_sum**2))

		return eff, effErr

	def virtual_cut(self, cut):
		
		self.all_data['physical_virtual'] = self.all_data['physical'].copy()
		self.all_data['processed_virtual'] = self.all_data['processed'].copy()

		gen_tot_val = self.all_data['physical'].shape[0]
		gen_tot_err = np.sqrt(gen_tot_val)

		if cut=='pass_stripping': # couldnt fix bug with query, this is work around
			self.all_data['physical_virtual'].reset_index(drop=True, inplace=True)
			self.all_data['processed_virtual'].reset_index(drop=True, inplace=True)
			passes = np.where(self.all_data['physical_virtual']['pass_stripping']>0.5)
			self.all_data['physical_virtual'] = self.all_data['physical_virtual'].iloc[passes]
		else:
			self.all_data['physical_virtual'] = self.all_data['physical_virtual'].query(cut)
		index = self.all_data['physical_virtual'].index
		
		if not self.turn_off_processing:
			self.all_data['processed_virtual'] = self.all_data['processed_virtual'].iloc[index]
			self.all_data['processed_virtual'] = self.all_data['processed_virtual'].reset_index(drop=True)

		self.all_data['physical_virtual'] = self.all_data['physical_virtual'].reset_index(drop=True)
		pass_tot_val = self.all_data['physical_virtual'].shape[0]
		pass_tot_err = np.sqrt(pass_tot_val)

		eff, effErr = self.getBinomialEff(pass_tot_val, gen_tot_val,
									 pass_tot_err, gen_tot_err)


		print(f'INFO cut(): {cut}, eff:{eff:.4f}+-{effErr:.4f}')


	def cut(self, cut):
		
		gen_tot_val = self.all_data['physical'].shape[0]
		gen_tot_err = np.sqrt(gen_tot_val)

		if cut=='pass_stripping': # couldnt fix bug with query, this is work around
			self.all_data['physical'].reset_index(drop=True, inplace=True)
			self.all_data['processed'].reset_index(drop=True, inplace=True)
			passes = np.where(self.all_data['physical']['pass_stripping']>0.5)
			self.all_data['physical'] = self.all_data['physical'].iloc[passes]
		else:
			self.all_data['physical'] = self.all_data['physical'].query(cut)
		index = self.all_data['physical'].index
		
		if not self.turn_off_processing:
			self.all_data['processed'] = self.all_data['processed'].iloc[index]
			self.all_data['processed'] = self.all_data['processed'].reset_index(drop=True)

		self.all_data['physical'] = self.all_data['physical'].reset_index(drop=True)
		pass_tot_val = self.all_data['physical'].shape[0]
		pass_tot_err = np.sqrt(pass_tot_val)

		eff, effErr = self.getBinomialEff(pass_tot_val, gen_tot_val,
									 pass_tot_err, gen_tot_err)


		print(f'INFO cut(): {cut}, eff:{eff:.4f}+-{effErr:.4f}')

	def getEff(self, cut):

		if isinstance(cut, dict):
			cut_string = ''
			for cut_idx, cut_i in enumerate(list(cut.keys())):
				if cut_idx > 0:
					cut_string += ' & '
				if cut_i == 'extra_cut':
					cut_string += f'{cut[cut_i]}'
				else:
					cut_string += f'{cut_i}{cut[cut_i]}'
			cut = cut_string   

		gen_tot_val = self.all_data['physical'].shape[0]
		gen_tot_err = np.sqrt(gen_tot_val)

		if 'B_plus_ENDVERTEX_NDOF' in cut:
			if 'B_plus_ENDVERTEX_NDOF' not in list(self.all_data['physical'].keys()):
				self.all_data['physical']['B_plus_ENDVERTEX_NDOF'] = 3

		cut_array = self.all_data['physical'].query(cut)
		pass_tot_val = cut_array.shape[0]
		pass_tot_err = np.sqrt(pass_tot_val)

		eff, effErr = self.getBinomialEff(pass_tot_val, gen_tot_val,
									 pass_tot_err, gen_tot_err)


		print(f'INFO getEff({cut}): {eff:.4f}+-{effErr:.4f} \t\t {self.name}')

		return eff, effErr

	def save_to_file(self, filename):
		
		write_df_to_root(self.all_data["physical"], filename)

	def add_branch_to_physical(self, name, values):

		self.all_data["physical"][name] = values
	
	def add_branch_and_process(self, name, recipe):

		self.all_data["physical"][name] = self.all_data["physical"].eval(recipe)
		physical_data = self.all_data["physical"]
		column = name

		data_array = np.asarray(physical_data[column]).copy()
		transformer_i = Transformer()
		transformer_i.fit(data_array, column)
		self.Transformers[column] = transformer_i

		self.all_data["processed"][column] = self.Transformers[column].process(
		np.asarray(physical_data[column]).copy()
		)

	def add_missing_mass_frac_branch(self):

		data = self.get_branches(['B_plus_TRUEID'],processed=False)
		data = np.asarray(data['B_plus_TRUEID']).astype(np.float64)
		unique, counts = np.unique(data, return_counts=True)

		found = [0,0]
		for i in range(np.shape(unique)[0]):
			# print(unique[i], counts[i])
			try:
				# print(Particle.from_pdgid(unique[i]).mass)
				found[0] += counts[i]
				data[np.where(data==unique[i])] = Particle.from_pdgid(int(unique[i])).mass*1E-3
			except:
				# print(f'{unique[i]} not found')
				found[1] += counts[i]
				data[np.where(data==unique[i])] = -1
		print(found)

		self.all_data["physical"]['MOTHER_TRUE_MASS'] = data

		self.cut("MOTHER_TRUE_MASS>=0")
		
		# self.add_branch_and_process('missing_mass_frac','(MOTHER_TRUE_MASS-B_plus_M_reco)/MOTHER_TRUE_MASS')
		self.add_branch_and_process('missing_mass_frac','(MOTHER_TRUE_MASS-B_plus_M)/MOTHER_TRUE_MASS')




	def convert_value_to_processed(self, name, value):
		print('convert_value_to_processed', name, value)
		print(self.Transformers[name].process(value))
		return self.Transformers[name].process(value)
	
	def plot(self,filename, variables=None,save_vars=False):

		if variables == None:
			variables = list(self.all_data["physical"].keys())

		if save_vars:
			vars_to_save = {}
			vars_to_save["physical"] = {}
			vars_to_save["processed"] = {}

		with PdfPages(filename) as pdf:

			for variable in variables:
				
				try:
					plt.figure(figsize=(10,8))

					plt.subplot(2,2,1)
					plt.title(variable)
					plt.hist(self.all_data["physical"][variable], bins=50, density=True, histtype='step')
					
					plt.subplot(2,2,2)
					plt.title(f'{variable} processed')
					plt.hist(self.all_data["processed"][variable], bins=50, density=True, histtype='step', range=[-1,1])

					plt.subplot(2,2,3)
					plt.hist(self.all_data["physical"][variable], bins=50, density=True, histtype='step')
					plt.yscale('log')
					
					plt.subplot(2,2,4)
					plt.hist(self.all_data["processed"][variable], bins=50, density=True, histtype='step', range=[-1,1])
					plt.yscale('log')

					pdf.savefig(bbox_inches="tight")
					plt.close()

					if save_vars:
						vars_to_save["physical"][variable] = np.asarray(self.all_data["physical"][variable])
						vars_to_save["processed"][variable] = np.asarray(self.all_data["processed"][variable])

				except:
					pdf.savefig(bbox_inches="tight")
					plt.close()
					pass
		
		if save_vars:
			with open(f'{filename[:-3]}.pickle', 'wb') as handle:
				pickle.dump(vars_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def get_file_names(self):
		return self.filenames


def convert_branches_to_RK_branch_names(columns, conversions):
	new_columns = []
	for column in columns:
		converted = False

		for conversion in conversions.keys():

			if conversion in column:
				if conversion == "MOTHER":
					if column[:6] == "MOTHER":
						new_column = conversions[conversion]+column[6:]
					elif "_MC_" in column:
						continue
					else:
						new_column = column.replace(conversion, conversions[conversion])
				else:
					new_column = column.replace(conversion, conversions[conversion])

				new_columns.append(new_column)
				converted = True
				break

		if not converted:
			new_columns.append(column)

	return new_columns

def load_data(path, equal_sizes=True, N=-1, transformers=None, convert_to_RK_branch_names=False, conversions=None, turn_off_processing=False,avoid_physics_variables=False, name='', testing_frac=0.1):

	if isinstance(path, list):
		for i in range(0, len(path)):
			if i == 0:
				if path[i][-5:] == '.root':
					file = uproot.open(path[i])['DecayTree']
					if N != -1:
						events = file.arrays(library='pd', entry_stop=N)
					else:
						events = file.arrays(library='pd')
					if convert_to_RK_branch_names:
						if conversions == None:
							print("must declare conversions, quitting...")
							quit()
						new_columns = convert_branches_to_RK_branch_names(events.columns, conversions)
						events.columns = new_columns
  
				else:
					events = pd.read_csv(path[i])
					if equal_sizes and N == -1:
						N = events.shape[0]
					elif equal_sizes:
						events = events.head(N)
			else:
				if path[i][-5:] == '.root':
					file = uproot.open(path[i])['DecayTree']
					if N != -1:
						events_i = file.arrays(library='pd', entry_stop=N)
					else:
						events_i = file.arrays(library='pd')
					if convert_to_RK_branch_names:
						if conversions == None:
							print("must declare conversions, quitting...")
							quit()
						new_columns = convert_branches_to_RK_branch_names(events_i.columns, conversions)
						events_i.columns = new_columns
				else:
					events_i = pd.read_csv(path[i])
					if equal_sizes:
						events_i = events_i.head(N)
				events = pd.concat([events, events_i], axis=0)
			events["file"] = np.asarray(np.ones(events.shape[0]) * i).astype("int")
	else:
		events = pd.read_csv(path)
		path = [path]

	events = events.loc[:, ~events.columns.str.contains("^Unnamed")]

	# if "nSPDHits" not in list(events.keys()):
	#     print(colored("WE DO NOT HAVE nSPD",'red'))
	#     shape_required = events.shape[0]
	#     N_load = 50000
	#     file_nSPD = uproot.open("datasets/general_sample_intermediate_All_more_vars_HEADfactor10.root")['DecayTree']
	#     events_nSPD = file_nSPD.arrays(['nSPDHits'], library='pd', entry_stop=N_load)
	#     nSPDHits = np.asarray(events_nSPD['nSPDHits'])
	#     nSPDHits = nSPDHits[np.random.randint(0, N_load, shape_required)]
	#     events['nSPDHits'] = nSPDHits
	#     # "datasets/general_sample_intermediate_All_more_vars_HEADfactor10.root",
	#     print(colored("filled nSPDHits",'green'))


	events_dataset = dataset(filenames=path, transformers=transformers, name=name)
	events_dataset.fill(events, turn_off_processing, avoid_physics_variables=avoid_physics_variables, testing_frac=testing_frac)

	return events_dataset
