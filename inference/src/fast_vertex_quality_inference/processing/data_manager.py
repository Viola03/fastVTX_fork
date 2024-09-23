import numpy as np
import uproot
import uproot3
import matplotlib.pyplot as plt
import pandas as pd
import fast_vertex_quality.tools.variables_tools as vt
from particle import Particle

masses = {}
masses[321] = 493.677
masses[211] = 139.57039
masses[13] = 105.66
masses[11] = 0.51099895000 * 1e-3
pid_list = [11,13,211,321]

class data_manager:

	def __init__(self, 
				tuple, 
				particles_TRUEID,
				mother_TRUEID,
				fully_reco,
				nPositive_missing_particles,
				nNegative_missing_particles,
				tree='DecayTree',
				particles=["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"],
				mother = 'MOTHER',
				intermediate = 'INTERMEDIATE',
				):

		self.rapidsim_raw = uproot.open(tuple)[tree]

		self.create_branch_converter(list(self.rapidsim_raw.keys()))
		
		self.particles_TRUEID = particles_TRUEID
		self.mother_TRUEID = mother_TRUEID
		self.fully_reco = fully_reco
		self.nPositive_missing_particles = nPositive_missing_particles
		self.nNegative_missing_particles = nNegative_missing_particles

		self.particles = particles
		self.mother = mother
		self.intermediate = intermediate

	def create_branch_converter(self, branches, extend=False):

		if not extend:
			self.branch_converter = {}

		for branch in branches:
						
			if "vtx" in branch:
				dim = branch[branch.index("vtx") + 3]
				if "TRUE" in branch: new_branch = branch.replace(f"vtx{dim}_TRUE",f"TRUEENDVERTEX_{dim}")
				else: new_branch = branch.replace(f"vtx{dim}",f"ENDVERTEX_{dim}")
				self.branch_converter[new_branch] = branch
			
			elif "orig" in branch:
				dim = branch[branch.index("orig") + 4]
				if "TRUE" in branch: new_branch = branch.replace(f"orig{dim}_TRUE",f"TRUEORIGINVERTEX_{dim}")
				else: new_branch = branch.replace(f"orig{dim}",f"ORIGINVERTEX_{dim}")
				self.branch_converter[new_branch] = branch
			
			elif "_P" in branch and "TRUE" in branch:
				dim = branch[branch.index("_P") + 2]
				if dim == '_':
					new_branch = branch.replace(f'P_TRUE',f'TRUEP')
				else:
					new_branch = branch.replace(f'P{dim}_TRUE',f'TRUEP_{dim}')

			else:
				new_branch = branch

			self.branch_converter[new_branch] = branch
			self.branch_converter[branch] = new_branch
	
	def get_branches(self, network, branches=None, process=False, convert=False):
		
		if not branches:
			branches = list(self.rapidsim_raw.keys())

		if convert:
			converted_branches = [self.branch_converter[branch] for branch in list(branches)]
			data = self.rapidsim_raw.arrays(converted_branches, library="pd")
		else:
			data = self.rapidsim_raw.arrays(branches, library="pd")

			data.columns = [self.branch_converter[branch] for branch in list(branches)]

		for branch in list(data.keys()):
			# if momentum branch convert to MeV
			if "_P" in branch or "TRUEP" in branch:
				data[branch] *= 1000.
		
		if process:
			
			for branch in branches:
				data[self.branch_converter[branch]] = network.Transformers[branch].process(np.asarray(data[self.branch_converter[branch]]))

			return np.asarray(data[converted_branches])

		else:
			return data

	# Function to compute the angles
	def compute_angles(self, origin, end):
		vector = np.array(end) - np.array(origin)
		theta = np.arccos(vector[2] / np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2))
		phi = np.arctan2(vector[1], vector[0])
		return theta, phi

	# Function to redefine the endpoint
	def redefine_endpoint(self, origin, theta, phi, new_distance):
		x = new_distance * np.sin(theta) * np.cos(phi)
		y = new_distance * np.sin(theta) * np.sin(phi)
		z = new_distance * np.cos(theta)
		return origin[0] + x, origin[1] + y, origin[2] + z

	def propagate_PV_smearing(self, full_tuple, smeared_info):

		distance_buffer = {}
		for particle in self.particles:
			for coordinate in ["X", "Y", "Z"]:
				distance_buffer[f"{particle}_{coordinate}"] = np.asarray(full_tuple[f"{particle}_TRUEORIGINVERTEX_{coordinate}"]-full_tuple[f"{self.mother}_TRUEENDVERTEX_{coordinate}"])

		B_plus_TRUEORIGINVERTEX = [full_tuple[f"{self.mother}_TRUEORIGINVERTEX_X"], full_tuple[f"{self.mother}_TRUEORIGINVERTEX_Y"], full_tuple[f"{self.mother}_TRUEORIGINVERTEX_Z"]]
		B_plus_TRUEENDVERTEX = [full_tuple[f"{self.mother}_TRUEENDVERTEX_X"], full_tuple[f"{self.mother}_TRUEENDVERTEX_Y"], full_tuple[f"{self.mother}_TRUEENDVERTEX_Z"]]
		theta, phi = self.compute_angles(B_plus_TRUEORIGINVERTEX, B_plus_TRUEENDVERTEX)

		for branch in list(smeared_info.keys()):
			full_tuple[branch] = smeared_info[branch]

		B_plus_TRUEORIGINVERTEX = [full_tuple[f"{self.mother}_TRUEORIGINVERTEX_X"], full_tuple[f"{self.mother}_TRUEORIGINVERTEX_Y"], full_tuple[f"{self.mother}_TRUEORIGINVERTEX_Z"]]
		full_tuple[f"{self.mother}_TRUEENDVERTEX_X"], full_tuple[f"{self.mother}_TRUEENDVERTEX_Y"], full_tuple[f"{self.mother}_TRUEENDVERTEX_Z"] = self.redefine_endpoint(B_plus_TRUEORIGINVERTEX, theta, phi, full_tuple[f"{self.mother}_TRUE_FD"])

		for particle in self.particles:
			for coordinate in ["X", "Y", "Z"]:
				full_tuple[f"{particle}_TRUEORIGINVERTEX_{coordinate}"] = full_tuple[f"{self.mother}_TRUEENDVERTEX_{coordinate}"]+ distance_buffer[f"{particle}_{coordinate}"]
		
		return full_tuple

	def update_conditional_branches(self, branches):
		updated_branches = []
		for branch in branches:
			branch = branch.replace('B_plus', self.mother)
			branch = branch.replace('K_Kst', self.particles[0])
			branch = branch.replace('e_plus',self.particles[1])
			branch = branch.replace('e_minus',self.particles[2])
			branch = branch.replace('J_psi_1S',self.intermediate)
			updated_branches.append(branch)
		return updated_branches

	def query_network(self, network, conditional_input, target_varaiables):

		N = np.shape(conditional_input)[0]

		generator_noise = np.random.normal(0, 1, (N, network.latent_dim))

		input_data = {
			network.input_names[0]: generator_noise.astype(np.float32),
			network.input_names[1]: conditional_input.astype(np.float32)
		}
		
		output = network.session.run(network.output_names, input_data)[0]

		df = {} 
		for idx, target in enumerate(target_varaiables):
			df[target] = output[:,idx]
		output = pd.DataFrame.from_dict(df)

		for branch in list(output.keys()):
			output[branch] = network.Transformers[branch].unprocess(np.asarray(output[branch]))

		return output


	def process(self, output_tuple, PV_smearing_network, vertexing_network):

		print(f"\n\n Processing... \n\n")

		full_rapid_sim_tuple = self.get_branches(vertexing_network)

		conditional_input = self.get_branches(PV_smearing_network, PV_smearing_network.conditions, process=True, convert=True)
		smeared_vertex = self.query_network(PV_smearing_network, conditional_input, PV_smearing_network.targets)

		full_rapid_sim_tuple_smeared = self.propagate_PV_smearing(full_rapid_sim_tuple, smeared_vertex)

		full_rapid_sim_tuple_appended = self.add_conditional_info(full_rapid_sim_tuple_smeared)
		converted_condition_branches = self.update_conditional_branches(vertexing_network.conditions)
		conditional_input = full_rapid_sim_tuple_appended[converted_condition_branches]
		for idx, branch in enumerate(converted_condition_branches):
			conditional_input[branch] = vertexing_network.Transformers[vertexing_network.conditions[idx]].process(np.asarray(conditional_input[branch]))
		conditional_input = np.asarray(conditional_input)

		generated_vertexing_info = self.query_network(vertexing_network, conditional_input, vertexing_network.targets)
		generated_vertexing_info.columns = self.update_conditional_branches(vertexing_network.targets)
		for branch in list(generated_vertexing_info.keys()):
			full_rapid_sim_tuple_smeared[branch] = generated_vertexing_info[branch]
		full_rapid_sim_tuple_smeared = full_rapid_sim_tuple_smeared.drop(columns=converted_condition_branches)
		# print('\n\n')
		# for branch in list(full_rapid_sim_tuple_smeared.keys()):
		# 	print(branch)

		self.write_df_to_root(full_rapid_sim_tuple_smeared, output_tuple)

		quit()

	def compute_distance_wrapped(self, df, A, A_var, B, B_var):

		A = vt.compute_distance(df, A, A_var, B, B_var)
		A = np.asarray(A)
		min_A = 5e-5
		A[np.where(A==0)] = min_A

		return A

	def add_conditional_info(self, data):

		

		for idx, particle in enumerate(self.particles):
			data[f'{particle}_TRUEID'] = self.particles_TRUEID[idx]

			mass = np.asarray(data[f'{particle}_TRUEID']).astype('float32')
			for pid in pid_list:
				mass[np.where(np.abs(mass)==pid)] = masses[pid]
			data[f'{particle}_mass'] = mass

		data[f"{self.mother}_FLIGHT"] = self.compute_distance_wrapped(data, self.mother, 'TRUEENDVERTEX', self.mother, 'TRUEORIGINVERTEX')
		data[f"{self.particles[0]}_FLIGHT"] = self.compute_distance_wrapped(data, self.particles[0], 'TRUEORIGINVERTEX', self.mother, 'TRUEENDVERTEX')
		data[f"{self.particles[1]}_FLIGHT"] = self.compute_distance_wrapped(data, self.particles[1], 'TRUEORIGINVERTEX', self.mother, 'TRUEENDVERTEX')
		data[f"{self.particles[2]}_FLIGHT"] = self.compute_distance_wrapped(data, self.particles[2], 'TRUEORIGINVERTEX', self.mother, 'TRUEENDVERTEX')



		# for particle in self.particles:
		# 	data[f"{particle}_P"], data[f"{particle}_PT"] = vt.compute_reconstructed_mother_momenta(data, particle)
		data[f"{self.mother}_P"], data[f"{self.mother}_PT"] = vt.compute_reconstructed_mother_momenta(data, self.mother)

		data[f"{self.intermediate}_P"], data[f"{self.intermediate}_PT"] = vt.compute_reconstructed_intermediate_momenta(data, [self.particles[1], self.particles[2]])

		data[f"missing_{self.mother}_P"], data[f"missing_{self.mother}_PT"] = vt.compute_missing_momentum(
			data, self.mother,self.particles
		)
		data[f"missing_{self.intermediate}_P"], data[f"missing_{self.intermediate}_PT"] = vt.compute_missing_momentum(
			data, self.mother,[self.particles[1], self.particles[2]]
		)

		# print(data['e_plus_P'])
		# print(data['e_plus_PT'])
		# print("C")
		# quit()

		for particle_i in range(0, len(self.particles)):
			(
				data[f"delta_{particle_i}_P"],
				data[f"delta_{particle_i}_PT"],
			) = vt.compute_reconstructed_momentum_residual(data,self.particles[particle_i])

		for particle in self.particles:
			data[f"angle_{particle}"] = vt.compute_angle(data, self.mother, f"{particle}")

		data[f"IP_{self.mother}_true_vertex"] = vt.compute_impactParameter(data,self.mother,self.particles,true_vertex=True)
		for particle in self.particles:
			data[f"IP_{particle}_true_vertex"] = vt.compute_impactParameter_i(data,self.mother, f"{particle}",true_vertex=True)
		data[f"FD_{self.mother}_true_vertex"] = vt.compute_flightDistance(data,self.mother,self.particles,true_vertex=True)
		data[f"DIRA_{self.mother}_true_vertex"] = vt.compute_DIRA(data,self.mother,self.particles,true_vertex=True)

		if self.fully_reco: data[f"fully_reco"] = 1.
		else: data[f"fully_reco"] = 0.

		data[f"{self.mother}_nPositive_missing"] = float(self.nPositive_missing_particles)
		data[f"{self.mother}_nNegative_missing"] = float(self.nPositive_missing_particles)


		for particle in self.particles:
			data[f"{particle}_eta"] = -np.log(
				np.tan(
					np.arcsin(
						data[f"{particle}_PT"]
						/ data[f"{particle}_P"]
					)
					/ 2.0
				)
			)

		# MOTHER_TRUE_MASS = Particle.from_pdgid(int(self.mother_TRUEID)).mass*1E-3
		# data["missing_mass_frac"] = (MOTHER_TRUE_MASS-data["B_plus_M"])/MOTHER_TRUE_MASS

		return data
	
	def write_df_to_root(self, df, output_name):
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
