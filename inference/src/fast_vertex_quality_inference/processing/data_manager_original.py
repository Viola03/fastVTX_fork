import numpy as np
import uproot
import uproot3
import matplotlib.pyplot as plt
import pandas as pd
import fast_vertex_quality_inference.processing.transformers as tfs
import fast_vertex_quality_inference.processing.processing_tools as pts

masses = {}
masses[321] = 493.677
masses[211] = 139.57039
masses[13] = 105.66
masses[11] = 0.51099895000 #* 1e-3
pid_list = [11,13,211,321]


class tuple_manager:

	def map_branch_names(self):
		branch_names = list(self.tuple.keys())
		branch_names = [branch.replace(self.mother_particle_name,self.mother) for branch in branch_names]
		branch_names = [branch.replace(self.intermediate_particle_name,self.intermediate) for branch in branch_names]
		branch_names = [branch.replace(self.daughter_particle_names[0],self.particles[0]) for branch in branch_names]
		branch_names = [branch.replace(self.daughter_particle_names[1],self.particles[1]) for branch in branch_names]
		branch_names = [branch.replace(self.daughter_particle_names[2],self.particles[2]) for branch in branch_names]
		self.tuple.columns = branch_names
	
	# def convert_GeV_MeV(self, factor):
	# 	for branch in self.tuple.columns:
	# 		if "_M_" in branch or branch[-2:] == "_M":
	# 			self.tuple[branch] *= factor
	# 		for P in ["P","PT","PX","PY","PZ"]:
	# 			if f"_{P}_" in branch or branch[-(len(P)+1):] == f"_{P}":
	# 				self.tuple[branch] *= factor

	def recompute_reconstructed_mass(self):
		
		df = self.tuple

		i = self.particles[0]
		j = self.particles[1]
		k = self.particles[2]

		mass_i = masses[self.particles_TRUEID[0]] * 1e-3
		mass_j = masses[self.particles_TRUEID[1]] * 1e-3
		mass_k = masses[self.particles_TRUEID[2]] * 1e-3

		PE = np.sqrt(
			mass_i**2 + df[f"{i}_PX"] ** 2 + df[f"{i}_PY"] ** 2 + df[f"{i}_PZ"] ** 2
		) + np.sqrt(
			mass_j**2 + df[f"{j}_PX"] ** 2 + df[f"{j}_PY"] ** 2 + df[f"{j}_PZ"] ** 2
		) + np.sqrt(
			mass_k**2 + df[f"{k}_PX"] ** 2 + df[f"{k}_PY"] ** 2 + df[f"{k}_PZ"] ** 2
		)
		PX = df[f"{i}_PX"] + df[f"{j}_PX"] + df[f"{k}_PX"]
		PY = df[f"{i}_PY"] + df[f"{j}_PY"] + df[f"{k}_PY"]
		PZ = df[f"{i}_PZ"] + df[f"{j}_PZ"] + df[f"{k}_PZ"]

		mass = np.sqrt((PE**2 - PX**2 - PY**2 - PZ**2))

		return mass


	def __init__(self, 
				tuple_location, 
				particles_TRUEID,
				fully_reco,
				nPositive_missing_particles,
				nNegative_missing_particles,
				mother_particle_name,
				intermediate_particle_name, # make this optional
				daughter_particle_names,
				tree='DecayTree',
				entry_stop=None,
				):


		self.particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
		self.mother = 'MOTHER'
		self.intermediate = 'INTERMEDIATE'
		self.fully_reco = fully_reco
		self.nPositive_missing_particles = nPositive_missing_particles
		self.nNegative_missing_particles = nNegative_missing_particles

		self.particles_TRUEID = particles_TRUEID
		self.mother_particle_name = mother_particle_name
		self.intermediate_particle_name = intermediate_particle_name
		self.daughter_particle_names = daughter_particle_names

		self.tree = tree
		self.tuple_location = tuple_location

		self.raw_tuple = uproot.open(self.tuple_location)[self.tree]

		list_of_branches = list(self.raw_tuple.keys())
		list_of_branches = [branch for branch in list_of_branches if "COV" not in branch]

		# for branch in list_of_branches:
		# 	print(branch)

		if entry_stop:
			self.tuple = self.raw_tuple.arrays(list_of_branches, library="pd", entry_stop=entry_stop)
		else:
			self.tuple = self.raw_tuple.arrays(list_of_branches, library="pd")
		self.map_branch_names()
  
		# for branch in list_of_branches:
		# 	print(branch)
  
		# self.convert_GeV_MeV(1000.)

		self.original_branches = list(self.tuple.keys())

		self.tuple[f"{self.mother}_M"] = self.recompute_reconstructed_mass()
	
	def write(self, new_branches_to_keep, output_location=None):

		branches = self.original_branches + new_branches_to_keep
		tuple_to_write = self.tuple[branches]
		if not output_location:
			output_location = f'{self.tuple_location[:-5]}_reco.root'
		pts.write_df_to_root(tuple_to_write, output_location, self.tree)

	def add_branches(self, data_to_add):

		for branch in list(data_to_add.keys()):
			self.tuple[branch] = data_to_add[branch]
		

	def get_branches(self, branches, transformers=None, numpy=False):
		data = self.tuple[branches]

		if transformers:
			
			data = tfs.transform_df(data, transformers)

		if numpy:
			data = np.asarray(data[branches])
			
		return data

	def smearPV(self, smeared_PV_output):

		print("Need to implement function to move the origin vertex too")

		distance_buffer = {}
		for particle in self.particles:
			for coordinate in ["X", "Y", "Z"]:
				distance_buffer[f"{particle}_{coordinate}"] = np.asarray(self.tuple[f"{particle}_orig{coordinate}_TRUE"]-self.tuple[f"{self.mother}_vtx{coordinate}_TRUE"])

		B_plus_TRUEORIGINVERTEX = [self.tuple[f"{self.mother}_origX_TRUE"], self.tuple[f"{self.mother}_origY_TRUE"], self.tuple[f"{self.mother}_origZ_TRUE"]]
		B_plus_TRUEENDVERTEX = [self.tuple[f"{self.mother}_vtxX_TRUE"], self.tuple[f"{self.mother}_vtxY_TRUE"], self.tuple[f"{self.mother}_vtxZ_TRUE"]]
		theta, phi = pts.compute_angles(B_plus_TRUEORIGINVERTEX, B_plus_TRUEENDVERTEX)

		for branch in list(smeared_PV_output.keys()):
			self.tuple[branch] = smeared_PV_output[branch]

		B_plus_TRUEORIGINVERTEX = [self.tuple[f"{self.mother}_origX_TRUE"], self.tuple[f"{self.mother}_origY_TRUE"], self.tuple[f"{self.mother}_origZ_TRUE"]]
		self.tuple[f"{self.mother}_vtxX_TRUE"], self.tuple[f"{self.mother}_vtxY_TRUE"], self.tuple[f"{self.mother}_vtxZ_TRUE"] = pts.redefine_endpoint(B_plus_TRUEORIGINVERTEX, theta, phi, self.tuple[f"{self.mother}_TRUE_FD"])

		for particle in self.particles:
			for coordinate in ["X", "Y", "Z"]:
				self.tuple[f"{particle}_orig{coordinate}_TRUE"] = self.tuple[f"{self.mother}_vtx{coordinate}_TRUE"]+ distance_buffer[f"{particle}_{coordinate}"]

	def append_conditional_information(self):

		for idx, particle in enumerate(self.particles):
			self.tuple[f'{particle}_TRUEID'] = self.particles_TRUEID[idx]

			mass = np.asarray(self.tuple[f'{particle}_TRUEID']).astype('float32')
			for pid in pid_list:
				mass[np.where(np.abs(mass)==pid)] = masses[pid]
			self.tuple[f'{particle}_mass'] = mass

		self.tuple[f"{self.mother}_FLIGHT"] = pts.compute_distance_wrapped(self.tuple, self.mother, 'vtx', self.mother, 'orig')
		self.tuple[f"{self.particles[0]}_FLIGHT"] = pts.compute_distance_wrapped(self.tuple, self.particles[0], 'orig', self.mother, 'vtx')
		self.tuple[f"{self.particles[1]}_FLIGHT"] = pts.compute_distance_wrapped(self.tuple, self.particles[1], 'orig', self.mother, 'vtx')
		self.tuple[f"{self.particles[2]}_FLIGHT"] = pts.compute_distance_wrapped(self.tuple, self.particles[2], 'orig', self.mother, 'vtx')

		self.tuple[f"{self.mother}_P"], self.tuple[f"{self.mother}_PT"] = pts.compute_reconstructed_mother_momenta(self.tuple, self.mother)

		self.tuple[f"{self.intermediate}_P"], self.tuple[f"{self.intermediate}_PT"] = pts.compute_reconstructed_intermediate_momenta(self.tuple, [self.particles[1], self.particles[2]])

		# self.tuple[f"missing_{self.mother}_P"], self.tuple[f"missing_{self.mother}_PT"] = pts.compute_missing_momentum(
		# 	self.tuple, self.mother,self.particles
		# )
		# self.tuple[f"missing_{self.intermediate}_P"], self.tuple[f"missing_{self.intermediate}_PT"] = pts.compute_missing_momentum(
		# 	self.tuple, self.mother,[self.particles[1], self.particles[2]]
		# )
  

		# Manually set to zero for now
		self.tuple[f"missing_{self.mother}_P"], self.tuple[f"missing_{self.mother}_PT"] = float(0), float(0)
		self.tuple[f"missing_{self.intermediate}_P"], self.tuple[f"missing_{self.intermediate}_PT"] = float(0), float(0)
		
		for particle_i in range(0, len(self.particles)):
			(
				self.tuple[f"delta_{particle_i}_P"],
				self.tuple[f"delta_{particle_i}_PT"],
			) = pts.compute_reconstructed_momentum_residual(self.tuple,self.particles[particle_i])

		for particle in self.particles:
			self.tuple[f"angle_{particle}"] = pts.compute_angle(self.tuple, self.mother, f"{particle}")

		self.tuple[f"IP_{self.mother}_true_vertex"] = pts.compute_impactParameter(self.tuple,self.mother,self.particles)
		for particle in self.particles:
			self.tuple[f"IP_{particle}_true_vertex"] = pts.compute_impactParameter_i(self.tuple,self.mother, f"{particle}")
		self.tuple[f"FD_{self.mother}_true_vertex"] = pts.compute_flightDistance(self.tuple,self.mother,self.particles)
		self.tuple[f"DIRA_{self.mother}_true_vertex"] = pts.compute_DIRA(self.tuple,self.mother,self.particles)

		if self.fully_reco: self.tuple[f"fully_reco"] = 1.
		else: self.tuple[f"fully_reco"] = 0.

		self.tuple[f"{self.mother}_nPositive_missing"] = float(self.nPositive_missing_particles)
		self.tuple[f"{self.mother}_nNegative_missing"] = float(self.nPositive_missing_particles)


		for particle in self.particles:
			self.tuple[f"{particle}_eta"] = -np.log(
				np.tan(
					np.arcsin(
						self.tuple[f"{particle}_PT"]
						/ self.tuple[f"{particle}_P"]
					)
					/ 2.0
				)
			)






