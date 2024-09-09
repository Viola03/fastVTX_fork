from fast_vertex_quality.tools.config import read_definition, rd

import fast_vertex_quality.tools.data_loader as data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import vector
import fast_vertex_quality.tools.variables_tools as vt
import uproot
from matplotlib.backends.backend_pdf import PdfPages

masses = {}
masses[321] = 493.677
masses[211] = 139.57039
masses[13] = 105.66
masses[11] = 0.51099895000 * 1e-3

# file_name = 'Kee/Signal_tree.root'
# file_name = 'Kee/Signal_tree_LARGE.root'
file_name = 'Kee/Signal_tree_no_photos.root'
# file_name = 'Kmumu/Kmumu_tree.root'
particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
mother = 'MOTHER'
intermediate = 'INTERMEDIATE'

directory = '/users/am13743/fast_vertexing_variables/rapidsim/'
print("Opening file...")

file = uproot.open(f"{directory}/{file_name}:DecayTree")
branches = file.keys()
print(branches)
print('\n')
drop_idx = 0
new_branches = []

def drop(drop_idx):
	drop_idx += 1
	new_branches.append(f"drop_{drop_idx}")
	return drop_idx

for branch in branches:

	new_branch = branch

	if "B_plus" in branch:
		new_branch = new_branch.replace("B_plus","MOTHER")
	if "K_plus" in branch:
		new_branch = new_branch.replace("K_plus","DAUGHTER1")
	if "e_plus" in branch:
		new_branch = new_branch.replace("e_plus","DAUGHTER2")
	if "e_minus" in branch:
		new_branch = new_branch.replace("e_minus","DAUGHTER3")

	if "vtxX" in branch:
		# if branch[-4:] != "TRUE": 
		# 	drop_idx = drop(drop_idx)
		# 	continue
		if "TRUE" in branch: 
			new_branch = new_branch.replace("vtxX","TRUEENDVERTEX_X")
		else:
			new_branch = new_branch.replace("vtxX","ENDVERTEX_X")
		if branch[-4:] == "TRUE": 
			new_branch = new_branch[:-5] # remove _TRUE
	if "vtxY" in branch:
		# if branch[-4:] != "TRUE": 
		# 	drop_idx = drop(drop_idx)
		# 	continue
		if "TRUE" in branch: 
			new_branch = new_branch.replace("vtxY","TRUEENDVERTEX_Y")
		else:
			new_branch = new_branch.replace("vtxY","ENDVERTEX_Y")
		if branch[-4:] == "TRUE": 
			new_branch = new_branch[:-5] # remove _TRUE
	if "vtxZ" in branch:
		# if branch[-4:] != "TRUE": 
		# 	drop_idx = drop(drop_idx)
		# 	continue
		if "TRUE" in branch: 
			new_branch = new_branch.replace("vtxZ","TRUEENDVERTEX_Z")
		else:
			new_branch = new_branch.replace("vtxZ","ENDVERTEX_Z")
		if branch[-4:] == "TRUE": 
			new_branch = new_branch[:-5] # remove _TRUE

	if "origX" in branch:
		# if branch[-4:] != "TRUE": 
		# 	drop_idx = drop(drop_idx)
		# 	continue
		if "TRUE" in branch: 
			new_branch = new_branch.replace("origX","TRUEORIGINVERTEX_X")
		else:
			new_branch = new_branch.replace("origX","ORIGINVERTEX_X")
		if branch[-4:] == "TRUE": 
			new_branch = new_branch[:-5] # remove _TRUE
	if "origY" in branch:
		# if branch[-4:] != "TRUE": 
		# 	drop_idx = drop(drop_idx)
		# 	continue
		if "TRUE" in branch: 
			new_branch = new_branch.replace("origY","TRUEORIGINVERTEX_Y")
		else:
			new_branch = new_branch.replace("origY","ORIGINVERTEX_Y")
		if branch[-4:] == "TRUE": 
			new_branch = new_branch[:-5] # remove _TRUE
	if "origZ" in branch:
		# if branch[-4:] != "TRUE": 
		# 	drop_idx = drop(drop_idx)
		# 	continue
		if "TRUE" in branch: 
			new_branch = new_branch.replace("origZ","TRUEORIGINVERTEX_Z")
		else:
			new_branch = new_branch.replace("origZ","ORIGINVERTEX_Z")
		if branch[-4:] == "TRUE": 
			new_branch = new_branch[:-5] # remove _TRUE
	
	if "_P" in branch:
		if new_branch == 'MOTHER_P_TRUE' or new_branch == 'MOTHER_PT_TRUE':
			drop_idx = drop(drop_idx)
			continue
		if "TRUE" in branch:
			new_branch = new_branch[:-5] # remove _TRUE
			new_branch = new_branch[:-3]+'_TRUEP_'+new_branch[-1]

	new_branches.append(new_branch)
list_to_drop = ['nEvent', 'MOTHER_M_TRUE']
where = [i for i in range(len(new_branches)) if 'drop' not in new_branches[i] and new_branches[i] not in list_to_drop]
branches = list(np.asarray(branches)[where])
new_branches = list(np.asarray(new_branches)[where])

events = file.arrays(branches, library='pd')
events.columns = new_branches

for branch in new_branches:
	if "_P" in branch or "TRUEP" in branch:
	# if "TRUEP" in branch:
		events[branch] *= 1000.


use_network_to_adapt_vertex_locations = True
if use_network_to_adapt_vertex_locations:
	
	from fast_vertex_quality.training_schemes.primary_vertex import primary_vertex_trainer

	conditions = [
		f"{mother}_TRUEP",
		f"{mother}_TRUEP_T",
		f"{mother}_TRUEP_X",
		f"{mother}_TRUEP_Y",
		f"{mother}_TRUEP_Z",
	]

	targets = [
		f"{mother}_TRUE_FD",
		f"{mother}_TRUEORIGINVERTEX_X",
		f"{mother}_TRUEORIGINVERTEX_Y",
		f"{mother}_TRUEORIGINVERTEX_Z",
	]

	check_targets = [
		f"{mother}_TRUEORIGINVERTEX_X",
		f"{mother}_TRUEORIGINVERTEX_Y",
		f"{mother}_TRUEORIGINVERTEX_Z",
		f"{mother}_TRUEENDVERTEX_X",
		f"{mother}_TRUEENDVERTEX_Y",
		f"{mother}_TRUEENDVERTEX_Z",
	]
	print(mother)

	events[f"{mother}_TRUEP"] = np.sqrt(events[f"{mother}_TRUEP_X"]**2 + events[f"{mother}_TRUEP_Y"]**2 + events[f"{mother}_TRUEP_Z"]**2)
	events[f"{mother}_TRUEP_T"] = np.sqrt(events[f"{mother}_TRUEP_X"]**2 + events[f"{mother}_TRUEP_Y"]**2)
	
	# Function to compute the angles
	def compute_angles(origin, end):
		vector = np.array(end) - np.array(origin)
		theta = np.arccos(vector[2] / np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2))
		phi = np.arctan2(vector[1], vector[0])
		return theta, phi

	# Function to redefine the endpoint
	def redefine_endpoint(origin, theta, phi, new_distance):
		x = new_distance * np.sin(theta) * np.cos(phi)
		y = new_distance * np.sin(theta) * np.sin(phi)
		z = new_distance * np.cos(theta)
		# new_end = np.array(origin) + np.array([x, y, z])

		return origin[0] + x, origin[1] + y, origin[2] + z

	# Original coordinates
	B_plus_TRUEORIGINVERTEX = [events[f"{mother}_TRUEORIGINVERTEX_X"], events[f"{mother}_TRUEORIGINVERTEX_Y"], events[f"{mother}_TRUEORIGINVERTEX_Z"]]
	B_plus_TRUEENDVERTEX = [events[f"{mother}_TRUEENDVERTEX_X"], events[f"{mother}_TRUEENDVERTEX_Y"], events[f"{mother}_TRUEENDVERTEX_Z"]]
	
	# Compute the original angles
	theta, phi = compute_angles(B_plus_TRUEORIGINVERTEX, B_plus_TRUEENDVERTEX)
	# plt.hist(theta, bins=100)
	# plt.savefig('test')
	# quit()
	print(events[check_targets])

	distance_buffer = {}
	for particle in particles:
		for coordinate in ["X", "Y", "Z"]:
			distance_buffer[f"{particle}_{coordinate}"] = np.asarray(events[f"{particle}_TRUEORIGINVERTEX_{coordinate}"]-events[f"{mother}_TRUEENDVERTEX_{coordinate}"])

	# rd.latent = 50 
	rd.latent = 2 

	primary_vertex_trainer_obj = primary_vertex_trainer(
		None,
		conditions=conditions,
		targets=targets,
		beta=float(rd.beta),
		latent_dim=rd.latent,
		batch_size=64,
		D_architecture=[1000,2000,1000],
		G_architecture=[1000,2000,1000],
		network_option='VAE',
	)
	# primary_vertex_trainer_obj.load_state(tag=f"networks/primary_vertex_job2")
	# primary_vertex_trainer_obj.load_state(tag=f"networks/primary_vertex_job_general")
	primary_vertex_trainer_obj.load_state(tag=f"networks/primary_vertex_job_generalBplus")

	# BUG FIX?!
	events[f"{mother}_P"] *= 1./1000.

	output = primary_vertex_trainer_obj.predict_physical_from_physical_pandas(events[conditions], targets)

	events[targets] = output

	B_plus_TRUEORIGINVERTEX = [events[f"{mother}_TRUEORIGINVERTEX_X"], events[f"{mother}_TRUEORIGINVERTEX_Y"], events[f"{mother}_TRUEORIGINVERTEX_Z"]]
	events[f"{mother}_TRUEENDVERTEX_X"], events[f"{mother}_TRUEENDVERTEX_Y"], events[f"{mother}_TRUEENDVERTEX_Z"] = redefine_endpoint(B_plus_TRUEORIGINVERTEX, theta, phi, events[f"{mother}_TRUE_FD"])

	print(events[f"{mother}_TRUEORIGINVERTEX_Z"])
	print(events[f"{mother}_TRUE_FD"])
	print(events[f"{mother}_TRUEENDVERTEX_Z"])


	# for particle  A B C, apply diff in x y z
	for particle in particles:
		for coordinate in ["X", "Y", "Z"]:
			events[f"{particle}_TRUEORIGINVERTEX_{coordinate}"] = events[f"{mother}_TRUEENDVERTEX_{coordinate}"]+distance_buffer[f"{particle}_{coordinate}"]


	# B_plus_TRUEORIGINVERTEX = [events[f"{mother}_TRUEORIGINVERTEX_X"], events[f"{mother}_TRUEORIGINVERTEX_Y"], events[f"{mother}_TRUEORIGINVERTEX_Z"]]
	# B_plus_TRUEENDVERTEX = [events[f"{mother}_TRUEENDVERTEX_X"], events[f"{mother}_TRUEENDVERTEX_Y"], events[f"{mother}_TRUEENDVERTEX_Z"]]
	# theta, phi = compute_angles(B_plus_TRUEORIGINVERTEX, B_plus_TRUEENDVERTEX)
	# print(events[check_targets])



print(new_branches)

# set TRUEID
events['DAUGHTER1_TRUEID'] = 321
events['DAUGHTER2_TRUEID'] = 11
events['DAUGHTER3_TRUEID'] = 11

events['INTERMEDIATE_TRUEENDVERTEX_X'] = events['MOTHER_TRUEENDVERTEX_X'] 
events['INTERMEDIATE_TRUEENDVERTEX_Y'] = events['MOTHER_TRUEENDVERTEX_Y'] 
events['INTERMEDIATE_TRUEENDVERTEX_Z'] = events['MOTHER_TRUEENDVERTEX_Z'] 

# events['MOTHER_ENDVERTEX_X'] = events['MOTHER_TRUEENDVERTEX_X'] 
# events['MOTHER_ENDVERTEX_Y'] = events['MOTHER_TRUEENDVERTEX_Y'] 
# events['MOTHER_ENDVERTEX_Z'] = events['MOTHER_TRUEENDVERTEX_Z'] 

events['MOTHER_OWNPV_X'] = events['MOTHER_ORIGINVERTEX_Z'] 
events['MOTHER_OWNPV_Y'] = events['MOTHER_ORIGINVERTEX_Y'] 
events['MOTHER_OWNPV_Z'] = events['MOTHER_ORIGINVERTEX_Z'] 


print("Opened file as pd array")
print(events.shape)

for key in events.keys():
	print(key)

pid_list = [11,13,211,321]

for particle in particles:
	events = events[np.abs(events[f'{particle}_TRUEID']).isin(pid_list)]

for particle in particles:
	mass = np.asarray(events[f'{particle}_TRUEID']).astype('float32')
	for pid in pid_list:
		mass[np.where(np.abs(mass)==pid)] = masses[pid]
	events[f'{particle}_mass'] = mass

print(events.shape)


A = vt.compute_distance(events, mother, 'TRUEENDVERTEX', mother, 'TRUEORIGINVERTEX')
B = vt.compute_distance(events, particles[0], 'TRUEORIGINVERTEX', mother, 'TRUEENDVERTEX')
C = vt.compute_distance(events, particles[1], 'TRUEORIGINVERTEX', mother, 'TRUEENDVERTEX')
D = vt.compute_distance(events, particles[2], 'TRUEORIGINVERTEX', mother, 'TRUEENDVERTEX')

print(A,B,C,D)
A = np.asarray(A)
B = np.asarray(B)
C = np.asarray(C)
D = np.asarray(D)

min_A = 5e-5
min_B = 5e-5
min_C = 5e-5
min_D = 5e-5

A[np.where(A==0)] = min_A
B[np.where(B==0)] = min_B
C[np.where(C==0)] = min_C
D[np.where(D==0)] = min_D

events[f"{mother}_FLIGHT"] = A
events[f"{particles[0]}_FLIGHT"] = B
events[f"{particles[1]}_FLIGHT"] = C
events[f"{particles[2]}_FLIGHT"] = D


dist = vt.compute_intermediate_distance(events, intermediate, mother)
dist = np.asarray(dist)
print(f'fraction of intermediates that travel: {np.shape(dist[np.where(dist>0)])[0]/np.shape(dist)[0]}')
dist[np.where(dist==0)] = 1E-4
events[f"{intermediate}_FLIGHT"] = dist

# plt.hist(np.log10(dist[np.where(dist>0)]),bins=50)
# plt.xlabel('log10(distance intermediate travelled)')
# plt.title('9 percent travel')
# plt.yscale('log')
# plt.savefig('test')
# quit()


for particle_i in range(0, len(particles)):
	for particle_j in range(particle_i + 1, len(particles)):
		(
			events[f"m_{particle_i}{particle_j}"],
			events[f"m_{particle_i}{particle_j}_inside"],
		) = vt.compute_mass(
			events,
			particles[particle_i],
			particles[particle_j],
			events[f'{particles[particle_i]}_mass'],
			events[f'{particles[particle_j]}_mass'],
		)

events[f"{mother}_M"] = vt.compute_mass_3(events,
			particles[0],
			particles[1],
			particles[2],
			events[f'{particles[0]}_mass'],
			events[f'{particles[1]}_mass'],
			events[f'{particles[2]}_mass'],)

events[f"{mother}_M_Kee"] = vt.compute_mass_3(events,
			particles[0],
			particles[1],
			particles[2],
			masses[321],
			masses[11],
			masses[11],)

events[f"{mother}_M_reco"] = vt.compute_mass_3(events,
			particles[0],
			particles[1],
			particles[2],
			events[f'{particles[0]}_mass'],
			events[f'{particles[1]}_mass'],
			events[f'{particles[2]}_mass'], true_vars=False)

events[f"{mother}_M_Kee_reco"] = vt.compute_mass_3(events,
			particles[0],
			particles[1],
			particles[2],
			masses[321],
			masses[11],
			masses[11], true_vars=False)



for particle in particles:
	events[f"{particle}_P"], events[f"{particle}_PT"] = vt.compute_reconstructed_mother_momenta(events, particle)

events[f"{mother}_P"], events[f"{mother}_PT"] = vt.compute_reconstructed_mother_momenta(events, mother)

events[f"{intermediate}_P"], events[f"{intermediate}_PT"] = vt.compute_reconstructed_intermediate_momenta(events, [particles[1], particles[2]])

# events[f"B_P"], events[f"B_PT"] = vt.compute_reconstructed_mother_momenta(events, 'M')
events[f"missing_{mother}_P"], events[f"missing_{mother}_PT"] = vt.compute_missing_momentum(
	events, mother,particles
)
events[f"missing_{intermediate}_P"], events[f"missing_{intermediate}_PT"] = vt.compute_missing_momentum(
	events, mother,[particles[1], particles[2]]
)

for particle_i in range(0, len(particles)):
	(
		events[f"delta_{particle_i}_P"],
		events[f"delta_{particle_i}_PT"],
	) = vt.compute_reconstructed_momentum_residual(events, particles[particle_i])

for m in ["m_01", "m_02", "m_12"]:
	events[m] = events[m].fillna(0)

################################################################################

for particle in particles:
	events[f"angle_{particle}"] = vt.compute_angle(events, mother, f"{particle}")

true_vertex = False
events[f"IP_{mother}"] = vt.compute_impactParameter(events,mother,particles,true_vertex=true_vertex)
for particle in particles:
	events[f"IP_{particle}"] = vt.compute_impactParameter_i(events,mother, f"{particle}",true_vertex=true_vertex)
events[f"FD_{mother}"] = vt.compute_flightDistance(events,mother,particles,true_vertex=true_vertex)
events[f"DIRA_{mother}"] = vt.compute_DIRA(events,mother,particles,true_vertex=true_vertex)

true_vertex = True
events[f"IP_{mother}_true_vertex"] = vt.compute_impactParameter(events,mother,particles,true_vertex=true_vertex)
for particle in particles:
	events[f"IP_{particle}_true_vertex"] = vt.compute_impactParameter_i(events,mother, f"{particle}",true_vertex=true_vertex)
events[f"FD_{mother}_true_vertex"] = vt.compute_flightDistance(events,mother,particles,true_vertex=true_vertex)
events[f"DIRA_{mother}_true_vertex"] = vt.compute_DIRA(events,mother,particles,true_vertex=true_vertex)


print(events)

import uproot3

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

if use_network_to_adapt_vertex_locations:
	if file_name[-5:] == '.root':
		write_df_to_root(events, f"{directory}{file_name[:-5]}_NNvertex_more_vars.root")
	else:
		events.to_csv(f"{directory}{file_name[:-4]}_NNvertex_more_vars.csv")

else:
	if file_name[-5:] == '.root':
		write_df_to_root(events, f"{directory}{file_name[:-5]}_more_vars.root")
	else:
		events.to_csv(f"{directory}{file_name[:-4]}_more_vars.csv")
