from fast_vertex_quality.tools.config import read_definition, rd

import fast_vertex_quality.tools.data_loader as data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import vector
import fast_vertex_quality.tools.variables_tools as vt
import uproot

masses = {}
masses[321] = 493.677
masses[211] = 139.57039
masses[13] = 105.66
masses[11] = 0.51099895000 * 1e-3

use_intermediate = True
# use_intermediate = False

# file_name = 'cocktail_hierarchy_cut.root'
# particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
# mother = 'MOTHER'
# intermediate = 'INTERMEDIATE'


# file_name = 'dedicated_Kee_MC_hierachy_cut.root'
# particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
# mother = 'MOTHER'
# intermediate = 'INTERMEDIATE'

# file_name = 'dedicated_Kmumu_MC_hierachy_cut.root'
# particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
# mother = 'MOTHER'
# intermediate = 'INTERMEDIATE'

# file_name = 'general_sample.root'
file_name = 'general_sample_intermediate.root'
particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
mother = 'MOTHER'
intermediate = 'INTERMEDIATE'

# file_name = 'dedicated_Kstee_MC_hierachy_cut.root'
# particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
# mother = 'MOTHER'
# intermediate = 'INTERMEDIATE'

# file_name = 'dedicated_BuD0enuKenu_MC_hierachy_cut.root'
# particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
# mother = 'MOTHER'
# intermediate = 'INTERMEDIATE'

# file_name = 'dedicated_BuD0piKenu_MC_hierachy_cut.root'
# particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
# mother = 'MOTHER'
# intermediate = 'INTERMEDIATE'

# file_name = 'cocktail_x5_MC_hierachy_cut.root'
# particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
# mother = 'MOTHER'
# intermediate = 'INTERMEDIATE'



# particles = ["K_Kst", "e_minus", "e_plus"]
# mother = 'B_plus'
# intermediate = "J_psi_1S"
# file_name = 'Kee_cut.root'

# particles = ["K_Kst", "e_minus", "e_plus"]
# mother = 'B_plus'
# intermediate = "J_psi_1S"
# file_name = 'Kstee_cut.root'


directory = '/users/am13743/fast_vertexing_variables/datasets/'
print("Opening file...")

if file_name[-5:] == '.root':
    file = uproot.open(f"{directory}/{file_name}:DecayTree")
    branches = file.keys()
    events = file.arrays(library='pd')
else:
    events = pd.read_csv(f"{directory}/{file_name}")

print("Opened file as pd array")
print(events.shape)

# for particle in particles:
#     print(np.unique(np.abs(events[f'{particle}_TRUEID'])))
# quit()

pid_list = [11,13,211,321]

for particle in particles:
    events = events[np.abs(events[f'{particle}_TRUEID']).isin(pid_list)]

for particle in particles:
    mass = np.asarray(events[f'{particle}_TRUEID']).astype('float32')
    for pid in pid_list:
        mass[np.where(np.abs(mass)==pid)] = masses[pid]
    events[f'{particle}_mass'] = mass

# print(events.shape)

print(events[f'{mother}_TRUEP_Z'])

A = vt.compute_distance(events, mother, 'TRUEENDVERTEX', mother, 'TRUEORIGINVERTEX')
B = vt.compute_distance(events, particles[0], 'TRUEORIGINVERTEX', mother, 'TRUEENDVERTEX')
C = vt.compute_distance(events, particles[1], 'TRUEORIGINVERTEX', mother, 'TRUEENDVERTEX')
D = vt.compute_distance(events, particles[2], 'TRUEORIGINVERTEX', mother, 'TRUEENDVERTEX')

A = np.asarray(A)
B = np.asarray(B)
C = np.asarray(C)
D = np.asarray(D)

# min_A = np.amin(A[np.where(A>0)])/2.
# min_B = np.amin(B[np.where(B>0)])/2.
# min_C = np.amin(C[np.where(C>0)])/2.
# min_D = np.amin(D[np.where(D>0)])/2.

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


if use_intermediate:
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

# print(events[f'{mother}_TRUEP_Z'])
# quit()
events[f"{mother}_P"], events[f"{mother}_PT"] = vt.compute_reconstructed_mother_momenta(events, mother)

if use_intermediate:
    events[f"{intermediate}_P"], events[f"{intermediate}_PT"] = vt.compute_reconstructed_intermediate_momenta(events, [particles[1], particles[2]])
# print(events[f'{mother}_TRUEP_Z'])
# quit()
# events[f"B_P"], events[f"B_PT"] = vt.compute_reconstructed_mother_momenta(events, 'M')
events[f"missing_{mother}_P"], events[f"missing_{mother}_PT"] = vt.compute_missing_momentum(
    events, mother,particles
)
if use_intermediate:
    events[f"missing_{intermediate}_P"], events[f"missing_{intermediate}_PT"] = vt.compute_missing_momentum(
        events, mother,[particles[1], particles[2]]
    )
# print(events[f'{mother}_TRUEP_Z'])
# quit()
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


print(events[f'{mother}_TRUEP_Z'])

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


cut_condition = "(MOTHER_TRUEID != 0) & (MOTHER_BKGCAT < 60)"
events = events.query(cut_condition)  

if file_name[-5:] == '.root':
    write_df_to_root(events, f"{directory}{file_name[:-5]}_more_vars.root")
else:
    events.to_csv(f"{directory}{file_name[:-4]}_more_vars.csv")
