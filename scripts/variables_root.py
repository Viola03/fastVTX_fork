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

# file_name = 'B2KEE_three_body_cut.root'
file_name = 'cocktail_hierarchy_merge.root'
# file_name = 'cocktail_three_body_cut.root'
# file_name = 'B2KEE_three_body_cut_SMALL.root'
particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
mother = 'MOTHER'
intermediate = 'INTERMEDIATE'

# particles = ["K_Kst", "e_minus", "e_plus"]
# mother = 'B_plus'
# intermediate = "J_psi_1S"
# # file_name = 'Kee_2018_truthed.csv'
# file_name = 'Kstee_2018_truthed.csv'
# # file_name = 'B2Kee_2018_CommonPresel.csv'
# # file_name = 'JPSIX_2018_truthed.csv'

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

print(events.shape)

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

if file_name[-5:] == '.root':
    write_df_to_root(events, f"{directory}{file_name[:-5]}_more_vars.root")
else:
    events.to_csv(f"{directory}{file_name[:-4]}_more_vars.csv")
