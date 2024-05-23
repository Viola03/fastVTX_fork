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


print("Opening file...")

file = uproot.open("/users/am13743/fast_vertexing_variables/datasets/B2KEE_three_body_cut.root:DecayTree")
branches = file.keys()
# events = file.arrays(library='pd')
print(branches)
# events = file.arrays(library='pd', entry_stop=10000)
branches = ['MOTHER_DIRA_OWNPV', 'MOTHER_ENDVERTEX_CHI2', 'MOTHER_ENDVERTEX_X', 'MOTHER_ENDVERTEX_Y', 'MOTHER_ENDVERTEX_Z', 'MOTHER_FDCHI2_OWNPV', 'MOTHER_IPCHI2_OWNPV', 'MOTHER_OWNPV_X', 'MOTHER_OWNPV_Y', 'MOTHER_OWNPV_Z', 'MOTHER_PX', 'MOTHER_PY', 'MOTHER_PZ', 'MOTHER_TRUEP_X', 'MOTHER_TRUEP_Y', 'MOTHER_TRUEP_Z', 'MOTHER_TRUEID', 'MOTHER_BKGCAT', 'DAUGHTER1_ID', 'DAUGHTER1_IPCHI2_OWNPV', 'DAUGHTER1_PX', 'DAUGHTER1_PY', 'DAUGHTER1_PZ', 'DAUGHTER1_TRACK_CHI2NDOF', 'DAUGHTER1_TRUEID', 'DAUGHTER1_TRUEP_X', 'DAUGHTER1_TRUEP_Y', 'DAUGHTER1_TRUEP_Z', 'DAUGHTER2_ID', 'DAUGHTER2_IPCHI2_OWNPV', 'DAUGHTER2_PX', 'DAUGHTER2_PY', 'DAUGHTER2_PZ', 'DAUGHTER2_TRACK_CHI2NDOF', 'DAUGHTER2_TRUEID', 'DAUGHTER2_TRUEP_X', 'DAUGHTER2_TRUEP_Y', 'DAUGHTER2_TRUEP_Z', 'DAUGHTER3_ID', 'DAUGHTER3_IPCHI2_OWNPV', 'DAUGHTER3_PX', 'DAUGHTER3_PY', 'DAUGHTER3_PZ', 'DAUGHTER3_TRACK_CHI2NDOF', 'DAUGHTER3_TRUEID', 'DAUGHTER3_TRUEP_X', 'DAUGHTER3_TRUEP_Y', 'DAUGHTER3_TRUEP_Z', 'nSPDHits', 'nTracks']

for branch in branches:
    print('\n',branch)
    try:
        events = file.arrays([branch],library='pd')
        print(events.shape)
    except:
         print("FAILED")
quit()
print("Opened file as pd array")
print(events.shape)

pid_list = [11,13,211,321]

particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
mother = 'MOTHER'

for particle in particles:
    events = events[np.abs(events[f'{particle}_TRUEID']).isin(pid_list)]

for particle in particles:
    mass = np.asarray(events[f'{particle}_TRUEID']).astype('float32')
    for pid in pid_list:
        mass[np.where(np.abs(mass)==pid)] = masses[pid]
    events[f'{particle}_mass'] = mass

print(events.shape)

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

events[f"{mother}_P"], events[f"{mother}_PT"] = vt.compute_reconstructed_mother_momenta(events, mother)
# events[f"B_P"], events[f"B_PT"] = vt.compute_reconstructed_mother_momenta(events, 'M')
events[f"missing_{mother}_P"], events[f"missing_{mother}_PT"] = vt.compute_missing_momentum(
    events, mother,particles
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
    events[f"angle_DAUGHTER{particle}"] = vt.compute_angle(events, mother, f"{particle}")

events[f"IP_{mother}"] = vt.compute_impactParameter(events,mother,particles)
for particle in particles:
    events[f"IP_{particle}"] = vt.compute_impactParameter_i(events,mother, f"{particle}")
events[f"FD_{mother}"] = vt.compute_flightDistance(events,mother,particles)
events[f"DIRA_{mother}"] = vt.compute_DIRA(events,mother,particles)

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

write_df_to_root(events, "/users/am13743/fast_vertexing_variables/datasets/B2KEE_three_body_cut_more_vars.root")
