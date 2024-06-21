
import uproot
import numpy as np
import pandas as pd
import uproot3 
from particle import Particle

def get_chain(row, particle, property=''):
	if property == 'name':
		chain = []
		try:chain.append(Particle.from_pdgid(row[f'{particle}_TRUEID']).name)
		except: chain.append('N/A')
		try:chain.append(Particle.from_pdgid(row[f'{particle}_MC_MOTHER_ID']).name)
		except: chain.append('N/A')
		try:chain.append(Particle.from_pdgid(row[f'{particle}_MC_GD_MOTHER_ID']).name)
		except: chain.append('N/A')
		try:chain.append(Particle.from_pdgid(row[f'{particle}_MC_GD_GD_MOTHER_ID']).name)
		except: chain.append('N/A')
	else:
		if property == 'code':
			property = ''
		chain = [row[f'{particle}_TRUEID{property}'], row[f'{particle}_MC_MOTHER_ID{property}'], row[f'{particle}_MC_GD_MOTHER_ID{property}'], row[f'{particle}_MC_GD_GD_MOTHER_ID{property}']]
	return chain

file = uproot.open("/users/am13743/fast_vertexing_variables/datasets/cocktail_hierarchy_merge_more_vars.root:DecayTree")

data = file.arrays(library='pd')

data = data.query("abs(DAUGHTER2_MC_MOTHER_ID)==421 or abs(DAUGHTER2_MC_GD_MOTHER_ID)==421")

for i in range(25):

	row = data.iloc[i]

	print('\n',i)
	
	for property in ['code','name','_mass','_width']:

		mother_chain = get_chain(row, 'MOTHER',property=property)
		intermediate_chain = get_chain(row, 'INTERMEDIATE',property=property)
		daughter1_chain = get_chain(row, 'DAUGHTER1',property=property)
		daughter2_chain = get_chain(row, 'DAUGHTER2',property=property)
		daughter3_chain = get_chain(row, 'DAUGHTER3',property=property)

		print("MOTHER", mother_chain[0])
		print("INTERMEDIATE", intermediate_chain)
		print("DAUGHTER1", daughter1_chain)
		print("DAUGHTER2", daughter2_chain)
		print("DAUGHTER3", daughter3_chain)


	print("Conditions:")
	mother_chain = get_chain(row, 'MOTHER',property='code')
	intermediate_chain = get_chain(row, 'INTERMEDIATE',property=property)
	daughter1_chain = get_chain(row, 'DAUGHTER1',property=property)
	daughter2_chain = get_chain(row, 'DAUGHTER2',property=property)
	daughter3_chain = get_chain(row, 'DAUGHTER3',property=property)

	print("MOTHER", mother_chain[0])
	print("INTERMEDIATE", intermediate_chain)
	print("DAUGHTER1", daughter1_chain)
	print("DAUGHTER2", daughter2_chain)
	print("DAUGHTER3", daughter3_chain)



# import uproot
# import numpy as np

# file = uproot.open("/users/am13743/fast_vertexing_variables/datasets/B2KEE_three_body_cut.root:DecayTree")

# def print_unique_PIDs(branches):

#     pids = np.empty(0)

#     arrays = file.arrays(branches, library='pd')

#     for item in branches:
		
#         pids_i = np.unique(np.asarray(arrays[item]))
#         pids = np.append(pids, pids_i)

#     for pid in np.unique(pids):
#         print(int(pid))

# print_unique_PIDs(["M_TRUEID"])
# print('\n')

# print_unique_PIDs(["A_TRUEID", "B_TRUEID", "C_TRUEID"])
# print('\n')



# arrays = file.arrays(["A_TRUEID", "B_TRUEID", "C_TRUEID"], library='pd')

# print(arrays.shape)
# pre = arrays.shape[0]

# pid_list = [-321,-211,-13,-11,11,13,211,321]

# arrays = arrays[arrays['A_TRUEID'].isin(pid_list)]
# arrays = arrays[arrays['B_TRUEID'].isin(pid_list)]
# arrays = arrays[arrays['C_TRUEID'].isin(pid_list)]

# print(np.unique(arrays['A_TRUEID']))
# print(arrays.shape)
# post = arrays.shape[0]
# print(post/pre)
