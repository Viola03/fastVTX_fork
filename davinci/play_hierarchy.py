
import uproot
import numpy as np
import pandas as pd
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

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


branches = [
	"MOTHER_TRUEID",
	"MOTHER_MC_MOTHER_ID",
	"MOTHER_MC_GD_MOTHER_ID",
	"MOTHER_MC_GD_GD_MOTHER_ID",
	"INTERMEDIATE_TRUEID",
	"INTERMEDIATE_MC_MOTHER_ID",
	"INTERMEDIATE_MC_GD_MOTHER_ID",
	"INTERMEDIATE_MC_GD_GD_MOTHER_ID",
	"DAUGHTER1_TRUEID",
	"DAUGHTER1_MC_MOTHER_ID",
	"DAUGHTER1_MC_GD_MOTHER_ID",
	"DAUGHTER1_MC_GD_GD_MOTHER_ID",
	"DAUGHTER2_TRUEID",
	"DAUGHTER2_MC_MOTHER_ID",
	"DAUGHTER2_MC_GD_MOTHER_ID",
	"DAUGHTER2_MC_GD_GD_MOTHER_ID",
	"DAUGHTER3_TRUEID",
	"DAUGHTER3_MC_MOTHER_ID",
	"DAUGHTER3_MC_GD_MOTHER_ID",
	"DAUGHTER3_MC_GD_GD_MOTHER_ID",
]

pdg_codes = {}
pdg_codes = []

from particle import Particle

# ##### Open file
with uproot.open('/users/am13743/fast_vertexing_variables/datasets/cocktail_hierarchy_cut.root') as ur_file:
	tree = ur_file["DecayTree"]
	# data = tree.arrays(branches, library="pd")
	data = tree.arrays(list(tree.keys()), library="pd")
# data = data.sample(frac=1)
data = data.sample(frac=0.01)

# data = data.query("MOTHER_TRUEID<5000")
# data = data.query("MOTHER_TRUEID>-5000")
data = data[data['MOTHER_TRUEID'].isin([-511,-521,-531,-541,511,521,531,541])]



# ##### Set to 0 anything before mother, in all lists
print("Cleaning up pdg codes of stuff before labelled mother, setting to 0...")
for mother_col in ['MOTHER_MC_MOTHER_ID', 'MOTHER_MC_GD_MOTHER_ID', 'MOTHER_MC_GD_GD_MOTHER_ID']:
	for particle in ['INTERMEDIATE', "DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]:
		for intermediate_col in [f'{particle}_MC_MOTHER_ID', f'{particle}_MC_GD_MOTHER_ID', f'{particle}_MC_GD_GD_MOTHER_ID']:
			mask = data[mother_col] == data[intermediate_col]
			data.loc[mask, intermediate_col] = 0
	data[mother_col] = np.zeros(np.shape(data[mother_col]))




# ##### Get list of pdg codes present
full_pdg_list = []
for branch in branches:
	pdg_codes_i = np.unique(data[branch])
	full_pdg_list.extend(pdg_codes_i)
print(f'len(unique pdg codes): {len(full_pdg_list)}')





# ##### Create pdg look up table for masses and widths, save pdg codes not recognised 

alex_lookup_by_hand = {}
alex_lookup_by_hand[4334] = 1.0
alex_lookup_by_hand[4322] = 1.0
alex_lookup_by_hand[4312] = 1.0
alex_lookup_by_hand[533] = 1.0
alex_lookup_by_hand[523] = 1.0
alex_lookup_by_hand[513] = 1.0
alex_lookup_by_hand[433] = 1.0
alex_lookup_by_hand[311] = 7.3508e-13 # unclear if KS or KL



# Rouge pdg codes: [-4334.0, -4322.0, -4312.0, -2212.0, -533.0, -513.0, -433.0, -311.0, -11.0, 11.0, 22.0, 311.0, 433.0, 513.0, 523.0, 533.0, 2212.0, 4312.0, 4322.0, 4334.0]


lookup_table = np.empty((0, 3))
deletes = []
deletes_no_width = []
for item in np.unique(np.asarray(full_pdg_list)):
	try:
		if Particle.from_pdgid(item).mass != None and Particle.from_pdgid(item).width != None:
			new_row = np.asarray([[item, Particle.from_pdgid(item).mass, Particle.from_pdgid(item).width]])
			print(item,'\t',Particle.from_pdgid(item).name,'\t',Particle.from_pdgid(item).mass,'\t',Particle.from_pdgid(item).width)
		else:
			try:
				new_row = np.asarray([[item, Particle.from_pdgid(item).mass, alex_lookup_by_hand[abs(int(item))]]])
				print(f"{bcolors.WARNING}{item} {Particle.from_pdgid(item).width}: \t {Particle.from_pdgid(item).name} \t {Particle.from_pdgid(item).mass} \t {new_row[-1][-1]} {bcolors.ENDC}")
			except:
				new_row = np.asarray([[item, 0., 0.]])
				deletes_no_width.append(item)
	except:
		print(f"{bcolors.FAIL}Failure! {item} {bcolors.ENDC}")
		new_row = np.asarray([[item, 0., 0.]])
		deletes.append(item)
	lookup_table = np.append(lookup_table, new_row, axis=0)
lookup_table = pd.DataFrame({'pdg': lookup_table[:, 0], 'mass': lookup_table[:, 1], 'width': lookup_table[:, 2]})




# ##### Remove events with unrecognised pdg codes
print(f'Rouge pdg codes: {deletes}')
pre = data.shape[0]
for delete in deletes:
	if delete != 0.:
		for branch in branches:
			data = data.query(f'{branch} != {delete}')
post = data.shape[0]
print(f'Removing events with any rouge pdg codes, eff: {post/pre:.4f}')

# ##### Remove events with unrecognised pdg codes
print(f'Rouge pdg codes: {deletes_no_width}')
pre = data.shape[0]
for delete in deletes_no_width:
	if delete != 0.:
		for branch in branches:
			data = data.query(f'{branch} != {delete}')
post = data.shape[0]
print(f'Removing events with any rouge pdg codes, eff: {post/pre:.4f}')





# ##### Add mass and width data
def add_mass_width_columns(data, lookup_table, column_name):
	merge_df = data[[column_name]].merge(lookup_table, how='left', left_on=column_name, right_on='pdg')
	
	mass = np.asarray(merge_df['mass']).astype('float64')
	width = np.asarray(merge_df['width']).astype('float64')

	data[f"{column_name}_mass"] = mass
	data[f"{column_name}_width"] = width

	print(column_name, np.min(mass), np.max(mass), np.min(width), np.max(width))

	return data

for column in branches:
	data = add_mass_width_columns(data, lookup_table, column)

write_df_to_root(data, '/users/am13743/fast_vertexing_variables/datasets/cocktail_hierarchy_merge.root')


# ##### PRINTING SOME EXAMPLES

def get_chain(row, particle, property=''):
	chain = [row[f'{particle}_TRUEID{property}'], row[f'{particle}_MC_MOTHER_ID{property}'], row[f'{particle}_MC_GD_MOTHER_ID{property}'], row[f'{particle}_MC_GD_GD_MOTHER_ID{property}']]
	return chain

def find_daughters(chains, mother):
	daughters = []
	for chain in chains:
		for particle_idx, particle in enumerate(chain):
			if particle == mother and particle_idx != 0:
				daughters.append(p)
			p = particle
	return list(np.unique(daughters))

def print_property(chains, property):

	chains_strings = chains.copy().astype('str')
	for ii in range(np.shape(chains)[0]):
		for jj in range(np.shape(chains)[1]):
			try:
				if property == 'name':
					chains_strings[ii][jj] = Particle.from_pdgid(chains[ii][jj]).name
				if property == 'mass':
					mass = Particle.from_pdgid(chains[ii][jj]).mass
					# if mass == 0.0:
					# 	mass = -20
					# else:
					# 	mass = np.log10(mass)
					chains_strings[ii][jj] = mass
				if property == 'width':
					width = Particle.from_pdgid(chains[ii][jj]).width
					# if width == 0.0:
					# 	width = -20
					# else:
					# 	width = np.log10(width)
					chains_strings[ii][jj] = width
			except:
				chains_strings[ii][jj] = 'null'
				continue

	print(chains_strings)

condition_branches = [
"MOTHER_TRUEID",
"MOTHER_TRUEID_width",
"INTERMEDIATE_TRUEID_width",
"INTERMEDIATE_MC_MOTHER_ID_width",
"INTERMEDIATE_MC_GD_MOTHER_ID_width",
"INTERMEDIATE_MC_GD_GD_MOTHER_ID_width",
"DAUGHTER1_TRUEID",
"DAUGHTER1_MC_MOTHER_ID_width",
"DAUGHTER1_MC_GD_MOTHER_ID_width",
"DAUGHTER1_MC_GD_GD_MOTHER_ID_width",
"DAUGHTER2_TRUEID",
"DAUGHTER2_MC_MOTHER_ID_width",
"DAUGHTER2_MC_GD_MOTHER_ID_width",
"DAUGHTER2_MC_GD_GD_MOTHER_ID_width",
"DAUGHTER3_TRUEID",
"DAUGHTER3_MC_MOTHER_ID_width",
"DAUGHTER3_MC_GD_MOTHER_ID_width",
"DAUGHTER3_MC_GD_GD_MOTHER_ID_width",
"MOTHER_TRUEID_mass",
"INTERMEDIATE_TRUEID_mass",
"INTERMEDIATE_MC_MOTHER_ID_mass",
"INTERMEDIATE_MC_GD_MOTHER_ID_mass",
"INTERMEDIATE_MC_GD_GD_MOTHER_ID_mass",
"DAUGHTER1_MC_MOTHER_ID_mass",
"DAUGHTER1_MC_GD_MOTHER_ID_mass",
"DAUGHTER1_MC_GD_GD_MOTHER_ID_mass",
"DAUGHTER2_MC_MOTHER_ID_mass",
"DAUGHTER2_MC_GD_MOTHER_ID_mass",
"DAUGHTER2_MC_GD_GD_MOTHER_ID_mass",
"DAUGHTER3_MC_MOTHER_ID_mass",
"DAUGHTER3_MC_GD_MOTHER_ID_mass",
"DAUGHTER3_MC_GD_GD_MOTHER_ID_mass",
]

for i in range(25):

	row = data.iloc[i]

	print('\n',i)

	# print(row[condition_branches])
	# for property in ['','_mass','_width']:
	for property in ['']:

		mother_chain = get_chain(row, 'MOTHER',property=property)
		intermediate_chain = get_chain(row, 'INTERMEDIATE',property=property)
		daughter1_chain = get_chain(row, 'DAUGHTER1',property=property)
		daughter2_chain = get_chain(row, 'DAUGHTER2',property=property)
		daughter3_chain = get_chain(row, 'DAUGHTER3',property=property)

		chains = np.asarray([mother_chain, intermediate_chain, daughter1_chain, daughter2_chain, daughter3_chain])

		print(chains)



