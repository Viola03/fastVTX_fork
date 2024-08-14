import pickle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from fast_vertex_quality.tools.config import read_definition, rd
import fast_vertex_quality.tools.data_loader as data_loader
import numpy as np

import tensorflow as tf
from fast_vertex_quality.training_schemes.primary_vertex import primary_vertex_trainer

rd.latent = 2 # noise dims
# rd.beta = 1000

rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'
rd.intermediate_particle = 'J_psi_1S'

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        # "datasets/cocktail_hierarchy_cut_more_vars.root",
        # "datasets/cocktail_x5_MC_hierachy_cut_more_vars.root",
        "datasets/general_sample_intermediate_more_vars.root",
    ],
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'},
    # N=2500,
)
training_data_loader.add_branch_and_process(name='B_plus_TRUE_FD',recipe="sqrt((B_plus_TRUEENDVERTEX_X-B_plus_TRUEORIGINVERTEX_X)**2 + (B_plus_TRUEENDVERTEX_Y-B_plus_TRUEORIGINVERTEX_Y)**2 + (B_plus_TRUEENDVERTEX_Z-B_plus_TRUEORIGINVERTEX_Z)**2)")
training_data_loader.add_branch_and_process(name='B_plus_TRUEP',recipe="sqrt((B_plus_TRUEP_X)**2 + (B_plus_TRUEP_Y)**2 + (B_plus_TRUEP_Z)**2)")
training_data_loader.add_branch_and_process(name='B_plus_TRUEP_T',recipe="sqrt((B_plus_TRUEP_X)**2 + (B_plus_TRUEP_Y)**2)")

transformers = training_data_loader.get_transformers()

training_data_loader.cut('B_plus_TRUEP_Z>0')
training_data_loader.cut('abs(B_plus_TRUEID)>521')
# training_data_loader.print_branches()



conditions = [
    "B_plus_TRUEP",
    "B_plus_TRUEP_T",
    "B_plus_TRUEP_X",
    "B_plus_TRUEP_Y",
    "B_plus_TRUEP_Z",
]

targets = [
    "B_plus_TRUE_FD",
	"B_plus_TRUEORIGINVERTEX_X",
	"B_plus_TRUEORIGINVERTEX_Y",
	"B_plus_TRUEORIGINVERTEX_Z",
]


# check_list = [
# 	"B_plus_TRUEORIGINVERTEX_X",
# 	"B_plus_TRUEORIGINVERTEX_Y",
# 	"B_plus_TRUEORIGINVERTEX_Z",
# 	"B_plus_TRUEENDVERTEX_X",
# 	"B_plus_TRUEENDVERTEX_Y",
# 	"B_plus_TRUEENDVERTEX_Z",
# ]
# training_data_loader.plot('conditions.pdf',check_list)
# quit()

# events = training_data_loader.get_branches([f"B_plus_TRUEORIGINVERTEX_Z", f"B_plus_TRUE_FD"],processed=False)
# plt.hist2d(events[f"B_plus_TRUEORIGINVERTEX_Z"], events[f"B_plus_TRUE_FD"], bins=100, norm=LogNorm(), range=[[-250,250],[0,300]])
# plt.savefig('TRAIN')
# quit()

primary_vertex_trainer_obj = primary_vertex_trainer(
    training_data_loader,
    conditions=conditions,
    targets=targets,
    beta=float(rd.beta),
    latent_dim=rd.latent,
    batch_size=64,
    D_architecture=[1000,2000,1000],
    G_architecture=[1000,2000,1000],
    network_option='VAE',
    # network_option='WGAN',
)

steps_for_plot = 5000
primary_vertex_trainer_obj.train(steps=steps_for_plot)
# primary_vertex_trainer_obj.save_state(tag=f"networks/primary_vertex_job2")
primary_vertex_trainer_obj.save_state(tag=f"networks/primary_vertex_job_generalBplus")
primary_vertex_trainer_obj.make_plots(filename=f'vertex_plots_0.pdf',testing_file=training_data_loader.get_file_names())

for i in range(100):
    primary_vertex_trainer_obj.train_more_steps(steps=steps_for_plot)
    # primary_vertex_trainer_obj.save_state(tag=f"networks/primary_vertex_job2")
    primary_vertex_trainer_obj.save_state(tag=f"networks/primary_vertex_job_generalBplus")
    primary_vertex_trainer_obj.make_plots(filename=f'vertex_plots_{i+1}.pdf',testing_file=training_data_loader.get_file_names())



quit()



























def mag(vec):
    sum_sqs = 0
    for component in vec:
        sum_sqs += component**2
    mag = np.sqrt(sum_sqs)
    return mag


def norm(vec):
    mag_vec = mag(vec)
    for component_idx in range(np.shape(vec)[0]):
        vec[component_idx] *= 1.0 / mag_vec
    return vec


def dot(vec1, vec2):
    dot = 0
    for component_idx in range(np.shape(vec1)[0]):
        dot += vec1[component_idx] * vec2[component_idx]
    return dot


def compute_DIRA(df, mother):

    PX = df[f"{mother}_TRUEP_X"]
    PY = df[f"{mother}_TRUEP_Y"]
    PZ = df[f"{mother}_TRUEP_Z"]

    A = norm(np.asarray([
        PX,
        PY,
        PZ]))

    B = norm(
        np.asarray(
            [
                df[f"{mother}_TRUEENDVERTEX_X"] - df[f"{mother}_TRUEORIGINVERTEX_X"],
                df[f"{mother}_TRUEENDVERTEX_Y"] - df[f"{mother}_TRUEORIGINVERTEX_Y"],
                df[f"{mother}_TRUEENDVERTEX_Z"] - df[f"{mother}_TRUEORIGINVERTEX_Z"],
            ]
        )
    )

    # dira = dot(A, B) / np.sqrt(mag(A) ** 2 * mag(B) ** 2)
    dira = dot(A, B) / (mag(A) * mag(B))

    return dira

# example = training_data_loader.get_branches(conditions+targets, processed=False)
# # example = training_data_loader_Kee.get_branches(conditions+targets, processed=False)
# # print(example)
# # print(compute_DIRA(example))
# example['DIRA_physical'] = compute_DIRA(example)
# example['DIRA_processed'] = transformers['DIRA_B_plus_true_vertex'].process(example['DIRA_physical'])
# print(example['DIRA_physical'])
# print(example['DIRA_processed'])

# plt.subplot(1,2,1)
# plt.hist(example['DIRA_physical'],bins=100)
# plt.subplot(1,2,2)
# plt.hist(example['DIRA_processed'], range=[-1,1],bins=100)
# plt.savefig("DIRA")

# query = example.query('DIRA_processed<-0.5')
# print(query['DIRA_physical'])
# print(query['DIRA_processed'])

# quit()


import uproot
masses = {}
masses[321] = 493.677
masses[211] = 139.57039
masses[13] = 105.66
masses[11] = 0.51099895000 * 1e-3

file_name = 'Kee/Signal_tree.root'
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


conditions_test = [
    f"{mother}_P",
    f"{mother}_PT",
    f"{mother}_TRUEP_X",
    f"{mother}_TRUEP_Y",
    f"{mother}_TRUEP_Z",
]

targets_test = [
    f"{mother}_TRUEENDVERTEX_X",
    f"{mother}_TRUEENDVERTEX_Y",
    f"{mother}_TRUEENDVERTEX_Z",
    f"{mother}_TRUEORIGINVERTEX_X",
    f"{mother}_TRUEORIGINVERTEX_Y",
    f"{mother}_TRUEORIGINVERTEX_Z",
]

print(events[targets_test])
print(events[conditions_test])

# example = training_data_loader.get_branches(conditions+targets, processed=False)
# # example = training_data_loader_Kee.get_branches(conditions+targets, processed=False)
# # print(example)
print(compute_DIRA(events[targets_test+conditions_test], mother))

quit()









# with PdfPages('distances.pdf') as pdf:

# 	for variable in list(conditions):
		
# 		try:

# 			plt.figure(figsize=(10,8))

# 			plt.subplot(2,2,1)
# 			plt.title(variable)
# 			plt.hist(conditions_cocktail["physical"][variable], bins=50, density=True, histtype='step')
			
# 			plt.subplot(2,2,2)
# 			plt.title(f'{variable} processed')
# 			plt.hist(conditions_cocktail["processed"][variable], bins=50, density=True, histtype='step', range=[-1,1])

# 			plt.subplot(2,2,3)
# 			plt.hist(conditions_cocktail["physical"][variable], bins=50, density=True, histtype='step')
# 			plt.yscale('log')
			
# 			plt.subplot(2,2,4)
# 			plt.hist(conditions_cocktail["processed"][variable], bins=50, density=True, histtype='step', range=[-1,1])
# 			plt.yscale('log')

# 			pdf.savefig(bbox_inches="tight")
# 			plt.close()
# 		except:
# 			pass



primary_vertex_trainer_obj = primary_vertex_trainer(
    training_data_loader,
    conditions=conditions,
    targets=targets,
    beta=float(rd.beta),
    latent_dim=rd.latent,
    batch_size=64,
    D_architecture=[100,200,100],
    G_architecture=[100,200,100],
    network_option='VAE',
)

# training_data_loader.plot('conditions_TRAIN.pdf',conditions)
# quit()

steps_for_plot = 25
primary_vertex_trainer_obj.train(steps=steps_for_plot)
primary_vertex_trainer_obj.save_state(tag=f"networks/primary_vertex_jon")
primary_vertex_trainer_obj.make_plots(filename=f'plots_0.pdf',testing_file=training_data_loader.get_file_names())


output = primary_vertex_trainer_obj.predict_physical_from_physical_pandas(events[conditions_test], targets_test)
df = events[conditions_test]
for branch in list(output.keys()):
    df[branch] = output[branch]
example = training_data_loader.get_branches(conditions_test+targets_test, processed=False)
print(df, list(df.keys()))
DIRA_physical = compute_DIRA(df, 'MOTHER')
DIRA_processed = transformers['DIRA_B_plus_true_vertex'].process(DIRA_physical)
plt.subplot(1,2,1)
plt.hist(DIRA_physical,bins=100)
plt.subplot(1,2,2)
plt.hist(DIRA_processed, range=[-1,1],bins=100)
plt.savefig("train_DIRA_0")
plt.close('all')
quit()

for i in range(10):
    primary_vertex_trainer_obj.train_more_steps(steps=steps_for_plot)
    primary_vertex_trainer_obj.save_state(tag=f"networks/primary_vertex_job")
    primary_vertex_trainer_obj.make_plots(filename=f'plots_{i+1}.pdf',testing_file=training_data_loader.get_file_names())

    output = primary_vertex_trainer_obj.predict_physical_from_physical_pandas(events[conditions_test], targets_test)
    df = events[conditions_test]
    for branch in list(output.keys()):
        df[branch] = output[branch]
    example = training_data_loader.get_branches(conditions_test+targets_test, processed=False)
    DIRA_physical = compute_DIRA(df, 'MOTHER')
    DIRA_processed = transformers['DIRA_B_plus_true_vertex'].process(DIRA_physical)
    plt.subplot(1,2,1)
    plt.hist(DIRA_physical,bins=100)
    plt.subplot(1,2,2)
    plt.hist(DIRA_processed, range=[-1,1],bins=100)
    plt.savefig(f"train_DIRA_{i+1}")
    plt.close('all')

