import pickle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from fast_vertex_quality.tools.config import read_definition, rd
import fast_vertex_quality.tools.data_loader as data_loader
import numpy as np

conditions = [

    "B_plus_P",
    "B_plus_PT",
    "angle_K_Kst",
    "angle_e_plus",
    "angle_e_minus",
    "K_Kst_eta",
    "e_plus_eta",
    "e_minus_eta",
    # "IP_B_plus",
    # "IP_K_Kst",
    # "IP_e_plus",
    # "IP_e_minus",
    # "FD_B_plus",
    # "DIRA_B_plus",
	 "IP_B_plus_true_vertex",
    "IP_K_Kst_true_vertex",
    "IP_e_plus_true_vertex",
    "IP_e_minus_true_vertex",
    "FD_B_plus_true_vertex",
    "DIRA_B_plus_true_vertex",
    "missing_B_plus_P",
    "missing_B_plus_PT",
    "missing_J_psi_1S_P",
    "missing_J_psi_1S_PT",
    "m_01",
    "m_02",
    "m_12",


	# "B_plus_TRUEENDVERTEX_X",
	# "B_plus_TRUEENDVERTEX_Y",
	# "B_plus_TRUEENDVERTEX_Z",
	# "B_plus_TRUEORIGINVERTEX_X",
	# "B_plus_TRUEORIGINVERTEX_Y",
	# "B_plus_TRUEORIGINVERTEX_Z",

	# "e_plus_TRUEORIGINVERTEX_X",
	# "e_plus_TRUEORIGINVERTEX_Y",
	# "e_plus_TRUEORIGINVERTEX_Z",


    # "B_plus_TRUEP_X",
    # "B_plus_TRUEP_Y",
    # "B_plus_TRUEP_Z",
    # "K_Kst_TRUEP_X",
    # "K_Kst_TRUEP_Y",
    # "K_Kst_TRUEP_Z",
    # "K_Kst_PT",
    # "e_plus_PT",
    # "e_minus_PT",

	# "K_Kst_TRUEID",
	# "e_minus_TRUEID",
	# "e_plus_TRUEID",

    # "B_plus_FLIGHT",
    "K_Kst_FLIGHT",
    "e_plus_FLIGHT",
    "e_minus_FLIGHT",

]

rd.latent = 50 # noise dims

rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'
rd.intermediate_particle = 'J_psi_1S'

print(f"Loading data...")
training_data_loader_cocktail = data_loader.load_data(
    [
        "datasets/cocktail_hierarchy_cut_more_vars.root",
    ],
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'},
    # N=2500,
)
# training_data_loader_cocktail.cut('B_plus_TRUEP_Z>0')
transformers = training_data_loader_cocktail.get_transformers()

conditions_cocktail = {}
conditions_cocktail["processed"] = training_data_loader_cocktail.get_branches(conditions, processed=True)
conditions_cocktail["physical"] = training_data_loader_cocktail.get_branches(conditions, processed=False)

print(f"Loading data...")
training_data_loader = data_loader.load_data(
    [
        # "datasets/Kee_cut_more_vars.root",
        "datasets/Kstee_cut_more_vars.root",
    ],
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'},
    # N=2500,
	transformers=transformers
)
# training_data_loader.cut('B_plus_TRUEP_Z>0')
training_data_loader.cut('abs(K_Kst_TRUEID)==321')

conditions_notrapdsim = {}
conditions_notrapdsim["processed"] = training_data_loader.get_branches(conditions, processed=True)
conditions_notrapdsim["physical"] = training_data_loader.get_branches(conditions, processed=False)


training_data_loader_rapidsim = data_loader.load_data(
    [
        # "rapidsim/Kee/Signal_tree_more_vars.root",
        # "rapidsim/Kee/Signal_tree_NNvertex_more_vars.root",
        "rapidsim/Kstree/Partreco_tree_NNvertex_more_vars.root",
    ],
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'},
    # N=2500,
	transformers=transformers
)
training_data_loader_rapidsim.cut('K_Kst_PT>400')
training_data_loader_rapidsim.cut('e_minus_PT>300')
training_data_loader_rapidsim.cut('e_plus_PT>300')
# training_data_loader_rapidsim.cut('B_plus_TRUEP_Z>0')


conditions_rapdsim = {}
conditions_rapdsim["processed"] = training_data_loader_rapidsim.get_branches(conditions, processed=True)
conditions_rapdsim["physical"] = training_data_loader_rapidsim.get_branches(conditions, processed=False)

# conditions_rapdsim["physical"]["dist"] = np.sqrt((conditions_rapdsim["physical"]["B_plus_TRUEENDVERTEX_X"]-conditions_rapdsim["physical"]["B_plus_TRUEORIGINVERTEX_X"])**2+
# 												 (conditions_rapdsim["physical"]["B_plus_TRUEENDVERTEX_Y"]-conditions_rapdsim["physical"]["B_plus_TRUEORIGINVERTEX_Y"])**2+
# 												 (conditions_rapdsim["physical"]["B_plus_TRUEENDVERTEX_Z"]-conditions_rapdsim["physical"]["B_plus_TRUEORIGINVERTEX_Z"])**2
# 												 )
# conditions_notrapdsim["physical"]["dist"] = np.sqrt((conditions_notrapdsim["physical"]["B_plus_TRUEENDVERTEX_X"]-conditions_notrapdsim["physical"]["B_plus_TRUEORIGINVERTEX_X"])**2+
# 												 (conditions_notrapdsim["physical"]["B_plus_TRUEENDVERTEX_Y"]-conditions_notrapdsim["physical"]["B_plus_TRUEORIGINVERTEX_Y"])**2+
# 												 (conditions_notrapdsim["physical"]["B_plus_TRUEENDVERTEX_Z"]-conditions_notrapdsim["physical"]["B_plus_TRUEORIGINVERTEX_Z"])**2
# 												 )
def compare_in_2d(filename, var_1, var_2, processed=False):

	if processed:
		tag = "processed"
	else:
		tag = "physical"

	max_var_1 = np.amax([np.amax(conditions_rapdsim[tag][var_1]), np.amax(conditions_notrapdsim[tag][var_1])])
	min_var_1 = np.amin([np.amin(conditions_rapdsim[tag][var_1]), np.amin(conditions_notrapdsim[tag][var_1])])
	max_var_2 = np.amax([np.amax(conditions_rapdsim[tag][var_2]), np.amax(conditions_notrapdsim[tag][var_2])])
	min_var_2 = np.amin([np.amin(conditions_rapdsim[tag][var_2]), np.amin(conditions_notrapdsim[tag][var_2])])

	plt.figure(figsize=(8,8))
	plt.subplot(2,2,1)
	plt.title("Rapidsim")
	plt.hist2d(conditions_rapdsim[tag][var_1], conditions_rapdsim[tag][var_2], bins=100, norm=LogNorm(), range=[[min_var_1, max_var_1],[min_var_2, max_var_2]])
	plt.xlabel(var_1)
	plt.ylabel(var_2)

	plt.subplot(2,2,2)
	plt.title("Not rapidsim")
	plt.hist2d(conditions_notrapdsim[tag][var_1], conditions_notrapdsim[tag][var_2], bins=100, norm=LogNorm(), range=[[min_var_1, max_var_1],[min_var_2, max_var_2]])
	plt.xlabel(var_1)
	plt.ylabel(var_2)

	plt.subplot(2,2,3)
	plt.xlabel(var_1)
	plt.hist([conditions_rapdsim[tag][var_1], conditions_notrapdsim[tag][var_1]], bins=100, range=[min_var_1, max_var_1], label=['Rapidsim','Not rapdisim'], histtype='step', density=True)
	plt.legend()
	plt.subplot(2,2,4)
	plt.xlabel(var_2)
	plt.hist([conditions_rapdsim[tag][var_2], conditions_notrapdsim[tag][var_2]], bins=100, range=[min_var_2, max_var_2], label=['Rapidsim','Not rapdisim'], histtype='step', density=True)
	plt.legend()

	plt.savefig(filename)
	plt.close("all")

# with PdfPages('conditions_distances.pdf') as pdf:
with PdfPages('conditions_distances_NNvertex_Kstr.pdf') as pdf:

	for variable in list(conditions):
		
		try:

			plt.figure(figsize=(10,8))

			plt.subplot(2,2,1)
			plt.title(variable)
			plt.hist([conditions_rapdsim["physical"][variable], conditions_notrapdsim["physical"][variable], conditions_cocktail["physical"][variable]], bins=50, density=True, histtype='step', label=['Rapidsim','Kee MC','Cocktail MC'], color=['tab:red','tab:blue','tab:grey'])
			plt.legend()
			
			plt.subplot(2,2,2)
			plt.title(f'{variable} processed')
			plt.hist([conditions_rapdsim["processed"][variable], conditions_notrapdsim["processed"][variable], conditions_cocktail["processed"][variable]], bins=50, density=True, histtype='step', range=[-1,1], color=['tab:red','tab:blue','tab:grey'])

			plt.subplot(2,2,3)
			plt.hist([conditions_rapdsim["physical"][variable], conditions_notrapdsim["physical"][variable], conditions_cocktail["physical"][variable]], bins=50, density=True, histtype='step', color=['tab:red','tab:blue','tab:grey'])
			plt.yscale('log')
			
			plt.subplot(2,2,4)
			plt.hist([conditions_rapdsim["processed"][variable], conditions_notrapdsim["processed"][variable], conditions_cocktail["processed"][variable]], bins=50, density=True, histtype='step', range=[-1,1], color=['tab:red','tab:blue','tab:grey'])
			plt.yscale('log')

			pdf.savefig(bbox_inches="tight")
			plt.close()
		except:
			pass

                
