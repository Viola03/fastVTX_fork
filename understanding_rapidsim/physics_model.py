import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import alexPlot
from matplotlib.colors import LogNorm
from scipy.stats import norm

def q2(df):

    i = 'e_plus'
    j = 'e_minus'

    PE = np.sqrt(
        df[f"{i}_mass"]**2 + df[f"{i}_PX_TRUE"] ** 2 + df[f"{i}_PY_TRUE"] ** 2 + df[f"{i}_PZ_TRUE"] ** 2
    ) + np.sqrt(
        df[f"{i}_mass"]**2 + df[f"{j}_PX_TRUE"] ** 2 + df[f"{j}_PY_TRUE"] ** 2 + df[f"{j}_PZ_TRUE"] ** 2
    )
    PX = df[f"{i}_PX_TRUE"] + df[f"{j}_PX_TRUE"]
    PY = df[f"{i}_PY_TRUE"] + df[f"{j}_PY_TRUE"]
    PZ = df[f"{i}_PZ_TRUE"] + df[f"{j}_PZ_TRUE"]

    q2 = np.sqrt((PE**2 - PX**2 - PY**2 - PZ**2))**2

    return q2

masses = {}
masses['e_plus'] = 0.51099895000 * 1e-3
masses['e_minus'] = 0.51099895000 * 1e-3
masses['K_plus'] = 493.677 * 1e-3

# rapidsim = uproot.open('Signal4pi_tree.root')['DecayTree']
rapidsim = uproot.open('Signal4pi_tree_PHOTOS.root')['DecayTree']
rapidsim_branches = [
    "e_plus_PX_TRUE",
    "e_plus_PY_TRUE",
    "e_plus_PZ_TRUE",
    "e_minus_PX_TRUE",
    "e_minus_PY_TRUE",
    "e_minus_PZ_TRUE",
    "K_plus_PX_TRUE",
    "K_plus_PY_TRUE",
    "K_plus_PZ_TRUE",
]
rapidsim = rapidsim.arrays(library="pd")#, entry_stop=25000)
rapidsim = rapidsim.query('B_plus_PZ_TRUE>0.')
for particle in ['e_plus', 'e_minus', 'K_plus']:
    rapidsim[f'{particle}_mass'] = masses[particle]
rapidsim['q2'] = q2(rapidsim)


full_mc = uproot.open('tuples/Kee_2018_GenLevelNoDecProdCut.root')['DecayTree']
full_mc_branches = [
    "e_plus_TRUEP_X",
    "e_plus_TRUEP_Y",
    "e_plus_TRUEP_Z",
    "e_minus_TRUEP_X",
    "e_minus_TRUEP_Y",
    "e_minus_TRUEP_Z",
    "K_Kst_TRUEP_X",
    "K_Kst_TRUEP_Y",
    "K_Kst_TRUEP_Z",
]
full_mc = full_mc.arrays(full_mc_branches, library="pd")#, entry_stop=25000)
full_mc['K_plus_TRUEP_X'] = full_mc['K_Kst_TRUEP_X']
full_mc['K_plus_TRUEP_Y'] = full_mc['K_Kst_TRUEP_Y']
full_mc['K_plus_TRUEP_Z'] = full_mc['K_Kst_TRUEP_Z']
for particle in ['e_plus', 'e_minus', 'K_plus']:    
    full_mc[f'{particle}_mass'] = masses[particle]
    for P in ["PX", "PY", "PZ"]:
        full_mc[f'{particle}_{P}_TRUE'] = full_mc[f'{particle}_TRUEP_{P[1]}']
        full_mc[f'{particle}_{P}_TRUE'] *= 1E-3
full_mc['q2'] = q2(full_mc)


alexPlot.plot_data([full_mc['q2'], rapidsim['q2']], density=True, also_plot_hist=True, bins=100, label=["full mc", "rapidsim"], only_canvas=True, pulls=True)
plt.legend()
plt.xlabel('q2')
plt.savefig('q2_physics_PHOTOS.pdf', bbox_inches="tight")
plt.close()


