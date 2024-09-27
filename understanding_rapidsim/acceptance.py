import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import alexPlot
from matplotlib.colors import LogNorm
from scipy.stats import norm

masses = {}
masses['e_plus'] = 0.51099895000 * 1e-3
masses['e_minus'] = 0.51099895000 * 1e-3
masses['K_plus'] = 493.677 * 1e-3

def compute_P(df, particle):
    return df.eval(f"sqrt({particle}_PX_TRUE**2+{particle}_PY_TRUE**2+{particle}_PZ_TRUE**2)")

def compute_PT(df, particle):
    return df.eval(f"sqrt({particle}_PX_TRUE**2+{particle}_PY_TRUE**2)")

def compute_eta(df, particle):
    return 0.5 * np.log((df[f'{particle}_P_TRUE'] + df[f'{particle}_PZ_TRUE']) / (df[f'{particle}_P_TRUE'] - df[f'{particle}_PZ_TRUE']))

def compute_q2(df):

    i = 'e_plus'
    j = 'e_minus'

    df[f"{i}_mass"] = 0.51099895000 * 1e-3
    df[f"{j}_mass"] = 0.51099895000 * 1e-3

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

rapidsim = uproot.open('Signal_tree_PHOTOS.root')['DecayTree']
rapidsim_branches = [
    "B_plus_P_TRUE",
    "B_plus_PT_TRUE",
    "B_plus_PZ_TRUE",
    "K_plus_P_TRUE",
    "K_plus_PT_TRUE",
    "K_plus_PZ_TRUE",
    "e_plus_P_TRUE",
    "e_plus_PT_TRUE",
    "e_plus_PZ_TRUE",
    "e_minus_P_TRUE",
    "e_minus_PT_TRUE",
    "e_minus_PZ_TRUE",
]
rapidsim = rapidsim.arrays(library="pd")#, entry_stop=25000)
rapidsim = rapidsim.query('B_plus_PZ_TRUE>0.')
rapidsim['B_plus_eta_TRUE'] = compute_eta(rapidsim, 'B_plus')
rapidsim['K_plus_eta_TRUE'] = compute_eta(rapidsim, 'K_plus')
rapidsim['e_plus_eta_TRUE'] = compute_eta(rapidsim, 'e_plus')
rapidsim['e_minus_eta_TRUE'] = compute_eta(rapidsim, 'e_minus')
rapidsim['q2_TRUE'] = compute_q2(rapidsim)


rapidsim_4pi = uproot.open('Signal4pi_tree_PHOTOS.root')['DecayTree']
rapidsim_4pi = rapidsim_4pi.arrays(library="pd")#, entry_stop=25000)
rapidsim_4pi = rapidsim_4pi.query('B_plus_PZ_TRUE>0.')
rapidsim_4pi['B_plus_eta_TRUE'] = compute_eta(rapidsim_4pi, 'B_plus')
rapidsim_4pi['K_plus_eta_TRUE'] = compute_eta(rapidsim_4pi, 'K_plus')
rapidsim_4pi['e_plus_eta_TRUE'] = compute_eta(rapidsim_4pi, 'e_plus')
rapidsim_4pi['e_minus_eta_TRUE'] = compute_eta(rapidsim_4pi, 'e_minus')
rapidsim_4pi['q2_TRUE'] = compute_q2(rapidsim_4pi)


rapidsim_newacceptance = uproot.open('Signal_tree_PHOTOS_updatedAcceptance.root')['DecayTree']
rapidsim_newacceptance = rapidsim_newacceptance.arrays(library="pd")#, entry_stop=25000)
rapidsim_newacceptance = rapidsim_newacceptance.query('B_plus_PZ_TRUE>0.')
rapidsim_newacceptance['B_plus_eta_TRUE'] = compute_eta(rapidsim_newacceptance, 'B_plus')
rapidsim_newacceptance['K_plus_eta_TRUE'] = compute_eta(rapidsim_newacceptance, 'K_plus')
rapidsim_newacceptance['e_plus_eta_TRUE'] = compute_eta(rapidsim_newacceptance, 'e_plus')
rapidsim_newacceptance['e_minus_eta_TRUE'] = compute_eta(rapidsim_newacceptance, 'e_minus')
rapidsim_newacceptance['q2_TRUE'] = compute_q2(rapidsim_newacceptance)


def full_mc_convert_branches(df, particle):
    particle_out = particle
    if particle == 'K_plus':
        particle = 'K_Kst'
    df[f'{particle_out}_PX_TRUE'] = df.eval(f"{particle}_TRUEP_X")
    df[f'{particle_out}_PY_TRUE'] = df.eval(f"{particle}_TRUEP_Y")
    df[f'{particle_out}_PZ_TRUE'] = df.eval(f"{particle}_TRUEP_Z")
    for P in ["PX", "PY", "PZ"]:
        df[f'{particle_out}_{P}_TRUE'] *= 1E-3
    return df

full_mc = uproot.open('tuples/Kee_2018_GenLevelNoDecProdCut.root')['DecayTree']
full_mc_branches = [
    "B_plus_TRUEP_X",
    "B_plus_TRUEP_Y",
    "B_plus_TRUEP_Z",
    "K_Kst_TRUEP_X",
    "K_Kst_TRUEP_Y",
    "K_Kst_TRUEP_Z",
    "e_plus_TRUEP_X",
    "e_plus_TRUEP_Y",
    "e_plus_TRUEP_Z",
    "e_minus_TRUEP_X",
    "e_minus_TRUEP_Y",
    "e_minus_TRUEP_Z",
]
full_mc = full_mc.arrays(full_mc_branches, library="pd")#, entry_stop=25000)
for particle in ['B_plus', 'K_plus', 'e_plus', 'e_minus']:
    full_mc = full_mc_convert_branches(full_mc, particle)
    full_mc[f'{particle}_P_TRUE'] = compute_P(full_mc, particle)
    full_mc[f'{particle}_PT_TRUE'] = compute_PT(full_mc, particle)
    full_mc[f'{particle}_eta_TRUE'] = compute_eta(full_mc, particle)
full_mc['q2_TRUE'] = compute_q2(full_mc)







def full_mc_decproducut_convert_branches(df, particle):
    particle_out = particle
    if particle == 'K_plus':
        particle = 'K_Kst'
    df[f'{particle_out}_PX_TRUE'] = df.eval(f"{particle}_PX")
    df[f'{particle_out}_PY_TRUE'] = df.eval(f"{particle}_PY")
    df[f'{particle_out}_PZ_TRUE'] = df.eval(f"{particle}_PZ")
    for P in ["PX", "PY", "PZ"]:
        df[f'{particle_out}_{P}_TRUE'] *= 1E-3
    return df

full_mc_decproducut = uproot.open('tuples/Kee_2018_GenLevel.root')['DecayTree']
full_mc_branches = [
    "B_plus_PX",
    "B_plus_PY",
    "B_plus_PZ",
    "K_Kst_PX",
    "K_Kst_PY",
    "K_Kst_PZ",
    "e_plus_PX",
    "e_plus_PY",
    "e_plus_PZ",
    "e_minus_PX",
    "e_minus_PY",
    "e_minus_PZ",
]
full_mc_decproducut = full_mc_decproducut.arrays(full_mc_branches, library="pd")#, entry_stop=25000)
for particle in ['B_plus', 'K_plus', 'e_plus', 'e_minus']:
    full_mc_decproducut = full_mc_decproducut_convert_branches(full_mc_decproducut, particle)
    full_mc_decproducut[f'{particle}_P_TRUE'] = compute_P(full_mc_decproducut, particle)
    full_mc_decproducut[f'{particle}_PT_TRUE'] = compute_PT(full_mc_decproducut, particle)
    full_mc_decproducut[f'{particle}_eta_TRUE'] = compute_eta(full_mc_decproducut, particle)
full_mc_decproducut['q2_TRUE'] = compute_q2(full_mc_decproducut)





# ########################
# # Optimisation attempt - Failed
# ########################
# full_mc_decproducut_cut = full_mc_decproducut.query('B_plus_eta_TRUE>0.')
# particle = 'K_plus'
# target_values = full_mc_decproducut_cut[f'{particle}_eta_TRUE']
# # target_hist = np.histogram(target_values, bins=10, range=[1,6], density=True)
# target_hist = np.histogram(target_values, bins=50, range=[1.8,5], density=True)

# rapidsim_4pi_cut = rapidsim_4pi.query('B_plus_eta_TRUE>0.')

# def figure_of_merit(x):

#     param1, param2, param3 = x

#     df_i = rapidsim_4pi_cut.copy()
#     df_i = df_i.query(f'abs({particle}_PX_TRUE/{particle}_PZ_TRUE)<{param1}')
#     df_i = df_i.query(f'abs({particle}_PY_TRUE/{particle}_PZ_TRUE)<{param2}')
#     df_i = df_i.query(f'sqrt(({particle}_PX_TRUE/{particle}_PZ_TRUE)**2 + ({particle}_PY_TRUE/{particle}_PZ_TRUE)**2)>{param3}')

#     print(df_i.shape)
    
#     current_values = df_i[f'{particle}_eta_TRUE']

#     current_hist = np.histogram(current_values, bins=target_hist[1], density=True)

#     chi2 = np.sum((current_hist[0]-target_hist[0])**2/target_hist[0])

#     print(x, chi2)
#     return chi2

# initial_guess = [0.3, 0.25, 0.01]

# from scipy.optimize import minimize

# # result = minimize(figure_of_merit, initial_guess, method='BFGS') # step size too low
# # result = minimize(figure_of_merit, initial_guess, method='Nelder-Mead') # 0.43683481 0.18531885 0.01058462
# # result = minimize(figure_of_merit, initial_guess, method='Powell') # nan
# # result = minimize(figure_of_merit, initial_guess, method='Newton-CG') # Jacobian required
# # result = minimize(figure_of_merit, initial_guess, method='CG') # step size too low
# result = minimize(figure_of_merit, initial_guess, method='Nelder-Mead') # [0.35297561 0.18193612 0.01108896]


# print("Optimized parameters:", result.x) 
# print("Optimized figure of merit:", result.fun)

# quit()












# with PdfPages(f"acceptance_opt.pdf") as pdf:
with PdfPages(f"acceptance_opt_new.pdf") as pdf:


    rapidsim_4pi_cut = rapidsim_4pi.query('B_plus_eta_TRUE>0.')
    rapidsim_cut = rapidsim.query('B_plus_eta_TRUE>0.')
    full_mc_cut = full_mc.query('B_plus_eta_TRUE>0.')
    full_mc_decproducut_cut = full_mc_decproducut.query('B_plus_eta_TRUE>0.')

    rapidsim_newacceptance_cut = rapidsim_newacceptance.query('B_plus_eta_TRUE>0.')


    # for particle in ['B_plus', 'K_plus', 'e_plus', 'e_minus']:
        
    #     plt.hist2d(full_mc_cut[f'{particle}_PT_TRUE'], full_mc_cut[f'{particle}_eta_TRUE'], range=[[0,10],[1,6]], bins=40, norm=LogNorm())
    #     plt.legend()
    #     plt.ylabel(f'{particle}_eta_TRUE')
    #     plt.xlabel(f'{particle}_PT_TRUE')
    #     pdf.savefig(bbox_inches="tight")
    #     plt.close()

    #     plt.hist2d(full_mc_decproducut_cut[f'{particle}_PT_TRUE'], full_mc_decproducut_cut[f'{particle}_eta_TRUE'], range=[[0,10],[1,6]], bins=40, norm=LogNorm())
    #     plt.legend()
    #     plt.ylabel(f'{particle}_eta_TRUE')
    #     plt.xlabel(f'{particle}_PT_TRUE')
    #     pdf.savefig(bbox_inches="tight")
    #     plt.close()


    # quit()
    
    for particle in ['K_plus', 'e_plus', 'e_minus']:
        
        hist_4pi = np.histogram(full_mc_cut[f'{particle}_eta_TRUE'], bins=250, range=[1,6])
        hist_decprodcut = np.histogram(full_mc_decproducut_cut[f'{particle}_eta_TRUE'], bins=hist_4pi[1])


        hist_decprodcut_values = np.asarray(hist_decprodcut[0], dtype=np.float64)
        hist_4pi_values = np.asarray(hist_4pi[0], dtype=np.float64)


        hist_decprodcut_errors = np.sqrt(np.asarray(hist_decprodcut[0], dtype=np.float64))
        hist_4pi_errors = np.sqrt(np.asarray(hist_4pi[0], dtype=np.float64))


        ratio = hist_4pi_values/hist_decprodcut_values
        max_ratio = np.amax(ratio[np.isfinite(ratio)])

        # converge
        hist_decprodcut_values *= max_ratio
        hist_decprodcut_errors *= max_ratio
        acceptance = hist_decprodcut_values/hist_4pi_values
        acceptance[np.logical_not(np.isfinite(acceptance))] = 0.

        acceptance_errors = acceptance * np.sqrt((hist_decprodcut_errors/hist_decprodcut_values)**2 + (hist_4pi_errors/hist_4pi_values)**2)

        plt.errorbar(hist_4pi[1][:-1]+(hist_4pi[1][1]-hist_4pi[1][0])/2., acceptance/np.sum(acceptance), yerr=acceptance_errors/np.sum(acceptance),label=particle,marker='o',fmt=' ',capsize=2,linewidth=1.75, markersize=8)

    plt.legend()
    plt.xlabel("eta")
    plt.ylabel("acceptance")
    pdf.savefig(bbox_inches="tight")
    plt.close()


    # rapidsim_alex_immitation = rapidsim_4pi.copy()
    # # for particle in ['K_plus', 'e_plus', 'e_minus']:
    # for particle in ['K_plus']:
        

    #     acceptance_bins = hist_4pi[1]

    #     print('\n',rapidsim_alex_immitation.shape)

    #     rapidsim_alex_immitation = rapidsim_alex_immitation.query(f'{particle}_eta_TRUE>{acceptance_bins[0]} and {particle}_eta_TRUE<{acceptance_bins[-1]}')
    #     print(rapidsim_alex_immitation.shape)

    #     eta_i = rapidsim_alex_immitation[f'{particle}_eta_TRUE']

    #     indexes = np.digitize(eta_i, acceptance_bins)

    #     normalised_acceptance = acceptance/np.amax(acceptance)

    #     keep_probabilities = normalised_acceptance[indexes-1]
    #     rand = np.random.uniform(0,1,np.shape(keep_probabilities))

    #     rapidsim_alex_immitation['keep_probabilities'] = keep_probabilities
    #     rapidsim_alex_immitation['rand'] = rand
    #     print(rapidsim_alex_immitation.shape)
    #     rapidsim_alex_immitation = rapidsim_alex_immitation.query("rand < keep_probabilities")
    #     print(rapidsim_alex_immitation.shape)



    ###
    # Changing RAPIDSIM CUTS 
    ###
    rapidsim_alex_immitation = rapidsim_4pi.copy()

    #the following matches RapidSim 
    #geometry : LHCb
    #acceptance : AllIn
    for particle in ['K_plus', 'e_plus', 'e_minus']:
        # params = [0.3, 0.25, 0.01]
        # params = [0.35297561, 0.18193612, 0.01108896]
        # param1 = params[0]
        # param2 = params[1]
        # param3 = params[2]
        # rapidsim_alex_immitation = rapidsim_alex_immitation.query(f'abs({particle}_PX_TRUE/{particle}_PZ_TRUE)<{param1}')
        # rapidsim_alex_immitation = rapidsim_alex_immitation.query(f'abs({particle}_PY_TRUE/{particle}_PZ_TRUE)<{param2}')
        # rapidsim_alex_immitation = rapidsim_alex_immitation.query(f'sqrt(({particle}_PX_TRUE/{particle}_PZ_TRUE)**2 + ({particle}_PY_TRUE/{particle}_PZ_TRUE)**2)>{param3}')


        rapidsim_alex_immitation = rapidsim_alex_immitation.query(f'{particle}_eta_TRUE>1.595 and {particle}_eta_TRUE<5.3')



    #     rapidsim_alex_immitation = rapidsim_alex_immitation.query(f'abs({particle}_PX_TRUE/{particle}_PZ_TRUE)<0.3')
    #     rapidsim_alex_immitation = rapidsim_alex_immitation.query(f'abs({particle}_PY_TRUE/{particle}_PZ_TRUE)<0.25')
    #     rapidsim_alex_immitation = rapidsim_alex_immitation.query(f'sqrt(({particle}_PX_TRUE/{particle}_PZ_TRUE)**2 + ({particle}_PY_TRUE/{particle}_PZ_TRUE)**2)>0.01')


    

    for particle in ['B_plus', 'K_plus', 'e_plus', 'e_minus']:

        ####################

        alexPlot.plot_data([full_mc_cut[f'{particle}_eta_TRUE'], rapidsim_4pi_cut[f'{particle}_eta_TRUE']], density=True, also_plot_hist=True, bins=100, label=["full mc", "rapidsim"], only_canvas=True, pulls=True)
        plt.legend()
        plt.title("4pi")
        plt.xlabel(f'{particle}_eta_TRUE')
        pdf.savefig(bbox_inches="tight")
        plt.close()

        alexPlot.plot_data([full_mc_decproducut_cut[f'{particle}_eta_TRUE'], rapidsim_cut[f'{particle}_eta_TRUE'], rapidsim_newacceptance_cut[f'{particle}_eta_TRUE']], density=True, also_plot_hist=True, bins=100, label=["full mc", "rapidsim","new rapidsim"], only_canvas=True, pulls=True)
        plt.legend()
        plt.title("Acceptance")
        plt.xlabel(f'{particle}_eta_TRUE')
        pdf.savefig(bbox_inches="tight")
        plt.close()

        # ####################


        # alexPlot.plot_data([full_mc_decproducut_cut[f'{particle}_eta_TRUE'], rapidsim_cut[f'{particle}_eta_TRUE'], rapidsim_alex_immitation[f'{particle}_eta_TRUE']], density=True, also_plot_hist=True, bins=100, label=["full mc", "rapidsim"], only_canvas=True, pulls=True, xmin=1.4,xmax=1.8)
        # plt.legend()
        # plt.title("Acceptance")
        # plt.xlabel(f'{particle}_eta_TRUE')
        # pdf.savefig(bbox_inches="tight")
        # plt.close()


        # ####################

        # alexPlot.plot_data([full_mc_decproducut_cut[f'{particle}_eta_TRUE'], rapidsim_cut[f'{particle}_eta_TRUE'], rapidsim_alex_immitation[f'{particle}_eta_TRUE']], density=True, also_plot_hist=True, bins=100, label=["full mc", "rapidsim"], only_canvas=True, pulls=True, xmin=5.2,xmax=5.6)
        # plt.legend()
        # plt.title("Acceptance")
        # plt.xlabel(f'{particle}_eta_TRUE')
        # pdf.savefig(bbox_inches="tight")
        # plt.close()

        # ####################

    alexPlot.plot_data([full_mc_cut[f'q2_TRUE'], rapidsim_4pi_cut[f'q2_TRUE']], density=True, also_plot_hist=True, bins=100, label=["full mc", "rapidsim"], only_canvas=True, pulls=True,xmin=0,xmax=23)
    plt.title("4pi")
    plt.legend()
    plt.xlabel(f'q2')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    alexPlot.plot_data([full_mc_decproducut_cut[f'q2_TRUE'], rapidsim_cut[f'q2_TRUE'], rapidsim_newacceptance_cut[f'q2_TRUE']], density=True, also_plot_hist=True, bins=100, label=["full mc", "rapidsim", "new rapidsim"], only_canvas=True, pulls=True,xmin=0,xmax=23)
    plt.title("Acceptance")
    plt.legend()
    plt.xlabel(f'q2')
    pdf.savefig(bbox_inches="tight")
    plt.close()


    alexPlot.plot_data([full_mc_decproducut_cut[f'q2_TRUE'], rapidsim_newacceptance_cut[f'q2_TRUE']], density=True, also_plot_hist=True, bins=100, label=["full mc", "new rapidsim"], only_canvas=True, pulls=True,xmin=0,xmax=23)
    plt.title("Acceptance")
    plt.legend()
    plt.xlabel(f'q2')
    pdf.savefig(bbox_inches="tight")
    plt.close()




    quit()





    plt.title('RapidSim 4pi')
    hist_4pi_RS = plt.hist2d(np.log10(rapidsim_4pi_cut['B_plus_PT_TRUE']), rapidsim_4pi_cut['B_plus_eta_TRUE'], bins=100, norm=LogNorm(),density=True)
    plt.xlabel('log10(B_plus_PT_TRUE)')
    plt.ylabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    
    plt.title('RapidSim')
    hist_RS = plt.hist2d(np.log10(rapidsim_cut['B_plus_PT_TRUE']), rapidsim_cut['B_plus_eta_TRUE'], bins=[hist_4pi_RS[1],hist_4pi_RS[2]], norm=LogNorm(),density=True)
    plt.xlabel('log10(B_plus_PT_TRUE)')
    plt.ylabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    
    plt.title('MC 4pi')
    hist_4pi = plt.hist2d(np.log10(full_mc_cut['B_plus_PT_TRUE']), full_mc_cut['B_plus_eta_TRUE'], bins=[hist_4pi_RS[1],hist_4pi_RS[2]], norm=LogNorm(),density=True)
    plt.xlabel('log10(B_plus_PT_TRUE)')
    plt.ylabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    plt.title('MC')
    hist = plt.hist2d(np.log10(full_mc_decproducut_cut['B_plus_PT_TRUE']), full_mc_decproducut_cut['B_plus_eta_TRUE'], bins=[hist_4pi_RS[1],hist_4pi_RS[2]], norm=LogNorm(), density=True)
    plt.xlabel('log10(B_plus_PT_TRUE)')
    plt.ylabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()


    # Calculate the ratio of the histograms, avoiding division by zero
    epsilon = 1e-10
    ratio = hist_RS[0] / (hist_4pi_RS[0] + epsilon)

    # Step 2: Plot the ratio
    plt.title('Ratio: RapidSim / RapidSim 4pi')
    # Plot using pcolormesh for better control over bin edges
    plt.pcolormesh(hist_4pi_RS[1], hist_4pi_RS[2], ratio.T, cmap='viridis', vmin=0, vmax=2)
    plt.colorbar(label='Ratio')
    plt.xlabel('log10(B_plus_PT_TRUE)')
    plt.ylabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    # Calculate the ratio of the histograms, avoiding division by zero
    epsilon = 1e-10
    ratio = hist[0] / (hist_4pi[0] + epsilon)

    # Step 2: Plot the ratio
    plt.title('Ratio: MC / MC 4pi')
    # Plot using pcolormesh for better control over bin edges
    plt.pcolormesh(hist_4pi[1], hist_4pi[2], ratio.T, cmap='viridis', vmin=0, vmax=2)
    plt.colorbar(label='Ratio')
    plt.xlabel('log10(B_plus_PT_TRUE)')
    plt.ylabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()



