import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import alexPlot
from matplotlib.colors import LogNorm
from scipy.stats import norm


# # rapidsim = uproot.open('new_momenta4pi.root')['DecayTree']
# rapidsim = uproot.open('new_momenta.root')['DecayTree']
# rapidsim_branches = [
#     "B_plus_P_TRUE",
#     "B_plus_PT_TRUE"
#     "B_plus_PZ_TRUE"
# ]
# rapidsim = rapidsim.arrays(library="pd")#, entry_stop=25000)
# rapidsim = rapidsim.query('B_plus_PZ_TRUE>0.')
# rapidsim['B_plus_eta_TRUE'] = 0.5 * np.log((rapidsim['B_plus_P_TRUE'] + rapidsim['B_plus_PZ_TRUE']) / (rapidsim['B_plus_P_TRUE'] - rapidsim['B_plus_PZ_TRUE']))


# plt.hist2d(np.log10(rapidsim['B_plus_PT_TRUE']), rapidsim['B_plus_eta_TRUE'], bins=50, norm=LogNorm())
# plt.xlabel('log10(B_plus_PT_TRUE)')
# plt.ylabel('B_plus_eta_TRUE')
# plt.savefig('test_Allin.png',bbox_inches="tight")
# plt.close()

# quit()


def get_rapidsim_hist_values(energy, variable):
    file = uproot.open(f'rapidsim_fonll/LHCb{energy}.root')
    if variable == 'eta':
        eta_hist = file['eta;1']
        eta_values, eta_edges = eta_hist.to_numpy()
        eta_values = eta_values[200:]
        eta_edges = eta_edges[200:]
        return eta_values, eta_edges
    elif variable == 'pT':
        eta_hist = file['pT;1']
        pt_values, pt_edges = eta_hist.to_numpy()
        return pt_values, pt_edges


rapidsim = uproot.open('tuples/Signal_tree_BTOSLLBALL6_4pi.root')['DecayTree']
# rapidsim = uproot.open('new_momenta4pi.root')['DecayTree']
# rapidsim = uproot.open('Signal4pi_tree.root')['DecayTree']
rapidsim_branches = [
    "B_plus_P_TRUE",
    "B_plus_PT_TRUE"
    "B_plus_PZ_TRUE"
]
rapidsim = rapidsim.arrays(library="pd")#, entry_stop=25000)
rapidsim = rapidsim.query('B_plus_PZ_TRUE>0.')
rapidsim['B_plus_eta_TRUE'] = 0.5 * np.log((rapidsim['B_plus_P_TRUE'] + rapidsim['B_plus_PZ_TRUE']) / (rapidsim['B_plus_P_TRUE'] - rapidsim['B_plus_PZ_TRUE']))
# one sided
# full_mc['B_plus_eta_TRUE'] = -np.log(np.tan(np.arcsin(full_mc[f"B_plus_PT_TRUE"]/ full_mc[f"B_plus_P_TRUE"])/ 2.0))


full_mc = uproot.open('tuples/Kee_2018_GenLevelNoDecProdCut.root')['DecayTree']
full_mc_branches = [
    "B_plus_TRUEP_X",
    "B_plus_TRUEP_Y",
    "B_plus_TRUEP_Z",
    "B_plus_TRUEPT",
]
full_mc = full_mc.arrays(full_mc_branches, library="pd")#, entry_stop=25000)
full_mc['B_plus_PT_TRUE'] = full_mc.eval("B_plus_TRUEPT")
full_mc['B_plus_PX_TRUE'] = full_mc.eval("B_plus_TRUEP_X")
full_mc['B_plus_PY_TRUE'] = full_mc.eval("B_plus_TRUEP_Y")
full_mc['B_plus_PZ_TRUE'] = full_mc.eval("B_plus_TRUEP_Z")
for P in ["PT","PX", "PY", "PZ"]:
    full_mc[f'B_plus_{P}_TRUE'] *= 1E-3
full_mc['B_plus_P_TRUE'] = full_mc.eval("sqrt(B_plus_PX_TRUE**2+B_plus_PY_TRUE**2+B_plus_PZ_TRUE**2)")
full_mc['B_plus_eta_TRUE'] = 0.5 * np.log((full_mc['B_plus_P_TRUE'] + full_mc['B_plus_PZ_TRUE']) / (full_mc['B_plus_P_TRUE'] - full_mc['B_plus_PZ_TRUE']))


full_mc_2011 = uproot.open('tuples/Kee_2011_GenLevelNoDecProdCut.root')['DecayTree']
full_mc_branches = [
    "B_plus_TRUEP_X",
    "B_plus_TRUEP_Y",
    "B_plus_TRUEP_Z",
    "B_plus_TRUEPT",
]
full_mc_2011 = full_mc_2011.arrays(full_mc_branches, library="pd")#, entry_stop=25000)
full_mc_2011['B_plus_PT_TRUE'] = full_mc_2011.eval("B_plus_TRUEPT")
full_mc_2011['B_plus_PX_TRUE'] = full_mc_2011.eval("B_plus_TRUEP_X")
full_mc_2011['B_plus_PY_TRUE'] = full_mc_2011.eval("B_plus_TRUEP_Y")
full_mc_2011['B_plus_PZ_TRUE'] = full_mc_2011.eval("B_plus_TRUEP_Z")
for P in ["PT","PX", "PY", "PZ"]:
    full_mc_2011[f'B_plus_{P}_TRUE'] *= 1E-3
full_mc_2011['B_plus_P_TRUE'] = full_mc_2011.eval("sqrt(B_plus_PX_TRUE**2+B_plus_PY_TRUE**2+B_plus_PZ_TRUE**2)")
full_mc_2011['B_plus_eta_TRUE'] = 0.5 * np.log((full_mc_2011['B_plus_P_TRUE'] + full_mc_2011['B_plus_PZ_TRUE']) / (full_mc_2011['B_plus_P_TRUE'] - full_mc_2011['B_plus_PZ_TRUE']))





# full_mc_cut = full_mc.query('B_plus_eta_TRUE>0.')
# hist_data, x_edges, y_edges = np.histogram2d(np.log10(full_mc_cut['B_plus_PT_TRUE']), 
#                                              full_mc_cut['B_plus_eta_TRUE'], bins=250)
# f = uproot.recreate('momenta_hist.root')#, compression=uproot3.ZLIB(4))
# f["h"] = np.histogram2d(np.log10(full_mc_cut['B_plus_PT_TRUE']), 
#                                              full_mc_cut['B_plus_eta_TRUE'], bins=250)
# f.close()
# quit()

Nbins = 150

with PdfPages(f"B_momenta_pre.pdf") as pdf:


    # full_mc_cut = full_mc.query('B_plus_eta_TRUE>0.5 and B_plus_eta_TRUE<6.5')
    full_mc_cut = full_mc.query('B_plus_eta_TRUE>0.')

    plt.title('MC')
    hist = plt.hist2d(np.log10(full_mc_cut['B_plus_PT_TRUE']), full_mc_cut['B_plus_eta_TRUE'], bins=100, norm=LogNorm())
    plt.xlabel('log10(B_plus_PT_TRUE)')
    plt.ylabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    # rapidsim_cut = rapidsim.query('B_plus_eta_TRUE>0.5 and B_plus_eta_TRUE<6.5')
    rapidsim_cut = rapidsim.query('B_plus_eta_TRUE>0.')

    plt.title('RapidSim')
    hist = plt.hist2d(np.log10(rapidsim_cut['B_plus_PT_TRUE']), rapidsim_cut['B_plus_eta_TRUE'], bins=[hist[1],hist[2]], norm=LogNorm())
    plt.xlabel('log10(B_plus_PT_TRUE)')
    plt.ylabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()


    bin_edges = np.linspace(1,6,7+1)

    gaussian = np.empty((0,3))

    for bin_idx in range(7):
        eta_min = bin_edges[bin_idx]
        eta_max = bin_edges[bin_idx+1]

        alexPlot.plot_data([np.log10(full_mc.query(f'B_plus_eta_TRUE>{eta_min} and B_plus_eta_TRUE<{eta_max}')['B_plus_PT_TRUE']), np.log10(rapidsim['B_plus_PT_TRUE'])], density=True, also_plot_hist=True, bins=Nbins, label=["full mc", "rapidsim"], only_canvas=True, pulls=True, xmin=-1,xmax=2)
        plt.legend()

        # alexPlot.plot_data([np.log10(full_mc.query(f'B_plus_eta_TRUE>{eta_min} and B_plus_eta_TRUE<{eta_max}')['B_plus_PT_TRUE']), np.log10(rapidsim.query(f'B_plus_eta_TRUE>{eta_min} and B_plus_eta_TRUE<{eta_max}')['B_plus_PT_TRUE'])], density=True, also_plot_hist=True, bins=Nbins, label=["full mc", "rapidsim"], only_canvas=True, pulls=True, xmin=-1,xmax=2)
        # plt.legend()


        
        # xmin, xmax = plt.xlim()  # Get the limits of the current x-axis
        # x = np.linspace(xmin, xmax, 100)

        # mean = np.mean(np.log10(rapidsim['B_plus_PT_TRUE']))
        # std = np.std(np.log10(rapidsim['B_plus_PT_TRUE']))

        # p = norm.pdf(x, mean, std)
        # plt.plot(x, p, 'b',zorder=105)

        # mean = np.mean(np.log10(full_mc.query(f'B_plus_eta_TRUE>{eta_min} and B_plus_eta_TRUE<{eta_max}')['B_plus_PT_TRUE']))
        # std = np.std(np.log10(full_mc.query(f'B_plus_eta_TRUE>{eta_min} and B_plus_eta_TRUE<{eta_max}')['B_plus_PT_TRUE']))

        # p = norm.pdf(x, mean, std)
        # plt.plot(x, p, 'r',zorder=105)        

        plt.title(f'B_plus_eta_TRUE ({eta_min} to {eta_max})')
        plt.xlabel('log10(B_plus_PT_TRUE)')
        pdf.savefig(bbox_inches="tight")
        plt.close() 

        # gaussian = np.append(gaussian, [[np.mean([eta_min, eta_max]), mean, std]], axis=0)
    
    # plt.subplot(1,2,1)
    # plt.plot(gaussian[:,0], gaussian[:,1])
    # plt.axhline(y=np.mean(np.log10(rapidsim['B_plus_PT_TRUE'])))
    # plt.ylabel("mean of MC Gaussian")
    # plt.subplot(1,2,2)
    # plt.plot(gaussian[:,0], gaussian[:,2])
    # plt.axhline(y=np.std(np.log10(rapidsim['B_plus_PT_TRUE'])))
    # plt.ylabel("std of MC Gaussian")
    # pdf.savefig(bbox_inches="tight")
    # plt.close() 

    
    plt.hist2d(np.log10(full_mc['B_plus_PT_TRUE']), full_mc['B_plus_eta_TRUE'], bins=100, norm=LogNorm())
    plt.xlabel('log10(B_plus_PT_TRUE)')
    plt.ylabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()


    # alexPlot.plot_data([full_mc['B_plus_PT_TRUE'], rapidsim['B_plus_PT_TRUE']], density=True, also_plot_hist=True, bins=Nbins, label=["full mc", "rapidsim"], only_canvas=True, pulls=True)
    # plt.xlabel('B_plus_PT_TRUE')
    # pdf.savefig(bbox_inches="tight")
    # plt.close()

    # alexPlot.plot_data([full_mc['B_plus_PT_TRUE'], rapidsim['B_plus_PT_TRUE']], density=True, also_plot_hist=True, bins=Nbins, label=["full mc", "rapidsim"], only_canvas=True, pulls=True,log=True)
    # plt.xlabel('B_plus_PT_TRUE')
    # pdf.savefig(bbox_inches="tight")
    # plt.close()

    # alexPlot.plot_data([np.log10(full_mc['B_plus_PT_TRUE']), np.log10(rapidsim['B_plus_PT_TRUE'])], density=True, also_plot_hist=True, bins=Nbins, label=["full mc", "rapidsim"], only_canvas=True, pulls=True,log=True)
    # plt.xlabel('log10(B_plus_PT_TRUE)')
    # pdf.savefig(bbox_inches="tight")
    # plt.close()

    pT_values, pT_edges = get_rapidsim_hist_values('13', variable='pT')

    alexPlot.plot_data(rapidsim['B_plus_PT_TRUE'], density=True, also_plot_hist=True, bins=pT_edges, only_canvas=True)
    for energy in ['13','14','7','8']:
        pT_values, pT_edges = get_rapidsim_hist_values(energy, variable='pT')
        plt.step(pT_edges[:-1], pT_values/np.sum(pT_values)/(pT_edges[1]-pT_edges[0]), where='mid',zorder=105,label=f'energy: {energy}')
    plt.legend()
    plt.xlabel('B_plus_PT_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    alexPlot.plot_data(rapidsim['B_plus_PT_TRUE'], density=True, also_plot_hist=True, bins=pT_edges, only_canvas=True, log=True)
    for energy in ['13','14','7','8']:
        pT_values, pT_edges = get_rapidsim_hist_values(energy, variable='pT')
        plt.step(pT_edges[:-1], pT_values/np.sum(pT_values)/(pT_edges[1]-pT_edges[0]), where='mid',zorder=105,label=f'energy: {energy}')
    plt.legend()
    plt.xlabel('B_plus_PT_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    alexPlot.plot_data([full_mc['B_plus_PT_TRUE'], rapidsim['B_plus_PT_TRUE'], full_mc_2011['B_plus_PT_TRUE']], density=True, also_plot_hist=True, bins=pT_edges, label=["full mc", "rapidsim", "full mc 2011"], only_canvas=True, pulls=True)
    plt.legend()
    plt.xlabel('B_plus_PT_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    alexPlot.plot_data([full_mc['B_plus_PT_TRUE'], rapidsim['B_plus_PT_TRUE'], full_mc_2011['B_plus_PT_TRUE']], density=True, also_plot_hist=True, bins=pT_edges, label=["full mc", "rapidsim", "full mc 2011"], only_canvas=True, pulls=True, log=True)
    plt.legend()
    plt.xlabel('B_plus_PT_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()


    alexPlot.plot_data([np.log10(full_mc['B_plus_PT_TRUE']), np.log10(rapidsim['B_plus_PT_TRUE']), np.log10(full_mc_2011['B_plus_PT_TRUE'])], density=True, also_plot_hist=True, bins=Nbins, label=["full mc", "rapidsim", "full mc 2011"], only_canvas=True, pulls=True)
    plt.legend()
    plt.xlabel('log10(B_plus_PT_TRUE)')
    pdf.savefig(bbox_inches="tight")
    plt.close()




    eta_values, eta_edges = get_rapidsim_hist_values('13', variable='eta')

    alexPlot.plot_data(rapidsim['B_plus_eta_TRUE'], density=True, also_plot_hist=True, bins=eta_edges, only_canvas=True)
    for energy in ['13','14','7','8']:
        eta_values, eta_edges = get_rapidsim_hist_values(energy, variable='eta')
        plt.step(eta_edges[:-1], eta_values/np.sum(eta_values)/(eta_edges[1]-eta_edges[0]), where='mid',zorder=105,label=f'energy: {energy}')
    plt.legend()
    plt.xlabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()

    alexPlot.plot_data([full_mc['B_plus_eta_TRUE'], rapidsim['B_plus_eta_TRUE'], full_mc_2011['B_plus_eta_TRUE']], density=True, also_plot_hist=True, bins=eta_edges, label=["full mc", "rapidsim", "full mc 2011"], only_canvas=True, pulls=True)
    plt.xlim(np.amin(eta_edges), np.amax(eta_edges))
    plt.legend()
    plt.xlabel('B_plus_eta_TRUE')
    pdf.savefig(bbox_inches="tight")
    plt.close()


