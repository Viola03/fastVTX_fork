from fast_vertex_quality.tools.config import rd, read_definition

import fast_vertex_quality.tools.data_loader as data_loader
import pickle

rd.daughter_particles = ["K_Kst", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'

transformers = pickle.load(open("networks/vertex_job_WGAN_transfomers.pkl", "rb"))

loader = data_loader.load_data(
    [
        "datasets/B2KEE_three_body_cut_more_vars.root",
    ],
    transformers=transformers,
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'},
    # turn_off_processing = True
)

loader_gen = data_loader.load_data(
    [
        # "saved_output.root",
        "saved_output_WGAN.root",
    ],
    transformers=transformers,
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'},
    # turn_off_processing = True
)

# loader.print_branches()

loader.cut("(abs(K_Kst_TRUEID)==321 & abs(e_plus_TRUEID)==11 & abs(e_minus_TRUEID)==11)")

cuts = {}
cuts['B_plus_FDCHI2_OWNPV'] = ">100."
cuts['B_plus_DIRA_OWNPV'] = ">0.9995"
cuts['B_plus_IPCHI2_OWNPV'] = "<25"
cuts['(B_plus_ENDVERTEX_CHI2/B_plus_ENDVERTEX_NDOF)'] = "<9"
# cuts['J_psi_1S_PT'] = ">0"
cuts['J_psi_1S_FDCHI2_OWNPV'] = ">16"
cuts['J_psi_1S_IPCHI2_OWNPV'] = ">0"
for lepton in ['e_minus', 'e_plus']:
    cuts[f'{lepton}_IPCHI2_OWNPV'] = ">9"
    # cuts[f'{lepton}_PT'] = ">300"
for hadron in ['K_Kst']:
    cuts[f'{hadron}_IPCHI2_OWNPV'] = ">9"
    # cuts[f'{hadron}_PT'] = ">400"
# cuts['m_12'] = "<5500"
# cuts['B_plus_M_Kee_reco'] = ">(5279.34-1500)"
# cuts['B_plus_M_Kee_reco'] = "<(5279.34+1500)"





from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

with PdfPages(f"cuts_eff.pdf") as pdf:

    for cut in list(cuts.keys()):

        if cut == '(B_plus_ENDVERTEX_CHI2/B_plus_ENDVERTEX_NDOF)':
            continue
        else:    
            true_vec = loader.get_branches(cut, processed=True)
            gen_vec = loader_gen.get_branches(cut, processed=True)
        
        cut_value = cuts[cut].replace('>','').replace('<','')

        true_vec = np.asarray(true_vec).flatten()
        gen_vec = np.asarray(gen_vec).flatten()

        true_vec = true_vec[np.isfinite(true_vec)]
        gen_vec = gen_vec[np.isfinite(gen_vec)]

        eff_true, effErr_true = loader.getEff(f'{cut}{cuts[cut]}')
        eff_gen, effErr_gen = loader_gen.getEff(f'{cut}{cuts[cut]}')

        plt.hist([true_vec,gen_vec], bins=75, density=True, histtype='step', label=[f'{eff_true:.3f}+-{effErr_true:.3f}', f'{eff_gen:.3f}+-{effErr_gen:.3f}'])
        plt.legend()
        plt.xlabel(cut)
        plt.xlim(-1,1)
        plt.axvline(x=loader.convert_value_to_processed(cut, cut_value),c='k')
        pdf.savefig(bbox_inches="tight")
        plt.close()


effs_true = np.empty((0,2))
effs_gen = np.empty((0,2))

eff_true, effErr_true = loader.getEff(cuts)
eff_gen, effErr_gen = loader_gen.getEff(cuts)

effs_true = np.append(effs_true, [[eff_true, effErr_true]], axis=0)
effs_gen = np.append(effs_gen, [[eff_gen, effErr_gen]], axis=0)

for cut in list(cuts.keys()):
    print('\n')
    eff_true, effErr_true = loader.getEff(f'{cut}{cuts[cut]}')
    eff_gen, effErr_gen = loader_gen.getEff(f'{cut}{cuts[cut]}')
    
    effs_true = np.append(effs_true, [[eff_true, effErr_true]], axis=0)
    effs_gen = np.append(effs_gen, [[eff_gen, effErr_gen]], axis=0)


plt.errorbar(np.arange(np.shape(effs_true)[0]), effs_true[:,0], yerr=effs_true[:,1])
plt.errorbar(np.arange(np.shape(effs_true)[0]), effs_gen[:,0], yerr=effs_gen[:,1])
plt.savefig('comp')