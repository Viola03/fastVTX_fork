from fast_vertex_quality.tools.config import read_definition, rd

from fast_vertex_quality.models.conditional_VAE import VAE_builder
import tensorflow as tf
import numpy as np
from fast_vertex_quality.tools.training import train_step
import fast_vertex_quality.tools.plotting as plotting
import pickle
import fast_vertex_quality.tools.data_loader as data_loader
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from particle import Particle
import uproot3
import pandas as pd

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

class BDT_tester:

    def __init__(
        self,
        transformers,
        tag,
        train=True,
        BDT_vars=[
            "B_plus_ENDVERTEX_CHI2",
            "B_plus_IPCHI2_OWNPV",
            "B_plus_FDCHI2_OWNPV",
            "B_plus_DIRA_OWNPV",
            "K_Kst_IPCHI2_OWNPV",
            "K_Kst_TRACK_CHI2NDOF",
            "e_minus_IPCHI2_OWNPV",
            "e_minus_TRACK_CHI2NDOF",
            "e_plus_IPCHI2_OWNPV",
            "e_plus_TRACK_CHI2NDOF",
        ],
        signal="datasets/Kee_2018_truthed_more_vars.csv",
        background="datasets/B2Kee_2018_CommonPresel.csv",
        signal_label="Train - sig",
        background_label="Train - comb",
        gen_track_chi2=True,
    ):

        self.log_columns = [
            f"{rd.mother_particle}_FDCHI2_OWNPV",
            f"{rd.daughter_particles[0]}_IPCHI2_OWNPV",
            f"{rd.daughter_particles[1]}_IPCHI2_OWNPV",
            f"{rd.daughter_particles[2]}_IPCHI2_OWNPV",
            f"{rd.daughter_particles[0]}_PZ",
            f"{rd.daughter_particles[1]}_PZ",
            f"{rd.daughter_particles[2]}_PZ",
            f"{rd.mother_particle}_ENDVERTEX_CHI2",
            f"{rd.mother_particle}_IPCHI2_OWNPV",
            f"IP_{rd.mother_particle}",
            f"{rd.intermediate_particle}_FDCHI2_OWNPV",
            f"{rd.intermediate_particle}_FLIGHT",

            f"{rd.mother_particle}_P",
            f"{rd.mother_particle}_PT",
            f"IP_{rd.daughter_particles[0]}",
            f"IP_{rd.daughter_particles[1]}",
            f"IP_{rd.daughter_particles[2]}",
            f"FD_{rd.mother_particle}",
            f"IP_{rd.daughter_particles[0]}_true_vertex",
            f"IP_{rd.daughter_particles[1]}_true_vertex",
            f"IP_{rd.daughter_particles[2]}_true_vertex",
            f"FD_{rd.mother_particle}_true_vertex",

        ]

        self.one_minus_log_columns = [f"{rd.mother_particle}_DIRA_OWNPV", f"DIRA_{rd.mother_particle}", f"DIRA_{rd.mother_particle}_true_vertex"]


        self.signal_label = signal_label
        self.background_label = background_label

        self.BDT_vars = BDT_vars

        if gen_track_chi2:
            self.BDT_vars_gen = [
                x.replace("CHI2NDOF", "CHI2NDOF_gen") for x in self.BDT_vars
            ]
        else:
            self.BDT_vars_gen = self.BDT_vars

        self.transformers = transformers

        if train:
            event_loader_MC = data_loader.load_data(
                [
                    signal,
                ],
                transformers=self.transformers,
                convert_to_RK_branch_names=True,
                conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
            )
            stripping_eff_signal = self.get_stripping_eff(event_loader_MC)
            event_loader_MC.select_randomly(Nevents=50000)

            events_MC = event_loader_MC.get_branches(
                self.BDT_vars + ["kFold"], processed=False
            )

            event_loader_data = data_loader.load_data(
                [
                    background,
                ],
                transformers=self.transformers,
            )
            event_loader_data.select_randomly(Nevents=50000)

            events_data = event_loader_data.get_branches(
                self.BDT_vars + ["kFold"], processed=False
            )

            self.BDTs = {}

            for kFold in range(10):

                print(f"Training kFold {kFold}...")

                clf = GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=4
                )

                events_data_i = events_data.query(f"kFold!={kFold}")
                events_MC_i = events_MC.query(f"kFold!={kFold}")

                events_data_i = events_data_i.drop("kFold", axis=1)
                events_MC_i = events_MC_i.drop("kFold", axis=1)

                real_training_data = np.squeeze(np.asarray(events_MC_i[self.BDT_vars]))

                fake_training_data = np.squeeze(
                    np.asarray(events_data_i[self.BDT_vars])
                )

                size = 25000
                real_training_data = real_training_data[:size]
                fake_training_data = fake_training_data[:size]

                real_training_labels = np.ones(size)

                fake_training_labels = np.zeros(size)

                total_training_data = np.concatenate(
                    (real_training_data, fake_training_data)
                )

                total_training_labels = np.concatenate(
                    (real_training_labels, fake_training_labels)
                )

                clf.fit(total_training_data, total_training_labels)

                self.BDTs[kFold] = {}
                self.BDTs[kFold]["BDT"] = clf
                self.BDTs[kFold]["signal_sample"] = real_training_data
                self.BDTs[kFold]["bkg_sample"] = fake_training_data
                self.BDTs[kFold]["signal_stripping_eff"] = stripping_eff_signal

                break

            for kFold in range(10):

                clf = self.BDTs[kFold]["BDT"]

                events_data_i = events_data.query(f"kFold=={kFold}")
                events_MC_i = events_MC.query(f"kFold=={kFold}")

                events_data_i = events_data_i.drop("kFold", axis=1)
                events_MC_i = events_MC_i.drop("kFold", axis=1)

                real_testing_data = np.squeeze(np.asarray(events_MC_i[self.BDT_vars]))

                fake_testing_data = np.squeeze(np.asarray(events_data_i[self.BDT_vars]))

                size = 25000
                real_testing_data = real_testing_data[:size]
                fake_testing_data = fake_testing_data[:size]

                self.BDTs[kFold]["values_sig"] = clf.predict_proba(real_testing_data)[
                    :, 1
                ]

                self.BDTs[kFold]["values_bkg"] = clf.predict_proba(fake_testing_data)[
                    :, 1
                ]

                break

            pickle.dump(
                self.BDTs,
                open(
                    f"{tag}.pkl",
                    "wb",
                ),
            )

        else:

            self.BDTs = pickle.load(open(f"{tag}.pkl", "rb"))

    def get_BDT(self):

        return self.BDTs[0]["BDT"]

    def get_sample(
        self,
        sample_loc,
        vertex_quality_trainer_obj,
        generate,
        cut=None,
        convert_branches=False,
        N=10000,
    ):

        if convert_branches:
            event_loader = data_loader.load_data(
                [
                    sample_loc,
                ],
                transformers=self.transformers,
                convert_to_RK_branch_names=True,
                conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
            )
        else:
            event_loader = data_loader.load_data(
                [
                    sample_loc,
                ],
                transformers=self.transformers,
            )

        if cut is not None:
            print(event_loader.shape())
            event_loader.cut(cut)
            print(event_loader.shape())
        
        try:
            event_loader.select_randomly(Nevents=N)
        except:
            pass
        

        if generate:

            event_loader = vertex_quality_trainer_obj.predict_from_data_loader(
                event_loader
            )

            query = event_loader.get_branches(self.BDT_vars_gen, processed=False)

            query = np.squeeze(np.asarray(query[self.BDT_vars_gen]))

        else:
            query = event_loader.get_branches(self.BDT_vars, processed=False)

            query = np.squeeze(np.asarray(query[self.BDT_vars]))

        return query

    def get_jpsiX_samples(
        self,
        sample_loc,
        vertex_quality_trainer_obj,
        generate,
        N=10000,
    ):

        event_loader = data_loader.load_data(
            [
                sample_loc,
            ],
            transformers=self.transformers,
        )

        lost_mass_cuts = [125, 250, 750, 1250]

        event_loader.select_randomly(Nevents=N * (len(lost_mass_cuts) + 1))
        if generate:

            event_loader = vertex_quality_trainer_obj.predict_from_data_loader(
                event_loader
            )

            query = event_loader.get_branches(
                self.BDT_vars_gen + ["lost_mass"], processed=False
            )

            queries = []
            for idx in range(len(lost_mass_cuts) + 1):
                if idx == 0:
                    query_i = query.query(f"lost_mass<{lost_mass_cuts[idx]}")
                elif idx < len(lost_mass_cuts):
                    query_i = query.query(
                        f"lost_mass<{lost_mass_cuts[idx]} and lost_mass>{lost_mass_cuts[idx-1]}"
                    )
                else:
                    query_i = query.query(f"lost_mass>{lost_mass_cuts[idx-1]}")

                query_i = np.squeeze(np.asarray(query_i[self.BDT_vars_gen]))
                queries.append(query_i)

        else:
            query = event_loader.get_branches(
                self.BDT_vars + ["lost_mass"], processed=False
            )

            queries = []
            for idx in range(len(lost_mass_cuts) + 1):
                if idx == 0:
                    query_i = query.query(f"lost_mass<{lost_mass_cuts[idx]}")
                elif idx < len(lost_mass_cuts):
                    query_i = query.query(
                        f"lost_mass<{lost_mass_cuts[idx]} and lost_mass>{lost_mass_cuts[idx-1]}"
                    )
                else:
                    query_i = query.query(f"lost_mass>{lost_mass_cuts[idx-1]}")

                query_i = np.squeeze(np.asarray(query_i[self.BDT_vars]))
                queries.append(query_i)

        return queries


    def get_sample_Kee(
        self,
        sample_loc,
        vertex_quality_trainer_obj,
        generate,
        cut=None,
        convert_branches=False,
        N=10000,
        rapidsim=False,
        return_data_loader=False
    ):

        conditions = [
    "B_plus_P",
    "B_plus_PT",
    "angle_K_Kst",
    "angle_e_plus",
    "angle_e_minus",
    "K_Kst_eta",
    "e_plus_eta",
    "e_minus_eta",
    "IP_B_plus",
    "IP_K_Kst",
    "IP_e_plus",
    "IP_e_minus",
    "FD_B_plus",
    "DIRA_B_plus",
    "missing_B_plus_P",
    "missing_B_plus_PT",
    "missing_J_psi_1S_P",
    "missing_J_psi_1S_PT",
    "m_01",
    "m_02",
    "m_12",

    "B_plus_FLIGHT",
    "K_Kst_FLIGHT",
    "e_plus_FLIGHT",
    "e_minus_FLIGHT",
        ]
        if rapidsim:
            event_loader = data_loader.load_data(
                [
                    sample_loc,
                ],
                transformers=self.transformers,
                convert_to_RK_branch_names=True,
                conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
            )
            # event_loader.plot('conditions_rapdsim.pdf',variables=conditions, save_vars=True)
            # event_loader.plot('conditions_rapdsim.pdf', save_vars=True)
            # A = event_loader.get_branches(['pass_stripping'])
            # print(np.asarray(A), np.shape(np.where(A!=1)))
            # quit()
            # event_loader.cut('K_Kst_PT>400')
            # event_loader.cut('e_minus_PT>300')
            # event_loader.cut('e_plus_PT>300')

            if "Partreco" in sample_loc:
                event_loader_target = data_loader.load_data(
                    [
                        "datasets/Kstee_cut_more_vars.root",
                    ],
                    transformers=self.transformers,
                    convert_to_RK_branch_names=True,
                    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
                )
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['K_Kst_eta','e_minus_eta','e_plus_eta'])
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['B_plus_P','B_plus_PT'])
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['B_plus_P','B_plus_PT','K_Kst_eta','e_minus_eta','e_plus_eta','m_01','m_02','m_12'])
                # best
                event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['m_01','m_02','m_12'])
            if "BuD0enuKenu" in sample_loc:
                # event_loader_target = data_loader.load_data(
                #     [
                #         "datasets/dedicated_BuD0enuKenu_MC_hierachy_cut_more_vars.root",
                #     ],
                #     transformers=self.transformers,
                #     convert_to_RK_branch_names=True,
                #     conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
                # )
                # event_loader_target.cut("pass_stripping")
                event_loader.cut('m_12>3.674')
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['m_01','m_02','m_12'])
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['FD_B_plus_true_vertex','IP_e_minus_true_vertex','IP_e_plus_true_vertex','IP_K_Kst_true_vertex'])
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['B_plus_P','B_plus_PT'])
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['angle_K_Kst','angle_e_plus','angle_e_minus'])
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['K_Kst_eta','e_plus_eta','e_minus_eta'])
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['FD_B_plus_true_vertex'])
            # if "BuD0piKenu" in sample_loc:

        else:
            if convert_branches:
                event_loader = data_loader.load_data(
                    [
                        sample_loc,
                    ],
                    transformers=self.transformers,
                    convert_to_RK_branch_names=True,
                    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
                )
            else:
                event_loader = data_loader.load_data(
                    [
                        sample_loc,
                    ],
                    transformers=self.transformers,
                )
            # event_loader.plot('conditions_notrapdsim.pdf',variables=conditions, save_vars=True)
            # event_loader.plot('conditions_notrapdsim.pdf', save_vars=True)

        # Jpsi_ID = 443

        # new_condition_dict = {}
        # new_condition_dict["B_plus_TRUEID"] = 521
        # new_condition_dict["J_psi_1S_TRUEID_width"] = Particle.from_pdgid(Jpsi_ID).width
        # new_condition_dict["J_psi_1S_MC_MOTHER_ID_width"] = Particle.from_pdgid(521).width
        # new_condition_dict["J_psi_1S_MC_GD_MOTHER_ID_width"] = 0.
        # new_condition_dict["J_psi_1S_MC_GD_GD_MOTHER_ID_width"] = 0.
        # new_condition_dict["K_Kst_TRUEID"] = 321
        # new_condition_dict["K_Kst_MC_MOTHER_ID_width"] = Particle.from_pdgid(521).width
        # new_condition_dict["K_Kst_MC_GD_MOTHER_ID_width"] = 0.
        # new_condition_dict["K_Kst_MC_GD_GD_MOTHER_ID_width"] = 0.
        # new_condition_dict["e_plus_TRUEID"] = 11
        # new_condition_dict["e_plus_MC_MOTHER_ID_width"] = Particle.from_pdgid(Jpsi_ID).width
        # new_condition_dict["e_plus_MC_GD_MOTHER_ID_width"] = Particle.from_pdgid(521).width
        # new_condition_dict["e_plus_MC_GD_GD_MOTHER_ID_width"] = 0.
        # new_condition_dict["e_minus_TRUEID"] = 11
        # new_condition_dict["e_minus_MC_MOTHER_ID_width"] = Particle.from_pdgid(Jpsi_ID).width
        # new_condition_dict["e_minus_MC_GD_MOTHER_ID_width"] = Particle.from_pdgid(521).width
        # new_condition_dict["e_minus_MC_GD_GD_MOTHER_ID_width"] = 0.
        # new_condition_dict["J_psi_1S_MC_MOTHER_ID_mass"] = Particle.from_pdgid(521).width
        # new_condition_dict["J_psi_1S_MC_GD_MOTHER_ID_mass"] = 0.
        # new_condition_dict["J_psi_1S_MC_GD_GD_MOTHER_ID_mass"] = 0.
        # new_condition_dict["K_Kst_MC_MOTHER_ID_mass"] = Particle.from_pdgid(521).mass
        # new_condition_dict["K_Kst_MC_GD_MOTHER_ID_mass"] = 0.
        # new_condition_dict["K_Kst_MC_GD_GD_MOTHER_ID_mass"] = 0.
        # new_condition_dict["e_plus_MC_MOTHER_ID_mass"] = Particle.from_pdgid(Jpsi_ID).mass
        # new_condition_dict["e_plus_MC_GD_MOTHER_ID_mass"] = Particle.from_pdgid(521).mass
        # new_condition_dict["e_plus_MC_GD_GD_MOTHER_ID_mass"] = 0.
        # new_condition_dict["e_minus_MC_MOTHER_ID_mass"] = Particle.from_pdgid(Jpsi_ID).mass
        # new_condition_dict["e_minus_MC_GD_MOTHER_ID_mass"] = Particle.from_pdgid(521).mass
        # new_condition_dict["e_minus_MC_GD_GD_MOTHER_ID_mass"] = 0.

        # event_loader.fill_new_condition(new_condition_dict)

        if cut is not None:
            print(event_loader.shape())
            event_loader.cut(cut)
            print(event_loader.shape())
        
        try:
            event_loader.select_randomly(Nevents=N)
        except:
            pass
        

        if generate:

            event_loader = vertex_quality_trainer_obj.predict_from_data_loader(
                event_loader
            )
            event_loader.fill_stripping_bool()
            event_loader.cut("pass_stripping")

            query = event_loader.get_branches(self.BDT_vars_gen, processed=False)

            query = np.squeeze(np.asarray(query[self.BDT_vars_gen]))

        else:  
            
            event_loader.fill_stripping_bool()
            event_loader.cut("pass_stripping")
     
            query = event_loader.get_branches(self.BDT_vars, processed=False)

            query = np.squeeze(np.asarray(query[self.BDT_vars]))

        if return_data_loader:
            return query, event_loader
        else:
            return query
    

    def get_vars_of_samples_that_pass_a_cut(self,
                                vertex_quality_trainer_obj,
                                target_vars,save=False,filename=''):
        
        BuD0enuKenu_MC, BuD0enuKenu_MC_event_loader = self.get_sample_Kee(
            "datasets/dedicated_BuD0enuKenu_MC_hierachy_cut_more_vars.root",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
            rapidsim=False,
            convert_branches=True,
            return_data_loader=True,
        )  
        # event_loader

        query = BuD0enuKenu_MC_event_loader.get_branches(self.BDT_vars, processed=False)
        target_vars_set = BuD0enuKenu_MC_event_loader.get_branches(target_vars, processed=False)

        kFold = 0
        clf = self.BDTs[kFold]["BDT"]

        query = np.squeeze(np.asarray(query[self.BDT_vars]))
        target_vars_set = np.squeeze(np.asarray(target_vars_set[target_vars]))

        sample_values = clf.predict_proba(query)[:, 1]

        where = np.where(sample_values>0.95)
        target_vars_set = target_vars_set[where]

        data = {}
        for var_idx, var in enumerate(target_vars):
            data[var] = target_vars_set[:,var_idx]
        data = pd.DataFrame.from_dict(data)

        if save:
            if filename == '':
                print("BDT.py, get_vars_of_samples_that_pass_a_cut, Must set filename.. quitting..")
                quit()
            
            write_df_to_root(data, filename)

        return data

    def make_BDT_plot_hierarchy(
        self,
        vertex_quality_trainer_obj,
        filename,
        include_combinatorial=False,
        include_jpsiX=False,
    ):  
        
        



        signal_gen = self.get_sample_Kee(
            # "datasets/Kee_2018_truthed_more_vars.csv",
            "datasets/Kee_cut_more_vars.root",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
        )  
        
        signal_gen_rapidsim = self.get_sample_Kee(
            "/users/am13743/fast_vertexing_variables/rapidsim/Kee/Signal_tree_NNvertex_more_vars.root",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
            rapidsim=True,
        )  


        part_reco_gen = self.get_sample_Kee(
            "datasets/Kstee_cut_more_vars.root",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
            rapidsim=False,
        )  
        
        part_reco_gen_rapidsim = self.get_sample_Kee(
            "/users/am13743/fast_vertexing_variables/rapidsim/Kstree/Partreco_tree_NNvertex_more_vars.root",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
            rapidsim=True,
        )  
        
        part_reco_MC = self.get_sample_Kee(
            "datasets/Kstee_cut_more_vars.root",
            vertex_quality_trainer_obj,
            generate=False,
            N=10000,
        )  





        

        samples = [signal_gen, signal_gen_rapidsim, part_reco_gen, part_reco_gen_rapidsim, part_reco_MC]
        labels = [self.signal_label, self.background_label, "sig - gen", "sig - gen (rapidsim)", "prc - gen", "prc - gen (rapidsim)", "prc - MC"]
        colours = ["tab:blue", "tab:red", "tab:green", "tab:orange", "k", "violet", "tab:purple"]

        scores = self.query_and_plot_samples(
            samples,
            labels,
            colours=colours,
            filename=filename,
            include_combinatorial=include_combinatorial,
            only_hists=True,
        )


        BuD0enuKenu_gen = self.get_sample_Kee(
            "datasets/dedicated_BuD0enuKenu_MC_hierachy_cut_more_vars.root",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
            rapidsim=False,
            convert_branches=True,
        )  
        
        BuD0enuKenu_gen_rapidsim = self.get_sample_Kee(
            "/users/am13743/fast_vertexing_variables/rapidsim/BuD0enuKenu/BuD0enuKenu_tree_NNvertex_more_vars.root",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
            rapidsim=True,
        )  
        
        BuD0enuKenu_MC = self.get_sample_Kee(
            "datasets/dedicated_BuD0enuKenu_MC_hierachy_cut_more_vars.root",
            vertex_quality_trainer_obj,
            generate=False,
            N=10000,
            convert_branches=True,
        )  


        samples = [signal_gen, signal_gen_rapidsim, BuD0enuKenu_gen, BuD0enuKenu_gen_rapidsim, BuD0enuKenu_MC]
        labels = [self.signal_label, self.background_label, "sig - gen", "sig - gen (rapidsim)", "BuD0enuKenu - gen", "BuD0enuKenu - gen (rapidsim)", "BuD0enuKenu - MC"]
        colours = ["tab:blue", "tab:red", "tab:green", "tab:orange", "k", "violet", "tab:purple"]

        scores = self.query_and_plot_samples(
            samples,
            labels,
            colours=colours,
            filename=filename.replace('.pdf','_BuD0enuKenu.pdf'),
            include_combinatorial=include_combinatorial,
            only_hists=True,
        )




        BuD0piKenu_gen = self.get_sample_Kee(
            "datasets/dedicated_BuD0piKenu_MC_hierachy_cut_more_vars.root",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
            rapidsim=False,
            convert_branches=True,
        )  
        
        BuD0piKenu_gen_rapidsim = self.get_sample_Kee(
            "/users/am13743/fast_vertexing_variables/rapidsim/BuD0piKenu/BuD0piKenu_tree_NNvertex_more_vars.root",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
            rapidsim=True,
        )  
        
        BuD0piKenu_MC = self.get_sample_Kee(
            "datasets/dedicated_BuD0piKenu_MC_hierachy_cut_more_vars.root",
            vertex_quality_trainer_obj,
            generate=False,
            N=10000,
            convert_branches=True,
        )  

        
        samples = [signal_gen, signal_gen_rapidsim, BuD0piKenu_gen, BuD0piKenu_gen_rapidsim, BuD0piKenu_MC]
        labels = [self.signal_label, self.background_label, "sig - gen", "sig - gen (rapidsim)", "BuD0piKenu - gen", "BuD0piKenu - gen (rapidsim)", "BuD0piKenu - MC"]
        colours = ["tab:blue", "tab:red", "tab:green", "tab:orange", "k", "violet", "tab:purple"]

        scores = self.query_and_plot_samples(
            samples,
            labels,
            colours=colours,
            filename=filename.replace('.pdf','_BuD0piKenu.pdf'),
            include_combinatorial=include_combinatorial,
            only_hists=True,
        )

        return scores
    
    
    
    def make_BDT_plot(
        self,
        vertex_quality_trainer_obj,
        filename,
        include_combinatorial=False,
        include_jpsiX=False,
    ):
        signal_gen = self.get_sample(
            "datasets/Kee_2018_truthed_more_vars.csv",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
        )

        prc_MC = self.get_sample(
            "datasets/Kstee_2018_truthed_more_vars.csv",
            None,
            generate=False,
            N=10000,
        )
        
        prc_gen = self.get_sample(
            "datasets/Kstee_2018_truthed_more_vars.csv",
            vertex_quality_trainer_obj,
            generate=True,
            N=10000,
        )

        print(np.shape(signal_gen))
        print(np.shape(prc_MC))
        print(np.shape(prc_gen))
        print(np.shape(self.BDTs[0]["signal_sample"]))
        print(np.shape(self.BDTs[0]["bkg_sample"]))

        with PdfPages(f"BDT_distributions.pdf") as pdf:
            
            for i in range(np.shape(signal_gen)[1]):
                
                data = [np.asarray(self.BDTs[0]["signal_sample"])[:,i], np.asarray(self.BDTs[0]["bkg_sample"])[:,i], np.asarray(signal_gen)[:,i], np.asarray(prc_MC)[:,i], np.asarray(prc_gen)[:,i]]

                if self.BDT_vars[i] in self.log_columns:
                    for j in range(len(data)):
                        data[j] = np.log10(data[j])
                # elif self.BDT_vars[i] in self.one_minus_log_columns:
                #     data[j][np.where(data[j]==1)] = 1.-1E-15
                #     data[j][np.where(np.isnan(data[j]))] = 1.-1E-15
                #     data[j][np.where(np.isinf(data[j]))] = 1.-1E-15
                #     for j in range(len(data)):
                #         data[j] = np.log10(1.-data[j])

                plt.hist(data, density=True, histtype='step', color=["tab:blue", "tab:red", "tab:green", "tab:purple", "k"], bins=75,label=["sig - MC","bkg","sig - gen", "prc - MC", "prc - gen"])
                if self.BDT_vars[i] in self.log_columns:
                    plt.xlabel(f'log({self.BDT_vars[i]})')
                # elif self.BDT_vars[i] in self.one_minus_log_columns:
                #     plt.xlabel(f'log(1-{self.BDT_vars[i]})')
                else:
                    plt.xlabel(self.BDT_vars[i])
                plt.legend()
                pdf.savefig(bbox_inches="tight")
                plt.close()


        samples = [signal_gen, prc_MC, prc_gen]
        labels = ["sig - gen", "prc - MC", "prc - gen"]
        colours = ["tab:blue", "tab:red", "tab:green", "tab:purple", "k"]

        if include_combinatorial:

            combi_gen = self.get_sample(
                "datasets/B2Kee_2018_CommonPresel_more_vars.csv",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
            )

            colours.append("tab:orange")
            samples.append(combi_gen)
            labels.append("combi - gen")

        if include_jpsiX:

            jpsix_MC = self.get_jpsiX_samples(
                "datasets/JPSIX_2018_truthed_more_vars.csv",
                None,
                generate=False,
                N=50000,
            )
            jpsix_gen = self.get_jpsiX_samples(
                "datasets/JPSIX_2018_truthed_more_vars.csv",
                vertex_quality_trainer_obj,
                generate=True,
                N=50000,
            )

            scores = self.query_and_plot_samples_jpsiX(
                jpsix_MC,
                jpsix_gen,
                filename=filename.replace(".pdf", "") + "_jpsiX.pdf",
                include_combinatorial=False,
            )

            # colours.append("tab:orange")
            # samples.append(combi_gen)
            # labels.append("combi - gen")

        scores = self.query_and_plot_samples(
            samples,
            labels,
            colours=colours,
            filename=filename,
            include_combinatorial=include_combinatorial,
        )

        return scores

    
    def make_BDT_plot_intermediates(
        self,
        vertex_quality_trainer_obj,
        filename,
        include_combinatorial=False,
        include_jpsiX=False,
        intermediate_IDs=None
    ):

        # if intermediate_IDs == None:
        #     intermediate_IDs = [421]

        # for intermediate_ID in intermediate_IDs:

        #     cocktail_gen = self.get_sample(
        #         "datasets/cocktail_three_body_cut_more_vars.root",
        #         vertex_quality_trainer_obj,
        #         generate=True,
        #         N=10000,
        #         convert_branches=True,
        #         cut=f'abs(B_plus_TRUEID)==521 & abs(J_psi_1S_TRUEID)=={intermediate_ID} & pass_stripping == 1',
        #     )

        #     cocktail = self.get_sample(
        #         "datasets/cocktail_three_body_cut_more_vars.root",
        #         None,
        #         generate=False,
        #         N=10000,
        #         convert_branches=True,
        #         cut=f'abs(B_plus_TRUEID)==521 & abs(J_psi_1S_TRUEID)=={intermediate_ID} & pass_stripping == 1',
        #     )

        #     samples = [cocktail, cocktail_gen]
        #     labels = [f'cocktail - {intermediate_ID}', f'cocktail - {intermediate_ID} gen']
        #     # colours = ["tab:blue", "tab:red", "tab:purple", 'k']
        #     colours = ["tab:purple", 'k']

        #     scores = self.query_and_plot_samples(
        #         samples,
        #         labels,
        #         colours=colours,
        #         filename=f"{filename[:-4]}_{intermediate_ID}.pdf",
        #         include_combinatorial=include_combinatorial,
        #         only_hists=True,
        #         plot_training=False,
        #     )

        if intermediate_IDs == None:
            intermediate_IDs = [421]

        for intermediate_ID in intermediate_IDs:

            cocktail_gen = self.get_sample(
                "datasets/cocktail_three_body_cut_more_vars.root",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
                convert_branches=True,
                cut=f'abs(B_plus_TRUEID)==521 & abs(J_psi_1S_TRUEID)=={intermediate_ID} & pass_stripping == 1',
            )

            cocktail = self.get_sample(
                "datasets/cocktail_three_body_cut_more_vars.root",
                None,
                generate=False,
                N=10000,
                convert_branches=True,
                cut=f'abs(B_plus_TRUEID)==521 & abs(J_psi_1S_TRUEID)=={intermediate_ID} & pass_stripping == 1',
            )

            samples = [cocktail, cocktail_gen]
            labels = [f'cocktail - {intermediate_ID}', f'cocktail - {intermediate_ID} gen']
            # colours = ["tab:blue", "tab:red", "tab:purple", 'k']
            colours = ["tab:purple", 'k']

            scores = self.query_and_plot_samples(
                samples,
                labels,
                colours=colours,
                filename=f"{filename[:-4]}_{intermediate_ID}.pdf",
                include_combinatorial=include_combinatorial,
                only_hists=True,
                plot_training=False,
            )


    
        return scores
    

    def query_and_plot_samples_jpsiX(
        self,
        samples_MC,
        samples_gen,
        filename="BDT.pdf",
        kFold=0,
        include_combinatorial=False,
    ):

        sample_values_MC = {}
        sample_values_gen = {}

        clf = self.BDTs[kFold]["BDT"]

        for i in range(len(samples_MC)):
            sample_values_MC[f"MC_{i}"] = clf.predict_proba(samples_MC[i])[:, 1]
        for i in range(len(samples_gen)):
            sample_values_gen[f"gen_{i}"] = clf.predict_proba(samples_gen[i])[:, 1]
        colours = [
            "tab:blue",
            "tab:red",
            "tab:orange",
            "tab:green",
            "tab:purple",
        ]
        with PdfPages(f"{filename}") as pdf:

            plt.figure(figsize=(8, 8))

            n_points = 75

            x = np.linspace(0, 0.99, n_points)

            for ii in range(5):

                colour = colours[ii]

                eff_true = np.empty(0)
                eff_fake = np.empty(0)

                for cut in x:

                    values = sample_values_MC[f"MC_{ii}"]
                    pass_i = np.shape(np.where(values > cut))[1]
                    eff_true = np.append(eff_true, pass_i / np.shape(values)[0])

                    values = sample_values_gen[f"gen_{ii}"]
                    pass_i = np.shape(np.where(values > cut))[1]
                    eff_fake = np.append(eff_fake, pass_i / np.shape(values)[0])

                plt.plot(x, eff_true, color=colour, linestyle="-")
                plt.plot(x, eff_fake, color=colour, linestyle="--")
                plt.fill_between(x, eff_true, eff_fake, color=colour, alpha=0.1)

            pdf.savefig(bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)

            hist = plt.hist(
                sample_values_MC.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_MC.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            hist = plt.hist(
                sample_values_gen.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_gen.keys()),
                histtype="step",
                range=[0, 1],
                alpha=0.5,
            )
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(2, 3, 2)
            hist = plt.hist(
                sample_values_MC.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_MC.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            hist = plt.hist(
                sample_values_gen.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_gen.keys()),
                histtype="step",
                range=[0, 1],
                alpha=0.5,
            )
            plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")

            pdf.savefig(bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.title("MC")

            hist = plt.hist(
                sample_values_MC.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_MC.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            plt.ylim(0.05, 16)
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(2, 3, 2)
            plt.title("MC")
            hist = plt.hist(
                sample_values_MC.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_MC.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            plt.ylim(0.05, 16)
            plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")

            pdf.savefig(bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.title("GEN")

            hist = plt.hist(
                sample_values_gen.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_gen.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            plt.ylim(0.05, 16)
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(2, 3, 2)
            plt.title("GEN")
            hist = plt.hist(
                sample_values_gen.values(),
                bins=25,
                color=colours,
                density=True,
                label=list(sample_values_gen.keys()),
                histtype="step",
                range=[0, 1],
                alpha=1.0,
            )
            plt.ylim(0.05, 16)
            plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")

            pdf.savefig(bbox_inches="tight")
            plt.close()

            for ii in range(5):

                plt.figure(figsize=(15, 10))
                plt.subplot(2, 3, 1)

                hist = plt.hist(
                    sample_values_MC[f"MC_{ii}"],
                    bins=25,
                    color=colours[ii],
                    density=True,
                    label=f"MC_{ii}",
                    histtype="step",
                    range=[0, 1],
                    alpha=1.0,
                )
                hist = plt.hist(
                    sample_values_gen[f"gen_{ii}"],
                    bins=25,
                    color=colours[ii],
                    density=True,
                    label=f"gen_{ii}",
                    histtype="step",
                    range=[0, 1],
                    alpha=0.5,
                )
                plt.xlabel(f"BDT output")
                plt.yscale("log")
                plt.subplot(2, 3, 2)
                hist = plt.hist(
                    sample_values_MC[f"MC_{ii}"],
                    bins=25,
                    color=colours[ii],
                    density=True,
                    label=f"MC_{ii}",
                    histtype="step",
                    range=[0, 1],
                    alpha=1.0,
                )
                hist = plt.hist(
                    sample_values_gen[f"gen_{ii}"],
                    bins=25,
                    color=colours[ii],
                    density=True,
                    label=f"gen_{ii}",
                    histtype="step",
                    range=[0, 1],
                    alpha=0.5,
                )
                plt.legend(loc="upper left")
                plt.xlabel(f"BDT output")

                pdf.savefig(bbox_inches="tight")
                plt.close()

        scores = None
        return scores

    def query_and_plot_samples(
        self,
        samples,
        labels,
        colours=["tab:blue", "tab:red", "tab:green", "tab:purple", "k"],
        filename="BDT.pdf",
        kFold=0,
        include_combinatorial=False,
        only_hists=False,
        plot_training=True,
    ):

        sample_values = {}
        if plot_training:
            sample_values[self.signal_label] = self.BDTs[kFold]["values_sig"]
            sample_values[self.background_label] = self.BDTs[kFold]["values_bkg"]

        clf = self.BDTs[kFold]["BDT"]

        for idx, sample in enumerate(samples):
            # print(idx, sample, np.where(np.isinf(sample)), np.where(np.isnan(sample)))
            sample_values[labels[idx+2]] = clf.predict_proba(sample)[:, 1]

            # if labels[idx+2] == 'BuD0enuKenu - MC':
                
            #     sample = np.asarray(sample)[np.where(sample_values[labels[idx+2]]>0.9)]
            #     data_for_root = {}
            #     for var_idx, var in enumerate(self.BDT_vars):
            #         data_for_root[var] = sample[:,var_idx]
            #     data_for_root = pd.DataFrame.from_dict(data_for_root)
            #     write_df_to_root(data_for_root, 'BuD0enuKenu_passing_BDT.root')
            #     quit()

        with PdfPages(f"{filename}") as pdf:

            plt.figure(figsize=(26, 7))
            plt.subplot(1, 4, 1)

            hist = plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                alpha=0.25,
                label=list(sample_values.keys()),
                density=True,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                density=True,
                histtype="step",
                range=[0, 1],
            )
            # plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(1, 4, 2)
            plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                alpha=0.25,
                label=list(sample_values.keys()),
                density=True,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                density=True,
                histtype="step",
                range=[0, 1],
            )
            plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")

            # if only_hists:
            #     pdf.savefig(bbox_inches="tight")
            #     plt.close()
            #     return None

            ax = plt.subplot(1, 4, 4)

            hist = plt.hist(
                sample_values.values(),
                bins=15,
                color=colours,
                alpha=0.25,
                label=list(sample_values.keys()),
                density=False,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                sample_values.values(),
                bins=15,
                color=colours,
                density=False,
                histtype="step",
                range=[0, 1],
            )
            # plt.title("Samples may not be correctly scaled") # set_visible(False)
            # plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            ax.set_visible(False)

            # plt.subplot(2, 3, 4)
            # x = hist[1][:-1] + (hist[1][1] - hist[1][0]) / 2.0
            # y = hist[0][0] / hist[0][3]
            # yerr = y * np.sqrt(
            #     (np.sqrt(hist[0][0]) / hist[0][0]) ** 2
            #     + (np.sqrt(hist[0][3]) / hist[0][3]) ** 2
            # )
            # y *= np.sum(hist[0][3]) / np.sum(hist[0][0])
            # plt.errorbar(
            #     x,
            #     y,
            #     yerr=yerr,
            #     label="MC",
            #     color="tab:blue",
            #     marker="o",
            #     fmt=" ",
            #     capsize=2,
            #     linewidth=1.75,
            # )

            # x = hist[1][:-1] + (hist[1][1] - hist[1][0]) / 2.0
            # y = hist[0][2] / hist[0][4]
            # yerr = y * np.sqrt(
            #     (np.sqrt(hist[0][2]) / hist[0][2]) ** 2
            #     + (np.sqrt(hist[0][4]) / hist[0][4]) ** 2
            # )
            # y *= np.sum(hist[0][4]) / np.sum(hist[0][2])
            # plt.errorbar(
            #     x,
            #     y,
            #     yerr=yerr,
            #     label="gen",
            #     color="tab:red",
            #     marker="o",
            #     fmt=" ",
            #     capsize=2,
            #     linewidth=1.75,
            # )

            # plt.ylabel("Signal/prc")
            # plt.xlabel(f"BDT output")
            # plt.legend()
            # plt.axhline(y=1, c="k")

            plt.subplot(1, 4, 3)

            n_points = 50

            effs = {}
            x = np.linspace(0, 0.99, n_points)

            # if include_combinatorial:
            #     sample_list = [
            #         self.signal_label,
            #         "sig - gen",
            #         "prc - MC",
            #         "prc - gen",
            #         self.background_label,
            #         "combi - gen",
            #     ]
            # else:
            #     sample_list = [self.signal_label, "sig - gen", "sig - gen (rapidsim)", "prc - MC", "prc - gen", "prc - gen (rapidsim)"]
            sample_list = list(sample_values.keys())

            for sample in sample_list:
                
                if sample == self.background_label:
                    continue

                eff = np.empty(0)
                for cut in x:

                    values = sample_values[sample]

                    pass_i = np.shape(np.where(values > cut))[1]
                    eff = np.append(eff, pass_i / np.shape(values)[0])
                effs[sample] = eff

                if "sig" in sample or sample == self.signal_label:
                    color = "tab:blue"
                else:
                    color = "tab:red"
                if "gen" in sample:
                    if "rapidsim" in sample:
                        style = "-."
                    else:
                        style = "--"
                else:
                    style = "-"

                if "combi" in sample or sample == self.background_label:
                    color = "tab:orange"

                plt.plot(x, effs[sample], label=sample, color=color, linestyle=style)

            # pairs = [[0, 1], [2, 3]]
            # if include_combinatorial:
            #     pairs.append([4, 5])
            scores = []
            # colors = ["tab:blue", "tab:red", "tab:orange"]
            # for idx, pair in enumerate(pairs):
            #     true = effs[list(effs.keys())[pair[0]]]
            #     false = effs[list(effs.keys())[pair[1]]]
            #     plt.fill_between(x, true, false, color=colors[idx], alpha=0.1)
            #     scores.append(np.sum(np.abs(true - false)) / n_points)

            plt.legend()
            plt.ylabel(f"Selection efficiency")
            plt.xlabel(f"BDT cut")

            # quit()

            pdf.savefig(bbox_inches="tight")
            plt.close()

        return scores

    def get_stripping_eff(self, loader):
        
        self.cuts = {}
        self.cuts['B_plus_FDCHI2_OWNPV'] = ">100."
        self.cuts['B_plus_DIRA_OWNPV'] = ">0.9995"
        self.cuts['B_plus_IPCHI2_OWNPV'] = "<25"
        self.cuts['(B_plus_ENDVERTEX_CHI2/B_plus_ENDVERTEX_NDOF)'] = "<9"
        # cuts['J_psi_1S_PT'] = ">0"
        self.cuts['J_psi_1S_FDCHI2_OWNPV'] = ">16"
        self.cuts['J_psi_1S_IPCHI2_OWNPV'] = ">0"
        for lepton in ['e_minus', 'e_plus']:
            self.cuts[f'{lepton}_IPCHI2_OWNPV'] = ">9"
            # cuts[f'{lepton}_PT'] = ">300"
        for hadron in ['K_Kst']:
            self.cuts[f'{hadron}_IPCHI2_OWNPV'] = ">9"
            # cuts[f'{hadron}_PT'] = ">400"
        # cuts['m_12'] = "<5500"
        # cuts['B_plus_M_Kee_reco'] = ">(5279.34-1500)"
        # cuts['B_plus_M_Kee_reco'] = "<(5279.34+1500)"

        effs_true = np.empty((0,2))
        eff_true, effErr_true = loader.getEff(self.cuts)
        effs_true = np.append(effs_true, [[eff_true, effErr_true]], axis=0)
        for cut in list(self.cuts.keys()):
            eff_true, effErr_true = loader.getEff(f'{cut}{self.cuts[cut]}')
            effs_true = np.append(effs_true, [[eff_true, effErr_true]], axis=0)

        return effs_true



    def get_sample_and_stripping_eff(
        self,
        sample_loc,
        vertex_quality_trainer_obj,
        generate,
        cut=None,
        convert_branches=False,
        N=10000,
        rapidsim=False,
        return_data_loader=False
    ):


        if rapidsim:
            event_loader = data_loader.load_data(
                [
                    sample_loc,
                ],
                transformers=self.transformers,
                convert_to_RK_branch_names=True,
                conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
            )

            if "Partreco" in sample_loc:
                event_loader_target = data_loader.load_data(
                    [
                        "datasets/Kstee_cut_more_vars.root",
                    ],
                    transformers=self.transformers,
                    convert_to_RK_branch_names=True,
                    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
                )
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['m_01','m_02','m_12'])
            if "BuD0enuKenu" in sample_loc:
                # event_loader_target = data_loader.load_data(
                #     [
                #         "datasets/dedicated_BuD0enuKenu_MC_hierachy_cut_more_vars.root",
                #     ],
                #     transformers=self.transformers,
                #     convert_to_RK_branch_names=True,
                #     conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
                # )
                # event_loader_target.cut("pass_stripping")
                event_loader.cut('m_12>3.674')
                # event_loader.sample_with_replacement_with_reweight(target_loader=event_loader_target, reweight_vars=['FD_B_plus_true_vertex'])
            # if "BuD0piKenu" in sample_loc:

        else:
            if convert_branches:
                event_loader = data_loader.load_data(
                    [
                        sample_loc,
                    ],
                    transformers=self.transformers,
                    convert_to_RK_branch_names=True,
                    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
                )
            else:
                event_loader = data_loader.load_data(
                    [
                        sample_loc,
                    ],
                    transformers=self.transformers,
                )

        if cut is not None:
            print(event_loader.shape())
            event_loader.cut(cut)
            print(event_loader.shape())
        
        try:
            event_loader.select_randomly(Nevents=N)
        except:
            pass
        

        if generate:

            event_loader = vertex_quality_trainer_obj.predict_from_data_loader(
                event_loader
            )
            # print(event_loader.print_branches())
            stripping_effs = self.get_stripping_eff(event_loader)
            event_loader.fill_stripping_bool()
            event_loader.cut("pass_stripping")
            sample_after_stripping = event_loader.get_branches(self.BDT_vars_gen, processed=False)
            sample_after_stripping = np.squeeze(np.asarray(sample_after_stripping[self.BDT_vars_gen]))

        else:  
            
            stripping_effs = self.get_stripping_eff(event_loader)
            event_loader.fill_stripping_bool()
            event_loader.cut("pass_stripping")
            sample_after_stripping = event_loader.get_branches(self.BDT_vars, processed=False)
            sample_after_stripping = np.squeeze(np.asarray(sample_after_stripping[self.BDT_vars]))

        if return_data_loader:
            return sample_after_stripping, stripping_effs, event_loader
        else:
            return sample_after_stripping, stripping_effs
        
        
    def compare_stripping_eff_plots(
        self,
        pdf,
        sample_loc,
        vertex_quality_trainer_obj,
        generate,
        convert_branches=False,
        N=10000,
        rapidsim=False,
        return_data_loader=False,
        extra_labels=[]
    ):
        loaders = []
        for index, sample_loc_i in enumerate(sample_loc):
            if rapidsim[index]:
                event_loader = data_loader.load_data(
                    [
                        sample_loc_i,
                    ],
                    transformers=self.transformers,
                    convert_to_RK_branch_names=True,
                    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
                )
            else:
                if convert_branches[index]:
                    event_loader = data_loader.load_data(
                        [
                            sample_loc_i,
                        ],
                        transformers=self.transformers,
                        convert_to_RK_branch_names=True,
                        conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
                    )
                else:
                    event_loader = data_loader.load_data(
                        [
                            sample_loc_i,
                        ],
                        transformers=self.transformers,
                    )

            
            if generate[index]:

                event_loader = vertex_quality_trainer_obj.predict_from_data_loader(
                    event_loader
                )

            loaders.append(event_loader)

        for cut in list(self.cuts.keys()):
            
            results = []
            results_physical = []
            if cut == '(B_plus_ENDVERTEX_CHI2/B_plus_ENDVERTEX_NDOF)':
                continue
            else:    
                for loader in loaders:
                    results.append(loader.get_branches(cut, processed=True))
                    results_physical.append(loader.get_branches(cut, processed=False))
            
            cut_value = self.cuts[cut].replace('>','').replace('<','')

            for index in range(len(results)):
                results[index] = np.asarray(results[index]).flatten()
                results[index] = results[index][np.isfinite(results[index])]

                results_physical[index] = np.asarray(results_physical[index]).flatten()
                results_physical[index] = results_physical[index][np.isfinite(results_physical[index])]

            labels = []
            for index in range(len(results)):

                eff, effErr = loaders[index].getEff(f'{cut}{self.cuts[cut]}')
                if len(extra_labels)==0:
                    labels.append(f'{eff:.3f}+-{effErr:.3f}')
                else:
                    labels.append(f'{extra_labels[index]} {eff:.3f}+-{effErr:.3f}')

            # plt.subplot(1,2,1)
            plt.hist(results, bins=75, density=True, histtype='step', label=labels)
            plt.legend()
            plt.xlabel(cut)
            plt.xlim(-1,1)
            plt.axvline(x=loader.convert_value_to_processed(cut, cut_value),c='k')

            # plt.subplot(1,2,2)
            # plt.hist(results_physical, bins=75, density=True, histtype='step', label=labels)
            # plt.legend()
            # plt.xlabel(cut)

            pdf.savefig(bbox_inches="tight")
            plt.close()



    def plot_detailed_metrics(
        self,
        conditions,
        targets,
        vertex_quality_trainer_obj,
        filename,
    ):  
        self.conditions = conditions
        self.targets = targets
        

        with PdfPages(filename) as pdf:

            signal_gen, signal_gen_stripping_eff = self.get_sample_and_stripping_eff(
                "datasets/dedicated_Kee_MC_hierachy_cut_more_vars.root",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
                convert_branches=True,
            )  

            signal_gen_rapidsim, signal_gen_rapidsim_stripping_eff = self.get_sample_and_stripping_eff(
                "/users/am13743/fast_vertexing_variables/rapidsim/Kee/Signal_tree_NNvertex_more_vars.root",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
                rapidsim=True,
            )  

            samples = [signal_gen, signal_gen_rapidsim]
            labels = [self.signal_label, self.background_label, "sig - gen", "sig - gen (rapidsim)"]
            colours = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

            self.query_and_plot_samples_pages(
                pdf,
                samples,
                labels,
                colours=colours,
                filename=filename,
                only_hists=True,
            )

            plt.title("Kee")
            plt.errorbar(np.arange(np.shape(self.BDTs[0]["signal_stripping_eff"])[0]), self.BDTs[0]["signal_stripping_eff"][:,0], yerr=self.BDTs[0]["signal_stripping_eff"][:,1],label='MC',color='tab:blue',linestyle='-')
            plt.errorbar(np.arange(np.shape(signal_gen_stripping_eff)[0]), signal_gen_stripping_eff[:,0], yerr=signal_gen_stripping_eff[:,1],label='gen',color='tab:blue',linestyle='--')
            plt.errorbar(np.arange(np.shape(signal_gen_rapidsim_stripping_eff)[0]), signal_gen_rapidsim_stripping_eff[:,0], yerr=signal_gen_rapidsim_stripping_eff[:,1],label='gen (rapidsim)',color='tab:blue',linestyle='-.')
            plt.ylim(0,1)
            cuts_ticks = ['All']+list(self.cuts.keys())
            plt.xticks(np.arange(len(cuts_ticks)), cuts_ticks, rotation=90)
            for i in np.arange(len(cuts_ticks)):
                if i ==0:
                    plt.axvline(x=i, alpha=0.5, ls='-',c='k')
                else:
                    plt.axvline(x=i, alpha=0.5, ls='--',c='k')
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()

            self.compare_stripping_eff_plots(
                pdf,
                ["datasets/dedicated_Kee_MC_hierachy_cut_more_vars.root", "/users/am13743/fast_vertexing_variables/rapidsim/Kee/Signal_tree_NNvertex_more_vars.root", "datasets/cocktail_hierarchy_cut_more_vars.root", "datasets/dedicated_Kee_MC_hierachy_cut_more_vars.root"],
                vertex_quality_trainer_obj,
                generate=[False, True, False, True],
                convert_branches=[True, True, True, True],
                rapidsim=[False, True, False, False],
                N=10000,
                extra_labels=['MC','Rapidsim','Cocktail','MC - gen'],
            )  

            part_reco_gen, part_reco_gen_stripping_eff = self.get_sample_and_stripping_eff(
                "datasets/dedicated_Kstee_MC_hierachy_cut_more_vars.root",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
                rapidsim=False,
                convert_branches=True,
            )  
            part_reco_gen_rapidsim, part_reco_gen_rapidsim_stripping_eff = self.get_sample_and_stripping_eff(
                "/users/am13743/fast_vertexing_variables/rapidsim/Kstree/Partreco_tree_NNvertex_more_vars.root",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
                rapidsim=True,
            )  
            part_reco_MC, part_reco_MC_stripping_eff = self.get_sample_and_stripping_eff(
                "datasets/dedicated_Kstee_MC_hierachy_cut_more_vars.root",
                vertex_quality_trainer_obj,
                generate=False,
                N=10000,
                convert_branches=True,
            ) 

            samples = [part_reco_gen, part_reco_gen_rapidsim, part_reco_MC]
            labels = [self.signal_label, self.background_label, "prc - gen", "prc - gen (rapidsim)", "prc - MC"]
            colours = ["tab:blue", "tab:red", "k", "violet", "tab:purple"]

            self.query_and_plot_samples_pages(
                pdf,
                samples,
                labels,
                colours=colours,
                filename=filename,
                only_hists=True,
            )



            plt.title("K*ee")
            plt.errorbar(np.arange(np.shape(self.BDTs[0]["signal_stripping_eff"])[0]), self.BDTs[0]["signal_stripping_eff"][:,0], yerr=self.BDTs[0]["signal_stripping_eff"][:,1],label='MC',color='tab:blue',linestyle='-')
            plt.errorbar(np.arange(np.shape(part_reco_MC_stripping_eff)[0]), part_reco_MC_stripping_eff[:,0], yerr=part_reco_MC_stripping_eff[:,1],label='MC',color='tab:red',linestyle='-')
            plt.errorbar(np.arange(np.shape(part_reco_gen_stripping_eff)[0]), part_reco_gen_stripping_eff[:,0], yerr=part_reco_gen_stripping_eff[:,1],label='gen',color='tab:red',linestyle='--')
            plt.errorbar(np.arange(np.shape(part_reco_gen_rapidsim_stripping_eff)[0]), part_reco_gen_rapidsim_stripping_eff[:,0], yerr=part_reco_gen_rapidsim_stripping_eff[:,1],label='gen (rapidsim)',color='tab:red',linestyle='-.')
            plt.ylim(0,1)
            cuts_ticks = ['All']+list(self.cuts.keys())
            plt.xticks(np.arange(len(cuts_ticks)), cuts_ticks, rotation=90)
            for i in np.arange(len(cuts_ticks)):
                if i ==0:
                    plt.axvline(x=i, alpha=0.5, ls='-',c='k')
                else:
                    plt.axvline(x=i, alpha=0.5, ls='--',c='k')
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()














            BuD0enuKenu_gen, BuD0enuKenu_gen_stripping_eff = self.get_sample_and_stripping_eff(
                "datasets/dedicated_BuD0enuKenu_MC_hierachy_cut_more_vars.root",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
                rapidsim=False,
                convert_branches=True,
            )  
            
            BuD0enuKenu_gen_rapidsim, BuD0enuKenu_gen_rapidsim_stripping_eff = self.get_sample_and_stripping_eff(
                "/users/am13743/fast_vertexing_variables/rapidsim/BuD0enuKenu/BuD0enuKenu_tree_NNvertex_more_vars.root",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
                rapidsim=True,
            )  
            
            BuD0enuKenu_MC, BuD0enuKenu_MC_stripping_eff = self.get_sample_and_stripping_eff(
                "datasets/dedicated_BuD0enuKenu_MC_hierachy_cut_more_vars.root",
                vertex_quality_trainer_obj,
                generate=False,
                N=10000,
                convert_branches=True,
            )  


            samples = [BuD0enuKenu_gen, BuD0enuKenu_gen_rapidsim, BuD0enuKenu_MC]
            labels = [self.signal_label, self.background_label, "BuD0enuKenu - gen", "BuD0enuKenu - gen (rapidsim)", "BuD0enuKenu - MC"]
            colours = ["tab:blue", "tab:red", "k", "violet", "tab:purple"]

            self.query_and_plot_samples_pages(
                pdf,
                samples,
                labels,
                colours=colours,
                filename=filename.replace('.pdf','_BuD0enuKenu.pdf'),
                only_hists=True,
            )


            plt.title("BuD0enuKenu")
            plt.errorbar(np.arange(np.shape(self.BDTs[0]["signal_stripping_eff"])[0]), self.BDTs[0]["signal_stripping_eff"][:,0], yerr=self.BDTs[0]["signal_stripping_eff"][:,1],label='MC',color='tab:blue',linestyle='-')
            plt.errorbar(np.arange(np.shape(BuD0enuKenu_MC_stripping_eff)[0]), BuD0enuKenu_MC_stripping_eff[:,0], yerr=BuD0enuKenu_MC_stripping_eff[:,1],label='MC',color='tab:red',linestyle='-')
            plt.errorbar(np.arange(np.shape(BuD0enuKenu_gen_stripping_eff)[0]), BuD0enuKenu_gen_stripping_eff[:,0], yerr=BuD0enuKenu_gen_stripping_eff[:,1],label='gen',color='tab:red',linestyle='--')
            plt.errorbar(np.arange(np.shape(BuD0enuKenu_gen_rapidsim_stripping_eff)[0]), BuD0enuKenu_gen_rapidsim_stripping_eff[:,0], yerr=BuD0enuKenu_gen_rapidsim_stripping_eff[:,1],label='gen (rapidsim)',color='tab:red',linestyle='-.')
            plt.ylim(0,1)
            cuts_ticks = ['All']+list(self.cuts.keys())
            plt.xticks(np.arange(len(cuts_ticks)), cuts_ticks, rotation=90)
            for i in np.arange(len(cuts_ticks)):
                if i ==0:
                    plt.axvline(x=i, alpha=0.5, ls='-',c='k')
                else:
                    plt.axvline(x=i, alpha=0.5, ls='--',c='k')
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()







            BuD0piKenu_gen, BuD0piKenu_gen_stripping_eff = self.get_sample_and_stripping_eff(
                "datasets/dedicated_BuD0piKenu_MC_hierachy_cut_more_vars.root",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
                rapidsim=False,
                convert_branches=True,
            )  
            
            BuD0piKenu_gen_rapidsim, BuD0piKenu_gen_rapidsim_stripping_eff = self.get_sample_and_stripping_eff(
                "/users/am13743/fast_vertexing_variables/rapidsim/BuD0piKenu/BuD0piKenu_tree_NNvertex_more_vars.root",
                vertex_quality_trainer_obj,
                generate=True,
                N=10000,
                rapidsim=True,
            )  
            
            BuD0piKenu_MC, BuD0piKenu_MC_stripping_eff = self.get_sample_and_stripping_eff(
                "datasets/dedicated_BuD0piKenu_MC_hierachy_cut_more_vars.root",
                vertex_quality_trainer_obj,
                generate=False,
                N=10000,
                convert_branches=True,
            )  

            
            samples = [BuD0piKenu_gen, BuD0piKenu_gen_rapidsim, BuD0piKenu_MC]
            labels = [self.signal_label, self.background_label, "BuD0piKenu - gen", "BuD0piKenu - gen (rapidsim)", "BuD0piKenu - MC"]
            colours = ["tab:blue", "tab:red", "k", "violet", "tab:purple"]

            self.query_and_plot_samples_pages(
                pdf,
                samples,
                labels,
                colours=colours,
                filename=filename.replace('.pdf','_BuD0piKenu.pdf'),
                only_hists=True,
            )

            plt.title("BuD0piKenu")
            plt.errorbar(np.arange(np.shape(self.BDTs[0]["signal_stripping_eff"])[0]), self.BDTs[0]["signal_stripping_eff"][:,0], yerr=self.BDTs[0]["signal_stripping_eff"][:,1],label='MC',color='tab:blue',linestyle='-')
            plt.errorbar(np.arange(np.shape(BuD0piKenu_MC_stripping_eff)[0]), BuD0piKenu_MC_stripping_eff[:,0], yerr=BuD0piKenu_MC_stripping_eff[:,1],label='MC',color='tab:red',linestyle='-')
            plt.errorbar(np.arange(np.shape(BuD0piKenu_gen_stripping_eff)[0]), BuD0piKenu_gen_stripping_eff[:,0], yerr=BuD0piKenu_gen_stripping_eff[:,1],label='gen',color='tab:red',linestyle='--')
            plt.errorbar(np.arange(np.shape(BuD0piKenu_gen_rapidsim_stripping_eff)[0]), BuD0piKenu_gen_rapidsim_stripping_eff[:,0], yerr=BuD0piKenu_gen_rapidsim_stripping_eff[:,1],label='gen (rapidsim)',color='tab:red',linestyle='-.')
            plt.ylim(0,1)
            cuts_ticks = ['All']+list(self.cuts.keys())
            plt.xticks(np.arange(len(cuts_ticks)), cuts_ticks, rotation=90)
            for i in np.arange(len(cuts_ticks)):
                if i == 0:
                    plt.axvline(x=i, alpha=0.5, ls='-',c='k')
                else:
                    plt.axvline(x=i, alpha=0.5, ls='--',c='k')
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()





    def query_and_plot_samples_pages(
            self,
            pdf,
            samples,
            labels,
            colours=["tab:blue", "tab:red", "tab:green", "tab:purple", "k"],
            filename="BDT.pdf",
            kFold=0,
            include_combinatorial=False,
            only_hists=False,
            plot_training=True,
        ):

            sample_values = {}
            if plot_training:
                sample_values[self.signal_label] = self.BDTs[kFold]["values_sig"]
                sample_values[self.background_label] = self.BDTs[kFold]["values_bkg"]

            clf = self.BDTs[kFold]["BDT"]

            for idx, sample in enumerate(samples):
                sample_values[labels[idx+2]] = clf.predict_proba(sample)[:, 1]


            # plt.figure(figsize=(26, 7))
            # plt.subplot(1, 4, 1)

            hist = plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                alpha=0.25,
                label=list(sample_values.keys()),
                density=True,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                density=True,
                histtype="step",
                range=[0, 1],
            )
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            pdf.savefig(bbox_inches="tight")
            plt.close()


            plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                alpha=0.25,
                label=list(sample_values.keys()),
                density=True,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                sample_values.values(),
                bins=50,
                color=colours,
                density=True,
                histtype="step",
                range=[0, 1],
            )
            plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")
            pdf.savefig(bbox_inches="tight")
            plt.close()


         
            n_points = 50

            effs = {}
            x = np.linspace(0, 0.99, n_points)

            sample_list = list(sample_values.keys())

            for sample in sample_list:
                
                if sample == self.background_label:
                    continue

                eff = np.empty(0)
                for cut in x:

                    values = sample_values[sample]

                    pass_i = np.shape(np.where(values > cut))[1]
                    eff = np.append(eff, pass_i / np.shape(values)[0])
                effs[sample] = eff

                if "sig" in sample or sample == self.signal_label:
                    color = "tab:blue"
                else:
                    color = "tab:red"
                if "gen" in sample:
                    if "rapidsim" in sample:
                        style = "-."
                    else:
                        style = "--"
                else:
                    style = "-"

                if "combi" in sample or sample == self.background_label:
                    color = "tab:orange"

                plt.plot(x, effs[sample], label=sample, color=color, linestyle=style)

            plt.legend()
            plt.ylabel(f"Selection efficiency")
            plt.xlabel(f"BDT cut")

            pdf.savefig(bbox_inches="tight")
            plt.close()






    
    
    