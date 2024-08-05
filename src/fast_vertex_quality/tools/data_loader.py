import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from fast_vertex_quality.tools.config import rd, read_definition
import tensorflow as tf
import uproot

import uproot3 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pickle
from particle import Particle
from hep_ml.reweight import BinsReweighter, GBReweighter, FoldingReweighter

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



def produce_physics_variables(data):

    physics_variables = {}

    for particle_i in rd.daughter_particles:

        physics_variables[f"{particle_i}_P"] = np.sqrt(
            data[f"{particle_i}_PX"] ** 2
            + data[f"{particle_i}_PY"] ** 2
            + data[f"{particle_i}_PZ"] ** 2
        )

        physics_variables[f"{particle_i}_PT"] = np.sqrt(
            data[f"{particle_i}_PX"] ** 2 + data[f"{particle_i}_PY"] ** 2
        )

        physics_variables[f"{particle_i}_eta"] = -np.log(
            np.tan(
                np.arcsin(
                    physics_variables[f"{particle_i}_PT"]
                    / physics_variables[f"{particle_i}_P"]
                )
                / 2.0
            )
        )

    physics_variables["kFold"] = np.random.randint(
        low=0,
        high=9,
        size=np.shape(data[f"{rd.daughter_particles[0]}_PX"])[0],
    )

    electron_mass = 0.51099895000 * 1e-3

    PE = np.sqrt(
        electron_mass**2
        + data[f"{rd.daughter_particles[1]}_PX"] ** 2
        + data[f"{rd.daughter_particles[1]}_PY"] ** 2
        + data[f"{rd.daughter_particles[1]}_PZ"] ** 2
    ) + np.sqrt(
        electron_mass**2
        + data[f"{rd.daughter_particles[2]}_PX"] ** 2
        + data[f"{rd.daughter_particles[2]}_PY"] ** 2
        + data[f"{rd.daughter_particles[2]}_PZ"] ** 2
    )
    PX = data[f"{rd.daughter_particles[1]}_PX"] + data[f"{rd.daughter_particles[2]}_PX"]
    PY = data[f"{rd.daughter_particles[1]}_PY"] + data[f"{rd.daughter_particles[2]}_PY"]
    PZ = data[f"{rd.daughter_particles[1]}_PZ"] + data[f"{rd.daughter_particles[2]}_PZ"]

    physics_variables["q2"] = (PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6

    df = pd.DataFrame.from_dict(physics_variables)

    return df


class NoneError(Exception):
    pass


def symsqrt(x, c=1):
    """Apply symmetric logarithm transformation."""
    # return np.sign(x) * np.log10(c * np.abs(x) + 1)
    return np.sign(x) * np.sqrt(np.abs(x))

def inv_symsqrt(y, c=1):
    """Apply inverse symmetric logarithm transformation."""
    # return np.sign(y) * (10**np.abs(y) - 1) / c
    return np.sign(y) * np.abs(y)**2

class Transformer:

    def __init__(self):

        self.abs_columns = [
            f"{rd.mother_particle}_TRUEID",
            f"{rd.daughter_particles[0]}_TRUEID",
            f"{rd.daughter_particles[1]}_TRUEID",
            f"{rd.daughter_particles[2]}_TRUEID",
                            ]

        self.shift_and_symsqrt_columns = [
            f"{rd.mother_particle}_TRUEORIGINVERTEX_X",
            f"{rd.mother_particle}_TRUEORIGINVERTEX_Y"
        ]

        self.log_columns = [
            f"{rd.mother_particle}_FDCHI2_OWNPV",
            f"{rd.daughter_particles[0]}_IPCHI2_OWNPV",
            f"{rd.daughter_particles[1]}_IPCHI2_OWNPV",
            f"{rd.daughter_particles[2]}_IPCHI2_OWNPV",
            f"{rd.intermediate_particle}_IPCHI2_OWNPV",
            f"{rd.daughter_particles[0]}_PZ",
            f"{rd.daughter_particles[1]}_PZ",
            f"{rd.daughter_particles[2]}_PZ",
            f"{rd.mother_particle}_ENDVERTEX_CHI2",
            f"{rd.mother_particle}_IPCHI2_OWNPV",
            f"IP_{rd.mother_particle}",
            f"{rd.intermediate_particle}_FDCHI2_OWNPV",
            f"{rd.intermediate_particle}_FLIGHT",

            f"{rd.mother_particle}_TRUE_FD",

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

            f"{rd.intermediate_particle}_TRUEID_width",
            f"{rd.intermediate_particle}_MC_MOTHER_ID_width",
            f"{rd.intermediate_particle}_MC_GD_MOTHER_ID_width",
            f"{rd.intermediate_particle}_MC_GD_GD_MOTHER_ID_width",

            f"{rd.daughter_particles[0]}_MC_MOTHER_ID_width",
            f"{rd.daughter_particles[0]}_MC_GD_MOTHER_ID_width",
            f"{rd.daughter_particles[0]}_MC_GD_GD_MOTHER_ID_width",

            f"{rd.daughter_particles[1]}_MC_MOTHER_ID_width",
            f"{rd.daughter_particles[1]}_MC_GD_MOTHER_ID_width",
            f"{rd.daughter_particles[1]}_MC_GD_GD_MOTHER_ID_width",

            f"{rd.daughter_particles[2]}_MC_MOTHER_ID_width",
            f"{rd.daughter_particles[2]}_MC_GD_MOTHER_ID_width",
            f"{rd.daughter_particles[2]}_MC_GD_GD_MOTHER_ID_width",

            f"{rd.intermediate_particle}_MC_MOTHER_ID_mass",
            f"{rd.intermediate_particle}_MC_GD_MOTHER_ID_mass",
            f"{rd.intermediate_particle}_MC_GD_GD_MOTHER_ID_mass",

            f"{rd.daughter_particles[0]}_MC_MOTHER_ID_mass",
            f"{rd.daughter_particles[0]}_MC_GD_MOTHER_ID_mass",
            f"{rd.daughter_particles[0]}_MC_GD_GD_MOTHER_ID_mass",

            f"{rd.daughter_particles[1]}_MC_MOTHER_ID_mass",
            f"{rd.daughter_particles[1]}_MC_GD_MOTHER_ID_mass",
            f"{rd.daughter_particles[1]}_MC_GD_GD_MOTHER_ID_mass",

            f"{rd.daughter_particles[2]}_MC_MOTHER_ID_mass",
            f"{rd.daughter_particles[2]}_MC_GD_MOTHER_ID_mass",
            f"{rd.daughter_particles[2]}_MC_GD_GD_MOTHER_ID_mass",

            f"{rd.daughter_particles[0]}_FLIGHT",
            f"{rd.daughter_particles[1]}_FLIGHT",
            f"{rd.daughter_particles[2]}_FLIGHT",

            "delta_0_P",
            "delta_0_PT",
            "delta_1_P",
            "delta_1_PT",
            "delta_2_P",
            "delta_2_PT",

        ]

        self.one_minus_log_columns = [f"{rd.mother_particle}_DIRA_OWNPV", f"DIRA_{rd.mother_particle}", f"DIRA_{rd.mother_particle}_true_vertex"]
        
        self.min_fills = {}

    def fit(self, data_raw, column):

        self.column = column

        data = data_raw.copy()

        if column in self.log_columns:
            if "width" in self.column or "mass" in self.column:
                data[np.where(data==0)] = np.amin(data[np.where(data!=0)])/2.
                self.min_fills[self.column] = np.amin(data[np.where(data!=0)])/2.
            else:
                data[np.where(data==0)] = 1E-6
            data = np.log10(data)
        elif column in self.one_minus_log_columns:
            # print(column, data)
            data[np.where(data==1)] = 1.-1E-15
            data[np.where(data>1)] = 1.-1E-15
            data[np.where(np.isnan(data))] = 1.-1E-15
            data[np.where(np.isinf(data))] = 1.-1E-15
            # print(np.amin(data), np.amax(data))
            data = np.log10(1.0 - data)
            # print(np.amin(data), np.amax(data))
        elif self.column in self.abs_columns:
            data = np.abs(data)
        elif self.column in self.shift_and_symsqrt_columns:
            self.shift = np.mean(data)
            data = data - self.shift
            data = symsqrt(data)

        self.min = np.amin(data)
        self.max = np.amax(data)
        # if column in self.one_minus_log_columns:
        #     print(self.min)
        #     print(self.max)
        #     print('\n\n')

    def process(self, data_raw):
        
        try:
            data = data_raw.copy()
        except:
            # pass # value is likely a single element
            data = np.asarray(data_raw).astype('float64')

        if self.column in self.log_columns:
            try:
                if "width" in self.column or "mass" in self.column:
                    data[np.where(data==0)] = self.min_fills[self.column]
                else:
                    data[np.where(data==0)] = 1E-6
            except:
                pass
            data = np.log10(data)
        elif self.column in self.one_minus_log_columns:
            try:
                data[np.where(data==1)] = 1.-1E-15
                data[np.where(np.isnan(data))] = 1.-1E-15
                data[np.where(np.isinf(data))] = 1.-1E-15
            except Exception as e:
                print(f"\n\nAn error occurred: {e}")
            data = np.log10(1.0 - data)

        elif self.column in self.abs_columns:
            data = np.abs(data)

        elif self.column in self.shift_and_symsqrt_columns:
            data = data - self.shift
            data = symsqrt(data)

        if "DIRA" in self.column:
            where = np.where(np.isnan(data))

        data = data - self.min
        data = data / (self.max - self.min)
        data *= 2
        data += -1

        try:
            if "DIRA" in self.column:
                data[where] = -1
        except Exception as e:
            print("ERROR in data_loader:",e)
            print("Continuing, might not be essential")

        return data

    def unprocess(self, data_raw):

        data = data_raw.copy()

        data += 1
        data *= 0.5
        data = data * (self.max - self.min)
        data = data + self.min

        if self.column in self.log_columns:
            data = np.power(10, data)
        elif self.column in self.one_minus_log_columns:
            data = np.power(10, data)
            data = 1.0 - data

        elif self.column in self.shift_and_symsqrt_columns:
            data = inv_symsqrt(data)
            data = data + self.shift

        return data


class dataset:

    def __init__(self, filenames, transformers=None):

        self.Transformers = transformers
        self.all_data = {"processed": None, "physical": None}
        self.filenames = filenames

    def fill_stripping_bool(self):

        self.all_data["physical"]["pass_stripping"] = np.zeros(self.all_data["physical"].shape[0])

        # print(self.all_data["physical"]["pass_stripping"])

        if 'B_plus_ENDVERTEX_NDOF' not in list(self.all_data["physical"].keys()):
            self.all_data["physical"]["B_plus_ENDVERTEX_NDOF"] = np.ones(self.all_data["physical"].shape[0])*3.

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

        if isinstance(cuts, dict):
            cut_string = ''
            for cut_idx, cut_i in enumerate(list(cuts.keys())):
                if cut_idx > 0:
                    cut_string += ' & '
                if cut_i == 'extra_cut':
                    cut_string += f'{cuts[cut_i]}'
                else:
                    cut_string += f'{cut_i}{cuts[cut_i]}'
            cuts = cut_string   
        
        # gen_tot_val = self.all_data['physical'].shape[0]
        try:
            cut_array = self.all_data['physical'].query(cuts)
            self.all_data["physical"].loc[cut_array.index,"pass_stripping"] = 1.
        except Exception as e:
            # for key in list(self.all_data['physical'].keys()): print(key)
            print(f"\n\nAn error occurred: {e}")
            print("continuing with pass_stripping = 1\n")
            self.all_data["physical"]["pass_stripping"] = np.ones(self.all_data["physical"].shape[0])

        # print('\n pass_stripping',self.all_data["physical"]["pass_stripping"])

    def print_branches(self):
        for key in list(self.all_data["physical"].keys()):
            print(key)

    def sample_with_replacement_with_reweight(self, target_loader, reweight_vars):

        original = []
        for var in reweight_vars:
            original.append(self.all_data['processed'][var])
        
        target_branches = target_loader.get_branches(reweight_vars, processed=True)

        target = []
        for var in reweight_vars:
            target.append(target_branches[var])

        original = np.swapaxes(np.asarray(original),0,1)
        target = np.swapaxes(np.asarray(target),0,1)

        print("Using GBReweighter to reweight then re-select data...")
        reweighter_base = GBReweighter(max_depth=2, gb_args={'subsample': 0.5})
        reweighter = FoldingReweighter(reweighter_base, n_folds=3)
        reweighter.fit(original=original, target=target)
        MC_weights = reweighter.predict_weights(original)
        MC_weights = np.clip(MC_weights, a_min=0, a_max=5.)
        
        N = self.all_data['processed'].shape[0]
        indexes = np.random.choice(np.arange(N), size=N, replace=True, p=MC_weights/np.sum(MC_weights))

        self.all_data['processed'] = self.all_data['processed'].iloc[indexes]
        self.all_data['physical'] = self.all_data['physical'].iloc[indexes]

        self.all_data['processed'].reset_index(drop=True, inplace=True)
        self.all_data['physical'].reset_index(drop=True, inplace=True)


    def fill(self, data, turn_off_processing=False, avoid_physics_variables=False):

        self.turn_off_processing = turn_off_processing

        if not isinstance(data, pd.DataFrame):
            raise NoneError("Dataset must be a pd.dataframe.")

        self.all_data["physical"] = data
        if self.turn_off_processing:
            return
        
        if not avoid_physics_variables:
            self.physics_variables = produce_physics_variables(self.all_data["physical"])
            shared = list(
                set(list(self.physics_variables.keys())).intersection(
                    set(list(self.all_data["physical"].keys()))
                )
            )
            difference = list(
                set(list(self.physics_variables.keys())).difference(
                    set(list(self.all_data["physical"].keys()))
                )
            )
            if len(shared) > 0:
                for key in shared:
                    self.all_data["physical"][key] = self.physics_variables[key]
            if len(difference) > 0:
                self.physics_variables = self.physics_variables[difference]
                self.all_data["physical"] = pd.concat(
                    (self.all_data["physical"], self.physics_variables), axis=1
                )

            self.all_data["physical"] = self.all_data["physical"].loc[
                :, ~self.all_data["physical"].columns.str.contains("^Unnamed")
            ]

        self.fill_stripping_bool()

        self.all_data["processed"] = self.pre_process(self.all_data["physical"])

    def fill_chi2_gen(self, trackchi2_trainer_obj):

        for particle_i in rd.daughter_particles:

            # decoder_chi2 = tf.keras.models.load_model(
            #     f"save_state/track_chi2_decoder_{particle_i}.h5"
            # )
            latent_dim_chi2 = 1

            conditions_i = [
                f"{particle_i}_PX",
                f"{particle_i}_PY",
                f"{particle_i}_PZ",
                f"{particle_i}_P",
                f"{particle_i}_PT",
                f"{particle_i}_eta",
            ]

            X_test_conditions = self.get_branches(conditions_i, processed=True)
            X_test_conditions = X_test_conditions[conditions_i]
            X_test_conditions = np.asarray(X_test_conditions)

            gen_noise = np.random.normal(
                0, 1, (np.shape(X_test_conditions)[0], latent_dim_chi2)
            )

            images = np.squeeze(
                trackchi2_trainer_obj.predict(
                    particle_i, [gen_noise, X_test_conditions]
                )
            )

            self.fill_new_column(
                images,
                f"{particle_i}_TRACK_CHI2NDOF_gen",
                f"{particle_i}_TRACK_CHI2NDOF",
                processed=True,
            )

    def post_process(self, processed_data):

        df = {}

        for column in list(processed_data.keys()):

            if column == "file":
                df[column] = processed_data[column]
            else:
                df[column] = self.Transformers[column].unprocess(
                    np.asarray(processed_data[column]).copy()
                )

        return pd.DataFrame.from_dict(df)

    def update_transformer(self, variable, new_transformer):
        self.Transformers[variable] = new_transformer
        self.all_data["processed"][variable] = self.Transformers[variable].process(
            np.asarray(self.all_data["physical"][variable]).copy()
        )

    def fill_new_column(
        self, data, new_column_name, transformer_variable, processed=True
    ):

        if processed:

            self.all_data["processed"][new_column_name] = data

            data_physical = self.Transformers[transformer_variable].unprocess(
                np.asarray(data).copy()
            )

            self.all_data["physical"][new_column_name] = data_physical

            self.Transformers[new_column_name] = self.Transformers[transformer_variable]
        else:
            print("fill_new_column, processed False not implemented quitting...")
            quit()

    def fill_new_condition(self, conditon_dict):

        for condition in list(conditon_dict.keys()):
            self.all_data["physical"][condition] = np.ones(self.all_data["physical"].shape[0])*conditon_dict[condition]
            self.all_data["processed"][condition] = self.Transformers[condition].process(
                        np.asarray(self.all_data["physical"][condition]).copy()
                    )


    
    def fill_target(self, processed_data, targets=None):

        if targets == None:
            targets = rd.targets

        df_processed = pd.DataFrame(processed_data, columns=targets)

        df_physical = self.post_process(df_processed)

        for column in targets:
            self.all_data["processed"][column] = np.asarray(df_processed[column])
            self.all_data["physical"][column] = np.asarray(df_physical[column])

        self.fill_stripping_bool()
        self.all_data["processed"]["pass_stripping"] = self.all_data["physical"]["pass_stripping"]

    def select_randomly(self, Nevents):

        idx = np.random.choice(
            self.all_data["processed"].shape[0], replace=False, size=Nevents
        )

        self.all_data["processed"] = self.all_data["processed"].iloc[idx]
        self.all_data["physical"] = self.all_data["physical"].iloc[idx]

    def get_physical(self):
        return self.all_data["physical"]

    def get_branches(self, branches, processed=True):

        if not isinstance(branches, list):
            branches = [branches]

        if processed:
            missing = list(
                set(branches).difference(set(list(self.all_data["processed"].keys())))
            )
            branches = list(
                set(branches).intersection(set(list(self.all_data["processed"].keys())))
            )

            if len(missing) > 0:
                print(f"missing branches: {missing}\n {self.filenames} \n")

            output = self.all_data["processed"][branches]

        else:
            missing = list(
                set(branches).difference(set(list(self.all_data["physical"].keys())))
            )
            branches = list(
                set(branches).intersection(set(list(self.all_data["physical"].keys())))
            )

            if len(missing) > 0:
                print(f"missing branches: {missing}\n {self.filenames} \n")

            output = self.all_data["physical"][branches]

        return output

    def virtual_get_branches(self, branches, processed=True):

        if not isinstance(branches, list):
            branches = [branches]

        if processed:
            missing = list(
                set(branches).difference(set(list(self.all_data["processed_virtual"].keys())))
            )
            branches = list(
                set(branches).intersection(set(list(self.all_data["processed_virtual"].keys())))
            )

            if len(missing) > 0:
                print(f"missing branches: {missing}\n {self.filenames} \n")

            output = self.all_data["processed_virtual"][branches]

        else:
            missing = list(
                set(branches).difference(set(list(self.all_data["physical_virtual"].keys())))
            )
            branches = list(
                set(branches).intersection(set(list(self.all_data["physical_virtual"].keys())))
            )

            if len(missing) > 0:
                print(f"missing branches: {missing}\n {self.filenames} \n")

            output = self.all_data["physical_virtual"][branches]

        return output
    
    def pre_process(self, physical_data):

        df = {}

        if self.Transformers == None:
            fresh_transformers = True
            self.Transformers = {}
        else:
            print("using loaded transformers")
            fresh_transformers = False

        for column in list(physical_data.keys()):

            if column == "file" or column == "pass_stripping":
                df[column] = physical_data[column]
            else:
                try:
                    if fresh_transformers:
                        data_array = np.asarray(physical_data[column]).copy()
                        transformer_i = Transformer()
                        transformer_i.fit(data_array, column)
                        self.Transformers[column] = transformer_i

                    df[column] = self.Transformers[column].process(
                        np.asarray(physical_data[column]).copy()
                    )
                # except Exception as e:
                #     print(f"\n\n pre_process: An error occurred: {e}")
                except:
                    pass
            # print(np.shape(df[column]), column)

        return pd.DataFrame.from_dict(df)

    def get_transformers(self):
        return self.Transformers

    def shape(self):
        return self.all_data['physical'].shape
    
    def getBinomialEff(self, pass_sum, tot_sum, pass_sumErr, tot_sumErr):
        '''
        Function for computing efficiency (and uncertainty).
        '''
        eff = pass_sum/tot_sum # Easy part

        # Compute uncertainty taken from Eqs. (13) from LHCb-PUB-2016-021
        x = (1 - 2*eff)*(pass_sumErr*pass_sumErr)
        y = (eff*eff)*(tot_sumErr*tot_sumErr)

        effErr = np.sqrt(abs(x + y)/(tot_sum**2))

        return eff, effErr

    def virtual_cut(self, cut):
        
        self.all_data['physical_virtual'] = self.all_data['physical'].copy()
        self.all_data['processed_virtual'] = self.all_data['processed'].copy()

        gen_tot_val = self.all_data['physical'].shape[0]
        gen_tot_err = np.sqrt(gen_tot_val)

        if cut=='pass_stripping': # couldnt fix bug with query, this is work around
            self.all_data['physical_virtual'].reset_index(drop=True, inplace=True)
            self.all_data['processed_virtual'].reset_index(drop=True, inplace=True)
            passes = np.where(self.all_data['physical_virtual']['pass_stripping']>0.5)
            self.all_data['physical_virtual'] = self.all_data['physical_virtual'].iloc[passes]
        else:
            self.all_data['physical_virtual'] = self.all_data['physical_virtual'].query(cut)
        index = self.all_data['physical_virtual'].index
        
        if not self.turn_off_processing:
            self.all_data['processed_virtual'] = self.all_data['processed_virtual'].iloc[index]
            self.all_data['processed_virtual'] = self.all_data['processed_virtual'].reset_index(drop=True)

        self.all_data['physical_virtual'] = self.all_data['physical_virtual'].reset_index(drop=True)
        pass_tot_val = self.all_data['physical_virtual'].shape[0]
        pass_tot_err = np.sqrt(pass_tot_val)

        eff, effErr = self.getBinomialEff(pass_tot_val, gen_tot_val,
                                     pass_tot_err, gen_tot_err)


        print(f'INFO cut(): {cut}, eff:{eff:.4f}+-{effErr:.4f}')


    def cut(self, cut):
        
        gen_tot_val = self.all_data['physical'].shape[0]
        gen_tot_err = np.sqrt(gen_tot_val)

        if cut=='pass_stripping': # couldnt fix bug with query, this is work around
            self.all_data['physical'].reset_index(drop=True, inplace=True)
            self.all_data['processed'].reset_index(drop=True, inplace=True)
            passes = np.where(self.all_data['physical']['pass_stripping']>0.5)
            self.all_data['physical'] = self.all_data['physical'].iloc[passes]
        else:
            self.all_data['physical'] = self.all_data['physical'].query(cut)
        index = self.all_data['physical'].index
        
        if not self.turn_off_processing:
            self.all_data['processed'] = self.all_data['processed'].iloc[index]
            self.all_data['processed'] = self.all_data['processed'].reset_index(drop=True)

        self.all_data['physical'] = self.all_data['physical'].reset_index(drop=True)
        pass_tot_val = self.all_data['physical'].shape[0]
        pass_tot_err = np.sqrt(pass_tot_val)

        eff, effErr = self.getBinomialEff(pass_tot_val, gen_tot_val,
                                     pass_tot_err, gen_tot_err)


        print(f'INFO cut(): {cut}, eff:{eff:.4f}+-{effErr:.4f}')

    def getEff(self, cut):

        if isinstance(cut, dict):
            cut_string = ''
            for cut_idx, cut_i in enumerate(list(cut.keys())):
                if cut_idx > 0:
                    cut_string += ' & '
                if cut_i == 'extra_cut':
                    cut_string += f'{cut[cut_i]}'
                else:
                    cut_string += f'{cut_i}{cut[cut_i]}'
            cut = cut_string   

        gen_tot_val = self.all_data['physical'].shape[0]
        gen_tot_err = np.sqrt(gen_tot_val)

        if 'B_plus_ENDVERTEX_NDOF' in cut:
            if 'B_plus_ENDVERTEX_NDOF' not in list(self.all_data['physical'].keys()):
                self.all_data['physical']['B_plus_ENDVERTEX_NDOF'] = 3

        cut_array = self.all_data['physical'].query(cut)
        pass_tot_val = cut_array.shape[0]
        pass_tot_err = np.sqrt(pass_tot_val)

        eff, effErr = self.getBinomialEff(pass_tot_val, gen_tot_val,
                                     pass_tot_err, gen_tot_err)


        print(f'INFO getEff(): {eff:.4f}+-{effErr:.4f}')

        return eff, effErr

    def save_to_file(self, filename):
        
        write_df_to_root(self.all_data["physical"], filename)

    def add_branch_to_physical(self, name, values):

        self.all_data["physical"][name] = values
    
    def add_branch_and_process(self, name, recipe):

        self.all_data["physical"][name] = self.all_data["physical"].eval(recipe)
        physical_data = self.all_data["physical"]
        column = name

        data_array = np.asarray(physical_data[column]).copy()
        transformer_i = Transformer()
        transformer_i.fit(data_array, column)
        self.Transformers[column] = transformer_i

        self.all_data["processed"][column] = self.Transformers[column].process(
        np.asarray(physical_data[column]).copy()
        )


    def convert_value_to_processed(self, name, value):
        return self.Transformers[name].process(value)
    
    def plot(self,filename, variables=None,save_vars=False):

        if variables == None:
            variables = list(self.all_data["physical"].keys())

        if save_vars:
            vars_to_save = {}
            vars_to_save["physical"] = {}
            vars_to_save["processed"] = {}

        with PdfPages(filename) as pdf:

            for variable in variables:
                
                try:
                    plt.figure(figsize=(10,8))

                    plt.subplot(2,2,1)
                    plt.title(variable)
                    plt.hist(self.all_data["physical"][variable], bins=50, density=True, histtype='step')
                    
                    plt.subplot(2,2,2)
                    plt.title(f'{variable} processed')
                    plt.hist(self.all_data["processed"][variable], bins=50, density=True, histtype='step', range=[-1,1])

                    plt.subplot(2,2,3)
                    plt.hist(self.all_data["physical"][variable], bins=50, density=True, histtype='step')
                    plt.yscale('log')
                    
                    plt.subplot(2,2,4)
                    plt.hist(self.all_data["processed"][variable], bins=50, density=True, histtype='step', range=[-1,1])
                    plt.yscale('log')

                    pdf.savefig(bbox_inches="tight")
                    plt.close()

                    if save_vars:
                        vars_to_save["physical"][variable] = np.asarray(self.all_data["physical"][variable])
                        vars_to_save["processed"][variable] = np.asarray(self.all_data["processed"][variable])

                except:
                    pdf.savefig(bbox_inches="tight")
                    plt.close()
                    pass
        
        if save_vars:
            with open(f'{filename[:-3]}.pickle', 'wb') as handle:
                pickle.dump(vars_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_file_names(self):
        return self.filenames


def convert_branches_to_RK_branch_names(columns, conversions):
    new_columns = []
    for column in columns:
        converted = False

        for conversion in conversions.keys():

            if conversion in column:
                if conversion == "MOTHER":
                    if column[:6] == "MOTHER":
                        new_column = conversions[conversion]+column[6:]
                    elif "_MC_" in column:
                        continue
                    else:
                        new_column = column.replace(conversion, conversions[conversion])
                else:
                    new_column = column.replace(conversion, conversions[conversion])

                new_columns.append(new_column)
                converted = True
                break

        if not converted:
            new_columns.append(column)

    return new_columns

def load_data(path, equal_sizes=True, N=-1, transformers=None, convert_to_RK_branch_names=False, conversions=None, turn_off_processing=False,avoid_physics_variables=False):

    if isinstance(path, list):
        for i in range(0, len(path)):
            if i == 0:
                if path[i][-5:] == '.root':
                    file = uproot.open(path[i])['DecayTree']
                    if N != -1:
                        events = file.arrays(library='pd', entry_stop=N)
                    else:
                        events = file.arrays(library='pd')
                    if convert_to_RK_branch_names:
                        if conversions == None:
                            print("must declare conversions, quitting...")
                            quit()
                        new_columns = convert_branches_to_RK_branch_names(events.columns, conversions)
                        events.columns = new_columns
  
                else:
                    events = pd.read_csv(path[i])
                    if equal_sizes and N == -1:
                        N = events.shape[0]
                    elif equal_sizes:
                        events = events.head(N)
            else:
                if path[i][-5:] == '.root':
                    file = uproot.open(path[i])['DecayTree']
                    if N != -1:
                        events_i = file.arrays(library='pd', entry_stop=N)
                    else:
                        events_i = file.arrays(library='pd')
                    if convert_to_RK_branch_names:
                        if conversions == None:
                            print("must declare conversions, quitting...")
                            quit()
                        new_columns = convert_branches_to_RK_branch_names(events_i.columns, conversions)
                        events_i.columns = new_columns
                else:
                    events_i = pd.read_csv(path[i])
                    if equal_sizes:
                        events_i = events_i.head(N)
                events = pd.concat([events, events_i], axis=0)
            events["file"] = np.asarray(np.ones(events.shape[0]) * i).astype("int")
    else:
        events = pd.read_csv(path)
        path = [path]

    events = events.loc[:, ~events.columns.str.contains("^Unnamed")]

    events_dataset = dataset(filenames=path, transformers=transformers)
    events_dataset.fill(events, turn_off_processing, avoid_physics_variables=avoid_physics_variables)

    return events_dataset
