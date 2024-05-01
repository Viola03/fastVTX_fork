import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from fast_vertex_quality.tools.config import rd, read_definition


class NoneError(Exception):
    pass


class dataset:

    def __init__(self, generated=False, transformers=None):

        self.generated = generated

        self.log_columns = [
            "B_plus_FDCHI2_OWNPV",
            "K_Kst_IPCHI2_OWNPV",
            "e_minus_IPCHI2_OWNPV",
            "e_plus_IPCHI2_OWNPV",
            "K_Kst_PZ",
            "e_minus_PZ",
            "e_plus_PZ",
        ]

        self.one_minus_log_columns = ["B_plus_DIRA_OWNPV"]
        self.QuantileTransformers = transformers

    def select_randomly(self, Nevents):

        # print(self.all_data)
        self.all_data = self.all_data.sample(n=Nevents)
        # print(self.all_data)

    def fill(self, data, processed=False):

        if not isinstance(data, pd.DataFrame):
            raise NoneError("Dataset must be a pd.dataframe.")

        if not processed:

            data = data.add_suffix("_physical_data")
            self.processed_data = self.pre_process(data)
            self.physical_data = data
            self.physics_variables = self.produce_physics_variables(processed=False)
            self.all_data = pd.concat(
                [self.physical_data, self.processed_data, self.physics_variables],
                axis=1,
            )

        elif processed:

            # print("no longer used... quitting...")
            self.all_data = data

            # data = data.add_suffix("_processed_data")
            # self.processed_data = data
            # self.physical_data = self.post_process(data)
            # print(self.physical_data, list(self.physical_data.keys()))
            # print(list(self.processed_data.keys()))
            # quit()
            # self.physics_variables = self.produce_physics_variables()

    def get_transformers(self):
        return self.QuantileTransformers

    def apply_cut(self, cut):

        self.all_data = self.all_data.query(cut)

        # self.processed_data = self.processed_data.add_suffix('_processed_data')
        # self.physics_variables = self.physics_variables.add_suffix('_physics_variables')
        # self.physical_data = self.physical_data.add_suffix('_physical_data')

        # all_vars = pd.concat([self.physical_data, self.physics_variables, self.processed_data], axis=1)

        # try:
        # 	all_vars = all_vars.query(cut)
        # except:
        # 	print('cut query failed, try adding _processed_data, _physics_variables, or _physical_data?')
        # 	quit()

        # self.processed_data = all_vars[[col for col in all_vars.columns if '_processed_data' in col]]
        # self.physics_variables = all_vars[[col for col in all_vars.columns if '_physics_variables' in col]]
        # self.physical_data = all_vars[[col for col in all_vars.columns if '_physical_data' in col]]

        # self.processed_data.columns = [col.replace('_processed_data','') for col in self.processed_data.columns]
        # self.physics_variables.columns = [col.replace('_physics_variables','') for col in self.physics_variables.columns]
        # self.physical_data.columns = [col.replace('_physical_data','') for col in self.physical_data.columns]

    def fill_target(self, data):

        df = pd.DataFrame(data, columns=[s + "_processed_data" for s in rd.targets])

        for column in rd.targets:

            df[column + "_physical_data"] = np.squeeze(
                self.QuantileTransformers[column + "_processed_data"].inverse_transform(
                    np.asarray(df[column + "_processed_data"]).reshape(-1, 1)
                )
            )

            if column in self.log_columns:
                df[column + "_physical_data"] = np.power(
                    10, df[column + "_physical_data"]
                )
            elif column in self.one_minus_log_columns:
                df[column + "_physical_data"] = np.power(
                    10, df[column + "_physical_data"]
                )
                df[column + "_physical_data"] = 1.0 - df[column + "_physical_data"]

        for key in list(df.keys()):
            self.all_data[key] = np.asarray(df[key])

    def get_branches(self, branches, processed):

        if not isinstance(branches, list):
            branches = [branches]

        if processed:
            branches = [s + "_processed_data" for s in branches]

            missing = list(set(branches).difference(set(list(self.all_data.keys()))))
            branches = list(set(branches).intersection(set(list(self.all_data.keys()))))

            if len(missing) > 0:
                print(f"missing branches: {missing}\n")

            output = self.all_data[branches]
            output.columns = [s.replace("_processed_data", "") for s in output.columns]

        else:
            branches = [s + "_physical_data" for s in branches]

            missing = list(set(branches).difference(set(list(self.all_data.keys()))))
            branches = list(set(branches).intersection(set(list(self.all_data.keys()))))

            if len(missing) > 0:
                print(f"missing branches: {missing}\n")

            output = self.all_data[branches]
            output.columns = [s.replace("_physical_data", "") for s in output.columns]

        return output

    def pre_process(self, physical_data):

        df = {}

        for column in list(physical_data.keys()):

            if column in [s + "_physical_data" for s in self.log_columns]:
                df[column.replace("_physical_data", "_processed_data")] = np.log10(
                    physical_data[column]
                )
            elif column in [s + "_physical_data" for s in self.one_minus_log_columns]:
                df[column.replace("_physical_data", "_processed_data")] = np.log10(
                    1.0 - physical_data[column]
                )
            else:
                df[column.replace("_physical_data", "_processed_data")] = physical_data[
                    column
                ]

        if self.generated == False:
            rd.normalisation_constants = {}

        fresh_transformers = False
        if self.QuantileTransformers == None:
            fresh_transformers = True
        if fresh_transformers:
            self.QuantileTransformers = {}

        for column in rd.targets + rd.conditions:

            column = column + "_processed_data"

            if column not in list(df.keys()):
                continue

            if fresh_transformers:
                qt = QuantileTransformer(n_quantiles=50, output_distribution="normal")
                self.QuantileTransformers[column] = qt.fit(
                    np.asarray(df[column]).reshape(-1, 1)
                )

            df[column] = np.squeeze(
                self.QuantileTransformers[column].transform(
                    np.asarray(df[column]).reshape(-1, 1)
                )
            )

        return pd.DataFrame.from_dict(df)

    def produce_physics_variables(self, processed=True):

        physics_variables = {}

        physics_variables["K_Kst_P"] = np.sqrt(
            self.physical_data["K_Kst_PX_physical_data"] ** 2
            + self.physical_data["K_Kst_PY_physical_data"] ** 2
            + self.physical_data["K_Kst_PZ_physical_data"] ** 2
        )
        physics_variables["e_plus_P"] = np.sqrt(
            self.physical_data["e_plus_PX_physical_data"] ** 2
            + self.physical_data["e_plus_PY_physical_data"] ** 2
            + self.physical_data["e_plus_PZ_physical_data"] ** 2
        )
        physics_variables["e_minus_P"] = np.sqrt(
            self.physical_data["e_minus_PX_physical_data"] ** 2
            + self.physical_data["e_minus_PY_physical_data"] ** 2
            + self.physical_data["e_minus_PZ_physical_data"] ** 2
        )
        physics_variables["kFold"] = np.random.randint(
            low=0,
            high=9,
            size=np.shape(self.physical_data["K_Kst_PX_physical_data"])[0],
        )

        electron_mass = 0.51099895000 * 1e-3

        PE = np.sqrt(
            electron_mass**2
            + self.physical_data["e_plus_PX_physical_data"] ** 2
            + self.physical_data["e_plus_PY_physical_data"] ** 2
            + self.physical_data["e_plus_PZ_physical_data"] ** 2
        ) + np.sqrt(
            electron_mass**2
            + self.physical_data["e_minus_PX_physical_data"] ** 2
            + self.physical_data["e_minus_PY_physical_data"] ** 2
            + self.physical_data["e_minus_PZ_physical_data"] ** 2
        )
        PX = (
            self.physical_data["e_plus_PX_physical_data"]
            + self.physical_data["e_minus_PX_physical_data"]
        )
        PY = (
            self.physical_data["e_plus_PY_physical_data"]
            + self.physical_data["e_minus_PY_physical_data"]
        )
        PZ = (
            self.physical_data["e_plus_PZ_physical_data"]
            + self.physical_data["e_minus_PZ_physical_data"]
        )

        physics_variables["q2"] = (PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6

        df = pd.DataFrame.from_dict(physics_variables)

        df.columns = [s + "_physical_data" for s in df.columns]

        if not processed:

            for column in list(df.keys()):

                qt = QuantileTransformer(n_quantiles=50, output_distribution="normal")

                self.QuantileTransformers[column.replace("_physical_data", "")] = (
                    qt.fit(np.asarray(df[column]).reshape(-1, 1))
                )

                df[column.replace("_physical_data", "_processed_data")] = np.squeeze(
                    self.QuantileTransformers[
                        column.replace("_physical_data", "")
                    ].transform(np.asarray(df[column]).reshape(-1, 1))
                )
        else:

            for column in list(df.keys()):

                df[column.replace("_physical_data", "_processed_data")] = np.squeeze(
                    self.QuantileTransformers[
                        column.replace("_physical_data", "")
                    ].transform(np.asarray(df[column]).reshape(-1, 1))
                )

        return df


def load_data(path, part_reco, equal_sizes=True, N=-1, transformers=None):

    if isinstance(path, list):
        if not isinstance(part_reco, list):
            print("path is list, part_reco must be too, quitting..")
            quit()
        if len(part_reco) != len(path):
            print("path and part_reco must have same lengths, quitting..")
            quit()
        for i in range(0, len(path)):
            if i == 0:
                events = pd.read_csv(path[i])
                events["part_reco"] = part_reco[i]
                if equal_sizes and N == -1:
                    N = events.shape[0]
                elif equal_sizes:
                    events = events.sample(n=N)
            else:
                events_i = pd.read_csv(path[i])
                events_i["part_reco"] = part_reco[i]
                if equal_sizes:
                    events_i = events_i.sample(n=N)
                events = pd.concat([events, events_i], axis=0)
    else:
        events = pd.read_csv(path)
        events["part_reco"] = part_reco

    events_dataset = dataset(generated=False, transformers=transformers)
    events_dataset.fill(events, processed=False)

    return events_dataset
