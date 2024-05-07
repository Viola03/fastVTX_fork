import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from fast_vertex_quality.tools.config import rd, read_definition


def produce_physics_variables(data):

    physics_variables = {}

    physics_variables["K_Kst_P"] = np.sqrt(
        data["K_Kst_PX"] ** 2 + data["K_Kst_PY"] ** 2 + data["K_Kst_PZ"] ** 2
    )
    physics_variables["e_plus_P"] = np.sqrt(
        data["e_plus_PX"] ** 2 + data["e_plus_PY"] ** 2 + data["e_plus_PZ"] ** 2
    )
    physics_variables["e_minus_P"] = np.sqrt(
        data["e_minus_PX"] ** 2 + data["e_minus_PY"] ** 2 + data["e_minus_PZ"] ** 2
    )
    physics_variables["kFold"] = np.random.randint(
        low=0,
        high=9,
        size=np.shape(data["K_Kst_PX"])[0],
    )

    electron_mass = 0.51099895000 * 1e-3

    PE = np.sqrt(
        electron_mass**2
        + data["e_plus_PX"] ** 2
        + data["e_plus_PY"] ** 2
        + data["e_plus_PZ"] ** 2
    ) + np.sqrt(
        electron_mass**2
        + data["e_minus_PX"] ** 2
        + data["e_minus_PY"] ** 2
        + data["e_minus_PZ"] ** 2
    )
    PX = data["e_plus_PX"] + data["e_minus_PX"]
    PY = data["e_plus_PY"] + data["e_minus_PY"]
    PZ = data["e_plus_PZ"] + data["e_minus_PZ"]

    physics_variables["q2"] = (PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6

    df = pd.DataFrame.from_dict(physics_variables)

    return df


class NoneError(Exception):
    pass


class Transformer:

    def __init__(self):

        self.log_columns = [
            "B_plus_FDCHI2_OWNPV",
            "K_Kst_IPCHI2_OWNPV",
            "e_minus_IPCHI2_OWNPV",
            "e_plus_IPCHI2_OWNPV",
            "K_Kst_PZ",
            "e_minus_PZ",
            "e_plus_PZ",
            "B_plus_ENDVERTEX_CHI2",
            "B_plus_IPCHI2_OWNPV",
        ]

        self.one_minus_log_columns = ["B_plus_DIRA_OWNPV"]

    def fit(self, data_raw, column):

        self.column = column

        data = data_raw.copy()

        if column in self.log_columns:
            data = np.log10(data)
        elif column in self.one_minus_log_columns:
            data = np.log10(1.0 - data)

        self.min = np.amin(data)
        self.max = np.amax(data)

    def process(self, data_raw):

        data = data_raw.copy()

        if self.column in self.log_columns:
            data = np.log10(data)
        elif self.column in self.one_minus_log_columns:
            data = np.log10(1.0 - data)

        data = data - self.min
        data = data / (self.max - self.min)
        data *= 2
        data += -1

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

        return data


class dataset:

    def __init__(self, transformers=None):

        self.Transformers = transformers
        self.all_data = {"processed": None, "physical": None}

    def fill(self, data):

        if not isinstance(data, pd.DataFrame):
            raise NoneError("Dataset must be a pd.dataframe.")

        self.all_data["physical"] = data
        self.physics_variables = produce_physics_variables(self.all_data["physical"])
        self.all_data["physical"] = pd.concat(
            (self.all_data["physical"], self.physics_variables), axis=1
        )

        self.all_data["physical"] = self.all_data["physical"].loc[
            :, ~self.all_data["physical"].columns.str.contains("^Unnamed")
        ]

        self.all_data["processed"] = self.pre_process(self.all_data["physical"])

    def post_process(self, processed_data):

        df = {}

        for column in list(processed_data.keys()):

            df[column] = self.Transformers[column].unprocess(
                np.asarray(processed_data[column]).copy()
            )

        return pd.DataFrame.from_dict(df)

    def fill_target(self, processed_data):

        df_processed = pd.DataFrame(processed_data, columns=rd.targets)

        df_physical = self.post_process(df_processed)

        for column in rd.targets:
            self.all_data["processed"][column] = np.asarray(df_processed[column])
            self.all_data["physical"][column] = np.asarray(df_physical[column])

    def select_randomly(self, Nevents):

        idx = np.random.choice(
            self.all_data["processed"].shape[0], replace=False, size=Nevents
        )

        self.all_data["processed"] = self.all_data["processed"].iloc[idx]
        self.all_data["physical"] = self.all_data["physical"].iloc[idx]

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
                print(f"missing branches: {missing}\n")

            output = self.all_data["processed"][branches]

        else:
            missing = list(
                set(branches).difference(set(list(self.all_data["physical"].keys())))
            )
            branches = list(
                set(branches).intersection(set(list(self.all_data["physical"].keys())))
            )

            if len(missing) > 0:
                print(f"missing branches: {missing}\n")

            output = self.all_data["physical"][branches]

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

            if fresh_transformers:
                data_array = np.asarray(physical_data[column]).copy()
                transformer_i = Transformer()
                transformer_i.fit(data_array, column)
                self.Transformers[column] = transformer_i

            df[column] = self.Transformers[column].process(
                np.asarray(physical_data[column]).copy()
            )

        return pd.DataFrame.from_dict(df)

    def get_transformers(self):
        return self.Transformers


def load_data(path, equal_sizes=True, N=-1, transformers=None):

    if isinstance(path, list):
        for i in range(0, len(path)):
            if i == 0:
                events = pd.read_csv(path[i])
                if equal_sizes and N == -1:
                    N = events.shape[0]
                elif equal_sizes:
                    events = events.sample(n=N)
            else:
                events_i = pd.read_csv(path[i])
                if equal_sizes:
                    events_i = events_i.sample(n=N)
                events = pd.concat([events, events_i], axis=0)
    else:
        events = pd.read_csv(path)

    events_dataset = dataset(transformers=transformers)
    events_dataset.fill(events)

    return events_dataset
