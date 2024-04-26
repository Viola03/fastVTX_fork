import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from fast_vertex_quality.tools.config import rd, read_definition


class NoneError(Exception):
    pass


rd.targets = [
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
]


rd.conditions = [
    "K_Kst_PX",
    "K_Kst_PY",
    "K_Kst_PZ",
    "e_minus_PX",
    "e_minus_PY",
    "e_minus_PZ",
    "e_plus_PX",
    "e_plus_PY",
    "e_plus_PZ",
    "nTracks",
    "nSPDHits",
]


class dataset:

    def __init__(self, generated=False):

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

    def fill(self, data, processed=False):

        if not isinstance(data, pd.DataFrame):
            raise NoneError("Dataset must be a pd.dataframe.")

        if not processed:
            
            data = data.add_suffix("_physical_data")
            self.processed_data = self.pre_process(data)
            self.physical_data = data

        elif processed:

            data = data.add_suffix("_processed_data")
            self.processed_data = data
            self.physical_data = self.post_process(data)

        self.physics_variables = self.produce_physics_variables()

        self.all_data = pd.concat([self.physical_data, self.processed_data, self.physics_variables], axis=1)

    def get_branches(self, branches, processed):
        
        if not isinstance(branches, list):
            branches = [branches]

        if processed:
            branches = [s + "_processed_data" for s in branches]
            output = self.all_data[branches]
            output.columns = [s.replace("_processed_data", "") for s in output.columns]

        else:
            branches = [s + "_physical_data" for s in branches]
            output = self.all_data[branches]
            output.columns = [s.replace("_physical_data", "") for s in output.columns]

        return output


    def post_process(self, processed_data):

        df = {}

        for column in list(processed_data.keys()):
            df[column.replace('_processed_data','_physical_data')] = processed_data[column]

        for column in rd.targets+rd.conditions:
            
            column = column+"_processed_data"
            
            df[column.replace('_processed_data','_physical_data')] = np.squeeze(
                rd.QuantileTransformers[column].inverse_transform(
                    np.asarray(df[column.replace('_processed_data','_physical_data')]).reshape(-1, 1)
                )
            )

            if column in [s + "_processed_data" for s in self.log_columns]:
                df[column.replace('_processed_data','_physical_data')] = np.power(10, df[column.replace('_processed_data','_physical_data')])
            elif column in [s + "_processed_data" for s in self.one_minus_log_columns]:
                df[column.replace('_processed_data','_physical_data')] = np.power(10, df[column.replace('_processed_data','_physical_data')])
                df[column.replace('_processed_data','_physical_data')] = 1.0 - df[column.replace('_processed_data','_physical_data')]

        return pd.DataFrame.from_dict(df)



    def pre_process(self, physical_data):
       
        df = {}
        
        for column in list(physical_data.keys()):

            if column in [s + "_physical_data" for s in self.log_columns]:
                df[column.replace("_physical_data", "_processed_data")] = np.log10(physical_data[column])
            elif column in [s + "_physical_data" for s in self.one_minus_log_columns]:
                df[column.replace("_physical_data", "_processed_data")] = np.log10(1.0 - physical_data[column])
            else:
                df[column.replace("_physical_data", "_processed_data")] = physical_data[column]

        if self.generated == False:
            rd.normalisation_constants = {}

        rd.QuantileTransformers = {}

        for column in rd.targets+rd.conditions:
        # for column in list(physical_data.keys()):
            
            column = column+"_processed_data"

            qt = QuantileTransformer(n_quantiles=50, output_distribution="normal")

            rd.QuantileTransformers[column] = qt.fit(
                np.asarray(df[column]).reshape(-1, 1)
            )

            df[column] = np.squeeze(
                rd.QuantileTransformers[column].transform(
                    np.asarray(df[column]).reshape(-1, 1)
                )
            )

        return pd.DataFrame.from_dict(df)


    def produce_physics_variables(self):

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
            low=0, high=9, size=np.shape(self.physical_data["K_Kst_PX_physical_data"])[0]
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
        PX = self.physical_data["e_plus_PX_physical_data"] + self.physical_data["e_minus_PX_physical_data"]
        PY = self.physical_data["e_plus_PY_physical_data"] + self.physical_data["e_minus_PY_physical_data"]
        PZ = self.physical_data["e_plus_PZ_physical_data"] + self.physical_data["e_minus_PZ_physical_data"]

        physics_variables["q2"] = (PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6

        df = pd.DataFrame.from_dict(physics_variables)

        df.columns = [s + "_processed_data" for s in df.columns]

        for column in list(df.keys()):
            df[column.replace("_processed_data", "_physical_data")] = df[column]
        
        return df


def load_data(path):

    events = pd.read_csv(path)

    events_dataset = dataset(generated=False)
    events_dataset.fill(events, processed=False)

    return events_dataset