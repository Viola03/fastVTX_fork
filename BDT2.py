from fast_vertex_quality.tools.config import read_definition, rd

import fast_vertex_quality.tools.data_loader as data_loader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fast_vertex_quality.tools.config import read_definition, rd

import tensorflow as tf

from sklearn.ensemble import GradientBoostingClassifier
import fast_vertex_quality.tools.plotting as plotting
import pickle
from matplotlib.backends.backend_pdf import PdfPages

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
    # "B_P",
    # "B_PT",
    "missing_B_P",
    "missing_B_PT",
    # "delta_0_P",
    # "delta_0_PT",
    # "delta_1_P",
    # "delta_1_PT",
    # "delta_2_P",
    # "delta_2_PT",
    # "m_01",
    # "m_02",
    # "m_12",
]

decoder = tf.keras.models.load_model("save_state/decoder.h5")
latent_dim = 8
transformers = pickle.load(open("save_state/QuantileTransformers.pkl", "rb"))

event_loader_MC = data_loader.load_data(
    [
        "datasets/Kee_2018_truthed_more_vars.csv",
    ],
    part_reco=[-1],
    transformers=transformers,
)
event_loader_MC.select_randomly(Nevents=50000)
events_MC = event_loader_MC.get_branches(rd.targets + ["kFold"], processed=False)

event_loader_data = data_loader.load_data(
    [
        "datasets/B2Kee_2018_CommonPresel.csv",
    ],
    part_reco=[-1],
    transformers=transformers,
)
event_loader_data.select_randomly(Nevents=50000)
events_data = event_loader_data.get_branches(rd.targets + ["kFold"], processed=False)


# event_loader_MC = data_loader.load_data("datasets/Kee_2018_truthed.csv")
# event_loader_data = data_loader.load_data("datasets/B2Kee_2018_CommonPresel.csv")

# events_MC, throw, throw = event_loader_MC.get_physical_data()
# events_data, throw, throw = event_loader_data.get_physical_data()

# events_MC_pp, throw, throw = event_loader_MC.get_processed_data()
# events_data_pp, throw, throw = event_loader_data.get_processed_data()

# events_data = events_data.query("B_plus_IPCHI2_OWNPV>0")

# events_data_physics_variables = event_loader_data.get_physics_variables()
# events_data = pd.concat([events_data, events_data_physics_variables], axis=1)

# events_MC_physics_variables = event_loader_MC.get_physics_variables()
# events_MC = pd.concat([events_MC, events_MC_physics_variables], axis=1)


BDTs = {}

for kFold in range(10):

    print(f"Training kFold {kFold}...")

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

    events_data_i = events_data.query(f"kFold!={kFold}")
    events_MC_i = events_MC.query(f"kFold!={kFold}")

    events_data_i = events_data_i.drop("kFold", axis=1)
    events_MC_i = events_MC_i.drop("kFold", axis=1)

    real_training_data = np.squeeze(np.asarray(events_MC_i[rd.targets]))

    fake_training_data = np.squeeze(np.asarray(events_data_i[rd.targets]))

    size = 25000
    real_training_data = real_training_data[:size]
    fake_training_data = fake_training_data[:size]

    real_training_labels = np.ones(size)

    fake_training_labels = np.zeros(size)

    total_training_data = np.concatenate((real_training_data, fake_training_data))

    total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

    clf.fit(total_training_data, total_training_labels)

    BDTs[kFold] = clf

    break


for kFold in range(10):

    events_data_i = events_data.query(f"kFold=={kFold}")
    events_MC_i = events_MC.query(f"kFold=={kFold}")

    events_data_i = events_data_i.drop("kFold", axis=1)
    events_MC_i = events_MC_i.drop("kFold", axis=1)

    real_testing_data = np.squeeze(np.asarray(events_MC_i[rd.targets]))

    fake_testing_data = np.squeeze(np.asarray(events_data_i[rd.targets]))

    size = 25000
    real_testing_data = real_testing_data[:size]
    fake_testing_data = fake_testing_data[:size]

    out_real = clf.predict_proba(real_testing_data)

    out_fake = clf.predict_proba(fake_testing_data)

    ########################
    event_loader_gen = data_loader.load_data(
        [
            "datasets/Kee_2018_truthed_more_vars.csv",
        ],
        part_reco=[-1],
        transformers=transformers,
    )
    event_loader_gen.select_randomly(Nevents=10000)
    events_gen = event_loader_gen.get_branches(rd.conditions, processed=True)

    events_gen = np.asarray(events_gen)

    gen_noise = np.random.normal(0, 1, (10000, latent_dim))
    images = np.squeeze(decoder.predict([gen_noise, events_gen]))

    event_loader_gen.fill_target(images)

    events_gen_query = event_loader_gen.get_branches(
        rd.targets + ["kFold"], processed=False
    )
    events_gen_query = np.squeeze(np.asarray(events_gen_query[rd.targets]))

    out_gen = clf.predict_proba(events_gen_query)

    ################################
    event_loader_prc = data_loader.load_data(
        [
            "datasets/Kstee_2018_truthed_more_vars.csv",
        ],
        part_reco=[1],
        transformers=transformers,
    )
    event_loader_prc.select_randomly(Nevents=20000)
    events_prc_targets = event_loader_prc.get_branches(rd.targets, processed=False)
    events_prc_query = np.squeeze(np.asarray(events_prc_targets[rd.targets])[:10000])
    out_prc = clf.predict_proba(events_prc_query)

    events_prc_conditions = event_loader_prc.get_branches(rd.conditions, processed=True)
    events_prc_conditions = np.asarray(events_prc_conditions)[-10000:]
    gen_noise = np.random.normal(0, 1, (10000, latent_dim))
    images = np.squeeze(decoder.predict([gen_noise, events_prc_conditions]))
    images = np.concatenate((images, images), axis=0)
    event_loader_prc.fill_target(images)
    events_gen_prc_query = event_loader_prc.get_branches(rd.targets, processed=False)
    events_gen_prc_query = np.squeeze(
        np.asarray(events_prc_targets[rd.targets])[-10000:]
    )
    out_gen_prc = clf.predict_proba(events_gen_prc_query)

    ################################

    with PdfPages(f"BDT.pdf") as pdf:

        samples = {}

        samples["sig - MC"] = {}
        samples["sig - MC"]["data"] = out_real[:, 1]
        samples["sig - MC"]["c"] = "tab:blue"

        samples["bkg - umsb"] = {}
        samples["bkg - umsb"]["data"] = out_fake[:, 1]
        samples["bkg - umsb"]["c"] = "tab:red"

        samples["sig - gen"] = {}
        samples["sig - gen"]["data"] = out_gen[:, 1]
        samples["sig - gen"]["c"] = "tab:green"

        samples["prc - MC"] = {}
        samples["prc - MC"]["data"] = out_prc[:, 1]
        samples["prc - MC"]["c"] = "tab:purple"

        samples["prc - gen"] = {}
        samples["prc - gen"]["data"] = out_gen_prc[:, 1]
        samples["prc - gen"]["c"] = "k"

        figures = [
            ["sig - MC", "bkg - umsb"],
            ["sig - MC", "bkg - umsb", "sig - gen"],
            ["sig - MC", "bkg - umsb", "sig - gen", "prc - MC", "prc - gen"],
        ]

        for figure_config in figures:

            hists = []
            colours = []
            for plot_i in figure_config:
                hists.append(samples[plot_i]["data"])
                colours.append(samples[plot_i]["c"])

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)

            plt.hist(
                hists,
                bins=50,
                color=colours,
                alpha=0.25,
                label=figure_config,
                density=True,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                hists,
                bins=50,
                color=colours,
                density=True,
                histtype="step",
                range=[0, 1],
            )
            # plt.legend(loc="upper left")
            plt.xlabel(f"BDT output")
            plt.yscale("log")
            plt.subplot(1, 2, 2)
            plt.hist(
                hists,
                bins=50,
                color=colours,
                alpha=0.25,
                label=figure_config,
                density=True,
                histtype="stepfilled",
                range=[0, 1],
            )
            plt.hist(
                hists,
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

    break
