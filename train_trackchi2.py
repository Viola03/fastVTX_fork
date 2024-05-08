from fast_vertex_quality.tools.config import read_definition, rd

import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

_EPSILON = K.epsilon()

import glob
import os
import time

import fast_vertex_quality.tools.plotting as plotting

# import fast_vertex_quality.tools.data_loader as data_loader
from fast_vertex_quality.tools.training import train_step
from fast_vertex_quality.models.conditional_VAE import VAE_builder

import fast_vertex_quality.tools.new_data_loader as data_loader

print(tf.__version__)

import pandas as pd

rd.targets = [
    "e_plus_TRACK_CHI2NDOF",
]

rd.conditions = [
    "e_plus_PX",
    "e_plus_PY",
    "e_plus_PZ",
    "e_plus_P",
    "e_plus_PT",
    "e_plus_eta",
]


# X_train_data_loader = data_loader.load_data(
#     [
#         "datasets/Kee_2018_truthed_more_vars.csv",
#     ],
#     N=50000,
# )
# physical_X_train_data_loader = X_train_data_loader.get_physical()
# physical_X_train_data_loader_K_Kst = physical_X_train_data_loader.copy()
# physical_X_train_data_loader_e_plus = physical_X_train_data_loader.copy()
# physical_X_train_data_loader_e_minus = physical_X_train_data_loader.copy()

# for var in list(physical_X_train_data_loader_K_Kst.keys()):
#     if "K_Kst" in var:
#         var_eplus = var.replace("K_Kst", "e_plus")
#         physical_X_train_data_loader_K_Kst[var_eplus] = (
#             physical_X_train_data_loader_K_Kst[var]
#         )

# for var in list(physical_X_train_data_loader_e_minus.keys()):
#     if "e_minus" in var:
#         var_eplus = var.replace("e_minus", "e_plus")
#         physical_X_train_data_loader_e_minus[var_eplus] = (
#             physical_X_train_data_loader_e_minus[var]
#         )


# events = pd.concat(
#     (
#         physical_X_train_data_loader_e_plus,
#         physical_X_train_data_loader_e_minus,
#         physical_X_train_data_loader_K_Kst,
#     ),
#     axis=0,
# )
# events = events.sample(frac=1.0)
# events.to_csv(f"datasets/Kee_2018_truthed_more_vars_chi2.csv")
# print("saved")

kl_factor = 1.0
reco_factor = 100.0

batch_size = 50

target_dim = len(rd.targets)
conditions_dim = len(rd.conditions)
latent_dim = 1

cut_idx = target_dim

VAE = VAE_builder(
    E_architecture=[25, 25],
    D_architecture=[25, 25],
    target_dim=target_dim,
    conditions_dim=conditions_dim,
    latent_dim=latent_dim,
)
rd.encoder = VAE.encoder
rd.decoder = VAE.decoder
rd.vae = VAE.vae

rd.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

start = time.time()

iteration = -1

loss_list = np.empty((0, 3))

t0 = time.time()

save_interval = 10000
# save_interval = 5

X_train_data_loader = data_loader.load_data(
    [
        # "datasets/Kee_2018_truthed_more_vars.csv",
        "datasets/Kee_2018_truthed_more_vars_chi2.csv",
    ],
    N=50000,
)


# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt

# targets = X_train_data_loader.get_branches(rd.targets, processed=False)

# X_train_data_loader_star = data_loader.load_data(
#     [
#         "datasets/Kstee_2018_truthed_more_vars.csv",
#     ],
#     N=50000,
# )
# targets_str = X_train_data_loader_star.get_branches(rd.targets, processed=False)


# plt.figure(figsize=(12, 4))
# plt.subplot(2, 3, 1)
# plt.hist(
#     [
#         targets["K_Kst_TRACK_CHI2NDOF"],
#         targets["e_plus_TRACK_CHI2NDOF"],
#         targets["e_minus_TRACK_CHI2NDOF"],
#     ],
#     bins=50,
#     histtype="step",
# )
# plt.subplot(2, 3, 4)
# plt.hist(
#     [
#         targets_str["K_Kst_TRACK_CHI2NDOF"],
#         targets_str["e_plus_TRACK_CHI2NDOF"],
#         targets_str["e_minus_TRACK_CHI2NDOF"],
#     ],
#     bins=50,
#     histtype="step",
# )

# plt.subplot(2, 3, 2)
# plt.hist(
#     [
#         targets["K_Kst_TRACK_CHI2NDOF"],
#         targets_str["K_Kst_TRACK_CHI2NDOF"],
#     ],
#     bins=50,
#     histtype="step",
# )

# plt.subplot(2, 3, 3)
# plt.hist(
#     [
#         targets["e_plus_TRACK_CHI2NDOF"],
#         targets_str["e_plus_TRACK_CHI2NDOF"],
#     ],
#     bins=50,
#     histtype="step",
# )

# plt.subplot(2, 3, 5)
# plt.hist(
#     [
#         targets["e_minus_TRACK_CHI2NDOF"],
#         targets_str["e_minus_TRACK_CHI2NDOF"],
#     ],
#     bins=50,
#     histtype="step",
# )

# plt.savefig("test")
# quit()

transformers = X_train_data_loader.get_transformers()

current_MSE = 1.0

for epoch in range(int(1e30)):

    X_train_data_all_pp = X_train_data_loader.get_branches(
        rd.targets + rd.conditions, processed=True
    )

    X_train_data_all_pp = X_train_data_all_pp.sample(frac=1)
    X_train_data_all_pp = X_train_data_all_pp[rd.targets + rd.conditions]

    X_train_raw = np.asarray(X_train_data_all_pp)

    X_train = np.expand_dims(X_train_raw, 1).astype("float32")

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(X_train)
        .batch(batch_size, drop_remainder=True)
        .repeat(1)
    )

    for samples_for_batch in train_dataset:

        iteration += 1

        if iteration % 100 == 0:
            print("Iteration:", iteration)

        # toggle_kl_value = 1.0
        if iteration % 1000 == 0:  # annealing https://arxiv.org/pdf/1511.06349.pdf
            toggle_kl_value = 0.0
        elif iteration % 1000 < 500:
            toggle_kl_value += 1.0 / 500.0
        else:
            toggle_kl_value = 1.0
        toggle_kl = tf.convert_to_tensor(toggle_kl_value)

        kl_loss_np, reco_loss_np, reco_loss_np_raw = train_step(
            samples_for_batch,
            cut_idx,
            tf.convert_to_tensor(kl_factor),
            tf.convert_to_tensor(reco_factor),
            toggle_kl,
        )

        loss_list = np.append(
            loss_list, [[iteration, kl_loss_np, reco_loss_np]], axis=0
        )

        if iteration % save_interval == 0 and iteration > 0:

            t1 = time.time()

            total = t1 - t0

            plotting.loss_plot(loss_list, 1.0, 1.0, "LOSSES.png")

            gen_noise = np.random.normal(0, 1, (10000, latent_dim))

            X_test_data_loader = data_loader.load_data(
                [
                    # "datasets/Kee_2018_truthed_more_vars.csv",
                    "datasets/Kee_2018_truthed_more_vars_chi2.csv",
                ],
                transformers=transformers,
            )

            X_test_data_loader.select_randomly(Nevents=10000)
            train = X_test_data_loader.get_branches(
                rd.targets + rd.conditions, processed=False
            )
            X_test_conditions = X_test_data_loader.get_branches(
                rd.conditions, processed=True
            )
            X_test_conditions = X_test_conditions[rd.conditions]
            X_test_conditions = np.asarray(X_test_conditions)

            images = np.squeeze(rd.decoder.predict([gen_noise, X_test_conditions]))

            X_test_data_loader.fill_target(images)

            # train = X_train_data_loader.get_branches(
            #     rd.targets + rd.conditions, processed=False
            # )
            test = X_test_data_loader.get_branches(
                rd.targets + rd.conditions, processed=False
            )

            # plt.figure(figsize=(12, 8))
            # plt.subplot(2, 3, 1)
            # plt.hist2d(
            #     train["e_plus_TRACK_CHI2NDOF"],
            #     train["e_plus_PX"],
            #     bins=75,
            #     norm=LogNorm(),
            # )
            # plt.subplot(2, 3, 2)
            # plt.hist2d(
            #     train["e_plus_TRACK_CHI2NDOF"],
            #     train["e_plus_PY"],
            #     bins=75,
            #     norm=LogNorm(),
            # )
            # plt.subplot(2, 3, 3)
            # plt.hist2d(
            #     train["e_plus_TRACK_CHI2NDOF"],
            #     train["e_plus_PZ"],
            #     bins=75,
            #     norm=LogNorm(),
            # )
            # plt.subplot(2, 3, 4)
            # plt.hist2d(
            #     test["e_plus_TRACK_CHI2NDOF"],
            #     test["e_plus_PX"],
            #     bins=75,
            #     norm=LogNorm(),
            # )
            # plt.subplot(2, 3, 5)
            # plt.hist2d(
            #     test["e_plus_TRACK_CHI2NDOF"],
            #     test["e_plus_PY"],
            #     bins=75,
            #     norm=LogNorm(),
            # )
            # plt.subplot(2, 3, 6)
            # plt.hist2d(
            #     test["e_plus_TRACK_CHI2NDOF"],
            #     test["e_plus_PZ"],
            #     bins=75,
            #     norm=LogNorm(),
            # )
            # plt.savefig("test")
            # plt.close("all")

            plotting.plot(
                X_train_data_loader,
                X_test_data_loader,
                f"plots_{iteration}",
                Nevents=10000,
            )

            print("Saving complete...")

            for file in glob.glob("save_state/*"):
                os.remove(file)
            rd.decoder.save("save_state/track_chi2_decoder.h5")
            pickle.dump(
                transformers,
                open("save_state/track_chi2_QuantileTransformers.pkl", "wb"),
            )
