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

import fast_vertex_quality.tools.data_loader as data_loader

from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

now = datetime.now()
time_now = now.strftime("%H_%M_%S")

print(tf.__version__)

rd.targets = [
    "B_plus_ENDVERTEX_CHI2",
    "B_plus_IPCHI2_OWNPV",
    "B_plus_FDCHI2_OWNPV",
    "B_plus_DIRA_OWNPV",
    "K_Kst_IPCHI2_OWNPV",
    # "K_Kst_TRACK_CHI2NDOF",
    "e_minus_IPCHI2_OWNPV",
    # "e_minus_TRACK_CHI2NDOF",
    "e_plus_IPCHI2_OWNPV",
    # "e_plus_TRACK_CHI2NDOF",
]

rd.conditions = [
    "B_P",
    "B_PT",
    # "angle_K_Kst",
    # "angle_e_plus",
    # "angle_e_minus",
    "K_Kst_eta",
    "e_plus_eta",
    "e_minus_eta",
    "IP_B",
    "IP_K_Kst",
    "IP_e_plus",
    "IP_e_minus",
    "FD_B",
    "DIRA_B",
    # "delta_0_P",
    # "delta_0_PT",
    # "delta_1_P",
    # "delta_1_PT",
    # "delta_2_P",
    # "delta_2_PT",
    "K_Kst_TRACK_CHI2NDOF_gen",
    "e_minus_TRACK_CHI2NDOF_gen",
    "e_plus_TRACK_CHI2NDOF_gen",
]

kl_factor = 1.0
reco_factor = 5000.0

batch_size = 50

target_dim = len(rd.targets)
conditions_dim = len(rd.conditions)
latent_dim = 5

cut_idx = target_dim

VAE = VAE_builder(
    # E_architecture=[50, 150, 50],
    # D_architecture=[50, 150, 50],
    E_architecture=[150, 250, 150],
    D_architecture=[150, 250, 150],
    # E_architecture=[150, 250, 250, 150],
    # D_architecture=[150, 250, 250, 150],
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

# save_interval = 10000
save_interval = 25000

transformers = pickle.load(
    open("save_state/track_chi2_QuantileTransformers_e_minus.pkl", "rb")
)

X_train_data_loader = data_loader.load_data(
    [
        "datasets/Kee_2018_truthed_more_vars.csv",
        "datasets/Kstee_2018_truthed_more_vars.csv",
    ],
    N=50000,
    transformers=transformers,
)

X_train_data_loader.fill_chi2_gen()
##

# physical = X_train_data_loader.get_physical()

# with PdfPages(f"e_minus.pdf") as pdf:

#     for key in list(physical.keys()):
#         if "e_minus" in key:
#             print(key)

#             plt.hist2d(
#                 physical["e_minus_TRACK_CHI2NDOF"],
#                 physical[key],
#                 bins=50,
#                 norm=LogNorm(),
#             )
#             plt.ylabel(key)
#             pdf.savefig(bbox_inches="tight")
#             plt.close()
# quit()

##


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

targets = X_train_data_loader.get_branches(rd.targets, processed=False)
targets_pp = X_train_data_loader.get_branches(rd.targets, processed=True)
with PdfPages(f"targets.pdf") as pdf:
    for target in rd.targets:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(targets[target], bins=50)
        plt.xlabel(target)
        plt.subplot(1, 2, 2)
        plt.hist(targets_pp[target], bins=50, range=[-1, 1])
        plt.xlabel(target)
        pdf.savefig(bbox_inches="tight")
        plt.close()

# plt.hist(
#     [
#         targets["K_Kst_TRACK_CHI2NDOF"],
#         targets["e_plus_TRACK_CHI2NDOF"],
#         targets["e_minus_TRACK_CHI2NDOF"],
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

            X_test_data_loader = data_loader.load_data(
                [
                    "datasets/Kee_2018_truthed_more_vars.csv",
                    "datasets/Kstee_2018_truthed_more_vars.csv",
                ],
                transformers=transformers,
            )

            X_test_data_loader.select_randomly(Nevents=10000)
            X_test_data_loader.fill_chi2_gen()

            gen_noise = np.random.normal(0, 1, (10000, latent_dim))
            X_test_conditions = X_test_data_loader.get_branches(
                rd.conditions, processed=True
            )
            X_test_conditions = X_test_conditions[rd.conditions]
            X_test_conditions = np.asarray(X_test_conditions)

            images = np.squeeze(rd.decoder.predict([gen_noise, X_test_conditions]))

            X_test_data_loader.fill_target(images)

            plotting.plot(
                X_train_data_loader,
                X_test_data_loader,
                f"plots_{iteration}_{time_now}_reco{reco_factor}",
                Nevents=10000,
            )

            print("Saving complete...")

            rd.decoder.save("save_state/decoder.h5")
            pickle.dump(
                transformers,
                open("save_state/QuantileTransformers.pkl", "wb"),
            )
            quit()
