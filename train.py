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
import fast_vertex_quality.tools.data_loader as data_loader
from fast_vertex_quality.tools.training import train_step

from fast_vertex_quality.models.conditional_VAE import VAE_builder


print(tf.__version__)


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


# rd.conditions = [
#     "K_Kst_PX",
#     "K_Kst_PY",
#     "K_Kst_PZ",
#     "e_minus_PX",
#     "e_minus_PY",
#     "e_minus_PZ",
#     "e_plus_PX",
#     "e_plus_PY",
#     "e_plus_PZ",
#     "nTracks",
#     "nSPDHits",
# ]

# rd.conditions = [
#     "q2",
# ]

rd.conditions = [
    "B_P",
    "B_PT",
    "missing_B_P",
    "missing_B_PT",
    "delta_0_P",
    "delta_0_PT",
    "delta_1_P",
    "delta_1_PT",
    "delta_2_P",
    "delta_2_PT",
    "m_01",
    "m_02",
    "m_12",
]


latent_dim = 6
kl_factor = 1.0
# reco_factor = 1E3
reco_factor = 100.0

batch_size = 50

target_dim = len(rd.targets)
conditions_dim = len(rd.conditions)
latent_dim = 6

cut_idx = target_dim

VAE = VAE_builder(
    E_architecture=[50, 250, 250, 50],
    D_architecture=[50, 250, 250, 50],
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


save_interval = 2500
# save_interval = 25

for epoch in range(int(1e30)):

    # X_train_data_loader = data_loader.load_data("datasets/Kee_2018_truthed.csv")
    X_train_data_loader = data_loader.load_data(
        "datasets/Kee_2018_truthed_more_vars.csv"
    )

    X_train_data_all_pp = X_train_data_loader.get_branches(
        rd.targets + rd.conditions, processed=True
    )

    X_train_raw = np.asarray(X_train_data_all_pp)

    X_train = np.expand_dims(X_train_raw, 1).astype("float32")

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(X_train)
        .batch(batch_size, drop_remainder=True)
        .repeat(1)
    )

    for samples_for_batch in train_dataset:

        if iteration % 100 == 0:
            print("Iteration:", iteration)

        iteration += 1

        if iteration % 1000 == 0:  # annealing https://arxiv.org/pdf/1511.06349.pdf
            toggle_kl_value = 0.0
        elif iteration % 1000 < 500:
            toggle_kl_value += 1 / 500
        else:
            toggle_kl_value = 1.0
        toggle_kl = tf.convert_to_tensor(toggle_kl_value)

        kl_loss_np, reco_loss_np = train_step(
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

            plotting.loss_plot(loss_list, reco_factor, kl_factor, "LOSSES.png")

            gen_noise = np.random.normal(0, 1, (10000, latent_dim))

            # X_test_data_loader = data_loader.load_data("datasets/Kee_2018_truthed.csv")
            X_test_data_loader = data_loader.load_data(
                "datasets/Kee_2018_truthed_more_vars.csv"
            )
            X_test_data_loader.select_randomly(Nevents=10000)
            X_test_conditions = X_test_data_loader.get_branches(
                rd.conditions, processed=True
            )
            X_test_conditions = np.asarray(X_test_conditions)

            images = np.squeeze(rd.decoder.predict([gen_noise, X_test_conditions]))

            X_test_data_loader.fill_target(images)

            plotting.plot(
                X_train_data_loader,
                X_test_data_loader,
                f"plots_{iteration}",
                Nevents=10000,
            )

            print("Saving complete...")

            for file in glob.glob("save_state/*"):
                os.remove(file)
            rd.decoder.save("save_state/decoder.h5")
            pickle.dump(
                rd.QuantileTransformers,
                open("save_state/QuantileTransformers.pkl", "wb"),
            )

            if iteration > 1e4:
                quit()
