from fast_vertex_quality.tools.config import read_definition, rd

from fast_vertex_quality.models.conditional_VAE import VAE_builder
import tensorflow as tf
import numpy as np
from fast_vertex_quality.tools.training import train_step_vertexing as train_step
import fast_vertex_quality.tools.plotting as plotting
import pickle
import fast_vertex_quality.tools.data_loader as data_loader
import matplotlib.pyplot as plt


class vertex_quality_trainer:

    def __init__(
        self, data_loader_obj, trackchi2_trainer, targets=None, conditions=None
    ):

        self.trackchi2_trainer = trackchi2_trainer

        if targets == None:
            self.targets = [
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
        else:
            self.targets = targets

        if conditions == None:
            # self.conditions = [
            #     "B_P",
            #     "B_PT",
            #     # "angle_K_Kst",
            #     # "angle_e_plus",
            #     # "angle_e_minus",
            #     "K_Kst_eta",
            #     "e_plus_eta",
            #     "e_minus_eta",
            #     "IP_B",
            #     "IP_K_Kst",
            #     "IP_e_plus",
            #     "IP_e_minus",
            #     "FD_B",
            #     "DIRA_B",
            #     # "delta_0_P",
            #     # "delta_0_PT",
            #     # "delta_1_P",
            #     # "delta_1_PT",
            #     # "delta_2_P",
            #     # "delta_2_PT",
            #     "K_Kst_TRACK_CHI2NDOF_gen",
            #     "e_minus_TRACK_CHI2NDOF_gen",
            #     "e_plus_TRACK_CHI2NDOF_gen",
            # ]

            self.conditions = [
                "IP_B",
                "DIRA_B",
            ]
        else:
            self.conditions = conditions

        self.kl_factor = 1.0
        self.reco_factor = 1000.0
        self.batch_size = 50

        self.target_dim = len(self.targets)
        self.conditions_dim = len(self.conditions)
        self.latent_dim = 4
        self.cut_idx = self.target_dim

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

        self.encoder, self.decoder, self.vae = self.build_VAE()

        self.initialised_weights = self.get_weights()

        self.data_loader_obj = data_loader_obj
        self.transformers = self.data_loader_obj.get_transformers()

        self.trained_weights = {}

    def get_decoder(self):
        return self.decoder

    def get_weights(self):

        weights = {}
        weights["vae"] = self.vae.get_weights()
        weights["decoder"] = self.decoder.get_weights()
        weights["encoder"] = self.encoder.get_weights()

        return weights

    def set_initialised_weights(self):

        self.vae.set_weights(self.initialised_weights["vae"])
        self.decoder.set_weights(self.initialised_weights["decoder"])
        self.encoder.set_weights(self.initialised_weights["encoder"])

    def set_trained_weights(self):

        self.vae.set_weights(self.trained_weights["vae"])
        self.decoder.set_weights(self.trained_weights["decoder"])
        self.encoder.set_weights(self.trained_weights["encoder"])

        # decoder = tf.keras.models.load_model("save_state/decoder.h5")
        # self.trained_weights = decoder.get_weights()
        # self.decoder.set_weights(self.trained_weights)

    def build_VAE(self):

        VAE = VAE_builder(
            E_architecture=[150, 250, 150],
            D_architecture=[150, 250, 150],
            # E_architecture=[75, 150, 75],
            # D_architecture=[75, 150, 75],
            # E_architecture=[150, 250, 250, 150],
            # D_architecture=[150, 250, 250, 150],
            target_dim=self.target_dim,
            conditions_dim=self.conditions_dim,
            latent_dim=self.latent_dim,
        )
        return VAE.encoder, VAE.decoder, VAE.vae

    def train_more_steps(self, steps=10000):

        private_iteration = -1

        break_option = False
        for epoch in range(int(1e30)):

            X_train_data_all_pp = self.data_loader_obj.get_branches(
                self.targets + self.conditions, processed=True
            )

            X_train_data_all_pp = X_train_data_all_pp.sample(frac=1)
            X_train_data_all_pp = X_train_data_all_pp[self.targets + self.conditions]

            X_train_raw = np.asarray(X_train_data_all_pp)

            X_train = np.expand_dims(X_train_raw, 1).astype("float32")

            train_dataset = (
                tf.data.Dataset.from_tensor_slices(X_train)
                .batch(self.batch_size, drop_remainder=True)
                .repeat(1)
            )

            for samples_for_batch in train_dataset:

                self.iteration += 1
                private_iteration += 1

                if self.iteration % 100 == 0:
                    print("Iteration:", self.iteration)

                if (
                    self.iteration % 1000 == 0
                ):  # annealing https://arxiv.org/pdf/1511.06349.pdf
                    self.toggle_kl_value = 0.0
                elif self.iteration % 1000 < 500:
                    self.toggle_kl_value += 1.0 / 500.0
                else:
                    self.toggle_kl_value = 1.0
                toggle_kl = tf.convert_to_tensor(self.toggle_kl_value)

                kl_loss_np, reco_loss_np, reco_loss_np_raw = train_step(
                    self.vae,
                    self.optimizer,
                    samples_for_batch,
                    self.cut_idx,
                    tf.convert_to_tensor(self.kl_factor),
                    tf.convert_to_tensor(self.reco_factor),
                    toggle_kl,
                )

                self.loss_list = np.append(
                    self.loss_list,
                    [[self.iteration, kl_loss_np, reco_loss_np, reco_loss_np_raw]],
                    axis=0,
                )

                if private_iteration > steps:
                    break_option = True
                    break

            if break_option:
                break

        self.trained_weights = self.get_weights()

        plt.subplot(1, 3, 1)
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 1])
        plt.subplot(1, 3, 2)
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 2])
        plt.subplot(1, 3, 3)
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 3])
        plt.savefig("Losses.png")
        plt.close("all")

    def train(self, steps=10000):

        self.set_initialised_weights()

        self.iteration = -1

        self.loss_list = np.empty((0, 4))

        break_option = False
        for epoch in range(int(1e30)):

            X_train_data_all_pp = self.data_loader_obj.get_branches(
                self.targets + self.conditions, processed=True
            )

            X_train_data_all_pp = X_train_data_all_pp.sample(frac=1)
            X_train_data_all_pp = X_train_data_all_pp[self.targets + self.conditions]

            X_train_raw = np.asarray(X_train_data_all_pp)

            X_train = np.expand_dims(X_train_raw, 1).astype("float32")

            train_dataset = (
                tf.data.Dataset.from_tensor_slices(X_train)
                .batch(self.batch_size, drop_remainder=True)
                .repeat(1)
            )

            for samples_for_batch in train_dataset:

                self.iteration += 1

                if self.iteration % 100 == 0:
                    print("Iteration:", self.iteration)

                if (
                    self.iteration % 1000 == 0
                ):  # annealing https://arxiv.org/pdf/1511.06349.pdf
                    self.toggle_kl_value = 0.0
                elif self.iteration % 1000 < 500:
                    self.toggle_kl_value += 1.0 / 500.0
                else:
                    self.toggle_kl_value = 1.0
                toggle_kl = tf.convert_to_tensor(self.toggle_kl_value)

                kl_loss_np, reco_loss_np, reco_loss_np_raw = train_step(
                    self.vae,
                    self.optimizer,
                    samples_for_batch,
                    self.cut_idx,
                    tf.convert_to_tensor(self.kl_factor),
                    tf.convert_to_tensor(self.reco_factor),
                    toggle_kl,
                )

                self.loss_list = np.append(
                    self.loss_list,
                    [[self.iteration, kl_loss_np, reco_loss_np, reco_loss_np_raw]],
                    axis=0,
                )

                if self.iteration > steps:
                    break_option = True
                    break

            if break_option:
                break

        self.trained_weights = self.get_weights()

        plt.subplot(1, 3, 1)
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 1])
        plt.subplot(1, 3, 2)
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 2])
        plt.subplot(1, 3, 3)
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 3])
        plt.savefig("Losses.png")
        plt.close("all")

    def make_plots(self, N=10000):

        self.set_trained_weights()

        gen_noise = np.random.normal(0, 1, (N, self.latent_dim))

        X_test_data_loader = data_loader.load_data(
            [
                "datasets/Kee_2018_truthed_more_vars.csv",
            ],
            transformers=self.transformers,
        )

        X_test_data_loader.select_randomly(Nevents=N)

        X_test_data_loader.fill_chi2_gen(self.trackchi2_trainer)

        X_test_conditions = X_test_data_loader.get_branches(
            self.conditions, processed=True
        )
        X_test_conditions = X_test_conditions[self.conditions]
        X_test_conditions = np.asarray(X_test_conditions)

        images = np.squeeze(self.decoder.predict([gen_noise, X_test_conditions]))

        X_test_data_loader.fill_target(images, self.targets)

        plotting.plot(
            self.data_loader_obj,
            X_test_data_loader,
            f"plots",
            self.targets,
            Nevents=10000,
        )

    def save_state(self, tag):

        pickle.dump(
            self.transformers,
            open(
                f"{tag}_transfomers.pkl",
                "wb",
            ),
        )

        pickle.dump(
            self.trained_weights,
            open(
                f"{tag}_trained_weights.pkl",
                "wb",
            ),
        )

    def load_state(self, tag):

        self.transformers = pickle.load(open(f"{tag}_transfomers.pkl", "rb"))
        self.trained_weights = pickle.load(open(f"{tag}_trained_weights.pkl", "rb"))

        # decoder = tf.keras.models.load_model("save_state/decoder.h5")
        # self.trained_weights = decoder.get_weights()

    def predict(self, inputs):

        self.set_trained_weights()

        return self.decoder.predict(inputs)

    def predict_from_data_loader(self, data_loader_obj):

        self.set_trained_weights()

        data_loader_obj.fill_chi2_gen(self.trackchi2_trainer)

        events_gen = data_loader_obj.get_branches(self.conditions, processed=True)

        events_gen = np.asarray(events_gen[self.conditions])

        gen_noise = np.random.normal(0, 1, (10000, self.latent_dim))
        images = np.squeeze(self.decoder.predict([gen_noise, events_gen]))

        data_loader_obj.fill_target(images, self.targets)

        return data_loader_obj
