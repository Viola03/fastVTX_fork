from fast_vertex_quality.tools.config import read_definition, rd

from fast_vertex_quality.models.conditional_VAE import VAE_builder
import tensorflow as tf
import numpy as np
from fast_vertex_quality.tools.training import train_step
import fast_vertex_quality.tools.plotting as plotting
import pickle
import fast_vertex_quality.tools.new_data_loader as data_loader


class trackchi2_trainer:

    def __init__(self, data_loader_obj):

        self.set_training_vars("PARTICLE")

        self.kl_factor = 1.0
        self.reco_factor = 100.0
        self.batch_size = 50

        self.target_dim = len(self.targets)
        self.conditions_dim = len(self.conditions)
        self.latent_dim = 1
        self.cut_idx = self.target_dim

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

        self.encoder, self.decoder, self.vae = self.build_VAE()

        self.initialised_weights = self.get_weights()

        self.data_loader_obj = data_loader_obj
        self.transformers = self.data_loader_obj.get_transformers()

        self.trained_weights = {}

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

    def set_trained_weights(self, particle):

        self.vae.set_weights(self.trained_weights[particle]["vae"])
        self.decoder.set_weights(self.trained_weights[particle]["decoder"])
        self.encoder.set_weights(self.trained_weights[particle]["encoder"])

    def set_training_vars(self, particle):

        self.targets = [
            f"{particle}_TRACK_CHI2NDOF",
        ]

        self.conditions = [
            f"{particle}_PX",
            f"{particle}_PY",
            f"{particle}_PZ",
            f"{particle}_P",
            f"{particle}_PT",
            f"{particle}_eta",
        ]

    def build_VAE(self):

        VAE = VAE_builder(
            E_architecture=[25, 25],
            D_architecture=[25, 25],
            target_dim=self.target_dim,
            conditions_dim=self.conditions_dim,
            latent_dim=self.latent_dim,
        )
        return VAE.encoder, VAE.decoder, VAE.vae

    def train(self, particle, steps=5000):

        self.set_initialised_weights()
        self.set_training_vars(particle)

        iteration = -1

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

                iteration += 1

                if iteration % 100 == 0:
                    print("Iteration:", iteration)

                if (
                    iteration % 1000 == 0
                ):  # annealing https://arxiv.org/pdf/1511.06349.pdf
                    toggle_kl_value = 0.0
                elif iteration % 1000 < 500:
                    toggle_kl_value += 1.0 / 500.0
                else:
                    toggle_kl_value = 1.0
                toggle_kl = tf.convert_to_tensor(toggle_kl_value)

                kl_loss_np, reco_loss_np, reco_loss_np_raw = train_step(
                    self.vae,
                    self.optimizer,
                    samples_for_batch,
                    self.cut_idx,
                    tf.convert_to_tensor(self.kl_factor),
                    tf.convert_to_tensor(self.reco_factor),
                    toggle_kl,
                )

                if iteration > steps:
                    break_option = True
                    break

            if break_option:
                break

        self.trained_weights[particle] = self.get_weights()

    def make_plots(self, particle, N=10000):

        self.set_trained_weights(particle)
        self.set_training_vars(particle)

        gen_noise = np.random.normal(0, 1, (N, self.latent_dim))

        X_test_data_loader = data_loader.load_data(
            [
                "datasets/Kee_2018_truthed_more_vars.csv",
            ],
            transformers=self.transformers,
        )

        X_test_data_loader.select_randomly(Nevents=N)

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
            f"plots_{particle}",
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

    def predict(self, particle, inputs):

        self.set_trained_weights(particle)
        self.set_training_vars(particle)

        return self.decoder.predict(inputs)
