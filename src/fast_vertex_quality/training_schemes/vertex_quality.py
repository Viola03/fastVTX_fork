from fast_vertex_quality.tools.config import read_definition, rd

from fast_vertex_quality.models.conditional_VAE import VAE_builder
from fast_vertex_quality.models.conditional_GAN import GAN_builder
from fast_vertex_quality.models.conditional_WGAN import WGAN_builder
import tensorflow as tf
import numpy as np
from fast_vertex_quality.tools.training import train_step_vertexing as train_step
from fast_vertex_quality.tools.training import train_step_vertexing_GAN as train_step_GAN
from fast_vertex_quality.tools.training import train_step_vertexing_WGAN as train_step_WGAN
import fast_vertex_quality.tools.plotting as plotting
import pickle
import fast_vertex_quality.tools.data_loader as data_loader
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

class vertex_quality_trainer:

    def __init__(
        self,
        data_loader_obj,
        trackchi2_trainer=None,
        targets=None,
        conditions=None,
        beta=1000.0,
        latent_dim=4,
        batch_size=50,
        E_architecture=[150, 250, 150],
        D_architecture=[150, 250, 150],
        G_architecture=None,
        network_option='VAE',
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
            self.conditions = [
                "IP_B",
                "DIRA_B",
            ]
        else:
            self.conditions = conditions

        self.kl_factor = 1.0
        self.reco_factor = beta
        self.batch_size = batch_size

        self.E_architecture = E_architecture
        self.D_architecture = D_architecture
        self.G_architecture = G_architecture

        self.target_dim = len(self.targets)
        self.conditions_dim = len(self.conditions)
        self.latent_dim = latent_dim
        self.cut_idx = self.target_dim

        self.network_option = network_option

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        if self.network_option == 'GAN':
            self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.5, amsgrad=True)
            self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.5, amsgrad=True)
        elif self.network_option == 'WGAN':
            # self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
            # self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

            # self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=0.0001, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
            # self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

            # self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=0.00005, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
            # self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=0.0001, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

            gen_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.00005,
                    decay_steps=5000,
                    decay_rate=0.9,
                )
            self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=gen_lr_schedule, beta1=0.5)
            disc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.0001,
                    decay_steps=5000,
                    decay_rate=0.9,
                )
            self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=disc_lr_schedule, beta1=0.5)
            
            # self.gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0004)
            # self.disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0004)



        if self.network_option == 'VAE':
            self.encoder, self.decoder, self.vae = self.build_VAE()
        elif self.network_option == 'GAN':
            self.discriminator, self.generator, self.gan = self.build_GAN()
        elif self.network_option == 'WGAN':
            self.discriminator, self.generator, self.gan = self.build_WGAN()
        else:
            print('network_option not valid... quitting...')
            quit()



        self.initialised_weights = self.get_weights()

        self.data_loader_obj = data_loader_obj
        self.transformers = self.data_loader_obj.get_transformers()

        self.trained_weights = {}

    def get_decoder(self):
        return self.decoder
    
    def get_generator(self):
        return self.generator

    def get_weights(self):

        weights = {}
        if self.network_option == 'VAE':
            weights["vae"] = self.vae.get_weights()
            weights["decoder"] = self.decoder.get_weights()
            weights["encoder"] = self.encoder.get_weights()
        elif self.network_option == 'GAN' or self.network_option == 'WGAN':
            # weights["gan"] = self.gan.get_weights()
            weights["discriminator"] = self.discriminator.get_weights()
            weights["generator"] = self.generator.get_weights()

        return weights

    def set_initialised_weights(self):
        
        if self.network_option == 'VAE':
            self.vae.set_weights(self.initialised_weights["vae"])
            self.decoder.set_weights(self.initialised_weights["decoder"])
            self.encoder.set_weights(self.initialised_weights["encoder"])
        elif self.network_option == 'GAN' or self.network_option == 'WGAN':
            # self.gan.set_weights(self.initialised_weights["gan"])
            self.discriminator.set_weights(self.initialised_weights["discriminator"])
            self.generator.set_weights(self.initialised_weights["generator"])
        

    def set_trained_weights(self):

        if self.network_option == 'VAE':
            self.vae.set_weights(self.trained_weights["vae"])
            self.decoder.set_weights(self.trained_weights["decoder"])
            self.encoder.set_weights(self.trained_weights["encoder"])
        elif self.network_option == 'GAN' or self.network_option == 'WGAN':
            # self.gan.set_weights(self.trained_weights["gan"])
            self.discriminator.set_weights(self.trained_weights["discriminator"])
            self.generator.set_weights(self.trained_weights["generator"])


        # decoder = tf.keras.models.load_model("save_state/decoder.h5")
        # self.trained_weights = decoder.get_weights()
        # self.decoder.set_weights(self.trained_weights)

    def build_VAE(self):

        VAE = VAE_builder(
            E_architecture=self.E_architecture,
            D_architecture=self.D_architecture,
            # E_architecture=[75, 150, 75],
            # D_architecture=[75, 150, 75],
            # E_architecture=[150, 250, 250, 150],
            # D_architecture=[150, 250, 250, 150],
            target_dim=self.target_dim,
            conditions_dim=self.conditions_dim,
            latent_dim=self.latent_dim,
        )
        return VAE.encoder, VAE.decoder, VAE.vae
    
    def build_GAN(self):

        GAN = GAN_builder(
            G_architecture=self.G_architecture,
            D_architecture=self.D_architecture,
            target_dim=self.target_dim,
            conditions_dim=self.conditions_dim,
            latent_dim=self.latent_dim,
        )
        return GAN.discriminator, GAN.generator, GAN.gan

    def build_WGAN(self):

        GAN = WGAN_builder(
            G_architecture=self.G_architecture,
            D_architecture=self.D_architecture,
            target_dim=self.target_dim,
            conditions_dim=self.conditions_dim,
            latent_dim=self.latent_dim,
        )
        return GAN.discriminator, GAN.generator, GAN.gan

    def step(self, samples_for_batch):
        
        if self.network_option == 'VAE':
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

            return [self.iteration, kl_loss_np, reco_loss_np, reco_loss_np_raw]
        
        elif self.network_option == 'GAN':
            disc_loss_np, gen_loss_np = train_step_GAN(
                self.batch_size,
                self.generator,
                self.discriminator,
                self.gen_optimizer,
                self.disc_optimizer,
                samples_for_batch,
                self.cut_idx,
                self.latent_dim,
            )
            return [self.iteration, disc_loss_np, gen_loss_np, 0.]

        elif self.network_option == 'WGAN':
            disc_loss_np, gen_loss_np = train_step_WGAN(
                self.batch_size,
                self.generator,
                self.discriminator,
                self.gen_optimizer,
                self.disc_optimizer,
                samples_for_batch,
                self.cut_idx,
                self.latent_dim,
            )
            return [self.iteration, disc_loss_np, gen_loss_np, 0.]

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

            if self.network_option == 'VAE':
                train_dataset = (
                    tf.data.Dataset.from_tensor_slices(X_train)
                    .batch(self.batch_size, drop_remainder=True)
                    .repeat(1)
                )
            elif self.network_option == 'GAN' or self.network_option == 'WGAN':
                train_dataset = (
                    tf.data.Dataset.from_tensor_slices(X_train)
                    .batch(self.batch_size*3, drop_remainder=True)
                    .repeat(1)
                )

            for samples_for_batch in train_dataset:

                self.iteration += 1
                private_iteration += 1

                if self.iteration % 100 == 0:
                    print("Iteration:", self.iteration)

                loss_list_i = self.step(samples_for_batch)

                self.loss_list = np.append(
                    self.loss_list,
                    [loss_list_i],
                    axis=0,
                )

                if np.isnan(self.loss_list[-1]).any():
                    print(samples_for_batch)
                    print(
                        f"NaNs present in loss_list, quitting on iteration {self.iteration}..."
                    )
                    print(np.shape(np.asarray(samples_for_batch)))
                    print('\n')
                    print(np.where(np.isnan(np.asarray(samples_for_batch))))
                    print('\n')
                    print(np.where(np.isinf(np.asarray(samples_for_batch))))
                    quit()

                if private_iteration > steps:
                    break_option = True
                    break

            if break_option:
                break

        self.trained_weights = self.get_weights()

        plt.subplot(1, 3, 1)
        plt.title('disc')
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 1])
        plt.subplot(1, 3, 2)
        plt.title('gen')
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

            if self.network_option == 'VAE':
                train_dataset = (
                    tf.data.Dataset.from_tensor_slices(X_train)
                    .batch(self.batch_size, drop_remainder=True)
                    .repeat(1)
                )
            elif self.network_option == 'GAN' or self.network_option == 'WGAN':
                train_dataset = (
                    tf.data.Dataset.from_tensor_slices(X_train)
                    .batch(self.batch_size*3, drop_remainder=True)
                    .repeat(1)
                )

            for samples_for_batch in train_dataset:

                self.iteration += 1

                loss_list_i = self.step(samples_for_batch)
                
                self.loss_list = np.append(
                    self.loss_list,
                    [loss_list_i],
                    axis=0,
                )

                if self.iteration % 100 == 0:
                    print("Iteration:", self.iteration)
                

                if np.isnan(self.loss_list[-1]).any():
                    print(samples_for_batch)
                    print(
                        f"NaNs present in loss_list, quitting on iteration {self.iteration}..."
                    )
                    print(np.shape(np.asarray(samples_for_batch)))
                    print('\n')
                    print(np.where(np.isnan(np.asarray(samples_for_batch))))
                    print('\n')
                    print(np.where(np.isinf(np.asarray(samples_for_batch))))
                    quit()

                if self.iteration > steps:
                    break_option = True
                    break
                
                if self.iteration % 500 == 0 and self.iteration > 1:
                    plt.subplot(1, 3, 1)
                    plt.title('disc')
                    plt.plot(self.loss_list[:, 0], self.loss_list[:, 1])
                    plt.subplot(1, 3, 2)
                    plt.title('gen')
                    plt.plot(self.loss_list[:, 0], self.loss_list[:, 2])
                    plt.subplot(1, 3, 3)
                    plt.plot(self.loss_list[:, 0], self.loss_list[:, 3])
                    plt.savefig("Losses.png")
                    plt.close("all")

            if break_option:
                break

        self.trained_weights = self.get_weights()

        plt.subplot(1, 3, 1)
        plt.title('disc')
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 1])
        plt.subplot(1, 3, 2)
        plt.title('gen')
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 2])
        plt.subplot(1, 3, 3)
        plt.plot(self.loss_list[:, 0], self.loss_list[:, 3])
        plt.savefig("Losses.png")
        plt.close("all")

    def make_plots(self, N=10000, filename=f"plots", testing_file="datasets/B2KEE_three_body_cut_more_vars.root"):

        self.set_trained_weights()

        gen_noise = np.random.normal(0, 1, (N, self.latent_dim))

        # X_test_data_loader = data_loader.load_data(
        #     [
        #         # "datasets/Kee_2018_truthed_more_vars.csv",
        #         "datasets/Kee_2018_truthed_more_vars.csv",
        #     ],
        #     transformers=self.transformers,
        # )
        X_test_data_loader = data_loader.load_data(
                    testing_file,
                    transformers=self.transformers,
                    convert_to_RK_branch_names=True,
                    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
                )


        X_test_data_loader.select_randomly(Nevents=N)

        if self.trackchi2_trainer is not None:
            X_test_data_loader.fill_chi2_gen(self.trackchi2_trainer)

        X_test_conditions = X_test_data_loader.get_branches(
            self.conditions, processed=True
        )
        X_test_conditions = X_test_conditions[self.conditions]
        X_test_conditions = np.asarray(X_test_conditions)

        if self.network_option == 'VAE':
            images = np.squeeze(self.decoder.predict([gen_noise, X_test_conditions]))
        elif self.network_option == 'GAN' or self.network_option == 'WGAN':
            images = np.squeeze(self.generator.predict([gen_noise, X_test_conditions]))

        X_test_data_loader.fill_target(images, self.targets)

        plotting.plot(
            self.data_loader_obj,
            X_test_data_loader,
            filename,
            self.targets,
            Nevents=10000,
        )
    
    def gen_data(self, filename, N=10000):

        self.set_trained_weights()

        gen_noise = np.random.normal(0, 1, (N, self.latent_dim))

        X_test_data_loader = data_loader.load_data(
            [
                # "datasets/Kee_2018_truthed_more_vars.csv",
                # "datasets/B2KEE_three_body_cut_more_vars.root",
                # "datasets/dedicated_Kee_MC_hierachy_cut_more_vars.root",
                "datasets/dedicated_Kstee_MC_hierachy_cut_more_vars.root",
            ],
            transformers=self.transformers,
            convert_to_RK_branch_names=True,
            conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
        )

        X_test_data_loader.select_randomly(Nevents=N)

        if self.trackchi2_trainer is not None:
            X_test_data_loader.fill_chi2_gen(self.trackchi2_trainer)

        X_test_conditions = X_test_data_loader.get_branches(
            self.conditions, processed=True
        )
        X_test_conditions = X_test_conditions[self.conditions]
        X_test_conditions = np.asarray(X_test_conditions)

        if self.network_option == 'VAE':
            images = np.squeeze(self.decoder.predict([gen_noise, X_test_conditions]))
        elif self.network_option == 'GAN' or self.network_option == 'WGAN':
            images = np.squeeze(self.generator.predict([gen_noise, X_test_conditions]))

        X_test_data_loader.fill_target(images, self.targets)

        X_test_data_loader.add_branch_to_physical('B_plus_ENDVERTEX_NDOF', np.ones(N)*3.)

        X_test_data_loader.save_to_file(filename)




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

        if self.trackchi2_trainer is not None:
            data_loader_obj.fill_chi2_gen(self.trackchi2_trainer)

        events_gen = data_loader_obj.get_branches(self.conditions, processed=True)

        events_gen = np.asarray(events_gen[self.conditions])

        gen_noise = np.random.normal(0, 1, (np.shape(events_gen)[0], self.latent_dim))
        
        if self.network_option == 'VAE':
            images = np.squeeze(self.decoder.predict([gen_noise, events_gen]))
        elif self.network_option == 'GAN' or self.network_option == 'WGAN':
            images = np.squeeze(self.generator.predict([gen_noise, events_gen]))

        data_loader_obj.fill_target(images, self.targets)

        return data_loader_obj
