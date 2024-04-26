import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Dense, Dropout, Flatten,
                                     Input, Lambda, LeakyReLU, ReLU, Reshape)
from tensorflow.keras.models import Model

_EPSILON = K.epsilon()


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon


class VAE_builder:

    def __init__(self, E_architecture, D_architecture, target_dim, conditions_dim, latent_dim):

        self.E_architecture = E_architecture
        self.D_architecture = D_architecture
        self.target_dim = target_dim
        self.conditions_dim = conditions_dim
        self.latent_dim = latent_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()

    def build_encoder(self):

        input_vertex_info = Input(shape=(self.target_dim,))
        momentum_conditions = Input(shape=(self.conditions_dim,))

        encoder_network_input = Concatenate(axis=-1)(
            [input_vertex_info, momentum_conditions]
        )

        H = Dense(int(self.E_architecture[0]))(encoder_network_input)
        H = BatchNormalization()(H)
        H = LeakyReLU()(H)
        # H = Dropout(0.2)(H)

        for layer in self.E_architecture[1:]:
            H = Dense(int(layer))(H)
            H = BatchNormalization()(H)
            H = LeakyReLU()(H)
            # H = Dropout(0.2)(H)

        z_mean = Dense(self.latent_dim)(H)
        z_log_var = Dense(self.latent_dim)(H)

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        encoder = Model(
            inputs=[input_vertex_info, momentum_conditions],
            outputs=[z, z_mean, z_log_var],
        )

        return encoder

    def build_decoder(self):

        input_latent = Input(shape=(self.latent_dim))
        momentum_conditions = Input(shape=(self.conditions_dim,))

        decoder_network_input = Concatenate()([input_latent, momentum_conditions])

        H = Dense(int(self.D_architecture[0]))(decoder_network_input)
        H = BatchNormalization()(H)
        H = LeakyReLU()(H)
        # H = Dropout(0.2)(H)

        for layer in self.D_architecture[1:]:
            H = Dense(int(layer))(H)
            H = BatchNormalization()(H)
            H = LeakyReLU()(H)
            # H = Dropout(0.2)(H)

        # decoded_mean = Dense(target_dim,activation='tanh')(H)
        decoded_mean = Dense(self.target_dim)(H)
        decoder = Model(
            inputs=[input_latent, momentum_conditions], outputs=[decoded_mean]
        )

        return decoder

    def build_vae(self):

        input_sample = Input(shape=(self.target_dim,))
        momentum_conditions = Input(shape=(self.conditions_dim,))
        z, z_mean, z_log_var = self.encoder([input_sample, momentum_conditions])
        decoded_mean = self.decoder([z, momentum_conditions])
        vae = Model(
            inputs=[input_sample, momentum_conditions],
            outputs=[decoded_mean, z_mean, z_log_var],
        )

        return vae
