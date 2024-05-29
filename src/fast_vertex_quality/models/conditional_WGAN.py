import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    LeakyReLU,
    ReLU,
    Reshape,
    LayerNormalization,
)
from tensorflow.keras.models import Model

_EPSILON = K.epsilon()

class WGAN_builder:

    def __init__(
        self, G_architecture, D_architecture, conditions_dim, target_dim, latent_dim
    ):

        self.G_architecture = G_architecture
        self.D_architecture = D_architecture
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.conditions_dim = conditions_dim

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        # self.gan = self.build_gan()
        self.gan = None

    def build_generator(self):

        input_noise = Input(shape=(self.latent_dim,))
        momentum_conditions = Input(shape=(self.conditions_dim,))

        generator_network_input = Concatenate(axis=-1)(
            [input_noise, momentum_conditions]
        )

        H = Dense(int(self.G_architecture[0]))(generator_network_input)
        H = LeakyReLU()(H)
        H = BatchNormalization()(H)
        H = Concatenate(axis=-1)([H, momentum_conditions])

        for layer in self.G_architecture[1:]:
            H = Dense(int(layer))(H)
            H = LeakyReLU()(H)
            H = BatchNormalization()(H)
            H = Concatenate(axis=-1)([H, momentum_conditions])

        output = Dense(self.target_dim,activation='tanh')(H)

        generator = Model(
            inputs=[input_noise, momentum_conditions],
            outputs=[output],
        )

        return generator

    def build_discriminator(self):

        input_sample = Input(shape=(self.target_dim))
        momentum_conditions = Input(shape=(self.conditions_dim,))

        discrim_network_input = Concatenate()([input_sample, momentum_conditions])

        H = Dense(int(self.D_architecture[0]))(discrim_network_input)
        H = LeakyReLU()(H)
        H = LayerNormalization()(H)
        H = Dropout(0.2)(H)
        H = Concatenate(axis=-1)([H, momentum_conditions])

        for layer in self.D_architecture[1:]:
            H = Dense(int(layer))(H)
            H = LeakyReLU()(H)
            H = LayerNormalization()(H)
            H = Dropout(0.2)(H)
            H = Concatenate(axis=-1)([H, momentum_conditions])

        discrim_out = Dense(1, activation="linear")(H)

        discriminator = Model(
            inputs=[input_sample, momentum_conditions], outputs=[discrim_out]
        )

        return discriminator
