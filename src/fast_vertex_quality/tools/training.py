import tensorflow as tf
from tensorflow.keras import backend as K

from fast_vertex_quality.tools.config import rd, read_definition


def reco_loss(x, x_decoded_mean):
    xent_loss = tf.keras.losses.mean_squared_error(x, x_decoded_mean)
    return xent_loss


def kl_loss(z_mean, z_log_var):
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return kl_loss


@tf.function
def train_step(images, cut_idx, kl_factor, reco_factor, toggle_kl):

    sample_targets, sample_conditions = images[:, 0, :cut_idx], images[:, 0, cut_idx:]

    with tf.GradientTape() as tape:

        vae_out, vae_z_mean, vae_z_log_var = rd.vae([sample_targets, sample_conditions])

        vae_reco_loss = reco_loss(sample_targets, vae_out)
        vae_reco_loss_raw = tf.math.reduce_mean(vae_reco_loss)
        vae_reco_loss = vae_reco_loss_raw * reco_factor
        vae_kl_loss = kl_loss(vae_z_mean, vae_z_log_var)
        vae_kl_loss = tf.math.reduce_mean(vae_kl_loss) * toggle_kl * kl_factor

        vae_loss = vae_kl_loss + vae_reco_loss

    grad_vae = tape.gradient(vae_loss, rd.vae.trainable_variables)

    rd.optimizer.apply_gradients(zip(grad_vae, rd.vae.trainable_variables))

    return vae_kl_loss, vae_reco_loss, vae_reco_loss_raw


@tf.function
def train_step_regressor(images, cut_idx):

    sample_targets, sample_conditions = images[:, 0, :cut_idx], images[:, 0, cut_idx:]

    with tf.GradientTape() as tape:

        regressor_conditions = rd.regressor([sample_targets])

        regressor_reco_loss = tf.keras.losses.mean_squared_error(
            sample_conditions,
            regressor_conditions,
        )
        regressor_reco_loss = tf.math.reduce_mean(regressor_reco_loss)

    grad_regressor = tape.gradient(
        regressor_reco_loss, rd.regressor.trainable_variables
    )

    rd.optimizer.apply_gradients(zip(grad_regressor, rd.regressor.trainable_variables))

    return regressor_reco_loss
