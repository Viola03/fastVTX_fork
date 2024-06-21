import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
_EPSILON = K.epsilon()

from fast_vertex_quality.tools.config import rd, read_definition


def reco_loss(x, x_decoded_mean):
	xent_loss = tf.keras.losses.mean_squared_error(x, x_decoded_mean)
	return xent_loss


def kl_loss(z_mean, z_log_var):
	kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	return kl_loss


@tf.function
def train_step_vertexing(
	vae, optimizer, images, cut_idx, kl_factor, reco_factor, toggle_kl
):

	sample_targets, sample_conditions = images[:, 0, :cut_idx], images[:, 0, cut_idx:]

	with tf.GradientTape() as tape:

		vae_out, vae_z_mean, vae_z_log_var = vae([sample_targets, sample_conditions])

		vae_reco_loss = reco_loss(sample_targets, vae_out)
		vae_reco_loss_raw = tf.math.reduce_mean(vae_reco_loss)
		vae_reco_loss = vae_reco_loss_raw * reco_factor
		vae_kl_loss = kl_loss(vae_z_mean, vae_z_log_var)
		vae_kl_loss = tf.math.reduce_mean(vae_kl_loss) * toggle_kl * kl_factor

		vae_loss = vae_kl_loss + vae_reco_loss

	grad_vae = tape.gradient(vae_loss, vae.trainable_variables)

	optimizer.apply_gradients(zip(grad_vae, vae.trainable_variables))

	return vae_kl_loss, vae_reco_loss, vae_reco_loss_raw

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

@tf.function
def train_step_vertexing_GAN(
	batch_size, generator, discriminator, gen_optimizer, disc_optimizer, images, cut_idx, latent_dim
):

	sample_targets, sample_conditions = images[:, 0, :cut_idx], images[:, 0, cut_idx:]
	sample_targets_train_A = sample_targets[:batch_size]
	sample_targets_train_B = sample_targets[batch_size:2*batch_size]
	sample_targets_train_C = sample_targets[2*batch_size:]
	sample_conditions_train_A = sample_conditions[:batch_size]
	sample_conditions_train_B = sample_conditions[batch_size:2*batch_size]
	sample_conditions_train_C = sample_conditions[2*batch_size:]

	noise = tf.random.normal([batch_size, latent_dim])
	generated_images = generator([noise, sample_conditions_train_A])

	in_values = tf.concat([generated_images, sample_targets_train_B],0)
	in_values_labels = tf.concat([sample_conditions_train_A, sample_conditions_train_B],0)
	labels_D_0 = tf.zeros((batch_size, 1)) 
	labels_D_1 = tf.ones((batch_size, 1))
	labels_D = tf.concat([labels_D_0, labels_D_1],0)
	
	with tf.GradientTape(persistent=True) as disc_tape:
		out_values_choice = discriminator([in_values, in_values_labels], training=True)
		disc_loss = tf.keras.losses.binary_crossentropy(tf.squeeze(labels_D),tf.squeeze(out_values_choice))
	
	noise_stacked = tf.random.normal((batch_size, latent_dim), 0, 1)
	labels_stacked = tf.ones((batch_size, 1))
	
	with tf.GradientTape(persistent=True) as gen_tape:
		fake_images2 = generator([noise_stacked, sample_conditions_train_C], training=True)
		stacked_output_choice = discriminator([fake_images2, sample_conditions_train_C])
		gen_loss = _loss_generator(tf.squeeze(labels_stacked),tf.squeeze(stacked_output_choice))

	grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

	grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
	disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

	return gen_loss, disc_loss



def WGAN_discriminator_loss(real_img, fake_img):
	real_loss = tf.reduce_mean(real_img)
	fake_loss = tf.reduce_mean(fake_img)
	return fake_loss - real_loss

def WGAN_generator_loss(fake_img):
	return -tf.reduce_mean(fake_img)

def WGAN_gradient_penalty(discriminator, batch_size, real_images, real_conditions, fake_images):
	# Get the interpolated image
	alpha = tf.random.normal([batch_size, tf.shape(fake_images)[1]], 0.0, 1.0)
	diff = fake_images - real_images
	interpolated = real_images + alpha * diff

	with tf.GradientTape() as gp_tape:
		gp_tape.watch(interpolated)
		pred = discriminator([interpolated, real_conditions], training=True)

	grads = gp_tape.gradient(pred, [interpolated])[0]
	norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))

	liptschitz_penalty = False
	if not liptschitz_penalty:
		gp = tf.reduce_mean((norm - 1.0) ** 2)
	else:
		gp = tf.reduce_mean(tf.clip_by_value(norm - 1., 0., np.infty)**2)
	return gp


d_steps = 5 # number of steps to train D for every one generator step
# gp_weight = 10.
gp_weight = 0.1

@tf.function
def train_step_vertexing_WGAN(
	batch_size, generator, discriminator, gen_optimizer, disc_optimizer, images, cut_idx, latent_dim
):

	sample_targets, sample_conditions = images[:, 0, :cut_idx], images[:, 0, cut_idx:]
	sample_targets_train_A = sample_targets[:batch_size]
	sample_targets_train_B = sample_targets[batch_size:2*batch_size]
	sample_targets_train_C = sample_targets[2*batch_size:]
	sample_conditions_train_A = sample_conditions[:batch_size]
	sample_conditions_train_B = sample_conditions[batch_size:2*batch_size]
	sample_conditions_train_C = sample_conditions[2*batch_size:]


	for i in range(d_steps):
			# Get the latent vector
			random_latent_vectors = tf.random.normal((batch_size, latent_dim), 0, 1)
			with tf.GradientTape() as tape:
				# Generate fake images from the latent vector
				fake_images = generator([random_latent_vectors, sample_conditions_train_A], training=True)
				# Get the logits for the fake images
				fake_logits = discriminator([fake_images, sample_conditions_train_A], training=True)
				# Get the logits for the real images
				real_logits = discriminator([sample_targets_train_B, sample_conditions_train_B], training=True)

				# Calculate the discriminator loss using the fake and real image logits
				d_cost = WGAN_discriminator_loss(real_img=real_logits, fake_img=fake_logits)
				# Calculate the gradient penalty

				gp = WGAN_gradient_penalty(discriminator, batch_size, sample_targets_train_A, sample_conditions_train_A, fake_images)
				# Add the gradient penalty to the original discriminator loss
				d_loss = d_cost + gp * gp_weight

				# Get the gradients w.r.t the discriminator loss
				d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
				# Update the weights of the discriminator using the discriminator optimizer
				disc_optimizer.apply_gradients(zip(d_gradient, discriminator.trainable_variables))

	# Train the generator
	random_latent_vectors = tf.random.normal((batch_size, latent_dim), 0, 1)
	with tf.GradientTape() as tape:
		# Generate fake images using the generator
		generated_images = generator([random_latent_vectors, sample_conditions_train_C], training=True)
		# Get the discriminator logits for fake images
		gen_img_logits = discriminator([generated_images, sample_conditions_train_C], training=True)
		# Calculate the generator loss
		g_loss = WGAN_generator_loss(gen_img_logits)

		# Get the gradients w.r.t the generator loss
		gen_gradient = tape.gradient(g_loss, generator.trainable_variables)
		# Update the weights of the generator using the generator optimizer
		gen_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))

	return d_loss, g_loss





@tf.function  # function repeated because tf compiles this, if shape of network changes it aint happy
def train_step(vae, optimizer, images, cut_idx, kl_factor, reco_factor, toggle_kl):

	sample_targets, sample_conditions = images[:, 0, :cut_idx], images[:, 0, cut_idx:]

	with tf.GradientTape() as tape:

		vae_out, vae_z_mean, vae_z_log_var = vae([sample_targets, sample_conditions])

		vae_reco_loss = reco_loss(sample_targets, vae_out)
		vae_reco_loss_raw = tf.math.reduce_mean(vae_reco_loss)
		vae_reco_loss = vae_reco_loss_raw * reco_factor
		vae_kl_loss = kl_loss(vae_z_mean, vae_z_log_var)
		vae_kl_loss = tf.math.reduce_mean(vae_kl_loss) * toggle_kl * kl_factor

		vae_loss = vae_kl_loss + vae_reco_loss

	grad_vae = tape.gradient(vae_loss, vae.trainable_variables)

	optimizer.apply_gradients(zip(grad_vae, vae.trainable_variables))

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
