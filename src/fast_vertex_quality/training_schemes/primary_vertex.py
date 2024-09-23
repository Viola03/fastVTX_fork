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
import pandas as pd
from tensorflow.keras.optimizers.legacy import Adam

class primary_vertex_trainer:

	def __init__(
		self,
		data_loader_obj=None,
		targets=None,
		conditions=None,
		beta=1000.0,
		latent_dim=4,
		batch_size=50,
		E_architecture=[150, 250, 150],
		D_architecture=[150, 250, 150],
		G_architecture=None,
		network_option='VAE',
		load_config=None,
	):
	
		if load_config:

			tag = load_config
			try:
				rd.network_option, rd.latent, rd.D_architecture, rd.G_architecture, rd.beta, rd.conditions, rd.targets, rd.use_QuantileTransformer = pickle.load(open(f"{tag}_configs.pkl", "rb"))
			except:
				rd.network_option, rd.latent, rd.D_architecture, rd.G_architecture, rd.beta, rd.conditions, rd.targets = pickle.load(open(f"{tag}_configs.pkl", "rb"))

			self.targets = rd.targets
			self.conditions = rd.conditions
			self.network_option = rd.network_option
			self.latent_dim = rd.latent
			self.G_architecture = rd.G_architecture
			self.D_architecture = rd.D_architecture
			self.reco_factor = rd.beta

			print('\n\n')
			print(f"network_option: {self.network_option}")
			print(f"G_architecture: {self.G_architecture}")
			print(f"D_architecture: {self.D_architecture}")
			print(f"latent_dim: {self.latent_dim}")
			print(f"beta reco_factor: {self.reco_factor}")
			print(f"targets: {self.targets}")
			print('\n')
			print(f"conditions: {self.conditions}")
			print('\n\n')

		else:
			if targets == None:
				self.targets = [
					"B_plus_TRUE_FD",
					"B_plus_TRUEORIGINVERTEX_X",
					"B_plus_TRUEORIGINVERTEX_Y",
					"B_plus_TRUEORIGINVERTEX_Z",
				]
			else:
				self.targets = targets

			if conditions == None:
				self.conditions = [
					"B_plus_TRUEP",
					"B_plus_TRUEP_T",
					"B_plus_TRUEP_X",
					"B_plus_TRUEP_Y",
					"B_plus_TRUEP_Z",
				]
			else:
				self.conditions = conditions

			self.reco_factor = beta
			self.D_architecture = D_architecture
			self.G_architecture = G_architecture
			self.latent_dim = latent_dim
			self.network_option = network_option

		self.kl_factor = 1.0
		self.batch_size = batch_size

		self.E_architecture = E_architecture
		
		self.target_dim = len(self.targets)
		self.conditions_dim = len(self.conditions)
		self.cut_idx = self.target_dim

		self.optimizer = Adam(learning_rate=0.00005)

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

		if data_loader_obj:
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
			# if (
			#     self.iteration % 1000 == 0
			# ):  # annealing https://arxiv.org/pdf/1511.06349.pdf
			#     self.toggle_kl_value = 0.0
			# elif self.iteration % 1000 < 500:
			#     self.toggle_kl_value += 1.0 / 500.0
			# else:
			#     self.toggle_kl_value = 1.0
		
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
					print(
						f"NaNs present in loss_list, quitting on iteration {self.iteration}..."
					)
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
		X_test_data_loader.add_branch_and_process(name='B_plus_TRUE_FD',recipe="sqrt((B_plus_TRUEENDVERTEX_X-B_plus_TRUEORIGINVERTEX_X)**2 + (B_plus_TRUEENDVERTEX_Y-B_plus_TRUEORIGINVERTEX_Y)**2 + (B_plus_TRUEENDVERTEX_Z-B_plus_TRUEORIGINVERTEX_Z)**2)")
		X_test_data_loader.add_branch_and_process(name='B_plus_TRUEP',recipe="sqrt((B_plus_TRUEP_X)**2 + (B_plus_TRUEP_Y)**2 + (B_plus_TRUEP_Z)**2)")
		X_test_data_loader.add_branch_and_process(name='B_plus_TRUEP_T',recipe="sqrt((B_plus_TRUEP_X)**2 + (B_plus_TRUEP_Y)**2)")


		X_test_data_loader.select_randomly(Nevents=N)

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
				"datasets/B2KEE_three_body_cut_more_vars.root",
			],
			transformers=self.transformers,
			convert_to_RK_branch_names=True,
			conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_Kst', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
		)

		X_test_data_loader.select_randomly(Nevents=N)

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

		print("Saving state...")
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

		objects = (rd.network_option, rd.latent, rd.D_architecture, rd.G_architecture, rd.beta, rd.conditions, rd.targets, rd.use_QuantileTransformer)

		pickle.dump(
			objects,
			open(
				f"{tag}_configs.pkl",
				"wb",
			),
		)

		print("saved.")

	# def save_state(self, tag):

	#     pickle.dump(
	#         self.transformers,
	#         open(
	#             f"{tag}_transfomers.pkl",
	#             "wb",
	#         ),
	#     )

	#     pickle.dump(
	#         self.trained_weights,
	#         open(
	#             f"{tag}_trained_weights.pkl",
	#             "wb",
	#         ),
	#     )

	def load_state(self, tag):

		self.transformers = pickle.load(open(f"{tag}_transfomers.pkl", "rb"))
		self.trained_weights = pickle.load(open(f"{tag}_trained_weights.pkl", "rb"))

		self.set_trained_weights()

		# decoder = tf.keras.models.load_model("save_state/decoder.h5")
		# self.trained_weights = decoder.get_weights()

	def predict(self, inputs):

		self.set_trained_weights()

		return self.decoder.predict(inputs)

	def predict_from_data_loader(self, data_loader_obj):

		self.set_trained_weights()

		events_gen = data_loader_obj.get_branches(self.conditions, processed=True)

		events_gen = np.asarray(events_gen[self.conditions])

		gen_noise = np.random.normal(0, 1, (np.shape(events_gen)[0], self.latent_dim))
		
		if self.network_option == 'VAE':
			images = np.squeeze(self.decoder.predict([gen_noise, events_gen]))
		elif self.network_option == 'GAN' or self.network_option == 'WGAN':
			images = np.squeeze(self.generator.predict([gen_noise, events_gen]))

		data_loader_obj.fill_target(images, self.targets)

		return data_loader_obj

	def predict_physical_from_physical_pandas(self, conditions, targets):
		
		self.set_trained_weights()

		for branch in list(conditions.keys()):
			conditions[branch] = self.transformers[branch.replace('MOTHER','B_plus')].process(np.asarray(conditions[branch]))

		events_gen = np.asarray(conditions)
		gen_noise = np.random.normal(0, 1, (np.shape(events_gen)[0], self.latent_dim))

		if self.network_option == 'VAE':
			images = np.squeeze(self.decoder.predict([gen_noise, events_gen]))
		elif self.network_option == 'GAN' or self.network_option == 'WGAN':
			images = np.squeeze(self.generator.predict([gen_noise, events_gen]))


		for branch in list(conditions.keys()):
			
			conditions[branch] = self.transformers[branch.replace('MOTHER','B_plus')].unprocess(np.asarray(conditions[branch]))


		images_dict = {}

		for i in range(np.shape(images)[1]):
			images_dict[targets[i]] = images[:,i]
		
		images = pd.DataFrame(images_dict)

		for branch in list(images.keys()):
			
			images[branch] = self.transformers[branch.replace('MOTHER','B_plus')].unprocess(np.asarray(images[branch]))

		return images
