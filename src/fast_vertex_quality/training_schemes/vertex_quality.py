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
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.optimizers.legacy import SGD
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

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
		load_config=False,
	):

		self.trackchi2_trainer = trackchi2_trainer

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

		# self.optimizer = Adam(learning_rate=0.0005) # default
		self.optimizer = Adam(learning_rate=0.00005)
		
		# # self.optimizer = Adam(learning_rate=0.000005)
		# gen_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		# 			initial_learning_rate=0.000005,
		# 			decay_steps=5000,
		# 			decay_rate=0.5)
		# # self.optimizer = Adam(learning_rate=0.000005, amsgrad=True)
		# self.optimizer = Adam(learning_rate=0.000005)

		# self.optimizer = Adam(learning_rate=0.00075, beta_1=0.5, amsgrad=True)
		# self.optimizer = SGD(learning_rate=0.0005)

		if self.network_option == 'GAN':
			# self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.5, amsgrad=True)
			# self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.5, amsgrad=True)
			self.gen_optimizer = Adam(learning_rate=0.0004, beta_1=0.5, amsgrad=True)
			self.disc_optimizer = Adam(learning_rate=0.0004, beta_1=0.5, amsgrad=True)
		elif self.network_option == 'WGAN':
			# self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
			# self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

			# self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=0.0001, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
			# self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=0.00025, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

			# self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=0.00005, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
			# self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=0.0001, beta1=0.5)#tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

			# gen_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			# 		initial_learning_rate=0.00005,
			# 		# initial_learning_rate=0.000005,
			# 		decay_steps=5000,
			# 		decay_rate=0.9,
			# 	)
			# self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=gen_lr_schedule, beta1=0.5)
			# disc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			# 		initial_learning_rate=0.0001,
			# 		# initial_learning_rate=0.00001,
			# 		decay_steps=5000,
			# 		decay_rate=0.9,
			# 	)
			# self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=disc_lr_schedule, beta1=0.5)


			# self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=0.00001, beta1=0.5)
			# self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=0.000005, beta1=0.5)
			# self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=0.00005)
			# self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=0.000005)
			self.gen_optimizer = tfa.optimizers.Yogi(learning_rate=0.00005)
			self.disc_optimizer = tfa.optimizers.Yogi(learning_rate=0.0001)


			# self.gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0004)
			# self.disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0004)
			# self.gen_optimizer = RMSprop(learning_rate=0.0004)
			# self.disc_optimizer = RMSprop(learning_rate=0.0004)

			# self.gen_optimizer = Adam(learning_rate=0.0004, beta_1=0.5, amsgrad=True)
			# self.disc_optimizer = Adam(learning_rate=0.0004, beta_1=0.5, amsgrad=True)
			

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

	def initalise_BDT_test(self, BDT_tester_obj, BDT_cut=0.9):

		self.BDT_tester_obj = BDT_tester_obj
		# self.event_loader_MC, self.event_loader_gen_MC, self.event_loader_RapidSim = BDT_tester_obj.get_event_loaders_for_live_tests(self)
		self.event_loader_MC, self.event_loader_MC_stripping_effs = BDT_tester_obj.get_event_loaders_for_live_tests(self)
		self.BDT_cut = BDT_cut


	def run_BDT_test(self, filename='outBDT.pdf'):

		####
		###############
		self.event_loader_RapidSim = self.BDT_tester_obj.get_event_loader(
			# "/users/am13743/fast_vertexing_variables/rapidsim/Kee/Signal_tree_NNvertex_more_vars.root",
			"/users/am13743/fast_vertexing_variables/rapidsim/Kee/Signal_tree_LARGE_NNvertex_more_vars.root",
			self,
			generate=True,
			N=100000,
			# N=200000,
			# N=-1,
			convert_branches=True,
			rapidsim=True,
		)  
		
		event_loader_RapidSim_stripping_effs = self.BDT_tester_obj.get_stripping_eff(self.event_loader_RapidSim)

		self.event_loader_RapidSim.add_dalitz_masses()

		self.event_loader_RapidSim = self.predict_from_data_loader(
				self.event_loader_RapidSim
			)
		self.event_loader_RapidSim.fill_stripping_bool()
		self.event_loader_RapidSim.cut("pass_stripping")

		BDT_scores = self.BDT_tester_obj.get_BDT_scores(
			self.event_loader_RapidSim,
			generate=True
		)  
		self.event_loader_RapidSim.add_branch_to_physical("BDT_score", np.asarray(BDT_scores))
		########################

		self.event_loader_gen_MC = self.BDT_tester_obj.get_event_loader(
			"datasets/Kee_Merge_cut_chargeCounters_more_vars.root",
			self,
			generate=True,
			N=100000,
			# N=200000,
			# N=-1,
			convert_branches=True,
			rapidsim=False,
		)  

		event_loader_gen_MC_stripping_effs = self.BDT_tester_obj.get_stripping_eff(self.event_loader_gen_MC)

		self.event_loader_gen_MC.add_dalitz_masses()


		self.event_loader_gen_MC = self.predict_from_data_loader(
				self.event_loader_gen_MC
			)
		self.event_loader_gen_MC.fill_stripping_bool()
		self.event_loader_gen_MC.cut("pass_stripping")


		BDT_scores = self.BDT_tester_obj.get_BDT_scores(
			self.event_loader_gen_MC,
			generate=True
		)  

		self.event_loader_gen_MC.add_branch_to_physical("BDT_score", np.asarray(BDT_scores))

		chi2 = [0,0,0]
		with PdfPages(filename) as pdf:
			

			plt.title(r"$B^+\to K^+e^+e^-$")
			plt.errorbar(np.arange(np.shape(self.event_loader_MC_stripping_effs)[0]), self.event_loader_MC_stripping_effs[:,0], yerr=self.event_loader_MC_stripping_effs[:,1],label=r"$B^+\to K^+e^+e^-$ MC",color='tab:blue',linestyle='-')

			plt.errorbar(np.arange(np.shape(event_loader_gen_MC_stripping_effs)[0]), event_loader_gen_MC_stripping_effs[:,0], yerr=event_loader_gen_MC_stripping_effs[:,1],label=r"Generated $B^+\to K^+e^+e^-$ (MC)",color='tab:green')

			plt.errorbar(np.arange(np.shape(event_loader_RapidSim_stripping_effs)[0]), event_loader_RapidSim_stripping_effs[:,0], yerr=event_loader_RapidSim_stripping_effs[:,1],label=r"Generated $B^+\to K^+e^+e^-$ (Rapidsim)",color='tab:orange')

			plt.ylim(0,1)
			# cuts_ticks = ['All']+list(self.BDT_tester_obj.cuts.keys())
			# plt.xticks(np.arange(len(cuts_ticks)), cuts_ticks, rotation=90)
			# for i in np.arange(len(cuts_ticks)):
			# 	if i ==0:
			# 		plt.axvline(x=i, alpha=0.5, ls='-',c='k')
			# 	else:
			# 		plt.axvline(x=i, alpha=0.5, ls='--',c='k')
			plt.legend(frameon=False)
			plt.ylabel("Cut Efficiency")
			pdf.savefig(bbox_inches="tight")
			plt.close()


			eff_A, effErr_A, eff_C, effErr_C = self.BDT_tester_obj.plot_efficiency_as_a_function_of_variable(pdf, self.event_loader_MC, self.event_loader_gen_MC, self.event_loader_RapidSim, "q2", f"BDT_score>{self.BDT_cut}", [0,25], r"$B^+\to K^+e^+e^-$", xlabel=r'$q^2$ (GeV$^2$)', signal=True, return_values=True)

			where = np.where((effErr_A<0.2)&(effErr_C<2)&(effErr_A>0))
			dof = np.shape(where)[1]-1
			chi2[0] = np.sum((eff_C[where]-eff_A[where])**2/(effErr_A[where]))/dof

			eff_A, effErr_A, eff_C, effErr_C = self.BDT_tester_obj.plot_efficiency_as_a_function_of_variable(pdf, self.event_loader_MC, self.event_loader_gen_MC, self.event_loader_RapidSim, "sqrt_dalitz_mass_mkl", f"BDT_score>{self.BDT_cut}", [0,5.3], r"$B^+\to K^+e^+e^-$", xlabel=r'$m(Ke)$ (GeV)', signal=True, return_values=True)
			where = np.where((effErr_A<0.2)&(effErr_C<2)&(effErr_A>0))
			dof = np.shape(where)[1]-1
			chi2[1] = np.sum((eff_C[where]-eff_A[where])**2/(effErr_A[where]))/dof

			eff_A, effErr_A, eff_C, effErr_C = self.BDT_tester_obj.plot_efficiency_as_a_function_of_variable(pdf, self.event_loader_MC, self.event_loader_gen_MC, self.event_loader_RapidSim, "B_plus_M_Kee_reco", f"BDT_score>{self.BDT_cut}", [4,5.7], r"$B^+\to K^+e^+e^-$", xlabel=r'$m(Kee)$ (GeV)', signal=True, return_values=True)
			where = np.where((effErr_A<0.2)&(effErr_C<2)&(effErr_A>0))
			dof = np.shape(where)[1]-1
			chi2[2] = np.sum((eff_C[where]-eff_A[where])**2/(effErr_A[where]))/dof

		print(f'{filename} plotted.')
		return chi2


	def old_step(self, samples_for_batch):
		
		if self.network_option == 'VAE':
			# if (
			# 	self.iteration % 1000 == 0
			# ):  # annealing https://arxiv.org/pdf/1511.06349.pdf
			# 	self.toggle_kl_value = 0.0
			# # elif self.iteration % 1000 < 500:
			# 	# self.toggle_kl_value += 1.0 / 500.0
			# elif self.iteration % 1000 < 995:
			# 	self.toggle_kl_value += 1.0 / 995.0
			# else:
			# 	self.toggle_kl_value = 1.0


			# annealing_start = 0. # most aggressive
			# annealing_start = 0.1 # this means, the starting value of the anneals will be 10 times higher than the actual beta value
			annealing_start = 0.33 
			if (
				self.iteration % 1000 == 0
			):  # annealing https://arxiv.org/pdf/1511.06349.pdf
				self.toggle_kl_value = annealing_start
			elif self.iteration % 1000 < 995:
				self.toggle_kl_value += (1.0-annealing_start) / 995.0
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
				rd.current_mse_raw,
			)
			rd.current_mse_raw = reco_loss_np_raw

			plot_out_to = 5000
			if self.iteration == 0:
				self.annealing_history = np.empty((0,2))
				if self.toggle_kl_value == 0:
					self.toggle_kl_value = 1E-6
				self.annealing_history = np.append(self.annealing_history, [[self.reco_factor, self.kl_factor*self.toggle_kl_value]], axis=0)
			elif self.iteration < plot_out_to:
				if self.toggle_kl_value == 0:
					self.toggle_kl_value = 1E-6
				self.annealing_history = np.append(self.annealing_history, [[self.reco_factor, self.kl_factor*self.toggle_kl_value]], axis=0)
			if self.iteration == plot_out_to:
				y = self.annealing_history[:,0]/self.annealing_history[:,1]
				plt.plot(np.arange(plot_out_to), self.annealing_history[:,0]/self.annealing_history[:,1], label='Reco/KL')
				plt.ylabel(r'$\beta$')
				plt.xlabel("Iteration")
				if annealing_start == 0.:
					plt.ylim(100,500000)
				plt.xlim(0,plot_out_to)
				plt.yscale('log')
				plt.savefig('annealing')
				plt.close('all')


			return [self.iteration, kl_loss_np, reco_loss_np, reco_loss_np_raw, 0., 0.]
		
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
			return [self.iteration, disc_loss_np, gen_loss_np, 0., 0., 0.]

		elif self.network_option == 'WGAN':
			disc_loss_np, gen_loss_np, disc_grad_norm_np, gen_grad_norm_np = train_step_WGAN(
				self.batch_size,
				self.generator,
				self.discriminator,
				self.gen_optimizer,
				self.disc_optimizer,
				samples_for_batch,
				self.cut_idx,
				self.latent_dim,
			)
			# print('loss', disc_loss_np, gen_loss_np)
			# print('norms', disc_grad_norm_np, gen_grad_norm_np)
			return [self.iteration, disc_loss_np, gen_loss_np, 0.,disc_grad_norm_np, gen_grad_norm_np]
			
	def step(self, samples_for_batch):
		
		if self.network_option == 'VAE':
			
			self.toggle_kl_value = 1.0

			# self.iteration

			toggle_kl = tf.convert_to_tensor(self.toggle_kl_value)


			if rd.use_beta_schedule:
				inital_boost_factor = 5. # initial boost factor, start reco loss x5 the default
				inital_boost_halflife = 10. # short halflife of decaying initial boost
				long_term_KL_boost_term_halflife = 50000. # halflife of long slow rise in KL loss
				cap_KL_boost = 3. # at most the KL boost will be x3 the default

				initial_boost_term = (inital_boost_factor-1.)*self.reco_factor*np.exp(-np.log(2.)*(self.iteration/inital_boost_halflife))    
				long_term_KL_boost_term = (1./np.exp(np.log(2.)*(self.iteration/long_term_KL_boost_term_halflife)))*(1.-1/cap_KL_boost)+(1./cap_KL_boost)
				self.reco_factor_employ = (self.reco_factor + initial_boost_term)*long_term_KL_boost_term
			else:
				self.reco_factor_employ = self.reco_factor
				
		
			kl_loss_np, reco_loss_np, reco_loss_np_raw = train_step(
				self.vae,
				self.optimizer,
				samples_for_batch,
				self.cut_idx,
				tf.convert_to_tensor(self.kl_factor), # 1.
				tf.convert_to_tensor(self.reco_factor_employ, dtype=tf.float32),
				toggle_kl,
				rd.current_mse_raw,
			)
			rd.current_mse_raw = reco_loss_np_raw
			
			return [self.iteration, kl_loss_np, reco_loss_np, reco_loss_np_raw, 0., 0.]
		
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
			return [self.iteration, disc_loss_np, gen_loss_np, 0., 0., 0.]

		elif self.network_option == 'WGAN':
			disc_loss_np, gen_loss_np, disc_grad_norm_np, gen_grad_norm_np = train_step_WGAN(
				self.batch_size,
				self.generator,
				self.discriminator,
				self.gen_optimizer,
				self.disc_optimizer,
				samples_for_batch,
				self.cut_idx,
				self.latent_dim,
			)
			# print('loss', disc_loss_np, gen_loss_np)
			# print('norms', disc_grad_norm_np, gen_grad_norm_np)
			return [self.iteration, disc_loss_np, gen_loss_np, 0.,disc_grad_norm_np, gen_grad_norm_np]

	def train_more_steps(self, steps=10000, reset_optimizer_state=False):
		
		if rd.network_option == 'WGAN':
			if reset_optimizer_state:
				self.gen_optimizer.set_weights(self.gen_optimizer_weights)
				self.disc_optimizer.set_weights(self.disc_optimizer_weights)

		private_iteration = -1

		break_option = False
		for epoch in range(int(1e30)):

			if self.data_loader_obj.reweight_for_training_bool:
				print("loading training events with weights")
				X_train_data_all_pp = self.data_loader_obj.get_branches(
					self.targets + self.conditions + ['training_weight'], processed=True, option='training'
				)
				X_train_data_all_pp = X_train_data_all_pp.sample(frac=1, weights=X_train_data_all_pp['training_weight'],replace=True)
				X_train_data_all_pp = X_train_data_all_pp.drop(columns=['training_weight'])

			else:
				X_train_data_all_pp = self.data_loader_obj.get_branches(
					self.targets + self.conditions, processed=True, option='training'
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
				
				try:
					self.iteration += 1
				except:
					self.iteration = 0
					self.loss_list = np.empty((0,4))

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

		self.plot_losses()

	def plot_losses(self):

		plt.figure(figsize=(12,12))

		plt.subplot(3, 3, 1)
		plt.title('KL loss/critic loss')
		plt.plot(self.loss_list[:, 0], self.loss_list[:, 1])
		plt.plot(self.loss_list[:, 0][0::100], self.loss_list[:, 1][0::100])
		plt.subplot(3, 3, 2)
		plt.title('Reco loss/gen loss')
		plt.plot(self.loss_list[:, 0], self.loss_list[:, 2])
		plt.plot(self.loss_list[:, 0][0::100], self.loss_list[:, 2][0::100])
		plt.subplot(3, 3, 3)
		plt.title('Reco loss raw')
		plt.plot(self.loss_list[:, 0], self.loss_list[:, 3])
		plt.plot(self.loss_list[:, 0][0::100], self.loss_list[:, 3][0::100])

		try:
			plt.subplot(3, 3, 4)
			plt.title('KL loss/critic loss')
			plt.plot(self.loss_list[-5000:, 0], self.loss_list[-5000:, 1])
			plt.subplot(3, 3, 5)
			plt.title('Reco loss/ gen loss')
			plt.plot(self.loss_list[-5000:, 0], self.loss_list[-5000:, 2])
			plt.subplot(3, 3, 6)
			plt.title('Reco loss raw')
			plt.plot(self.loss_list[-5000:, 0], self.loss_list[-5000:, 3])
		except:
			pass

		try:
			plt.subplot(3, 3, 7)
			plt.title('norm grad critic')
			plt.plot(self.loss_list[:, 0], self.loss_list[:, 4])
			# plt.axhline(y=1,c='k',alpha=0.25)
			plt.subplot(3, 3, 9)
			plt.title('norm grad critic, last 100')
			plt.plot(self.loss_list[-100:, 0], self.loss_list[-100:, 4])
			# plt.axhline(y=1,c='k',alpha=0.25)
			plt.subplot(3, 3, 8)
			plt.title('norm grad gen')
			plt.plot(self.loss_list[:, 0], self.loss_list[:, 5])
			# plt.axhline(y=1,c='k',alpha=0.25)
		except:
			pass

		plt.savefig(f"{rd.test_loc}Losses.png")
		plt.close("all")

	def train(self, steps=10000):

		self.set_initialised_weights()

		self.iteration = -1

		self.loss_list = np.empty((0, 6))

		break_option = False
		for epoch in range(int(1e30)):

			if self.data_loader_obj.reweight_for_training_bool:
				print("loading training events with weights")
				X_train_data_all_pp = self.data_loader_obj.get_branches(
					self.targets + self.conditions + ['training_weight'], processed=True, option='training'
				)
				X_train_data_all_pp = X_train_data_all_pp.sample(frac=1, weights=X_train_data_all_pp['training_weight'],replace=True)
				X_train_data_all_pp = X_train_data_all_pp.drop(columns=['training_weight'])

			else:
				X_train_data_all_pp = self.data_loader_obj.get_branches(
					self.targets + self.conditions, processed=True, option='training'
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
				
				try:
					self.iteration += 1
				except:
					self.iteration = 0
					self.loss_list = np.empty((0, 4))

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
				
				# if self.iteration == 1:
				# 	self.gen_optimizer_weights = self.gen_optimizer.get_weights()
				# 	self.disc_optimizer_weights = self.disc_optimizer.get_weights()

				if self.iteration % 500 == 0 and self.iteration > 1:
					# plt.subplot(1, 3, 1)
					# plt.title('disc')
					# # for i in range(10):
					# # 	if i*100 <= self.iteration:
					# # 		plt.axvline(x=i*100.,c='k',alpha=0.25)
					# plt.plot(self.loss_list[:, 0], self.loss_list[:, 1])
					# plt.subplot(1, 3, 2)
					# plt.title('gen')
					# # for i in range(10):
					# # 	if i*100 <= self.iteration:
					# # 		plt.axvline(x=i*100.,c='k',alpha=0.25)
					# plt.plot(self.loss_list[:, 0], self.loss_list[:, 2])
					# plt.subplot(1, 3, 3)
					# plt.plot(self.loss_list[:, 0], self.loss_list[:, 3])
					# plt.savefig("Losses.png")
					# plt.close("all")

					self.plot_losses()

			if break_option:
				break

		self.trained_weights = self.get_weights()

		# plt.subplot(1, 3, 1)
		# plt.title('disc')
		# plt.plot(self.loss_list[:, 0], self.loss_list[:, 1])
		# plt.subplot(1, 3, 2)
		# plt.title('gen')
		# plt.plot(self.loss_list[:, 0], self.loss_list[:, 2])
		# plt.subplot(1, 3, 3)
		# plt.plot(self.loss_list[:, 0], self.loss_list[:, 3])
		# plt.savefig("Losses.png")
		# plt.close("all")
		self.plot_losses()

		if rd.network_option == 'WGAN':
			self.gen_optimizer_weights = self.gen_optimizer.get_weights()
			self.disc_optimizer_weights = self.disc_optimizer.get_weights()

	def make_plots(self, N=10000, filename=f"plots", testing_file="datasets/B2KEE_three_body_cut_more_vars.root", offline=False):

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
		X_test_data_loader.add_missing_mass_frac_branch()


		X_test_data_loader.select_randomly(Nevents=N)

		if self.trackchi2_trainer is not None:
			X_test_data_loader.fill_chi2_gen(self.trackchi2_trainer)

		X_test_conditions = X_test_data_loader.get_branches(
			self.conditions, processed=True
		)
		X_test_conditions = X_test_conditions[self.conditions]
		X_test_conditions = np.asarray(X_test_conditions)


		X_test_targets = X_test_data_loader.get_branches(
			self.targets, processed=True
		)
		X_test_targets = X_test_targets[self.targets]
		X_test_targets = np.asarray(X_test_targets)



		if self.network_option == 'VAE':
			latent = np.squeeze(self.encoder.predict([X_test_targets, X_test_conditions]))
			z = latent[0]

			with PdfPages(filename.replace('.pdf','_latent.pdf')) as pdf:
				for i in range(np.shape(z)[1]):
					for j in range(i+1, np.shape(z)[1]):
						plt.hist2d(z[:,i],z[:,j],bins=25,range=[[-5,5],[-5,5]],norm=LogNorm())
						pdf.savefig(bbox_inches="tight")
						plt.close()

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
			offline=offline
		)
	
	def gen_data(self, filename, N=10000):

		self.set_trained_weights()

		gen_noise = np.random.normal(0, 1, (N, self.latent_dim))

		X_test_data_loader = data_loader.load_data(
			[
				# "datasets/Kee_2018_truthed_more_vars.csv",
				# "datasets/B2KEE_three_body_cut_more_vars.root",
				"datasets/dedicated_Kee_MC_hierachy_cut_more_vars.root",
				# "datasets/dedicated_Kstee_MC_hierachy_cut_more_vars.root",
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

		# data_loader_obj.plot(f'conditions_{np.random.randint(0,9999)}.pdf',self.conditions)

		events_gen = data_loader_obj.get_branches(self.conditions, processed=True)

		events_gen = np.asarray(events_gen[self.conditions])

		gen_noise = np.random.normal(0, 1, (np.shape(events_gen)[0], self.latent_dim))
		
		if self.network_option == 'VAE':
			images = np.squeeze(self.decoder.predict([gen_noise, events_gen]))
		elif self.network_option == 'GAN' or self.network_option == 'WGAN':
			images = np.squeeze(self.generator.predict([gen_noise, events_gen]))

		# # add B_plus_ENDVERTEX_NDOF
		# # print(np.shape(images))
		# if 'B_plus_ENDVERTEX_NDOF' not in self.targets:
		#     images = np.concatenate((images, np.expand_dims(np.ones(np.shape(images)[0])*3,1)),axis=1)
		#     self.targets.append('B_plus_ENDVERTEX_NDOF')
		#     # print(np.shape(images))
		#     # print(len(self.targets))

		data_loader_obj.fill_target(images, self.targets)

		return data_loader_obj
