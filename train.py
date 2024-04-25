from fast_vertex_quality.tools.config import read_definition, rd




import fast_vertex_quality.tools.data_loader as data_loader
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import pandas as pd
import pickle
# values = np.empty(0)
# for iteration in range(10000):
# 	if iteration % 1000 == 0: # annealing https://arxiv.org/pdf/1511.06349.pdf
# 		toggle_kl_value = 0
# 	elif iteration % 1000 < 500:
# 		toggle_kl_value += 1/500
# 	else:
# 		toggle_kl_value = 1
# 	values = np.append(values, toggle_kl_value)
# plt.plot(values)
# plt.savefig('values')
# quit()

def plot(data, gen_data, filename, Nevents=10000):

	print(f'Plotting {filename}.pdf....')

	data_all, data_targets, data_condtions = data.get_physical_data()
	data_all_pp, data_targets_pp, data_condtions_pp = data.get_processed_data()
	data_physics = data.get_physics_variables()

	gen_data_all, gen_data_targets, gen_data_condtions = gen_data.get_physical_data()
	gen_data_all_pp, gen_data_targets_pp, gen_data_condtions_pp = gen_data.get_processed_data()
	gen_data_physics = gen_data.get_physics_variables()


	columns = list(data_all.keys())
	N = len(columns)
	
	with PdfPages(f'{filename}.pdf') as pdf:

		for i in range(0,N):
			
			if columns[i] in rd.targets:

				plt.figure(figsize=(8,4))
				plt.subplot(1,2,1)
				plt.hist([data_all[columns[i]][:Nevents], gen_data_all[columns[i]][:Nevents]], bins=75, histtype='step', label=['truth','gen'])
				plt.xlabel(columns[i])
				plt.legend()
				plt.subplot(1,2,2)
				plt.hist([data_all_pp[columns[i]][:Nevents], gen_data_all_pp[columns[i]][:Nevents]], bins=75, histtype='step')
				plt.xlabel(columns[i])
				pdf.savefig(bbox_inches='tight')
				plt.close()

		for particle in ['K_Kst','e_minus','e_plus']:

			plt.figure(figsize=(12,16))
			idx = 0
			for i in range(0,N):

				if "B_plus" in columns[i] or particle in columns[i] and columns[i] in rd.targets:

					idx+=1

					plt.subplot(4,3,idx)        
					hist = plt.hist2d(np.log10(data_physics[f'{particle}_P'][:Nevents]), data_all_pp[columns[i]][:Nevents], norm=LogNorm(),bins=35, cmap='Reds')
					plt.xlabel(f"log {particle} P")
					plt.ylabel(columns[i])

					plt.subplot(4,3,idx+3)        
					plt.hist2d(np.log10(gen_data_physics[f'{particle}_P'][:Nevents]), gen_data_all_pp[columns[i]][:Nevents], norm=LogNorm(),bins=[hist[1],hist[2]], cmap='Blues')
					plt.xlabel(f"log {particle} P")
					plt.ylabel(columns[i])

					if idx % 3 == 0 and idx > 0:
						idx += 3

			pdf.savefig(bbox_inches='tight')
			plt.close()


# training_data = data_loader.load_data('datasets/Kee_2018_truthed.csv')

# truth_plots = plot(training_data, training_data, 'truth', Nevents=10000)

# quit()




from fast_vertex_quality.tools.config import read_definition, rd
import fast_vertex_quality.tools.data_loader as data_loader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np


import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Concatenate, Lambda, ReLU, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
_EPSILON = K.epsilon()

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import scipy

import math
import glob
import time
import shutil
import os

from pickle import load
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=K.shape(z_mean), mean=0, stddev=1)
	return z_mean + K.exp(z_log_var / 2) * epsilon
	
def reco_loss(x, x_decoded_mean):
	# xent_loss = tf.keras.losses.mean_absolute_error(x, x_decoded_mean)
	xent_loss = tf.keras.losses.mean_squared_error(x, x_decoded_mean)
	# xent_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
	return xent_loss

def kl_loss(z_mean, z_log_var):
	kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	return kl_loss

def split_tensor(index, x):
	return Lambda(lambda x : x[:,index])(x)

print(tf.__version__)

latent_dim = 6
kl_factor = 1
# reco_factor = 1E3
reco_factor = 100

target_dim = 10
conditions_dim = 11

batch_size = 50

# E_architecture = [50,250,250]
# D_architecture = [250,250,50]

E_architecture = [50,250,250,50]
D_architecture = [50,250,250,50]

def split_tensor(index, x):
	return Lambda(lambda x : x[:,index])(x)

##############################################################################################################
# Build encoder model ...
input_vertex_info = Input(shape=(target_dim,))
momentum_conditions = Input(shape=(conditions_dim,))

encoder_network_input = Concatenate(axis=-1)([input_vertex_info,momentum_conditions])

H = Dense(int(E_architecture[0]))(encoder_network_input)
H = BatchNormalization()(H)
H = LeakyReLU()(H)
# H = Dropout(0.2)(H)

for layer in E_architecture[1:]:
	H = Dense(int(layer))(H)
	H = BatchNormalization()(H)
	H = LeakyReLU()(H)
	# H = Dropout(0.2)(H)

z_mean = Dense(latent_dim)(H)
z_log_var = Dense(latent_dim)(H)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model(inputs=[input_vertex_info, momentum_conditions], outputs=[z, z_mean, z_log_var])
##############################################################################################################

##############################################################################################################
# Build decoder model ...
input_latent = Input(shape=(latent_dim))
momentum_conditions = Input(shape=(conditions_dim,))

decoder_network_input = Concatenate()([input_latent,momentum_conditions])

H = Dense(int(D_architecture[0]))(decoder_network_input)
H = BatchNormalization()(H)
H = LeakyReLU()(H)
# H = Dropout(0.2)(H)

for layer in D_architecture[1:]:
	H = Dense(int(layer))(H)
	H = BatchNormalization()(H)
	H = LeakyReLU()(H)
	# H = Dropout(0.2)(H)

# decoded_mean = Dense(target_dim,activation='tanh')(H)
decoded_mean = Dense(target_dim)(H)
decoder = Model(inputs=[input_latent, momentum_conditions], outputs=[decoded_mean])
##############################################################################################################

input_sample = Input(shape=(target_dim,))
momentum_conditions = Input(shape=(conditions_dim,))
z, z_mean, z_log_var = encoder([input_sample,momentum_conditions])
decoded_mean = decoder([z, momentum_conditions])
vae = Model(inputs=[input_sample, momentum_conditions], outputs=[decoded_mean, z_mean, z_log_var])

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5, decay=0, amsgrad=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

@tf.function
def train_step(images, toggle_kl):

	sample_targets, sample_conditions = images[:,0,:10], images[:,0,10:]

	with tf.GradientTape() as tape:
		
		vae_out, vae_z_mean, vae_z_log_var = vae([sample_targets,sample_conditions])

		vae_reco_loss = reco_loss(sample_targets, vae_out)
		vae_reco_loss = tf.math.reduce_mean(vae_reco_loss)
		vae_kl_loss = kl_loss(vae_z_mean, vae_z_log_var)
		vae_kl_loss = tf.math.reduce_mean(vae_kl_loss)*toggle_kl

		vae_loss = vae_kl_loss*kl_factor + vae_reco_loss*reco_factor

	grad_vae = tape.gradient(vae_loss, vae.trainable_variables)

	optimizer.apply_gradients(zip(grad_vae, vae.trainable_variables))

	return vae_kl_loss, vae_reco_loss


start = time.time()

iteration = -1

loss_list = np.empty((0,3))

t0 = time.time()

save_interval = 2500

for epoch in range(int(1E30)):

	X_train_data_loader = data_loader.load_data('datasets/Kee_2018_truthed.csv')

	X_train_data_all_pp, X_train_data_targets_pp, X_train_data_condtions_pp = X_train_data_loader.get_processed_data()
	X_train_raw = np.asarray(X_train_data_all_pp)


	X_train = np.expand_dims(X_train_raw,1).astype("float32")

	train_dataset = (
		tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size,drop_remainder=True).repeat(1)
	)

	for samples_for_batch in train_dataset:

		if iteration % 100 == 0: print('Iteration:',iteration)

		iteration += 1

		# if iteration > 1000: # annealing https://arxiv.org/pdf/1511.06349.pdf
		# 	toggle_kl = tf.convert_to_tensor(1.)
		# else:
		# 	toggle_kl = tf.convert_to_tensor(0.)

		if iteration % 1000 == 0: # annealing https://arxiv.org/pdf/1511.06349.pdf
			toggle_kl_value = 0.
		elif iteration % 1000 < 500:
			toggle_kl_value += 1/500
		else:
			toggle_kl_value = 1.
		toggle_kl = tf.convert_to_tensor(toggle_kl_value)


		kl_loss_np, reco_loss_np = train_step(samples_for_batch, toggle_kl)

		loss_list = np.append(loss_list, [[iteration, kl_loss_np, reco_loss_np]], axis=0)
 
		if iteration % save_interval == 0 and iteration>0:

			t1 = time.time()

			total = t1-t0
			

			plt.figure(figsize=(12, 8))
			plt.subplot(2,3,1)
			plt.plot(loss_list[:,0], loss_list[:,1])
			plt.ylabel('kl Loss')
			plt.subplot(2,3,2)
			plt.plot(loss_list[:,0], loss_list[:,2])
			plt.ylabel('reco Loss')
			plt.subplot(2,3,3)
			plt.plot(loss_list[:,0], loss_list[:,1])
			plt.ylabel('kl Loss')
			plt.yscale('log')
			plt.subplot(2,3,4)
			plt.plot(loss_list[:,0], loss_list[:,2])
			plt.ylabel('reco Loss')
			plt.yscale('log')
			plt.subplot(2,3,5)
			plt.plot(loss_list[:,0], kl_factor*loss_list[:,1]+reco_factor*loss_list[:,2])
			plt.ylabel('TOTAL Loss')
			plt.yscale('log')
			plt.subplots_adjust(wspace=0.3, hspace=0.3)
			plt.savefig('LOSSES.png',bbox_inches='tight')
			plt.close('all')

			gen_noise = np.random.normal(0, 1, (10000, latent_dim))
			images = np.squeeze(decoder.predict([gen_noise,X_train_raw[-10000:,10:]]))

			gen_images = np.concatenate((images, X_train_raw[-10000:,10:]), axis=-1)

			gen_images = pd.DataFrame(gen_images, columns=rd.targets+rd.conditions)

			gen_events_dataset = data_loader.dataset(generated=True)
			gen_events_dataset.fill(gen_images, processed=True)

			plot(X_train_data_loader, gen_events_dataset, f'plots_{iteration}', Nevents=10000)

			# if iteration == save_interval:
			# 	plot(X_train_raw[:,:10], 'truth',Nevents=10000)
			
			# plot(images[:,:10], f'out_{iteration}',Nevents=10000)

			print("Saving complete...")

			for file in glob.glob('save_state/*'):
				os.remove(file)
			decoder.save('save_state/decoder.h5')
			pickle.dump(rd.QuantileTransformers, open('save_state/QuantileTransformers.pkl', 'wb') )

			if iteration > 1E4:
				quit()