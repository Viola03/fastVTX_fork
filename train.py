import pickle
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization,
									 Concatenate, Dense, Dropout, Flatten,
									 Input, Lambda, LeakyReLU, ReLU, Reshape)
from tensorflow.keras.models import Model

import fast_vertex_quality.tools.data_loader as data_loader
from fast_vertex_quality.tools.config import rd, read_definition

_EPSILON = K.epsilon()

import glob
import math
import os
import shutil
import time
from pickle import load

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from matplotlib.colors import LogNorm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

import fast_vertex_quality.tools.plotting as plotting

from fast_vertex_quality.tools.training import train_step

from fast_vertex_quality.models.conditional_VAE import VAE_builder


print(tf.__version__)


latent_dim = 6
kl_factor = 1.
# reco_factor = 1E3
reco_factor = 100.

batch_size = 50

target_dim = 10
cut_idx = target_dim

VAE = VAE_builder(
	E_architecture=[50, 250, 250, 50], D_architecture=[50, 250, 250, 50],
	target_dim=target_dim, conditions_dim=11, latent_dim=6
)
rd.encoder = VAE.encoder
rd.decoder = VAE.decoder
rd.vae = VAE.vae

rd.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

start = time.time()

iteration = -1

loss_list = np.empty((0, 3))

t0 = time.time()


save_interval = 2500

for epoch in range(int(1e30)):

	X_train_data_loader = data_loader.load_data("datasets/Kee_2018_truthed.csv")
	
	X_train_data_all_pp = X_train_data_loader.get_branches(rd.targets+rd.conditions, processed=True)

	print(X_train_data_all_pp)

	X_train_raw = np.asarray(X_train_data_all_pp)

	print(np.where(np.isinf(X_train_raw)))
	print(np.where(np.isnan(X_train_raw)))
	print(np.shape(X_train_raw))

	X_train = np.expand_dims(X_train_raw, 1).astype("float32")

	train_dataset = (
		tf.data.Dataset.from_tensor_slices(X_train)
		.batch(batch_size, drop_remainder=True)
		.repeat(1)
	)

	# latent_dim = 6
	# X_train_data_loader = data_loader.load_data("datasets/Kee_2018_truthed_head.csv")
	# conditions = X_train_data_loader.get_branches(rd.conditions, processed=True)
	# gen_noise = np.random.normal(0, 1, (99, latent_dim))
	# images = np.squeeze(rd.decoder.predict([gen_noise,conditions]))
	# print(images)
	# quit()

	for samples_for_batch in train_dataset:

		if iteration % 100 == 0:
			print("Iteration:", iteration)

		iteration += 1

		if iteration % 1000 == 0:  # annealing https://arxiv.org/pdf/1511.06349.pdf
			toggle_kl_value = 0.0
		elif iteration % 1000 < 500:
			toggle_kl_value += 1 / 500
		else:
			toggle_kl_value = 1.0
		toggle_kl = tf.convert_to_tensor(toggle_kl_value)

		kl_loss_np, reco_loss_np = train_step(
			samples_for_batch, cut_idx, tf.convert_to_tensor(kl_factor), tf.convert_to_tensor(reco_factor), toggle_kl
		)

		loss_list = np.append(
			loss_list, [[iteration, kl_loss_np, reco_loss_np]], axis=0
		)

		if iteration % save_interval == 0 and iteration > 0:

			t1 = time.time()

			total = t1 - t0

			plotting.loss_plot(loss_list, reco_factor, kl_factor, "LOSSES.png")

			gen_noise = np.random.normal(0, 1, (10000, latent_dim))
			images = np.squeeze(
				rd.decoder.predict([gen_noise, X_train_raw[-10000:, 10:]])
			)

			gen_images = np.concatenate((images, X_train_raw[-10000:, 10:]), axis=-1)

			gen_images = pd.DataFrame(gen_images, columns=rd.targets + rd.conditions)

			gen_events_dataset = data_loader.dataset(generated=True)
			gen_events_dataset.fill(gen_images, processed=True)

			plotting.plot(
				X_train_data_loader,
				gen_events_dataset,
				f"plots_{iteration}",
				Nevents=10000,
			)

			print("Saving complete...")

			for file in glob.glob("save_state/*"):
				os.remove(file)
			rd.decoder.save("save_state/decoder.h5")
			pickle.dump(
				rd.QuantileTransformers,
				open("save_state/QuantileTransformers.pkl", "wb"),
			)

			if iteration > 1e4:
				quit()
