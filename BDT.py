from fast_vertex_quality.tools.config import read_definition, rd

import fast_vertex_quality.tools.data_loader as data_loader
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import pandas as pd


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

event_loader_MC = data_loader.load_data('datasets/Kee_2018_truthed.csv')
event_loader_data = data_loader.load_data('datasets/B2Kee_2018_CommonPresel.csv')

events_MC, throw, throw = event_loader_MC.get_physical_data()
events_data, throw, throw = event_loader_data.get_physical_data()

events_MC_pp, throw, throw = event_loader_MC.get_processed_data()
events_data_pp, throw, throw = event_loader_data.get_processed_data()

events_data = events_data.query("B_plus_IPCHI2_OWNPV>0")

events_data_physics_variables = event_loader_data.get_physics_variables()
events_data = pd.concat([events_data, events_data_physics_variables], axis=1)

events_MC_physics_variables = event_loader_MC.get_physics_variables()
events_MC = pd.concat([events_MC, events_MC_physics_variables], axis=1)


BDTs = {}

for kFold in range(10):

	print(f'Training kFold {kFold}...')

	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

	events_data_i = events_data.query(f'kFold!={kFold}')
	events_MC_i = events_MC.query(f'kFold!={kFold}')

	real_training_data = np.squeeze(np.asarray(events_MC_i[rd.targets]))

	fake_training_data = np.squeeze(np.asarray(events_data_i[rd.targets]))

	size = 25000
	real_training_data = real_training_data[:size]
	fake_training_data = fake_training_data[:size]

	real_training_labels = np.ones(size)

	fake_training_labels = np.zeros(size)

	total_training_data = np.concatenate((real_training_data, fake_training_data))

	total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

	clf.fit(total_training_data, total_training_labels)

	BDTs[kFold] = clf

	break


for kFold in range(10):

	events_data_i = events_data.query(f'kFold=={kFold}')
	events_MC_i = events_MC.query(f'kFold=={kFold}')

	real_testing_data = np.squeeze(np.asarray(events_MC_i[rd.targets]))

	fake_testing_data = np.squeeze(np.asarray(events_data_i[rd.targets]))

	size = 25000
	real_testing_data = real_testing_data[:size]
	fake_testing_data = fake_testing_data[:size]


	out_real = clf.predict_proba(real_testing_data)

	out_fake = clf.predict_proba(fake_testing_data)

	########################
	X_train_data_loader = data_loader.load_data('datasets/Kee_2018_truthed.csv')
	X_train_data_all_pp, X_train_data_targets_pp, X_train_data_condtions_pp = X_train_data_loader.get_processed_data()
	X_train_raw = np.asarray(X_train_data_all_pp)

	decoder = tf.keras.models.load_model('save_state/decoder.h5')
	
	latent_dim = 6

	gen_noise = np.random.normal(0, 1, (10000, latent_dim))
	images = np.squeeze(decoder.predict([gen_noise,X_train_raw[-10000:,10:]]))

	gen_images = np.concatenate((images, X_train_raw[-10000:,10:]), axis=-1)

	gen_images = pd.DataFrame(gen_images, columns=rd.targets+rd.conditions)

	gen_events_dataset = data_loader.dataset(generated=True)
	gen_events_dataset.fill(gen_images, processed=True)
	
	events_gen, throw, throw = gen_events_dataset.get_physical_data()

	events_gen_query = np.squeeze(np.asarray(events_gen[rd.targets]))

	out_gen = clf.predict_proba(events_gen_query)


	plt.hist([out_real[:,1], out_fake[:,1], out_gen[:,1]], bins=50, color=['tab:blue','tab:red', 'tab:green'], alpha=0.25, label=['Background', "Signal - MC", "Signal - generated"], density=True, histtype='stepfilled',range=[0,1])
	plt.hist([out_real[:,1], out_fake[:,1], out_gen[:,1]], bins=50, color=['tab:blue','tab:red', 'tab:green'], density=True, histtype='step',range=[0,1])
	plt.legend()
	plt.xlabel(f'BDT output')
	plt.yscale('log')
	plt.savefig('BDT')

	break



