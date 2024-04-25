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

event_loader_MC_lowq2 = data_loader.load_data('datasets/Kee_2018_truthed.csv')
event_loader_MC_lowq2.apply_cut('q2_physics_variables<6')

event_loader_MC_highq2 = data_loader.load_data('datasets/Kee_2018_truthed.csv')
event_loader_MC_highq2.apply_cut('q2_physics_variables>15')

# plot(event_loader_MC_lowq2, event_loader_MC_highq2, 'q2', Nevents=10000)



event_loader_MC_lowq2, throw, throw = event_loader_MC_lowq2.get_processed_data()
X_train_raw_lowq2 = np.asarray(event_loader_MC_lowq2)

event_loader_MC_highq2, throw, throw = event_loader_MC_highq2.get_processed_data()
X_train_raw_highq2 = np.asarray(event_loader_MC_highq2)

decoder = tf.keras.models.load_model('save_state/decoder.h5')

latent_dim = 6




gen_noise = np.random.normal(0, 1, (10000, latent_dim))
images = np.squeeze(decoder.predict([gen_noise,X_train_raw_lowq2[-10000:,10:]]))
gen_images = np.concatenate((images, X_train_raw_lowq2[-10000:,10:]), axis=-1)
gen_images = pd.DataFrame(gen_images, columns=rd.targets+rd.conditions)
gen_events_dataset_lowq2 = data_loader.dataset(generated=True)
gen_events_dataset_lowq2.fill(gen_images, processed=True)

gen_noise = np.random.normal(0, 1, (10000, latent_dim))
images = np.squeeze(decoder.predict([gen_noise,X_train_raw_lowq2[-10000:,10:]]))
gen_images = np.concatenate((images, X_train_raw_highq2[-10000:,10:]), axis=-1)
gen_images = pd.DataFrame(gen_images, columns=rd.targets+rd.conditions)
gen_events_dataset_highq2 = data_loader.dataset(generated=True)
gen_events_dataset_highq2.fill(gen_images, processed=True)



plot(gen_events_dataset_lowq2, gen_events_dataset_highq2, 'q2_gen', Nevents=10000)

