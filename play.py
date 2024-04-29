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

from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Reshape,
    Dropout,
    Concatenate,
    Lambda,
    ReLU,
    Activation,
)
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

import fast_vertex_quality.tools.plotting as plotting

rd.targets = [
    "B_plus_ENDVERTEX_CHI2",
    "B_plus_IPCHI2_OWNPV",
    "B_plus_FDCHI2_OWNPV",
    "B_plus_DIRA_OWNPV",
    "K_Kst_IPCHI2_OWNPV",
    "K_Kst_TRACK_CHI2NDOF",
    "e_minus_IPCHI2_OWNPV",
    "e_minus_TRACK_CHI2NDOF",
    "e_plus_IPCHI2_OWNPV",
    "e_plus_TRACK_CHI2NDOF",
]


# rd.conditions = [
#     "K_Kst_PX",
#     "K_Kst_PY",
#     "K_Kst_PZ",
#     "e_minus_PX",
#     "e_minus_PY",
#     "e_minus_PZ",
#     "e_plus_PX",
#     "e_plus_PY",
#     "e_plus_PZ",
#     "nTracks",
#     "nSPDHits",
# ]

rd.conditions = [
    "q2",
]

event_loader_MC_lowq2 = data_loader.load_data("datasets/Kee_2018_truthed.csv")
event_loader_MC_lowq2.apply_cut("q2_physical_data<6")

event_loader_MC_highq2 = data_loader.load_data("datasets/Kee_2018_truthed.csv")
event_loader_MC_highq2.apply_cut("q2_physical_data>15")


plotting.plot(event_loader_MC_lowq2, event_loader_MC_highq2, "q2", Nevents=10000)


decoder = tf.keras.models.load_model("save_state/decoder.h5")

latent_dim = 6

event_loader_MC_lowq2.select_randomly(Nevents=10000)
gen_noise = np.random.normal(0, 1, (10000, latent_dim))
X_test_conditions = event_loader_MC_lowq2.get_branches(rd.conditions, processed=True)
X_test_conditions = np.asarray(X_test_conditions)
images = np.squeeze(decoder.predict([gen_noise, X_test_conditions]))
event_loader_MC_lowq2.fill_target(images)


event_loader_MC_highq2.select_randomly(Nevents=10000)
gen_noise = np.random.normal(0, 1, (10000, latent_dim))
X_test_conditions = event_loader_MC_highq2.get_branches(rd.conditions, processed=True)
X_test_conditions = np.asarray(X_test_conditions)
images = np.squeeze(decoder.predict([gen_noise, X_test_conditions]))
event_loader_MC_highq2.fill_target(images)

plotting.plot(event_loader_MC_lowq2, event_loader_MC_highq2, "q2_gen", Nevents=10000)
