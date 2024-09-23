import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from fast_vertex_quality.tools.config import rd, read_definition
import tensorflow as tf
import uproot

import uproot3 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pickle
from particle import Particle
from hep_ml.reweight import BinsReweighter, GBReweighter, FoldingReweighter
from termcolor import colored

class Transformer:

	def __init__(self):
		
		self.qt_fit = False
		self.clip_value = 4.

	def fit(self, data_raw, column):

		self.column = column

		self.qt = QuantileTransformer(
			n_quantiles=500, output_distribution="normal"
		)

	def process(self, data_raw):
		
		try:
			data = data_raw.copy()
		except:
			# pass # value is likely a single element
			data = np.asarray(data_raw).astype('float64')

		if "TRUEORIGINVERTEX_X" in self.column or "TRUEORIGINVERTEX_Y" in self.column:
			return data

		if "DIRA" in self.column:
			where = np.where(np.isnan(data))
			where_not_nan = np.where(np.logical_not(np.isnan(data)))
			data[where] = np.amin(data[where_not_nan])

		if 'VTXISOBDTHARD' in self.column:
			data[np.where(data==-1)] = np.random.uniform(low=-1.1,high=-1.0,size=np.shape(data[np.where(data==-1)]))
		if 'FLIGHT' in self.column or 'FD' in self.column or 'IP' in self.column:
			data[np.where(data==0)] = np.random.uniform(low=-0.1,high=0.0,size=np.shape(data[np.where(data==0)]))

		if not self.qt_fit:
			self.qt.fit(data.reshape(-1, 1))
			self.qt_fit = True
		
		data = self.qt.transform(data.reshape(-1, 1))[:,0]
		data = np.clip(data, -self.clip_value, self.clip_value)
		data = data/self.clip_value

		return data

	def unprocess(self, data_raw):
			
		data = data_raw.copy()

		if "TRUEORIGINVERTEX_X" in self.column or "TRUEORIGINVERTEX_Y" in self.column:
			return data

		data = data*self.clip_value
		data = self.qt.inverse_transform(data.reshape(-1, 1))[:,0]	

		if 'VTXISOBDTHARD' in self.column:
			data[np.where(data<-1)] = -1.
		if 'FLIGHT' in self.column or 'FD' in self.column or 'IP' in self.column:
			data[np.where(data<0)] = 0.

		return data

