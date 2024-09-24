import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from fast_vertex_quality.tools.config import rd, read_definition
import uproot

import uproot3 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pickle


def symlog(x, linthresh=1.0):
	sign = np.sign(x)
	abs_x = np.abs(x)
	return sign * np.log10(1 + abs_x / linthresh)

def invsymlog(y, linthresh=1.0):
	sign = np.sign(y)
	abs_y = np.abs(y)
	return sign * linthresh * (10**abs_y - 1)

def symsqrt(x, c=1):
	"""Apply symmetric logarithm transformation."""
	# return np.sign(x) * np.log10(c * np.abs(x) + 1)
	return np.sign(x) * np.sqrt(np.abs(x))

def inv_symsqrt(y, c=1):
	"""Apply inverse symmetric logarithm transformation."""
	# return np.sign(y) * (10**np.abs(y) - 1) / c
	return np.sign(y) * np.abs(y)**2

class OriginalTransformer:

	def __init__(self):

		self.abs_columns = [
			f"{rd.mother_particle}_TRUEID",
			f"{rd.daughter_particles[0]}_TRUEID",
			f"{rd.daughter_particles[1]}_TRUEID",
			f"{rd.daughter_particles[2]}_TRUEID",
							]

		self.shift_and_symsqrt_columns = [
			f"{rd.mother_particle}_TRUEORIGINVERTEX_X",
			f"{rd.mother_particle}_TRUEORIGINVERTEX_Y"
		]

		self.log_columns = [
			f"{rd.mother_particle}_FDCHI2_OWNPV",
			f"{rd.daughter_particles[0]}_IPCHI2_OWNPV",
			f"{rd.daughter_particles[1]}_IPCHI2_OWNPV",
			f"{rd.daughter_particles[2]}_IPCHI2_OWNPV",
			f"{rd.intermediate_particle}_IPCHI2_OWNPV",
			f"{rd.daughter_particles[0]}_PZ",
			f"{rd.daughter_particles[1]}_PZ",
			f"{rd.daughter_particles[2]}_PZ",
			f"{rd.mother_particle}_ENDVERTEX_CHI2",
			f"{rd.intermediate_particle}_ENDVERTEX_CHI2",
			f"{rd.mother_particle}_IPCHI2_OWNPV",
			f"IP_{rd.mother_particle}",
			f"{rd.intermediate_particle}_FDCHI2_OWNPV",
			f"{rd.intermediate_particle}_FLIGHT",

			f"{rd.mother_particle}_TRUE_FD",

			f"{rd.mother_particle}_P",
			f"{rd.mother_particle}_PT",
			f"IP_{rd.daughter_particles[0]}",
			f"IP_{rd.daughter_particles[1]}",
			f"IP_{rd.daughter_particles[2]}",
			f"FD_{rd.mother_particle}",
			f"IP_{rd.daughter_particles[0]}_true_vertex",
			f"IP_{rd.daughter_particles[1]}_true_vertex",
			f"IP_{rd.daughter_particles[2]}_true_vertex",
			f"IP_{rd.mother_particle}_true_vertex",
			f"FD_{rd.mother_particle}_true_vertex",

			f"{rd.intermediate_particle}_TRUEID_width",
			f"{rd.intermediate_particle}_MC_MOTHER_ID_width",
			f"{rd.intermediate_particle}_MC_GD_MOTHER_ID_width",
			f"{rd.intermediate_particle}_MC_GD_GD_MOTHER_ID_width",

			f"{rd.daughter_particles[0]}_MC_MOTHER_ID_width",
			f"{rd.daughter_particles[0]}_MC_GD_MOTHER_ID_width",
			f"{rd.daughter_particles[0]}_MC_GD_GD_MOTHER_ID_width",

			f"{rd.daughter_particles[1]}_MC_MOTHER_ID_width",
			f"{rd.daughter_particles[1]}_MC_GD_MOTHER_ID_width",
			f"{rd.daughter_particles[1]}_MC_GD_GD_MOTHER_ID_width",

			f"{rd.daughter_particles[2]}_MC_MOTHER_ID_width",
			f"{rd.daughter_particles[2]}_MC_GD_MOTHER_ID_width",
			f"{rd.daughter_particles[2]}_MC_GD_GD_MOTHER_ID_width",

			f"{rd.intermediate_particle}_MC_MOTHER_ID_mass",
			f"{rd.intermediate_particle}_MC_GD_MOTHER_ID_mass",
			f"{rd.intermediate_particle}_MC_GD_GD_MOTHER_ID_mass",

			f"{rd.daughter_particles[0]}_MC_MOTHER_ID_mass",
			f"{rd.daughter_particles[0]}_MC_GD_MOTHER_ID_mass",
			f"{rd.daughter_particles[0]}_MC_GD_GD_MOTHER_ID_mass",

			f"{rd.daughter_particles[1]}_MC_MOTHER_ID_mass",
			f"{rd.daughter_particles[1]}_MC_GD_MOTHER_ID_mass",
			f"{rd.daughter_particles[1]}_MC_GD_GD_MOTHER_ID_mass",

			f"{rd.daughter_particles[2]}_MC_MOTHER_ID_mass",
			f"{rd.daughter_particles[2]}_MC_GD_MOTHER_ID_mass",
			f"{rd.daughter_particles[2]}_MC_GD_GD_MOTHER_ID_mass",

			f"{rd.daughter_particles[0]}_FLIGHT",
			f"{rd.daughter_particles[1]}_FLIGHT",
			f"{rd.daughter_particles[2]}_FLIGHT",

			f"{rd.daughter_particles[0]}_TRACK_GhostProb",
			f"{rd.daughter_particles[1]}_TRACK_GhostProb",
			f"{rd.daughter_particles[2]}_TRACK_GhostProb",

			f"{rd.mother_particle}_cp_0.70",
			f"{rd.mother_particle}_cpt_0.70",
			


			"delta_0_P",
			"delta_0_PT",
			"delta_1_P",
			"delta_1_PT",
			"delta_2_P",
			"delta_2_PT",

		]

		self.one_minus_log_columns = [f"{rd.mother_particle}_DIRA_OWNPV", f"DIRA_{rd.mother_particle}", f"DIRA_{rd.mother_particle}_true_vertex", f"{rd.intermediate_particle}_DIRA_OWNPV", f"DIRA_{rd.intermediate_particle}", f"DIRA_{rd.intermediate_particle}_true_vertex"]
		

		self.symlog_columns = [f"{rd.mother_particle}_SmallestDeltaChi2OneTrack", f"{rd.mother_particle}_SmallestDeltaChi2TwoTracks"]

		self.min_fills = {}

		# self.trueID_map = {-11:1, -13:2, 211:3, 321:4} # positive particles
		# for pid in list(self.trueID_map.keys()):
		#     self.trueID_map[-pid] = -self.trueID_map[pid]
		self.trueID_map = {11:1, 13:2, 211:3, 321:4} # positive particles

		values = list(self.trueID_map.values())
		values_max = np.amax(values)
		values_min = np.amin(values)
		for pid in list(self.trueID_map.keys()):
			self.trueID_map[pid] = (((self.trueID_map[pid]-values_min)/(values_max-values_min))*2.-1.)*0.8

	def map_pdg_codes(self, data):
		mapped_values = np.vectorize(lambda pid: self.trueID_map.get(pid, -1 if pid < 0 else 1))(np.abs(data))
		return mapped_values.astype(np.float64)


	def fit(self, data_raw, column):

		self.column = column

		data = data_raw.copy()

		if column in self.log_columns:
			if "width" in self.column or "mass" in self.column:
				data[np.where(data==0)] = np.amin(data[np.where(data!=0)])/2.
				self.min_fills[self.column] = np.amin(data[np.where(data!=0)])/2.
			else:
				data[np.where(data==0)] = 1E-6
			data = np.log10(data)
		elif column in self.one_minus_log_columns:
			data[np.where(data==1)] = 1.-1E-15
			data[np.where(data>1)] = 1.-1E-15
			data[np.where(np.isnan(data))] = 1.-1E-15
			data[np.where(np.isinf(data))] = 1.-1E-15
			data = np.log10(1.0 - data)
		elif self.column in self.abs_columns:
			data = np.abs(data)
		elif self.column in self.shift_and_symsqrt_columns:
			self.shift = np.mean(data)
			data = data - self.shift
			data = symsqrt(data)
			
		elif self.column in self.symlog_columns:
			data = symlog(data)

		self.min = np.amin(data)
		self.max = np.amax(data)

		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.targets)):
		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.targets) or str(self.column) in list(rd.conditions)):
		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.conditions)):
		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.conditions) or str(self.column) in list(rd.targets)):
		if rd.use_QuantileTransformer and (str(self.column) in ['B_plus_VTXISOBDTHARDTHIRDVALUE']):
				self.qt = QuantileTransformer(
					n_quantiles=500, output_distribution="normal"
				)
				self.qt_fit = False


	def process(self, data_raw):
		
		try:
			if self.symlog_columns:
				pass
		except:
			self.symlog_columns = []

		try:
			data = data_raw.copy()
		except:
			# pass # value is likely a single element
			data = np.asarray(data_raw).astype('float64')

		block_scaling = False

		if "TRUEID" in self.column:
			# print(data)
			data = self.map_pdg_codes(data)
			# print(data)
			# print(np.where(np.abs(data)>1.))
			# quit()
			block_scaling = True
		elif self.column in self.log_columns:
			try:
				if "width" in self.column or "mass" in self.column:
					data[np.where(data==0)] = self.min_fills[self.column]
				else:
					data[np.where(data==0)] = 1E-6
			except:
				pass
			data = np.log10(data)
		elif self.column in self.one_minus_log_columns:
			try:
				data[np.where(data==1)] = 1.-1E-15
				data[np.where(np.isnan(data))] = 1.-1E-15
				data[np.where(np.isinf(data))] = 1.-1E-15
			except Exception as e:
				print(f"\n\nAn error occurred: {e}")
			data = np.log10(1.0 - data)

		elif self.column in self.abs_columns:
			data = np.abs(data)

		elif self.column in self.shift_and_symsqrt_columns:
			data = data - self.shift
			data = symsqrt(data)

		elif self.column in self.symlog_columns:
			data = symlog(data)


		# # # # # # # # # # # # # # # 
		# Threshold cut for DIRA and IP
		if "DIRA" in self.column and "true_vertex" in self.column and rd.mother_particle in self.column:
			where_over_threshold = np.where(data<-7.6)
			data[where_over_threshold] = -7.6
		if "IP" in self.column and "true_vertex" in self.column and rd.mother_particle in self.column:
			where_over_threshold = np.where(data<-2.6)
			data[where_over_threshold] = -2.6
		# # # # # # # # # # # # # # # 

		if "DIRA" in self.column:
			where = np.where(np.isnan(data))
			where_not_nan = np.where(np.logical_not(np.isnan(data)))

		if not block_scaling:
			data = data - self.min
			data = data / (self.max - self.min)
			data *= 2
			data += -1

		try:
			if "DIRA" in self.column:
				data[where] = np.amin(data[where_not_nan])
				# data[where] = -1
		except Exception as e:
			print("ERROR in data_loader:",e)
			print("Continuing, might not be essential")

		data = np.clip(data, -1, 1)

		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.targets)):
		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.targets) or str(self.column) in list(rd.conditions)):
		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.conditions)):
		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.conditions) or str(self.column) in list(rd.targets)):
		if rd.use_QuantileTransformer and (str(self.column) in ['B_plus_VTXISOBDTHARDTHIRDVALUE']):
				if not self.qt_fit:
					self.qt.fit(data.reshape(-1, 1))
					self.qt_fit = True
				data = self.qt.transform(data.reshape(-1, 1))[:,0]
				data = np.clip(data, -5, 5)
				data = data/5.

		return data

	def unprocess(self, data_raw):

		try:
			if self.symlog_columns:
				pass
		except:
			self.symlog_columns = []
			
		data = data_raw.copy()

		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.targets)):
		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.targets) or str(self.column) in list(rd.conditions)):
		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.conditions)):
		# if rd.use_QuantileTransformer and (str(self.column) in list(rd.conditions) or str(self.column) in list(rd.targets)):
		if rd.use_QuantileTransformer and (str(self.column) in ['B_plus_VTXISOBDTHARDTHIRDVALUE']):
			data = data*5.
			data = self.qt.inverse_transform(data.reshape(-1, 1))[:,0]
		

		data += 1
		data *= 0.5
		data = data * (self.max - self.min)
		data = data + self.min

		if "TRUEID" in self.column:
			pass # not currently inverting the processing of TRUEID values
		elif self.column in self.log_columns:
			data = np.power(10, data)
		elif self.column in self.one_minus_log_columns:
			data = np.power(10, data)
			data = 1.0 - data

		elif self.column in self.shift_and_symsqrt_columns:
			data = inv_symsqrt(data)
			data = data + self.shift

		elif self.column in self.symlog_columns:
			data = invsymlog(data)

		

		return data





class UpdatedTransformer:

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

		# if self.column == "FD_B_plus_true_vertex":

		# 	plt.hist()

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

