from fast_vertex_quality.tools.config import read_definition, rd

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

class NoneError(Exception):
    pass

rd.targets = ["B_plus_ENDVERTEX_CHI2",
		"B_plus_IPCHI2_OWNPV",
		"B_plus_FDCHI2_OWNPV",
		"B_plus_DIRA_OWNPV",
		"K_Kst_IPCHI2_OWNPV",
		"K_Kst_TRACK_CHI2NDOF",
		"e_minus_IPCHI2_OWNPV",
		"e_minus_TRACK_CHI2NDOF",
		"e_plus_IPCHI2_OWNPV",
		"e_plus_TRACK_CHI2NDOF"]
		
		
rd.conditions = ['K_Kst_PX','K_Kst_PY','K_Kst_PZ',
				'e_minus_PX','e_minus_PY','e_minus_PZ',
				'e_plus_PX','e_plus_PY','e_plus_PZ',
				'nTracks','nSPDHits'
				]


class dataset:

	def __init__(self, generated=False):

		self.generated = generated

		self.log_columns = ["B_plus_FDCHI2_OWNPV", "K_Kst_IPCHI2_OWNPV", 
						"e_minus_IPCHI2_OWNPV", "e_plus_IPCHI2_OWNPV",
						"K_Kst_PZ","e_minus_PZ","e_plus_PZ"]
		
		self.one_minus_log_columns = ["B_plus_DIRA_OWNPV"]

	def fill(self, data, processed=False):

		if not isinstance(data, pd.DataFrame):
			raise NoneError("Dataset must be a pd.dataframe.")

		if not processed:
			self.physical_data = data[rd.targets+rd.conditions]
			self.pre_process(data[rd.targets+rd.conditions])
		elif processed:
			self.processed_data = data[rd.targets+rd.conditions]
			self.post_process(data[rd.targets+rd.conditions])

		self.produce_physics_variables()

	def get_physical_data(self):
		return self.physical_data, self.physical_data[rd.targets], self.physical_data[rd.conditions]

	def get_processed_data(self):
		return self.processed_data, self.processed_data[rd.targets], self.processed_data[rd.conditions]

	def get_physics_variables(self):
		return self.physics_variables

	def apply_cut(self, cut):
		
		self.processed_data = self.processed_data.add_suffix('_processed_data')
		self.physics_variables = self.physics_variables.add_suffix('_physics_variables')
		self.physical_data = self.physical_data.add_suffix('_physical_data')

		all_vars = pd.concat([self.physical_data, self.physics_variables, self.processed_data], axis=1)

		try:
			all_vars = all_vars.query(cut)
		except:
			print('cut query failed, try adding _processed_data, _physics_variables, or _physical_data?')
			quit()

		self.processed_data = all_vars[[col for col in all_vars.columns if '_processed_data' in col]]
		self.physics_variables = all_vars[[col for col in all_vars.columns if '_physics_variables' in col]]
		self.physical_data = all_vars[[col for col in all_vars.columns if '_physical_data' in col]]

		self.processed_data.columns = [col.replace('_processed_data','') for col in self.processed_data.columns]
		self.physics_variables.columns = [col.replace('_physics_variables','') for col in self.physics_variables.columns]
		self.physical_data.columns = [col.replace('_physical_data','') for col in self.physical_data.columns]

	def produce_physics_variables(self):

		physics_variables = {}

		physics_variables["K_Kst_P"] = np.sqrt(self.physical_data['K_Kst_PX']**2+self.physical_data['K_Kst_PY']**2+self.physical_data['K_Kst_PZ']**2)
		physics_variables["e_plus_P"] = np.sqrt(self.physical_data['e_plus_PX']**2+self.physical_data['e_plus_PY']**2+self.physical_data['e_plus_PZ']**2)
		physics_variables["e_minus_P"] = np.sqrt(self.physical_data['e_minus_PX']**2+self.physical_data['e_minus_PY']**2+self.physical_data['e_minus_PZ']**2)
		physics_variables["kFold"] = np.random.randint(low=0, high=9, size=np.shape(self.physical_data["K_Kst_PX"])[0])

		electron_mass = 0.51099895000*1E-3

		PE = np.sqrt(electron_mass**2+self.physical_data['e_plus_PX']**2+self.physical_data['e_plus_PY']**2+self.physical_data['e_plus_PZ']**2) + np.sqrt(electron_mass**2+self.physical_data['e_minus_PX']**2+self.physical_data['e_minus_PY']**2+self.physical_data['e_minus_PZ']**2)
		PX = self.physical_data['e_plus_PX'] + self.physical_data['e_minus_PX']
		PY = self.physical_data['e_plus_PY'] + self.physical_data['e_minus_PY']
		PZ = self.physical_data['e_plus_PZ'] + self.physical_data['e_minus_PZ']

		physics_variables["q2"] = (PE**2-PX**2-PY**2-PZ**2)*1e-6

		self.physics_variables = pd.DataFrame.from_dict(physics_variables)

	# def pre_process(self, physical_data):

	# 	df = {}
		
	# 	for column in rd.targets+rd.conditions:
		
	# 		if column in self.log_columns:
	# 			df[column] = np.log10(physical_data[column])
	# 		elif column in self.one_minus_log_columns:
	# 			df[column] = np.log10(1.-physical_data[column])
	# 		else:
	# 			df[column] = physical_data[column]

	# 	if self.generated == False:
	# 		rd.normalisation_constants = {}

	# 	for column in list(df.keys()):

	# 		if self.generated == False:
	# 			rd.normalisation_constants[column] = {}
	# 			rd.normalisation_constants[column]['min'] = np.amin(df[column])

	# 		df[column] = df[column] - np.amin(df[column])

	# 		if self.generated == False: rd.normalisation_constants[column]['max'] = np.amax(df[column])

	# 		df[column] = df[column]/np.amax(df[column])
	# 		df[column] *= 1.9
	# 		df[column] += -0.95

	# 	self.processed_data = pd.DataFrame.from_dict(df)


	# def post_process(self, processed_data):

	# 	df = {}

	# 	for column in rd.targets+rd.conditions:
	# 		df[column] = processed_data[column]

	# 	for column in list(processed_data.keys()):
	# 		df[column] -= -0.95
	# 		df[column] *= 1./1.9
	# 		df[column] = df[column]*rd.normalisation_constants[column]['max']
	# 		df[column] = df[column] + rd.normalisation_constants[column]['min']

	# 		if column in self.log_columns:
	# 			df[column] = np.power(10, df[column])
	# 		elif column in self.one_minus_log_columns:
	# 			df[column] = np.power(10, df[column])
	# 			df[column] = 1.-df[column]

	# 	self.physical_data = pd.DataFrame.from_dict(df)


	def pre_process(self, physical_data):

		df = {}
		
		for column in rd.targets+rd.conditions:
		
			if column in self.log_columns:
				df[column] = np.log10(physical_data[column])
			elif column in self.one_minus_log_columns:
				df[column] = np.log10(1.-physical_data[column])
			else:
				df[column] = physical_data[column]

		if self.generated == False:
			rd.normalisation_constants = {}

		rd.QuantileTransformers = {}

		for column in list(df.keys()):
			
			qt = QuantileTransformer(n_quantiles=50, output_distribution="normal")
			rd.QuantileTransformers[column] = qt.fit(np.asarray(df[column]).reshape(-1, 1))
			
			df[column] = np.squeeze(rd.QuantileTransformers[column].transform(np.asarray(df[column]).reshape(-1, 1)))

			# if self.generated == False:
			# 	rd.normalisation_constants[column] = {}
			# 	rd.normalisation_constants[column]['min'] = np.mean(df[column])

			# df[column] = df[column] - rd.normalisation_constants[column]['min']

			# if self.generated == False: rd.normalisation_constants[column]['max'] = np.std(df[column])

			# df[column] = df[column]/rd.normalisation_constants[column]['max']
			# # df[column] *= 1.9
			# # df[column] += -0.95

		self.processed_data = pd.DataFrame.from_dict(df)


	def post_process(self, processed_data):

		df = {}

		for column in rd.targets+rd.conditions:
			df[column] = processed_data[column]

		for column in list(processed_data.keys()):
			# df[column] -= -0.95
			# df[column] *= 1./1.9
			# df[column] = df[column]*rd.normalisation_constants[column]['max']
			# df[column] = df[column] + rd.normalisation_constants[column]['min']

			df[column] = np.squeeze(rd.QuantileTransformers[column].inverse_transform(np.asarray(df[column]).reshape(-1, 1)))

			if column in self.log_columns:
				df[column] = np.power(10, df[column])
			elif column in self.one_minus_log_columns:
				df[column] = np.power(10, df[column])
				df[column] = 1.-df[column]

			


		self.physical_data = pd.DataFrame.from_dict(df)


def load_data(path):

	events = pd.read_csv(path)

	events_dataset = dataset(generated=False)
	events_dataset.fill(events, processed=False)

	return events_dataset


	# events = pre_process_pd(events)
	# # events = post_process_pd(events)
	# events = events[rd.targets+rd.conditions]
	# return events

	# target_array = np.asarray(events[targets])
	# condition_array = np.asarray(events[conditions])

	# full_array = np.concatenate((target_array,condition_array),axis=1)

	# # return events[targets], events[conditions]
	# return full_array, targets+conditions
