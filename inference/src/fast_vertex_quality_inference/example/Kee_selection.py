import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class analyser:


	def query_stripping(self):

		self.data["pass_stripping"] = np.zeros(self.data.shape[0])


		if f'{self.mother}_ENDVERTEX_NDOF' not in list(self.data.keys()):
			self.data[f"{self.mother}_ENDVERTEX_NDOF"] = np.ones(self.data.shape[0])*3.

		cuts = {}
		cuts[f'{self.mother}_FDCHI2_OWNPV'] = ">100."
		cuts[f'{self.mother}_DIRA_OWNPV'] = ">0.9995"
		cuts[f'{self.mother}_IPCHI2_OWNPV'] = "<25"
		cuts[f'({self.mother}_ENDVERTEX_CHI2/{self.mother}_ENDVERTEX_NDOF)'] = "<9"
		# cuts['J_psi_1S_PT'] = ">0"
		cuts[f'{self.intermediate}_FDCHI2_OWNPV'] = ">16"
		cuts[f'{self.intermediate}_IPCHI2_OWNPV'] = ">0"
		for lepton in [self.particles[1], self.particles[2]]:
			cuts[f'{lepton}_IPCHI2_OWNPV'] = ">9"
			# cuts[f'{lepton}_PT'] = ">300"
		for hadron in [self.particles[0]]:
			cuts[f'{hadron}_IPCHI2_OWNPV'] = ">9"
			# cuts[f'{hadron}_PT'] = ">400"
		# cuts['m_12'] = "<5500"
		# cuts['B_plus_M_Kee_reco'] = ">(5279.34-1500)"
		# cuts['B_plus_M_Kee_reco'] = "<(5279.34+1500)"

		if isinstance(cuts, dict):
			cut_string = ''
			for cut_idx, cut_i in enumerate(list(cuts.keys())):
				if cut_idx > 0:
					cut_string += ' & '
				if cut_i == 'extra_cut':
					cut_string += f'{cuts[cut_i]}'
				else:
					cut_string += f'{cut_i}{cuts[cut_i]}'
			cuts = cut_string   
		
		cut_array = self.data.query(cuts)
		self.data.loc[cut_array.index,self.stripping_branch_name] = 1.

	def query_BDT(self):

		BDT_targets = [
						f"{self.mother}_ENDVERTEX_CHI2",
						f"{self.mother}_IPCHI2_OWNPV",
						f"{self.mother}_FDCHI2_OWNPV",
						f"{self.mother}_DIRA_OWNPV",
						f"{self.particles[0]}_IPCHI2_OWNPV",
						f"{self.particles[0]}_TRACK_CHI2NDOF",
						f"{self.particles[1]}_IPCHI2_OWNPV",
						f"{self.particles[1]}_TRACK_CHI2NDOF",
						f"{self.particles[2]}_IPCHI2_OWNPV",
						f"{self.particles[2]}_TRACK_CHI2NDOF",
						f"{self.intermediate}_FDCHI2_OWNPV",
						f"{self.intermediate}_IPCHI2_OWNPV"
						]

		# clf = pickle.load(open(f"networks/BDT_sig_prc_WGANcocktail_newconditions.pkl", "rb"))[0]["BDT"]
		clf = pickle.load(open(f"networks/BDT_sig_comb_WGANcocktail_newconditions.pkl", "rb"))[0]["BDT"]
		sample = self.data[BDT_targets]

		# with PdfPages('BDT.pdf') as pdf:
		# 	for BDT_target in BDT_targets:
		# 		plt.title(BDT_target)
		# 		plt.hist(sample[BDT_target], bins=50)
		# 		pdf.savefig(bbox_inches="tight")
		# 		plt.close()
				
		
		# sample = np.squeeze(np.asarray(sample[BDT_targets]))

		# print(np.shape(sample))
		# print(np.where(np.isnan(sample)))
		# print(np.unique(np.where(np.isnan(sample))[0]))

		
		# self.data[self.BDT_branch_name] = clf.predict_proba(sample)[:, 1]

		# Convert sample to numpy array and squeeze
		sample = np.squeeze(np.asarray(sample[BDT_targets]))

		nan_rows = np.unique(np.where(np.isnan(sample))[0])

		# Initialize an array to store BDT responses, filled with NaN
		bdt_responses = np.full(len(sample), np.nan)

		# Identify the rows without NaN values
		non_nan_rows = np.setdiff1d(np.arange(len(sample)), nan_rows)

		# For rows without NaN values, make predictions
		if len(non_nan_rows) > 0:
			bdt_responses[non_nan_rows] = clf.predict_proba(sample[non_nan_rows])[:, 1]

		# Assign the responses back to the appropriate branch
		self.data[self.BDT_branch_name] = bdt_responses
		


	def __init__(self, 
				data,
				stripping_branch_name,
				BDT_branch_name,
				):

		self.data = data
		self.stripping_branch_name = stripping_branch_name
		self.BDT_branch_name = BDT_branch_name

		self.particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
		self.mother = 'MOTHER'
		self.intermediate = 'INTERMEDIATE'

		self.query_stripping()
		self.query_BDT()
	
				
				