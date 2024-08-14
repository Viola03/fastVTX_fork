import pickle
import numpy as np

# Search for the LFNs of a currently running ganga job with "jobs(2011).subjobs(286)", look for inputdata, LHCbCompressedFileSet and suffixes

target_LFN = "00116947_00000151_7.AllStreams.dst"


filehandler = open('targets_LFNs.pkl', 'rb')  
LFNs = pickle.load(filehandler)

total = 0
for target in list(LFNs.keys()):
	
	LFNs[target] = np.asarray(LFNs[target])

	found = any(target_LFN in string for string in LFNs[target])

	if found:
		print(target)