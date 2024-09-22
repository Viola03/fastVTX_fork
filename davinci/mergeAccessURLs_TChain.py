
# # source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102b_LHCB_Core x86_64-centos9-gcc11-opt

import pickle
import threading
from ROOT import TFile, TTree, TList
import time
import numpy as np
from termcolor import colored
import ROOT

import signal

class timeout_class:
	def __init__(self, seconds=1, error_message='Timeout'):
		self.seconds = seconds
		self.error_message = error_message
	def handle_timeout(self, signum, frame):
		raise TimeoutError(self.error_message)
	def __enter__(self):
		signal.signal(signal.SIGALRM, self.handle_timeout)
		signal.alarm(self.seconds)
	def __exit__(self, type, value, traceback):
		signal.alarm(0)

black_list_sites = ['xrootd.grid.surfsara.nl']

def open_file_with_timeout(name, timeout):
	handle = ROOT.TFile.AsyncOpen(name)
	while timeout > 0 and ROOT.TFile.GetAsyncOpenStatus(handle) == 1: # kAOSInProgress
		time.sleep(1)
		timeout -= 1
	if timeout == 0:
		print(colored(f"File {name} TIMED OUT.",'red'))
		return None
	tfile = ROOT.TFile.Open(handle)
	if tfile.IsOpen():
		return tfile
	return None
	
def merge_root_files(pickle_file, output_file_name_prefix, timeout=30, split_up=1, skip_splits=-1):
	# Load the list of paths from the pickle file
	with open(pickle_file, 'rb') as filehandler:
		pathList = pickle.load(filehandler)
	

	pathLists = np.array_split(pathList, split_up)

	for idx, pathList in enumerate(pathLists):
		
		if idx<=skip_splits: 
			print(f'Skipping {idx}')
			continue
		
		if split_up == 1:
			outname = f"{output_file_name_prefix}.root"		
		else:
			outname = f"{output_file_name_prefix}_{idx}.root"

		treeList = TList()
		outputFile = TFile(outname, 'recreate')
		pyfilelist = []
		pytreelist = []

		total_entries = 0
		len_pathList = len(pathList)

		for path_idx, path in enumerate(pathList):
			print(f"Path {path_idx}/{len_pathList}: {path}")

			contains_blacklist_string = any(blacklist_item in path for blacklist_item in black_list_sites)
			if contains_blacklist_string:
				print(colored(f"black_list_sites {path}",'red'))
				continue

			inputFile = open_file_with_timeout(path, timeout)

			if inputFile:
				print("Got file")
				pyfilelist.append(inputFile)  # Make this TFile survive the loop
				inputTree = inputFile.Get('DecayTree')
				entries = inputTree.GetEntries()

				total_entries += entries

				print('\t{:<15} : {:>12} : Total: {:>12}'.format('Entries', entries, total_entries))

				pytreelist.append(inputTree)  # Make this TTree survive the loop
				treeList.Add(inputTree)
			
		print('\t{:<15} : {:>12}'.format('*'*15, '*'*12))
		print('\t{:<15} : {:>12}'.format('TOTAL ENTRIES', total_entries))
			
		outputFile.cd()
		print("\n BEGIN MERGE...")
		outputTree = TTree.MergeTrees(treeList)
		outputFile.Write()
		outputFile.Close()
		print(f"{outname} DONE")

if __name__ == "__main__":
	# Path to the pickle file containing the list of input ROOT files
	pickle_file = "OutputDataAccessURLs_2282.pkl"
	
	# Name of the output ROOT file
	output_file_name_prefix = "BuD0piKenu_Merge"
	
	# Call the merge function with desired basket size and timeout
	merge_root_files(pickle_file, output_file_name_prefix, timeout=25, split_up=1)
	# merge_root_files(pickle_file, output_file_name_prefix, timeout=25, split_up=10, skip_splits=-1)
	# merge_root_files(pickle_file, output_file_name_prefix, timeout=25, split_up=10, skip_splits=5)
