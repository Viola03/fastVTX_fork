
# # source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102b_LHCB_Core x86_64-centos9-gcc11-opt

# import pickle
# from ROOT import TFile, TTree, TList

# filehandler = open("OutputDataAccessURLs.pkl", 'rb') 
# pathList = pickle.load(filehandler)
	
# treeList = TList()
# outputFile = TFile('MergeTest.root', 'recreate')
# pyfilelist = []
# pytreelist = []

# for path_idx, path in enumerate(pathList):
# 		print("Path", path)
# 		inputFile = TFile.Open(path, 'read')
# 		pyfilelist.append(inputFile) # Make this TFile survive the loop!
# 		inputTree = inputFile.Get('DecayTree')
# 		pytreelist.append(inputTree) # Make this TTree survive the loop!
# 		outputTree = inputTree.CloneTree() #instead of extensive processing
# 		treeList.Add(inputTree)

# outputFile.cd()
# outputTree = TTree.MergeTrees(treeList)
# outputFile.Write()
# outputFile.Close()






import pickle
import threading
from ROOT import TFile, TTree, TList
import time
import numpy as np
from termcolor import colored


black_list = ['xrootd.grid.surfsara.nl']

class FileLoader(threading.Thread):
	def __init__(self, path):
		threading.Thread.__init__(self)
		self.path = path
		self.file = None

	def run(self):
		# if np.random.uniform() < 0.5:
		# 	print('SLEEPY')
		# 	time.sleep(30)
		self.file = TFile.Open(self.path, 'read')

def open_file_with_timeout(path, timeout=30):
	file_loader = FileLoader(path)
	file_loader.start()
	file_loader.join(timeout)
	
	if file_loader.is_alive():
		print(colored(f"File {path} took too long to load and was skipped.",'red'))
		return None
	else:
		return file_loader.file
	
def merge_root_files(pickle_file, output_file_name_prefix, timeout=30, split_up=1):
	# Load the list of paths from the pickle file
	with open(pickle_file, 'rb') as filehandler:
		pathList = pickle.load(filehandler)
	

	pathLists = np.array_split(pathList, split_up)

	for idx, pathList in enumerate(pathLists):

		outname = f"{output_file_name_prefix}_{idx}.root"

		treeList = TList()
		outputFile = TFile(outname, 'recreate')
		pyfilelist = []
		pytreelist = []

		total_entries = 0
		len_pathList = len(pathList)

		for path_idx, path in enumerate(pathList):
			print(f"Path {path_idx}/{len_pathList}: {path}")

			contains_blacklist_string = any(blacklist_item in path for blacklist_item in black_list)

			if contains_blacklist_string:
				print(colored(f"black_list {path}",'red'))
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
	pickle_file = "OutputDataAccessURLs_2048.pkl"
	
	# Name of the output ROOT file
	output_file_name_prefix = "MergeTest"
	
	# Call the merge function with desired basket size and timeout
	merge_root_files(pickle_file, output_file_name_prefix, timeout=30, split_up=10)
