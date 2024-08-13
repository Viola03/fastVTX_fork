import os, re
import numpy as np
import time
import pickle 

class timer:
	def __init__(self):
		self.start_time = time.time()
		self.end_time = time.time()
	def hit_start(self):
		self.start_time = time.time()
	def hit_stop(self, print_time=True):
		self.end_time = time.time()
		total = self.end_time-self.start_time
		if print_time:
			print(f'Time: {total:.4f}s')
		return total

stopwatch = timer()
stopwatch_total = timer()

# LFNs = {}

# filehandler = open('targets.pkl', 'rb') 
# all_targets = pickle.load(filehandler)

# for sim in list(all_targets.keys()):

# 	targets = all_targets[sim]['Stripping34NoPrescalingFlagged']
# 	np.random.shuffle(targets)
# 	# targets = targets[:25]

# 	extend = False

# 	stopwatch_total.hit_start()

# 	for target_idx, target in enumerate(targets):
		
# 		try:
# 			stopwatch.hit_start()

# 			print('\n',target,f'{target_idx}/{len(targets)} in {stopwatch_total.hit_stop(print_time=False):.4f}s')
			
# 			PATH_name = f"/MC/2018/Beam6500GeV-2018-MagUp-Nu1.6-25ns-Pythia8/{sim}/Trig0x617d18a4/Reco18/Turbo05-WithTurcal/Stripping34NoPrescalingFlagged/{target}/ALLSTREAMS.DST"
# 			bk_query = BKQuery(path=PATH_name)
# 			dataset = bk_query.getDataset()

# 			LFNs[target] = [a.lfn for a in dataset]

# 			# files = []
# 			# for n in range(nfiles_per):
# 			# 	files.append(dataset[n].lfn)
# 			# if extend:
# 			# 	comp_dataset.extend(files)
# 			# else:
# 			# 	fileset = LHCbCompressedFileSet(files)
# 			# 	comp_dataset = LHCbCompressedDataset(fileset)
# 			# 	extend = True

# 			stopwatch.hit_stop()

# 			filehandler = open('targets_LFNs.pkl', 'wb')  
# 			print(len(list(LFNs.keys())))
# 			pickle.dump(LFNs, filehandler)

# 		except Exception as e:
# 			print(f"\n\nAn error occurred: {e}\n\n")


filehandler = open('targets_LFNs.pkl', 'rb')  
LFNs = pickle.load(filehandler)

extend = False


# nfiles_per_event_typs = 30
# filesPerJob = 5
# nfiles_per_event_typs = 150
# filesPerJob = 5
# nfiles_per_event_typs = 10
nfiles_per_event_typs = 3
filesPerJob = 2

# print(len(list(LFNs.keys())))
# quit()

total = 0
for target in list(LFNs.keys()):

	files = []
	LFNs[target] = np.asarray(LFNs[target])
	np.random.shuffle(LFNs[target])
	if len(LFNs[target]) < nfiles_per_event_typs:
		files = LFNs[target]
	else:
		for n in range(nfiles_per_event_typs):
			files.append(LFNs[target][n])
	
	total += len(files)

	if extend:
		comp_dataset.extend(files)
	else:
		fileset = LHCbCompressedFileSet(files)
		comp_dataset = LHCbCompressedDataset(fileset)
		extend = True

print(f'\n\ntotal number of files: {total}, jobs {total/filesPerJob:2f}\n\n')

# job_name = 'mix'
year = ["18"]
energy = ["6500"]
strip_v = ["34"]
Reco_v = ["18"]
polarity = ["Down"]
streams = ["ALLSTREAMS.DST"]
i = 0
job_name = (
    "20"
    + year[i]
    + "_Reco"
    + Reco_v[i]
    + "Strip"
    + strip_v[i]
    + "_"
    + polarity[i]
    + "_"
    + streams[i]
)

try:
    myApp = prepareGaudiExec("DaVinci", "v44r3", myPath=".")
except:
    myApp = GaudiExec()
    myApp.directory = "./DaVinciDev_v44r3"

# myApp.platform = "x86_64-slc6-gcc62-opt"
myApp.platform = "x86_64+avx2+fma-centos7-gcc62-opt"
# myApp.options = ["./davinci_intermediates.py", "./print_something.py"]
# myApp.options = ["./print_something.py", "./davinci_intermediates.py"]
# myApp.options = ["./davinci_intermediates.py"]
# myApp.options = ["./davinci_general_mcmatch.py"]
myApp.options = ["./davinci_general_mcmatch_intermediates.py"]

bck = Dirac()
# bck = Condor()
# bck = Local()
# bck = Interactive()

splitter = SplitByFiles()
splitter.ignoremissing = True
splitter.maxFiles = -1
splitter.filesPerJob = filesPerJob

job = Job(name=job_name, comment=job_name, backend=bck, splitter=splitter)

job.backend.downloadSandbox = False


Year = (
    bool("2011" in job_name) * ' "2011" '
    + bool("2012" in job_name) * ' "2012" '
    + bool("2015" in job_name) * ' "2015"  '
    + bool("2016" in job_name) * ' "2016"  '
    + bool("2017" in job_name) * ' "2017" '
    + bool("2018" in job_name) * ' "2018" '
)
job.do_auto_resubmit = False
job.application = myApp

job.application.extraOpts = (
    "from Configurables import DaVinci                     ; "
    + 'DaVinci().TupleFile     = "DTT_'
    + job_name
    + '.root"  ; '
    + "DaVinci().EvtMax        =              -1             ; "
    # + "DaVinci().EvtMax        =              5             ; "
    + "DaVinci().PrintFreq        =              2500             ; "
    + "DaVinci().VerboseMessages        =              False             ; "
    + "from Configurables import CondDB                      ; "
    + "CondDB( LatestGlobalTagByDataType = "
    + Year
    + ")     ; "
    + "DaVinci().DataType      =   "
    + Year
    + "              ; "
)

print("Create job for thee jobs: ", job.name)
# job.inputdata = comp_dataset[:5]
job.inputdata  = comp_dataset
# job.inputdata  = comp_dataset[:3]

# This throws the files on the grid personall space
# job.outputfiles = [
#     # DiracFile(namePattern="*.root"),
#     LocalFile(namePattern="DTT*.root"),
#     LocalFile("summary.xml"),
# ]  # keep my Tuples on grid element (retrive manually)

job.outputfiles = [
    DiracFile(namePattern="DTT*.root"),
]  # keep my Tuples on grid element (retrive manually)


jobs.parallel_submit = True
job.submit()
print("======================================")
print("job: ", job.name + " submitted")
print("======================================")

print(" Jobs submitted .... bye ")
