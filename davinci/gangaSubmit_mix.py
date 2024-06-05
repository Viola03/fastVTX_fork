import os, re
import numpy as np
import time

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

import pickle 
filehandler = open('targets.pkl', 'rb') 
all_targets = pickle.load(filehandler)
targets = all_targets['Sim09h']['Stripping34NoPrescalingFlagged']
np.random.shuffle(targets)
targets = targets[:25]

nfiles_per = 5

extend = False

stopwatch_total.hit_start()

for target_idx, target in enumerate(targets):
	
	try:
		stopwatch.hit_start()

		print('\n',target,f'{target_idx}/{len(targets)} in {stopwatch_total.hit_stop(print_time=False):.4f}s')
		
		PATH_name = f"/MC/2018/Beam6500GeV-2018-MagUp-Nu1.6-25ns-Pythia8/Sim09h/Trig0x617d18a4/Reco18/Turbo05-WithTurcal/Stripping34NoPrescalingFlagged/{target}/ALLSTREAMS.DST"
		bk_query = BKQuery(path=PATH_name)
		dataset = bk_query.getDataset()

		files = []
		for n in range(nfiles_per):
			files.append(dataset[n].lfn)
		if extend:
			comp_dataset.extend(files)
		else:
			fileset = LHCbCompressedFileSet(files)
			comp_dataset = LHCbCompressedDataset(fileset)
			extend = True

		stopwatch.hit_stop()

	except Exception as e:
		print(f"\n\nAn error occurred: {e}\n\n")

job_name = 'mix'

try:
    myApp = prepareGaudiExec("DaVinci", "v44r3", myPath=".")
except:
    myApp = GaudiExec()
    myApp.directory = "./DaVinciDev_v44r3"

myApp.platform = "x86_64-slc6-gcc62-opt"
myApp.options = ["./davinci_intermediates.py"]

bck = Dirac()
# bck = Condor()
# bck = Local()
# bck = Interactive()

splitter = SplitByFiles()
splitter.ignoremissing = True
splitter.maxFiles = -1
splitter.filesPerJob = 1

job = Job(name=job_name, comment=job_name, backend=bck, splitter=splitter)
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
    # + "DaVinci().EvtMax        =              25             ; "
    + "from Configurables import CondDB                      ; "
    + "CondDB( LatestGlobalTagByDataType = "
    + Year
    + ")     ; "
    + "DaVinci().DataType      =   "
    + Year
    + "              ; "
)

print("Create job for thee jobs: ", job.name)
job.inputdata  = comp_dataset

# This throws the files on the grid personall space
job.outputfiles = [
    DiracFile(namePattern="*.root"),
    LocalFile("summary.xml"),
]  # keep my Tuples on grid element (retrive manually)
# job.outputfiles= [LocalFile(namePattern='*.root') , LocalFile('summary.xml') ] # keep my Tuples on grid element (retrive manually)
jobs.parallel_submit = True
job.submit()
print("======================================")
print("job: ", job.name + " submitted")
print("======================================")

print(" Jobs submitted .... bye ")
