import os, re
import numpy as np
import time
import pickle 

# def dumpDataAccessURLs(jobs_i, output_name):
# 	files = jobs_i.backend.getOutputDataAccessURLs()
# 	filehandler = open(output_name, 'wb') 
# 	pickle.dump(files, filehandler)

# dumpDataAccessURLs(jobs(2048), "OutputDataAccessURLs_2048.pkl")

# quit()

# How do I run an executable job that uses input files on the Grid as arguments to the script?

JOB_ID = 2046

filesPerJob = 1

############################

extend = False
j = jobs(JOB_ID)
files_list = []
for lfn_idx, sj in enumerate(j.subjobs.select(status="completed")): 
	
	files = sj.backend.getOutputDataLFNs()
	
	if extend:
		ds.extend(files)
	else:
		fileset = LHCbCompressedFileSet(files)
		ds = LHCbCompressedDataset(fileset)
		extend = True
		
	files_Dirac = [DiracFile(lfn=file) for file in files]
	files_list.extend(files_Dirac)

	# if lfn_idx == 2:
	# 	break
	
job_name = "%s_Collapse"%JOB_ID

bck = Dirac()

job = Job(name=job_name, comment=job_name, backend=bck)

job.splitter = GenericSplitter(attribute = 'inputfiles', values = [[i, LocalFile('collapse_individual_tuples_GANGA.py')] for idx, i in enumerate(files_list)])

myApp = Executable()
myApp.exe = File('run_collapse_individual_tuples_GANGA.sh')
myApp.platform = "x86_64-centos7-gcc8-opt"
job.application = myApp

job.outputfiles = [
	DiracFile(namePattern="*merged*.root"),
]

job.backend.settings['BannedSites'] = ['LCG.Beijing.cn'] #We have issues with this grid site in particular

jobs.parallel_submit = True
job.submit()
print("======================================")
print("job: ", job.name + " submitted")
print("======================================")

print(" Jobs submitted .... bye ")
