import os, re
import numpy as np
import time
import pickle 

# def hadd_job_output(job, output_filename):
#     command = ["hadd", "-fk", output_filename] + job.backend.getOutputDataAccessURLs()
#     os.system(" ".join(command))

# How do I run an executable job that uses input files on the Grid as arguments to the script?

JOB_ID = 2011

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


	for file in files:
		print(file)
	files_Dirac = [DiracFile(lfn=file) for file in files]
	files_list.extend(files_Dirac)

	# if lfn_idx == 2:
	# 	break
	
job_name = "%s_Collapse"%JOB_ID


bck = Dirac()

job = Job(name=job_name, comment=job_name, backend=bck)

job.splitter = GenericSplitter(attribute = 'inputfiles', values = [[i, LocalFile('collapse_individual_tuples_GANGA.py')] for idx, i in enumerate(files_list)])
job.inputfiles = files_list

myApp = Executable()
myApp.exe = File('run_collapse_individual_tuples_GANGA.sh')
job.application = myApp

job.outputfiles = [
	DiracFile(namePattern="*merged.root"),
]  # keep my Tuples on grid element (retrive manually)

jobs.parallel_submit = True
job.submit()
print("======================================")
print("job: ", job.name + " submitted")
print("======================================")

print(" Jobs submitted .... bye ")