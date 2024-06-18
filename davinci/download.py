import glob
import os
import numpy as np
import signal
from contextlib import contextmanager

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TimeoutException(Exception): 
	print(f"{bcolors.WARNING}Timed out!{bcolors.ENDC}")
	pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def download_A(diracfile_loc, diracfile_loc_root, localDir, i):

	f = DiracFile(lfn=diracfile_loc)
	f.localDir = localDir
	f.get()
	os.rename(f'{localDir}/{diracfile_loc_root}.root',f'{localDir}/{diracfile_loc_root}_{i}.root')

import subprocess

def download_B(diracfile_loc, diracfile_loc_root, localDir, i):
	# bashCommand= "lb-dirac dirac-dms-get-file  {0} -D $PWD/{1}"
	# subprocesses.append(subprocess.Popen(bashCommand.format(line[0:-1],index),stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True))
	# subprocess.Popen(bashCommand= f"lb-dirac dirac-dms-get-file  LFN:{diracfile_loc}",stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
	subprocess.run(["lb-dirac", "dirac-dms-get-file", f"LFN:{diracfile_loc}", "-D", "{localDir}/{diracfile_loc_root}_{i}.root"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
	# print('run')
	# os.rename(f'{diracfile_loc_root}.root',f'{localDir}/{diracfile_loc_root}_{i}.root')



gangadir = '/afs/cern.ch/work/m/marshall/gangadir/'

mode = 'cocktail_mix'
sub_jobs = 1034 # number of subjobs
job_ID = 1406


localDir = '/eos/lhcb/user/m/marshall/gangaDownload/%d/'%job_ID
# localDir = '/eos/lhcb/user/m/marshall/gangaDownload/temp/'

try:
	os.mkdir(localDir)
except:
	pass

counts = np.zeros(2)

for i in range(0, sub_jobs+1):
	
	with time_limit(5):

		files = glob.glob(f'{gangadir}/workspace/marshall/LocalXML/{job_ID}/{i}/output/__postprocesslocations__')

		try:	

			file = files[0]
			print(' ')
			print(f'{i}/{sub_jobs}', file)
			
			f = open(file, "r")
			f_string = f.read()

			f_string = f_string.split('DiracFile')

			for item in f_string:
				if item != '' and 'hist' not in item and 'Hist' not in item:
					diracfile_loc = item

			diracfile_loc = diracfile_loc.split(diracfile_loc.split("/lhcb/",1)[0],1)[1]
			diracfile_loc = diracfile_loc.split(diracfile_loc.split(".root",1)[1],1)[0]

			diracfile_loc_root = diracfile_loc.split("/")[-1]
			diracfile_loc_root = diracfile_loc_root.split(".root")[0]

			print(f'Getting... {diracfile_loc}')
			# os.system(f'lb-dirac dirac-dms-lfn-accessURL LFN:{diracfile_loc}')

			# download_A(diracfile_loc, diracfile_loc_root, localDir, i)
			download_B(diracfile_loc, diracfile_loc_root, localDir, i)

			print(f'{bcolors.OKGREEN}Output: {localDir}/{diracfile_loc_root}_{i}.root {bcolors.ENDC}')

			counts[0] += 1

		except Exception as e:
			# try:
			# 	os.system(f'lb-dirac dirac-dms-lfn-accessURL LFN:{diracfile_loc}')
			# except:
			# 	pass
			print(f"{bcolors.FAIL}Failure! {e} {bcolors.ENDC}")
			counts[1] += 1
	
	if i == 15:
		quit()











# import glob
# import os
# import numpy as np


# gangadir = '/afs/cern.ch/work/m/marshall/gangadir/'

# # mode = 'B2KEE_three_body'
# # sub_jobs = 1105 # number of subjobs
# # # sub_jobs = 100 # number of subjobs
# # job_ID = 1344

# mode = 'cocktail_mix'
# sub_jobs = 1034 # number of subjobs
# job_ID = 1406


# localDir = '/eos/lhcb/user/m/marshall/gangaDownload/%d/'%job_ID
# # localDir = '/eos/lhcb/user/m/marshall/gangaDownload/temp/'

# try:
# 	os.mkdir(localDir)
# except:
# 	pass

# counts = np.zeros(2)

# # for i in range(1, sub_jobs+1):
# for i in range(0, sub_jobs+1):
	
# 	files = glob.glob(f'{gangadir}/workspace/marshall/LocalXML/{job_ID}/{i}/output/__postprocesslocations__')

# 	try:	
# 		file = files[0]
# 		print(' ')
# 		print(f'{i}/{sub_jobs}', file)
		
# 		f = open(file, "r")
# 		f_string = f.read()

# 		f_string = f_string.split('DiracFile')

# 		for item in f_string:
# 			if item != '' and 'hist' not in item and 'Hist' not in item:
# 				diracfile_loc = item

# 		diracfile_loc = diracfile_loc.split(diracfile_loc.split("/lhcb/",1)[0],1)[1]
# 		diracfile_loc = diracfile_loc.split(diracfile_loc.split(".root",1)[1],1)[0]

# 		diracfile_loc_root = diracfile_loc.split("/")[-1]
# 		diracfile_loc_root = diracfile_loc_root.split(".root")[0]

# 		print(f'Getting... {diracfile_loc}')

# 		f = DiracFile(lfn=diracfile_loc)

# 		f.localDir = localDir

# 		f.get()

# 		os.rename(f'{localDir}/{diracfile_loc_root}.root',f'{localDir}/{diracfile_loc_root}_{i}.root')
# 		print(f'Output: {localDir}/{diracfile_loc_root}_{i}.root')

# 		counts[0] += 1

# 	except:
# 		print('Failure?')
# 		counts[1] += 1
	
# # print('hadding...')
# # hadd_cmd = f'hadd -fk -k {localDir}/{mode}.root {localDir}/{diracfile_loc_root}_*.root'
# # os.system(hadd_cmd)
# # # rm_cmd = f'rm {localDir}/{diracfile_loc_root}_*.root'
# # # print('removing...')
# # # os.system(rm_cmd)

# # print(counts[0], counts[1])





