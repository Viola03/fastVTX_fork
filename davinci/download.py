import glob
import os
import numpy as np

gangadir = '/afs/cern.ch/work/m/marshall/gangadir/'

# mode = 'B2KEE_three_body'
# sub_jobs = 1105 # number of subjobs
# # sub_jobs = 100 # number of subjobs
# job_ID = 1344

mode = 'cocktail_mix'
sub_jobs = 175 # number of subjobs
job_ID = 1404


localDir = '/eos/lhcb/user/m/marshall/gangaDownload/%d/'%job_ID
# localDir = '/eos/lhcb/user/m/marshall/gangaDownload/temp/'

try:
	os.mkdir(localDir)
except:
	pass

counts = np.zeros(2)

# for i in range(1, sub_jobs+1):
for i in range(0, sub_jobs+1):
	
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

		f = DiracFile(lfn=diracfile_loc)

		f.localDir = localDir

		f.get()

		os.rename(f'{localDir}/{diracfile_loc_root}.root',f'{localDir}/{diracfile_loc_root}_{i}.root')
		print(f'Output: {localDir}/{diracfile_loc_root}_{i}.root')

		counts[0] += 1

	except:
		print('Failure?')
		counts[1] += 1
	
# print('hadding...')
# hadd_cmd = f'hadd -fk -k {localDir}/{mode}.root {localDir}/{diracfile_loc_root}_*.root'
# os.system(hadd_cmd)
# # rm_cmd = f'rm {localDir}/{diracfile_loc_root}_*.root'
# # print('removing...')
# # os.system(rm_cmd)

# print(counts[0], counts[1])





