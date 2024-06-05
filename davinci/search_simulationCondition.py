
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import numpy as np
import re
from bookkeeping_lib import Bookkeeping_Browser
import pickle 

year = '2018'

beam = 'Beam6500GeV-2018-MagUp-Nu1.6-25ns-Pythia8'
sim = 'Sim09i'
# sim = 'Sim09h'
# sim = 'Sim09k'
trig = 'Trig0x617d18a4'
reco = 'Reco18'
turbo = 'Turbo05-WithTurcal'

browser = Bookkeeping_Browser('/Users/am13743/Desktop/4qszh6o1.default-release', headless=False)
# browser.switch_to_eventType()
mother, mother_level = browser.open_slash_directory()
mother, mother_level = browser.open_MC_directory()

# Enter the chosen year
branches = browser.get_branches(mother=mother, mother_level=mother_level, quiet=True)
mother, mother_level = browser.expand_directory(element_id=branches[year], title=year)

for option in [beam, sim, trig, reco, turbo]:
	branches = browser.get_branches(mother=mother, mother_level=mother_level, quiet=True)
	mother, mother_level = browser.expand_directory(element_id=branches[option], title=option)

def filter_items(item_list):
    pattern = re.compile(r'^\d{8}$')
    filtered_list = [item for item in item_list if pattern.match(item)]
    return np.asarray(filtered_list).astype(np.uint32)

output = {}
output[sim] = {}

stripping_branches = browser.get_branches(mother=mother, mother_level=mother_level, quiet=True)
for stripping_branch in stripping_branches:
	if "Str" in stripping_branch:
		print('\n',stripping_branch)
		mother, mother_level = browser.expand_directory(element_id=stripping_branches[stripping_branch], title=stripping_branch)
		data_branches = browser.get_branches(mother=mother, mother_level=mother_level, quiet=True)
		data_branches = filter_items(list(data_branches.keys()))
		data_branches = data_branches[np.where(data_branches>11000000)]
		data_branches = data_branches[np.where(data_branches<15000000)]
		output[sim][stripping_branch] = data_branches
		# close directory again
		mother, mother_level = browser.expand_directory(element_id=stripping_branches[stripping_branch], title=stripping_branch)


print(sim)
for key in list(output[sim].keys()):
	print(key, output[sim][key])

try:
	filehandler = open('targets.pkl', 'rb') 
	all_targets = pickle.load(filehandler)
	all_targets[sim] = output[sim]
	filehandler = open('targets.pkl', 'wb') 
	pickle.dump(all_targets, filehandler)
except:
	filehandler = open(f'targets.pkl', 'wb') 
	pickle.dump(output, filehandler)



# PATH_name = "/MC/2018/Beam6500GeV-2018-MagDown-Nu1.6-25ns-Pythia8/Sim09i/Trig0x617d18a4/Reco18/Turbo05-WithTurcal/Stripping34NoPrescalingFlagged/12123003/ALLSTREAMS.DST"





