import argparse
import configparser
import distutils
import os
import zfit
import os
import socket
from str2bool import str2bool

parser = configparser.ConfigParser()
parser.optionxform = str # Allows variable names to be case sensitive

'''

Compatible types:
	
	- floats
	- bools (recognised by distutils.util.strtobool() so accepts 'yes' true 'no' 'false' False etc...)
	- int (recognised by lack of '.')
	- strings

'''


parser['DEFAULT'] = {

	# Here go default values of all definitions...
	'help': False,

	'processID':0,

}

# Can overwrite the above defaults in mainConfig.txt
parser.read("mainConfig.txt".format(**os.environ))

def read_definition(name, organising_types=False):
	''' Function to access a defintion from anywhere else '''
	try:
		value = parser.get('mainConfig',name)
	except:
		print("no mainConfig.txt in current directory...")
		print("\n\n quitting...")
		quit()
	if not organising_types:
		try:
			if value in ['0','1']: quit()
			else:
				str2bool_result = str2bool(value)
				if str2bool_result == None: quit()
				value = str2bool_result
		except:
			try:
				if '.' not in value:
					value = int(value)
				else:
					value = float(value)
			except ValueError:
				pass
			pass
	return value

def update_definition(name, value):
	''' Function to access a defintion from anywhere else '''
	value = parser.set('mainConfig',name, str(value))

## Make a list of possible arguments and their types...
arguments_libary = parser['DEFAULT']
arguments_types = {}
for key in arguments_libary.keys():
	arguments_types[key] = type(read_definition(key, organising_types=True))#.__name__

## Look for all arguments...
argparser = argparse.ArgumentParser()
for key in arguments_types.keys():
	argparser.add_argument('-%s'%key, action='store', dest='%s'%key, type=arguments_types[key])
args = argparser.parse_args().__dict__

if args['help']:
	print("\n\nAvaliable options:\n")
	for key in parser['DEFAULT']:
		print('{:<36s}  :  {:>36s}'.format(key,parser['DEFAULT'][key]))
	print('\n')
	quit()

info_string = "\n"
## Override if any existing...
for key in args.keys():
	if args[key] != None:
		info_string += f"Setting option {key} to {args[key]}\n"
		update_definition(key, args[key])

## Create a namespace...
from types import SimpleNamespace 
arguments = {}
for key in arguments_types.keys():
	value = read_definition(key)
	arguments[key] = value

arguments["info_string"] = info_string

rd = SimpleNamespace(**arguments)

####
####
####
####
####
####

rd.seed = -1
if rd.seed != -1:
	print('\nSetting seed to %d\n'%rd.seed)
	zfit.settings.set_seed(rd.seed)

os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"
print('\n\thost:', socket.gethostname())

if 'gpu01' in socket.gethostname() or 'gpu02' in socket.gethostname() or 'bc4' in socket.gethostname():
	os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%rd.processID
	assert rd.processID <= 5, '\n Only 6 GPUs on gpu01!'
	print("\tCUDA_VISIBLE_DEVICES = {CUDA_VISIBLE_DEVICES}".format(**os.environ))
else:
	import tensorflow as tf
	num_threads = 10
	num_threads += -1
	os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
	os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_threads}"
	os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_threads}"

	tf.config.threading.set_inter_op_parallelism_threads(
		num_threads
	)
	tf.config.threading.set_intra_op_parallelism_threads(
		num_threads
	)
	tf.config.set_soft_device_placement(True)

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

zfit.settings.advanced_warnings['sum_extended_frac'] = False

def print_splash():
	print('\n\n*****************************************************************')
	print('Fast generation of vertexing variables....')
	print('*****************************************************************') # https://patorjk.com/software/taag/#p=display&f=Mini&t=B%20-%3E%20h%20u%20u%20toys
print_splash()
print(rd.info_string)
print('\n')





