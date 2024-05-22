import uproot
import numpy as np

file = uproot.open("/users/am13743/fast_vertexing_variables/datasets/B2KEE_three_body_cut.root:DecayTree")

def print_unique_PIDs(branches):

    pids = np.empty(0)

    arrays = file.arrays(branches, library='pd')

    for item in branches:
        
        pids_i = np.unique(np.asarray(arrays[item]))
        pids = np.append(pids, pids_i)

    for pid in np.unique(pids):
        print(int(pid))

print_unique_PIDs(["M_TRUEID"])
print('\n')

print_unique_PIDs(["A_TRUEID", "B_TRUEID", "C_TRUEID"])
print('\n')



arrays = file.arrays(["A_TRUEID", "B_TRUEID", "C_TRUEID"], library='pd')

print(arrays.shape)
pre = arrays.shape[0]

pid_list = [-321,-211,-13,-11,11,13,211,321]

arrays = arrays[arrays['A_TRUEID'].isin(pid_list)]
arrays = arrays[arrays['B_TRUEID'].isin(pid_list)]
arrays = arrays[arrays['C_TRUEID'].isin(pid_list)]

print(np.unique(arrays['A_TRUEID']))
print(arrays.shape)
post = arrays.shape[0]
print(post/pre)
