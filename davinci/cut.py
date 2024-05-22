#source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh


import ROOT

mode = 'B2KEE_three_body'

job_ID = 716

localDir = '/eos/lhcb/user/m/marshall/gangaDownload/%d/'%job_ID

print("Open the original ROOT file")
file = ROOT.TFile(f"{localDir}/{mode}.root")

print("Get the TTree from the file")
tree = file.Get("B2Kee_Tuple/DecayTree")


# # List of branches to keep
# branches_to_keep = ["branch1", "branch2", "branch3"]

# # Disable all branches initially
# tree.SetBranchStatus("*", 0)

# # Enable only the branches you want to keep
# for branch in branches_to_keep:
#     tree.SetBranchStatus(branch, 1)



print("Define the cut")
cut = "(M_TRUEID!=0 && M_BKGCAT<60)"

print("Apply the cut and create a new TTree")
new_tree = tree.CopyTree(cut)

print("Save the new TTree to a new ROOT file")
new_file = ROOT.TFile(f"{localDir}/{mode}_cut.root", "RECREATE")
new_tree.Write()
new_file.Close()

print("Close the original file")
file.Close()
