# source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102b_LHCB_Core x86_64-centos9-gcc11-opt

import ROOT
from ROOT import TFile, TTree, TList
import glob
import os

branches_to_keep = [
    'MOTHER_DIRA_OWNPV', 'MOTHER_ENDVERTEX_CHI2', 'MOTHER_ENDVERTEX_NDOF', 'MOTHER_ENDVERTEX_X', 'MOTHER_ENDVERTEX_Y', 'MOTHER_ENDVERTEX_Z',
    'MOTHER_FDCHI2_OWNPV', 'MOTHER_IPCHI2_OWNPV', 'MOTHER_OWNPV_X', 'MOTHER_OWNPV_Y', 'MOTHER_OWNPV_Z', 'MOTHER_PX', 'MOTHER_PY', 
    'MOTHER_PZ', 'MOTHER_TRUEP_X', 'MOTHER_TRUEP_Y', 'MOTHER_TRUEP_Z', 'MOTHER_TRUEID', 'MOTHER_BKGCAT',
    'DAUGHTER1_ID', 'DAUGHTER1_IPCHI2_OWNPV', 'DAUGHTER1_PX', 'DAUGHTER1_PY', 'DAUGHTER1_PZ', 'DAUGHTER1_TRACK_CHI2NDOF', 'DAUGHTER1_TRUEID', 'DAUGHTER1_TRUEP_X', 
    'DAUGHTER1_TRUEP_Y', 'DAUGHTER1_TRUEP_Z', 'DAUGHTER2_ID', 'DAUGHTER2_IPCHI2_OWNPV', 'DAUGHTER2_PX', 'DAUGHTER2_PY', 'DAUGHTER2_PZ', 'DAUGHTER2_TRACK_CHI2NDOF', 
    'DAUGHTER2_TRUEID', 'DAUGHTER2_TRUEP_X', 'DAUGHTER2_TRUEP_Y', 'DAUGHTER2_TRUEP_Z', 'DAUGHTER3_ID', 'DAUGHTER3_IPCHI2_OWNPV', 'DAUGHTER3_PX', 'DAUGHTER3_PY', 
    'DAUGHTER3_PZ', 'DAUGHTER3_TRACK_CHI2NDOF', 'DAUGHTER3_TRUEID', 'DAUGHTER3_TRUEP_X', 'DAUGHTER3_TRUEP_Y', 'DAUGHTER3_TRUEP_Z', 'nSPDHits', 'nTracks',
    'INTERMEDIATE_TRUEID',
    'INTERMEDIATE_DIRA_OWNPV', 'INTERMEDIATE_ENDVERTEX_CHI2', 'INTERMEDIATE_ENDVERTEX_NDOF',
    'INTERMEDIATE_FDCHI2_OWNPV', 'INTERMEDIATE_IPCHI2_OWNPV',
    'MOTHER_TRUEENDVERTEX_X',
    'MOTHER_TRUEENDVERTEX_Y',
    'MOTHER_TRUEENDVERTEX_Z',

    'MOTHER_TRUEORIGINVERTEX_X',
    'MOTHER_TRUEORIGINVERTEX_Y',
    'MOTHER_TRUEORIGINVERTEX_Z',

    'INTERMEDIATE_TRUEENDVERTEX_X',
    'INTERMEDIATE_TRUEENDVERTEX_Y',
    'INTERMEDIATE_TRUEENDVERTEX_Z',
    'INTERMEDIATE_TRUEORIGINVERTEX_X',
    'INTERMEDIATE_TRUEORIGINVERTEX_Y',
    'INTERMEDIATE_TRUEORIGINVERTEX_Z',
    'EVT_GenEvent',

    'DAUGHTER1_TRUEENDVERTEX_X',
    'DAUGHTER1_TRUEENDVERTEX_Y',
    'DAUGHTER1_TRUEENDVERTEX_Z',
    'DAUGHTER1_TRUEORIGINVERTEX_X',
    'DAUGHTER1_TRUEORIGINVERTEX_Y',
    'DAUGHTER1_TRUEORIGINVERTEX_Z',

    'DAUGHTER2_TRUEENDVERTEX_X',
    'DAUGHTER2_TRUEENDVERTEX_Y',
    'DAUGHTER2_TRUEENDVERTEX_Z',
    'DAUGHTER2_TRUEORIGINVERTEX_X',
    'DAUGHTER2_TRUEORIGINVERTEX_Y',
    'DAUGHTER2_TRUEORIGINVERTEX_Z',

    'DAUGHTER3_TRUEENDVERTEX_X',
    'DAUGHTER3_TRUEENDVERTEX_Y',
    'DAUGHTER3_TRUEENDVERTEX_Z',
    'DAUGHTER3_TRUEORIGINVERTEX_X',
    'DAUGHTER3_TRUEORIGINVERTEX_Y',
    'DAUGHTER3_TRUEORIGINVERTEX_Z',
	
    "MOTHER_MC_MOTHER_ID",
    "MOTHER_MC_GD_MOTHER_ID",
    "MOTHER_MC_GD_GD_MOTHER_ID",
    "INTERMEDIATE_MC_MOTHER_ID",
    "INTERMEDIATE_MC_GD_MOTHER_ID",
    "INTERMEDIATE_MC_GD_GD_MOTHER_ID",
    "DAUGHTER1_MC_MOTHER_ID",
    "DAUGHTER1_MC_GD_MOTHER_ID",
    "DAUGHTER1_MC_GD_GD_MOTHER_ID",
    "DAUGHTER2_MC_MOTHER_ID",
    "DAUGHTER2_MC_GD_MOTHER_ID",
    "DAUGHTER2_MC_GD_GD_MOTHER_ID",
    "DAUGHTER3_MC_MOTHER_ID",
    "DAUGHTER3_MC_GD_MOTHER_ID",
    "DAUGHTER3_MC_GD_GD_MOTHER_ID",

    "INTERMEDIATE_ENDVERTEX_CHI2",
    "INTERMEDIATE_DIRA_OWNPV",
    "MOTHER_VTXISOBDTHARDFIRSTVALUE",
    "MOTHER_VTXISOBDTHARDSECONDVALUE",
    "MOTHER_VTXISOBDTHARDTHIRDVALUE",
    "MOTHER_SmallestDeltaChi2OneTrack",
    "MOTHER_SmallestDeltaChi2TwoTracks",
    "MOTHER_cp_0.70",
    "MOTHER_cpt_0.70",
    "MOTHER_cmult_0.70",
    "DAUGHTER1_TRACK_GhostProb",
    "DAUGHTER2_TRACK_GhostProb",
    "DAUGHTER3_TRACK_GhostProb",
    "MOTHER_OWNPV_X",
    "MOTHER_ENDVERTEX_X",
    "INTERMEDIATE_ENDVERTEX_X",
    "MOTHER_OWNPV_Y",
    "MOTHER_ENDVERTEX_Y",
    "INTERMEDIATE_ENDVERTEX_Y",
    "MOTHER_OWNPV_Z",
    "MOTHER_ENDVERTEX_Z",
    "INTERMEDIATE_ENDVERTEX_Z",
    "MOTHER_OWNPV_XERR",
    "MOTHER_ENDVERTEX_XERR",
    "INTERMEDIATE_ENDVERTEX_XERR",
    "MOTHER_OWNPV_YERR",
    "MOTHER_ENDVERTEX_YERR",
    "INTERMEDIATE_ENDVERTEX_YERR",
    "MOTHER_OWNPV_ZERR",
    "MOTHER_ENDVERTEX_ZERR",
    "INTERMEDIATE_ENDVERTEX_ZERR",
    "MOTHER_OWNPV_COV_", # these are 9 values each!
    "MOTHER_ENDVERTEX_COV_",
    "INTERMEDIATE_ENDVERTEX_COV_", # Also need XERR?
]



new_targets = [
    "J_psi_1S_ENDVERTEX_CHI2",
    "J_psi_1S_DIRA_OWNPV",
    # VertexIsoBDTInfo:
    "B_plus_VTXISOBDTHARDFIRSTVALUE",
    "B_plus_VTXISOBDTHARDSECONDVALUE",
    "B_plus_VTXISOBDTHARDTHIRDVALUE",
    # TupleToolVtxIsoln:
    "B_plus_SmallestDeltaChi2OneTrack",
    "B_plus_SmallestDeltaChi2TwoTracks",
    # TupleToolTrackIsolation:
    "B_plus_cp_0.70",
    "B_plus_cpt_0.70",
    "B_plus_cmult_0.70",
    # Ghost:
    "DAUGHTER1_TRACK_GhostProb",
    "DAUGHTER2_TRACK_GhostProb",
    "DAUGHTER3_TRACK_GhostProb",
    # Vertex info - for rerunning DecayTreeFitter (for mass constrained variables):
    "MOTHER_OWNPV_X",
    "MOTHER_ENDVERTEX_X",
    "INTERMEDIATE_ENDVERTEX_X",
    "MOTHER_OWNPV_Y",
    "MOTHER_ENDVERTEX_Y",
    "INTERMEDIATE_ENDVERTEX_Y",
    "MOTHER_OWNPV_Z",
    "MOTHER_ENDVERTEX_Z",
    "INTERMEDIATE_ENDVERTEX_Z",
    "MOTHER_OWNPV_COV_", # these are 9 values each!
    "MOTHER_ENDVERTEX_COV_",
    "INTERMEDIATE_ENDVERTEX_COV_", # Also need XERR?
]



def trim_file_DecayTree(filename):
    # Open the old file
    oldfile = ROOT.TFile(filename, "READ")
    oldtree = oldfile.Get("DecayTree")

    # Deactivate all branches
    oldtree.SetBranchStatus("*", 0)

    # Activate only specific branches
    for branch_name in branches_to_keep:
        oldtree.SetBranchStatus(branch_name, 1)

    # Create a new file and clone the old tree into the new file
    newfile = ROOT.TFile("small.root", "RECREATE")
    newtree = oldtree.CloneTree(-1, "fast")

    # Print tree information
    # newtree.Print()

    # Write and close the new file
    newfile.Write()
    newfile.Close()


def trim_file_directory_structure(filename_in, filename_out):
    # Open the old file
    oldfile = ROOT.TFile(filename_in, "READ")

    oldtrees = []

    for key in oldfile.GetListOfKeys():
        dir_name = key.GetName()
        directory = oldfile.Get(dir_name)
        oldtree = directory.Get('DecayTree')
        oldtree.SetBranchStatus("*", 0)
        for branch_name in branches_to_keep:
            oldtree.SetBranchStatus(branch_name, 1)
        oldtrees.append((dir_name, oldtree))

    # Create a new file and clone the old tree into the new file
    newfile = ROOT.TFile(filename_out, "RECREATE")

    # Clone the trees into the new file, preserving the directory structure
    for dir_name, oldtree in oldtrees:
        # Create the directory in the new file
        newfile.mkdir(dir_name)
        newfile.cd(dir_name)
        
        # Clone the tree and write it to the new directory
        newtree = oldtree.CloneTree(-1, "fast")
        newtree.Write('DecayTree')

    newfile.Write()
    newfile.Close()

def merge_directory_structure(inname, outname):

    # input_file = TFile("cocktail_general_MC_hierachy.root", 'read')
    input_file = TFile(inname, 'read')
    treeList = TList()
    outputFile = TFile(outname, 'recreate')

    for key in input_file.GetListOfKeys():
        dir_name = key.GetName()
        directory = input_file.Get(dir_name)
        inputTree = directory.Get('DecayTree')
        outputTree = inputTree.CloneTree() #instead of extensive processing
        treeList.Add(inputTree)

    outputFile.cd()
    outputTree = TTree.MergeTrees(treeList)
    outputFile.Write()
    outputFile.Close()


# trim_file_DecayTree("MergeTest.root")

# merge_directory_structure("DVntuple_general_full.root", "DVntuple_general.root")
merge_directory_structure("DVntuple.root", "DVntuple_collapse.root")
quit()

file_list = glob.glob("/afs/cern.ch/work/m/marshall/gangadir/workspace/marshall/LocalXML/2006/*/output/DTT_*.root")


for file_idx, file in enumerate(file_list):

    print('\n',file_idx,'/',len(file_list),file)
    file_trimmed = file[:-5]+'_trimmed.root'
    file_merged = file[:-5]+'_merged.root'

    trim_file_directory_structure(file, file_trimmed)
    merge_directory_structure(file_trimmed, file_merged)
    os.remove(file_trimmed)


