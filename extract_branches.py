import uproot

conditions = [
    "K_Kst_PX",
    "K_Kst_PY",
    "K_Kst_PZ",
    "e_minus_PX",
    "e_minus_PY",
    "e_minus_PZ",
    "e_plus_PX",
    "e_plus_PY",
    "e_plus_PZ",
    "nTracks",
    "nSPDHits",
]

conditions_TRUTH = [
    "K_Kst_TRUEP_X",
    "K_Kst_TRUEP_Y",
    "K_Kst_TRUEP_Z",
    "e_minus_TRUEP_X",
    "e_minus_TRUEP_Y",
    "e_minus_TRUEP_Z",
    "e_plus_TRUEP_X",
    "e_plus_TRUEP_Y",
    "e_plus_TRUEP_Z",
    "B_plus_TRUEP_X",
    "B_plus_TRUEP_Y",
    "B_plus_TRUEP_Z",
    "e_plus_ID",
    "e_minus_ID",
    "K_Kst_ID",
    "e_plus_TRUEID",
    "e_minus_TRUEID",
    "K_Kst_TRUEID",
]


targets = [
    "B_plus_ENDVERTEX_CHI2",
    "B_plus_IPCHI2_OWNPV",
    "B_plus_FDCHI2_OWNPV",
    "B_plus_DIRA_OWNPV",
    "K_Kst_IPCHI2_OWNPV",
    "K_Kst_TRACK_CHI2NDOF",
    "e_minus_IPCHI2_OWNPV",
    "e_minus_TRACK_CHI2NDOF",
    "e_plus_IPCHI2_OWNPV",
    "e_plus_TRACK_CHI2NDOF",
]

print("A")
file = uproot.open(
    "/eos/lhcb/wg/RD/RK-highq2/data/tuples/2018/Kee/MC/truthed/Kee_2018_truthed.root"
)["DecayTree"]
branches = list(conditions + targets + conditions_TRUTH)
events = file.arrays(branches, library="pd")  # [0]
# events.to_pickle('Kee_2018_truthed.pickle')
events.to_csv("Kee_2018_truthed.csv")

print("B")
file = uproot.open(
    "/eos/lhcb/wg/RD/RK-highq2/data/tuples/2018/Kee/MC/truthed/Kstee_2018_truthed.root"
)["DecayTree"]
branches = list(conditions + targets + conditions_TRUTH)
events = file.arrays(branches, library="pd")  # [0]
# # events.to_pickle('Kstee_2018_truthed.pickle')
events.to_csv("Kstee_2018_truthed.csv")

# file = uproot.open("/eos/lhcb/wg/RD/RK-highq2/data/tuples/2018/Kee/data/presel/B2Kee_2018_CommonPresel.root")['DecayTree']
# branches = list(conditions+targets+['B_plus_M'])
# events = file.arrays(branches, library="pd")#[0]
# # # events.to_pickle('Kstee_2018_truthed.pickle'
# events = events.query('B_plus_M>5600')
# events.to_csv('B2Kee_2018_CommonPresel.csv')
