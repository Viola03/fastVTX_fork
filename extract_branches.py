import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm


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

extras = [
    "e_minus_PX",
    "e_minus_PY",
    "e_minus_PZ",
    "e_plus_PX",
    "e_plus_PY",
    "e_plus_PZ",
    "K_Kst_PX",
    "K_Kst_PY",
    "K_Kst_PZ",
    "B_plus_PX",
    "B_plus_PY",
    "B_plus_PZ",
    "B_plus_OWNPV_X",
    "B_plus_OWNPV_Y",
    "B_plus_OWNPV_Z",
    "B_plus_ENDVERTEX_X",
    "B_plus_ENDVERTEX_Y",
    "B_plus_ENDVERTEX_Z",
    "B_plus_IPCHI2_OWNPV",
    "K_Kst_IPCHI2_OWNPV",
    "e_plus_IPCHI2_OWNPV",
    "e_minus_IPCHI2_OWNPV",
    "B_plus_FDCHI2_OWNPV",
    "B_plus_DIRA_OWNPV",
    "K_Kst_TRACK_CHI2NDOF",
    "e_plus_TRACK_CHI2NDOF",
    "e_minus_TRACK_CHI2NDOF",
]

extras_truth = [
    "B_plus_TRUEP_X",
    "B_plus_TRUEP_Y",
    "B_plus_TRUEP_Z",
    "e_minus_TRUEP_X",
    "e_minus_TRUEP_Y",
    "e_minus_TRUEP_Z",
    "e_plus_TRUEP_X",
    "e_plus_TRUEP_Y",
    "e_plus_TRUEP_Z",
    "K_Kst_TRUEP_X",
    "K_Kst_TRUEP_Y",
    "K_Kst_TRUEP_Z",
]


# print("A")
# file = uproot.open(
#     "/eos/lhcb/wg/RD/RK-highq2/data/tuples/2018/Kee/MC/truthed/Kee_2018_truthed.root"
# )["DecayTree"]
# branches = list(np.unique(conditions + targets + conditions_TRUTH + extras + extras_truth))
# events = file.arrays(branches, library="pd")  # [0]
# # events.to_pickle('Kee_2018_truthed.pickle')
# events.to_csv("Kee_2018_truthed.csv")

# print("B")
# file = uproot.open(
#     "/eos/lhcb/wg/RD/RK-highq2/data/tuples/2018/Kee/MC/truthed/Kstee_2018_truthed.root"
# )["DecayTree"]
# branches = list(np.unique(conditions + targets + conditions_TRUTH + extras + extras_truth))
# events = file.arrays(branches, library="pd")  # [0]
# # # events.to_pickle('Kstee_2018_truthed.pickle')
# events.to_csv("Kstee_2018_truthed.csv")

# print("C")
# file = uproot.open(
#     "/eos/lhcb/wg/RD/RK-highq2/data/tuples/2018/Kee/data/presel/B2Kee_2018_CommonPresel.root"
# )["DecayTree"]
# branches = list(np.unique(conditions + targets + ["B_plus_M"] + extras))
# events = file.arrays(branches, library="pd")  # [0]
# # # events.to_pickle('Kstee_2018_truthed.pickle'
# events = events.query("B_plus_M>5600")
# events.to_csv("B2Kee_2018_CommonPresel.csv")


print("D")


branches = [
    "B_plus_TRUEP_X",
    "B_plus_TRUEP_Y",
    "B_plus_TRUEP_Z",
    "e_minus_TRUEP_X",
    "e_minus_TRUEP_Y",
    "e_minus_TRUEP_Z",
    "e_plus_TRUEP_X",
    "e_plus_TRUEP_Y",
    "e_plus_TRUEP_Z",
    "K_Kst_TRUEP_X",
    "K_Kst_TRUEP_Y",
    "K_Kst_TRUEP_Z",
    "K_Kst_TRUEID",
    "e_plus_TRUEID",
    "e_minus_TRUEID",
]


file = uproot.open(
    "/eos/lhcb/wg/RD/RK-highq2/data/tuples/2018/Kee/MC/truthed/Bu2JPsiX_2018_KisK.root"
)["DecayTree"]
branches = list(
    np.unique(
        branches + conditions + targets + conditions_TRUTH + extras + extras_truth
    )
)
# events = file.arrays(branches, library="pd", entry_stop=25000)
events = file.arrays(branches, library="pd")
events = events.query("K_Kst_TRUEID == [-321, 321]")
events = events.query("e_plus_TRUEID == [-11, 11]")
events = events.query("e_minus_TRUEID == [-11, 11]")

print(events)

masses = {}
masses["K_Kst"] = 493.677
masses["e_plus"] = 0.51099895000 * 1e-3
masses["e_minus"] = 0.51099895000 * 1e-3


def compute_Bmass(df, masses):

    PE = (
        np.sqrt(
            masses["K_Kst"] ** 2
            + df[f"K_Kst_TRUEP_X"] ** 2
            + df[f"K_Kst_TRUEP_Y"] ** 2
            + df[f"K_Kst_TRUEP_Z"] ** 2
        )
        + np.sqrt(
            masses["e_plus"] ** 2
            + df[f"e_plus_TRUEP_X"] ** 2
            + df[f"e_plus_TRUEP_Y"] ** 2
            + df[f"e_plus_TRUEP_Z"] ** 2
        )
        + np.sqrt(
            masses["e_minus"] ** 2
            + df[f"e_minus_TRUEP_X"] ** 2
            + df[f"e_minus_TRUEP_Y"] ** 2
            + df[f"e_minus_TRUEP_Z"] ** 2
        )
    )
    PX = df[f"K_Kst_TRUEP_X"] + df[f"e_plus_TRUEP_X"] + df[f"e_minus_TRUEP_X"]
    PY = df[f"K_Kst_TRUEP_Y"] + df[f"e_plus_TRUEP_Y"] + df[f"e_minus_TRUEP_Y"]
    PZ = df[f"K_Kst_TRUEP_Z"] + df[f"e_plus_TRUEP_Z"] + df[f"e_minus_TRUEP_Z"]

    mass = np.sqrt((PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6)

    return mass


def compute_X_momenta(df):

    PX = (
        df["B_plus_TRUEP_X"]
        - df[f"K_Kst_TRUEP_X"]
        - df[f"e_plus_TRUEP_X"]
        - df[f"e_minus_TRUEP_X"]
    )
    PY = (
        df["B_plus_TRUEP_Y"]
        - df[f"K_Kst_TRUEP_Y"]
        - df[f"e_plus_TRUEP_Y"]
        - df[f"e_minus_TRUEP_Y"]
    )
    PZ = (
        df["B_plus_TRUEP_Z"]
        - df[f"K_Kst_TRUEP_Z"]
        - df[f"e_plus_TRUEP_Z"]
        - df[f"e_minus_TRUEP_Z"]
    )

    return PX, PY, PZ


def compute_lost_mass(df, masses):

    df[f"true_B_mass"] = 5.27934 * np.ones(np.shape(df[f"K_Kst_TRUEP_X"]))

    PE_1 = np.sqrt(
        masses["K_Kst"] ** 2
        + df[f"K_Kst_TRUEP_X"] ** 2
        + df[f"K_Kst_TRUEP_Y"] ** 2
        + df[f"K_Kst_TRUEP_Z"] ** 2
    )
    PE_2 = np.sqrt(
        masses["e_plus"] ** 2
        + df[f"e_plus_TRUEP_X"] ** 2
        + df[f"e_plus_TRUEP_Y"] ** 2
        + df[f"e_plus_TRUEP_Z"] ** 2
    )
    PE_3 = np.sqrt(
        masses["e_minus"] ** 2
        + df[f"e_minus_TRUEP_X"] ** 2
        + df[f"e_minus_TRUEP_Y"] ** 2
        + df[f"e_minus_TRUEP_Z"] ** 2
    )
    # PE_4 = lost_mass**2 + df[f"pX_x"] ** 2 + df[f"pX_y"] ** 2 + df[f"pX_z"] ** 2

    PX = (
        df[f"K_Kst_TRUEP_X"]
        + df[f"e_plus_TRUEP_X"]
        + df[f"e_minus_TRUEP_X"]
        + df["pX_x"]
    )
    PY = (
        df[f"K_Kst_TRUEP_Y"]
        + df[f"e_plus_TRUEP_Y"]
        + df[f"e_minus_TRUEP_Y"]
        + df["pX_y"]
    )
    PZ = (
        df[f"K_Kst_TRUEP_Z"]
        + df[f"e_plus_TRUEP_Z"]
        + df[f"e_minus_TRUEP_Z"]
        + df["pX_z"]
    )

    lost_mass = np.sqrt(
        (
            np.sqrt((df[f"true_B_mass"] ** 2 / 1e-6) + PX**2 + PY**2 + PZ**2)
            - PE_1
            - PE_2
            - PE_3
        )
        ** 2
        - df[f"pX_x"] ** 2
        - df[f"pX_y"] ** 2
        - df[f"pX_z"] ** 2
    )

    return lost_mass


events["mB"] = compute_Bmass(events, masses)

events["pX_x"], events["pX_y"], events["pX_z"] = compute_X_momenta(events)

events["lost_mass"] = compute_lost_mass(events, masses)

events = events.dropna()

events.to_csv("JPSIX_2018_truthed.csv")


# print(mB)

# plt.hist(mB, bins=50)
# plt.savefig("mB.png")

plt.axvline(x=139, c="k", alpha=0.25)
plt.axvline(x=497, c="k", alpha=0.25)
plt.axvline(x=938, c="k", alpha=0.25)
plt.hist(np.asarray(events["lost_mass"]), bins=100)
plt.yscale("log")
plt.savefig("mB_lost.png")
plt.close("all")

# with PdfPages(f"targets.pdf") as pdf:

#     for target in targets:

#         print(target)

#         plt.scatter(
#             np.asarray(events["lost_mass"]),
#             np.asarray(events[target]),
#             # bins=75,
#             # norm=LogNorm(),
#         )
#         plt.xlabel("lost_mass")
#         plt.ylabel(target)
#         pdf.savefig(bbox_inches="tight")
#         plt.close()
