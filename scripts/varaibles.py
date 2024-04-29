# ,K_Kst_PX,K_Kst_PY,K_Kst_PZ,e_minus_PX,e_minus_PY,e_minus_PZ,e_plus_PX,e_plus_PY,e_plus_PZ,nTracks,nSPDHits,B_plus_ENDVERTEX_CHI2,B_plus_IPCHI2_OWNPV,B_plus_FDCHI2_OWNPV,B_plus_DIRA_OWNPV,K_Kst_IPCHI2_OWNPV,K_Kst_TRACK_CHI2NDOF,e_minus_IPCHI2_OWNPV,e_minus_TRACK_CHI2NDOF,e_plus_IPCHI2_OWNPV,e_plus_TRACK_CHI2NDOF,K_Kst_TRUEP_X,K_Kst_TRUEP_Y,K_Kst_TRUEP_Z,e_minus_TRUEP_X,e_minus_TRUEP_Y,e_minus_TRUEP_Z,e_plus_TRUEP_X,e_plus_TRUEP_Y,e_plus_TRUEP_Z,B_plus_TRUEP_X,B_plus_TRUEP_Y,B_plus_TRUEP_Z
from fast_vertex_quality.tools.config import read_definition, rd

import fast_vertex_quality.tools.data_loader as data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def compute_mass(df, i, j, mass_i, mass_j):

    PE = np.sqrt(
        mass_i**2
        + df[f"{i}_TRUEP_X"] ** 2
        + df[f"{i}_TRUEP_Y"] ** 2
        + df[f"{i}_TRUEP_Z"] ** 2
    ) + np.sqrt(
        mass_j**2
        + df[f"{j}_TRUEP_X"] ** 2
        + df[f"{j}_TRUEP_Y"] ** 2
        + df[f"{j}_TRUEP_Z"] ** 2
    )
    PX = df[f"{i}_TRUEP_X"] + df[f"{j}_TRUEP_X"]
    PY = df[f"{i}_TRUEP_Y"] + df[f"{j}_TRUEP_Y"]
    PZ = df[f"{i}_TRUEP_Z"] + df[f"{j}_TRUEP_Z"]

    mass = np.sqrt((PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6)

    return mass, (PE**2 - PX**2 - PY**2 - PZ**2)


def compute_B_mom(df):

    PX = df[f"B_plus_TRUEP_X"]
    PY = df[f"B_plus_TRUEP_Y"]
    PZ = df[f"B_plus_TRUEP_Z"]

    return np.sqrt((PX**2 + PY**2 + PZ**2)) * 1e-3, np.sqrt((PX**2 + PY**2)) * 1e-3


def compute_miss_mom(df, particles):

    PX = df[f"B_plus_TRUEP_X"]
    PY = df[f"B_plus_TRUEP_Y"]
    PZ = df[f"B_plus_TRUEP_Z"]

    for particle in particles:
        PX += -1.0 * df[f"{particle}_TRUEP_X"]
        PY += -1.0 * df[f"{particle}_TRUEP_Y"]
        PZ += -1.0 * df[f"{particle}_TRUEP_Z"]

    return np.sqrt((PX**2 + PY**2 + PZ**2)) * 1e-3, np.sqrt((PX**2 + PY**2)) * 1e-3


def compute_delta_mom(df, particle):

    PX = df[f"{particle}_PX"] - df[f"{particle}_TRUEP_X"]
    PY = df[f"{particle}_PY"] - df[f"{particle}_TRUEP_Y"]
    PZ = df[f"{particle}_PZ"] - df[f"{particle}_TRUEP_Z"]

    return np.sqrt((PX**2 + PY**2 + PZ**2)) * 1e-3, np.sqrt((PX**2 + PY**2)) * 1e-3


masses = {}
masses["K_Kst"] = 493.677
masses["e_plus"] = 0.51099895000 * 1e-3
masses["e_minus"] = 0.51099895000 * 1e-3

for channel in ["Kee", "Kstee"]:

    print(channel)

    # events = pd.read_csv(f"datasets/{channel}_2018_truthed_head.csv")
    events = pd.read_csv(f"datasets/{channel}_2018_truthed.csv")

    print(events)

    particles = ["K_Kst", "e_plus", "e_minus"]

    for particle_i in range(0, len(particles)):
        for particle_j in range(particle_i + 1, len(particles)):
            print(particle_i, particle_j)

            (
                events[f"m_{particle_i}{particle_j}"],
                events[f"m_{particle_i}{particle_j}_inside"],
            ) = compute_mass(
                events,
                particles[particle_i],
                particles[particle_j],
                masses[particles[particle_i]],
                masses[particles[particle_j]],
            )

    events[f"B_P"], events[f"B_PT"] = compute_B_mom(events)
    events[f"missing_B_P"], events[f"missing_B_PT"] = compute_miss_mom(
        events, particles
    )

    for particle_i in range(0, len(particles)):
        (
            events[f"delta_{particle_i}_P"],
            events[f"delta_{particle_i}_PT"],
        ) = compute_delta_mom(events, particles[particle_i])

    for m in ["m_01", "m_02", "m_12"]:
        events[m] = events[m].fillna(0)
        # print(np.where(np.isnan(events[m])))

    plt.subplot(2, 4, 1)
    plt.hist2d(
        events[f"m_01"],
        events[f"m_02"],
        bins=40,
        norm=LogNorm(),
    )
    plt.subplot(2, 4, 2)
    plt.hist2d(
        events[f"m_01"],
        events[f"m_12"],
        bins=40,
        norm=LogNorm(),
    )

    plt.subplot(2, 4, 3)
    plt.hist2d(
        events[f"m_02"],
        events[f"m_12"],
        bins=40,
        norm=LogNorm(),
    )

    plt.subplot(2, 4, 4)
    plt.hist([events[f"B_P"], events[f"B_PT"]], bins=40, histtype="step")

    plt.subplot(2, 4, 5)
    plt.hist(
        [events[f"missing_B_P"], events[f"missing_B_PT"]], bins=40, histtype="step"
    )

    for particle_i in range(0, len(particles)):
        plt.subplot(2, 4, 5 + particle_i + 1)
        plt.hist2d(
            events[f"delta_{particle_i}_P"],
            events[f"delta_{particle_i}_PT"],
            bins=40,
            norm=LogNorm(),
        )

    plt.savefig(f"wip_{channel}.png")
    plt.close("all")

    events.to_csv(f"datasets/{channel}_2018_truthed_more_vars.csv")
