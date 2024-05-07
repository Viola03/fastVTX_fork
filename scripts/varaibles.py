# ,K_Kst_PX,K_Kst_PY,K_Kst_PZ,e_minus_PX,e_minus_PY,e_minus_PZ,e_plus_PX,e_plus_PY,e_plus_PZ,nTracks,nSPDHits,B_plus_ENDVERTEX_CHI2,B_plus_IPCHI2_OWNPV,B_plus_FDCHI2_OWNPV,B_plus_DIRA_OWNPV,K_Kst_IPCHI2_OWNPV,K_Kst_TRACK_CHI2NDOF,e_minus_IPCHI2_OWNPV,e_minus_TRACK_CHI2NDOF,e_plus_IPCHI2_OWNPV,e_plus_TRACK_CHI2NDOF,K_Kst_TRUEP_X,K_Kst_TRUEP_Y,K_Kst_TRUEP_Z,e_minus_TRUEP_X,e_minus_TRUEP_Y,e_minus_TRUEP_Z,e_plus_TRUEP_X,e_plus_TRUEP_Y,e_plus_TRUEP_Z,B_plus_TRUEP_X,B_plus_TRUEP_Y,B_plus_TRUEP_Z
from fast_vertex_quality.tools.config import read_definition, rd

import fast_vertex_quality.tools.data_loader as data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import vector


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


def ImpactParameter(point, tPos, tMom):
    t = 0
    if hasattr(tMom, "P"):
        P = tMom.P()
    else:
        P = tMom.Mag()
    for i in range(3):
        t += tMom(i) / P * (point(i) - tPos(i))
    dist = 0
    for i in range(3):
        dist += (point(i) - tPos(i) - t * tMom(i) / P) ** 2
    dist = ROOT.TMath.Sqrt(dist)
    return dist


def compute_impactParameter(df):

    momenta = vector.obj(
        px=df.K_Kst_TRUEP_X + df.e_minus_TRUEP_X + df.e_plus_TRUEP_X,
        py=df.K_Kst_TRUEP_Y + df.e_minus_TRUEP_Y + df.e_plus_TRUEP_Y,
        pz=df.K_Kst_TRUEP_Z + df.e_minus_TRUEP_Z + df.e_plus_TRUEP_Z,
    )
    end_vertex = np.asarray(
        [df.B_plus_ENDVERTEX_X, df.B_plus_ENDVERTEX_Y, df.B_plus_ENDVERTEX_Z]
    )
    primary_vertex = np.asarray(
        [df.B_plus_OWNPV_X, df.B_plus_OWNPV_Y, df.B_plus_OWNPV_Z]
    )
    # primary_vertex = np.asarray([np.zeros(np.shape(df.B_plus_OWNPV_X)), np.zeros(np.shape(df.B_plus_OWNPV_X)), np.zeros(np.shape(df.B_plus_OWNPV_X))])

    momenta_array = np.asarray([momenta.px, momenta.py, momenta.pz])
    P = momenta.mag
    t = 0
    for i in range(3):
        t += momenta_array[i] / P * (primary_vertex[i] - end_vertex[i])
    dist = 0
    for i in range(3):
        dist += (primary_vertex[i] - end_vertex[i] - t * momenta_array[i] / P) ** 2
    dist = np.sqrt(dist)

    return dist


def compute_impactParameter_i(df, particle):

    momenta = vector.obj(
        px=df[f"{particle}_TRUEP_X"],
        py=df[f"{particle}_TRUEP_Y"],
        pz=df[f"{particle}_TRUEP_Z"],
    )
    end_vertex = np.asarray(
        [df.B_plus_ENDVERTEX_X, df.B_plus_ENDVERTEX_Y, df.B_plus_ENDVERTEX_Z]
    )
    primary_vertex = np.asarray(
        [df.B_plus_OWNPV_X, df.B_plus_OWNPV_Y, df.B_plus_OWNPV_Z]
    )

    momenta_array = np.asarray([momenta.px, momenta.py, momenta.pz])
    P = momenta.mag
    t = 0
    for i in range(3):
        t += momenta_array[i] / P * (primary_vertex[i] - end_vertex[i])
    dist = 0
    for i in range(3):
        dist += (primary_vertex[i] - end_vertex[i] - t * momenta_array[i] / P) ** 2
    dist = np.sqrt(dist)

    return dist


def compute_angle(df, particle):
    momenta_B = np.asarray([df.B_plus_TRUEP_X, df.B_plus_TRUEP_Y, df.B_plus_TRUEP_Z])
    momenta_i = np.asarray(
        [
            df[f"{particle}_TRUEP_X"],
            df[f"{particle}_TRUEP_Y"],
            df[f"{particle}_TRUEP_Z"],
        ]
    )

    dot_prod = np.arccos(dot(momenta_B, momenta_i) / (mag(momenta_B) * mag(momenta_i)))

    dot_prod[np.where(np.isnan(dot_prod))] = 1e-6
    dot_prod[np.where(dot_prod == 0)] = 1e-6
    # print(np.where(np.isnan(dot_prod)))
    # print(np.where(np.isinf(dot_prod)))
    # print(np.amin(dot_prod), np.amax(dot_prod))
    return dot_prod


def compute_flightDistance(df):

    momenta = vector.obj(
        px=df.K_Kst_TRUEP_X + df.e_minus_TRUEP_X + df.e_plus_TRUEP_X,
        py=df.K_Kst_TRUEP_Y + df.e_minus_TRUEP_Y + df.e_plus_TRUEP_Y,
        pz=df.K_Kst_TRUEP_Z + df.e_minus_TRUEP_Z + df.e_plus_TRUEP_Z,
    )
    end_vertex = np.asarray(
        [df.B_plus_ENDVERTEX_X, df.B_plus_ENDVERTEX_Y, df.B_plus_ENDVERTEX_Z]
    )
    primary_vertex = np.asarray(
        [df.B_plus_OWNPV_X, df.B_plus_OWNPV_Y, df.B_plus_OWNPV_Z]
    )

    momenta_array = np.asarray([momenta.px, momenta.py, momenta.pz])
    P = momenta.mag
    t = 0
    for i in range(3):
        t += momenta_array[i] / P * (primary_vertex[i] - end_vertex[i])
    dist = 0
    for i in range(3):
        dist += (t * momenta_array[i] / P) ** 2
    dist = np.sqrt(dist)

    return dist


def compute_flightDistance2(df):

    end_vertex = np.asarray(
        [df.B_plus_ENDVERTEX_X, df.B_plus_ENDVERTEX_Y, df.B_plus_ENDVERTEX_Z]
    )
    primary_vertex = np.asarray(
        [df.B_plus_OWNPV_X, df.B_plus_OWNPV_Y, df.B_plus_OWNPV_Z]
    )
    dist = 0
    for i in range(3):
        dist += (end_vertex[i] - primary_vertex[i]) ** 2
    dist = np.sqrt(dist)

    return dist


def mag(vec):
    sum_sqs = 0
    for component in vec:
        sum_sqs += component**2
    mag = np.sqrt(sum_sqs)
    return mag


def norm(vec):
    mag_vec = mag(vec)
    for component_idx in range(np.shape(vec)[0]):
        vec[component_idx] *= 1.0 / mag_vec
    return vec


def dot(vec1, vec2):
    dot = 0
    for component_idx in range(np.shape(vec1)[0]):
        dot += vec1[component_idx] * vec2[component_idx]
    return dot


def compute_DIRA(df):

    A = norm(
        np.asarray(
            [
                df.K_Kst_TRUEP_X + df.e_minus_TRUEP_X + df.e_plus_TRUEP_X,
                df.K_Kst_TRUEP_Y + df.e_minus_TRUEP_Y + df.e_plus_TRUEP_Y,
                df.K_Kst_TRUEP_Z + df.e_minus_TRUEP_Z + df.e_plus_TRUEP_Z,
            ]
        )
    )

    # A = norm(np.asarray([
    #     df.K_Kst_PX+df.e_minus_PX+df.e_plus_PX,
    #     df.K_Kst_PY+df.e_minus_PY+df.e_plus_PY,
    #     df.K_Kst_PZ+df.e_minus_PZ+df.e_plus_PZ]))

    B = norm(
        np.asarray(
            [
                df.B_plus_ENDVERTEX_X - df.B_plus_OWNPV_X,
                df.B_plus_ENDVERTEX_Y - df.B_plus_OWNPV_Y,
                df.B_plus_ENDVERTEX_Z - df.B_plus_OWNPV_Z,
            ]
        )
    )
    dira = dot(A, B) / np.sqrt(mag(A) ** 2 * mag(B) ** 2)

    return dira


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

    ################################################################################

    for particle in ["K_Kst", "e_plus", "e_minus"]:
        events[f"angle_{particle}"] = compute_angle(events, f"{particle}")

    events["IP_B"] = compute_impactParameter(events)
    for particle in ["K_Kst", "e_plus", "e_minus"]:
        events[f"IP_{particle}"] = compute_impactParameter_i(events, f"{particle}")
    events["FD_B"] = compute_flightDistance(events)
    events["DIRA_B"] = compute_DIRA(events)

    events.to_csv(f"datasets/{channel}_2018_truthed_more_vars.csv")
