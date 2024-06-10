from fast_vertex_quality.tools.config import read_definition, rd

import fast_vertex_quality.tools.data_loader as data_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import vector



def compute_DIRA(df, mother, particles, true_vars=True):

    PX = 0
    PY = 0
    PZ = 0

    if true_vars:
        for particle in particles:
            PX += df[f"{particle}_TRUEP_X"]
            PY += df[f"{particle}_TRUEP_Y"]
            PZ += df[f"{particle}_TRUEP_Z"]
    else:
        for particle in particles:
            PX += df[f"{particle}_PX"]
            PY += df[f"{particle}_PY"]
            PZ += df[f"{particle}_PZ"]

    A = norm(np.asarray([
        PX,
        PY,
        PZ]))

    B = norm(
        np.asarray(
            [
                df[f"{mother}_ENDVERTEX_X"] - df[f"{mother}_OWNPV_X"],
                df[f"{mother}_ENDVERTEX_Y"] - df[f"{mother}_OWNPV_Y"],
                df[f"{mother}_ENDVERTEX_Z"] - df[f"{mother}_OWNPV_Z"],
            ]
        )
    )
    dira = dot(A, B) / np.sqrt(mag(A) ** 2 * mag(B) ** 2)

    return dira



def compute_flightDistance(df, mother, particles, true_vars=True):

    PX = 0
    PY = 0
    PZ = 0

    if true_vars:
        for particle in particles:
            PX += df[f"{particle}_TRUEP_X"]
            PY += df[f"{particle}_TRUEP_Y"]
            PZ += df[f"{particle}_TRUEP_Z"]
    else:
        for particle in particles:
            PX += df[f"{particle}_PX"]
            PY += df[f"{particle}_PY"]
            PZ += df[f"{particle}_PZ"]
        
    momenta = vector.obj(
        px=PX,
        py=PY,
        pz=PZ,
    )

    end_vertex = np.asarray(
        [df[f"{mother}_ENDVERTEX_X"], df[f"{mother}_ENDVERTEX_Y"], df[f"{mother}_ENDVERTEX_Z"]]
    )
    primary_vertex = np.asarray(
        [df[f"{mother}_OWNPV_X"], df[f"{mother}_OWNPV_Y"], df[f"{mother}_OWNPV_Z"]]
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

def compute_angle(df, mother, particle, true_vars=True):

    if true_vars:
        momenta_B = np.asarray(
            [df[f"{mother}_TRUEP_X"], df[f"{mother}_TRUEP_Y"], df[f"{mother}_TRUEP_Z"]]
        )
        momenta_i = np.asarray(
            [
                df[f"{particle}_TRUEP_X"],
                df[f"{particle}_TRUEP_Y"],
                df[f"{particle}_TRUEP_Z"],
            ]
        )
    else:
        momenta_B = np.asarray([df[f"{mother}_PX"], df[f"{mother}_PY"], df[f"{mother}_PZ"]])
        momenta_i = np.asarray(
            [
                df[f"{particle}_PX"],
                df[f"{particle}_PY"],
                df[f"{particle}_PZ"],
            ]
        )

    dot_prod = np.arccos(dot(momenta_B, momenta_i) / (mag(momenta_B) * mag(momenta_i)))

    dot_prod[np.where(np.isnan(dot_prod))] = 1e-6
    dot_prod[np.where(dot_prod == 0)] = 1e-6

    return dot_prod


def compute_impactParameter_i(df, mother, particle, true_vars=True):

    if true_vars:
        momenta = vector.obj(
            px=df[f"{particle}_TRUEP_X"],
            py=df[f"{particle}_TRUEP_Y"],
            pz=df[f"{particle}_TRUEP_Z"],
        )
    else:
        momenta = vector.obj(
            px=df[f"{particle}_PX"],
            py=df[f"{particle}_PY"],
            pz=df[f"{particle}_PZ"],
        )
    end_vertex = np.asarray(
        [df[f"{mother}_ENDVERTEX_X"], df[f"{mother}_ENDVERTEX_Y"], df[f"{mother}_ENDVERTEX_Z"]]
    )
    primary_vertex = np.asarray(
        [df[f"{mother}_OWNPV_X"], df[f"{mother}_OWNPV_Y"], df[f"{mother}_OWNPV_Z"]]
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



def compute_intermediate_distance(df, intermediate, mother):

    X = df[f'{intermediate}_TRUEENDVERTEX_X']-df[f'{mother}_TRUEENDVERTEX_X']
    Y = df[f'{intermediate}_TRUEENDVERTEX_Y']-df[f'{mother}_TRUEENDVERTEX_Y']
    Z = df[f'{intermediate}_TRUEENDVERTEX_Z']-df[f'{mother}_TRUEENDVERTEX_Z']
    dist = np.sqrt(X**2 + Y**2 + Z**2)
    return dist

def compute_impactParameter(df, mother, particles, true_vars=True):

    PX = 0
    PY = 0
    PZ = 0

    if true_vars:
        for particle in particles:
            PX += df[f"{particle}_TRUEP_X"]
            PY += df[f"{particle}_TRUEP_Y"]
            PZ += df[f"{particle}_TRUEP_Z"]
    else:
        for particle in particles:
            PX += df[f"{particle}_PX"]
            PY += df[f"{particle}_PY"]
            PZ += df[f"{particle}_PZ"]
        
    momenta = vector.obj(
        px=PX,
        py=PY,
        pz=PZ,
    )

    end_vertex = np.asarray(
        [df[f"{mother}_ENDVERTEX_X"], df[f"{mother}_ENDVERTEX_Y"], df[f"{mother}_ENDVERTEX_Z"]]
    )
    primary_vertex = np.asarray(
        [df[f"{mother}_OWNPV_X"], df[f"{mother}_OWNPV_Y"], df[f"{mother}_OWNPV_Z"]]
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






# def compute_delta_mom(df, particle):
def compute_reconstructed_momentum_residual(df, particle):

    PX = df[f"{particle}_PX"] - df[f"{particle}_TRUEP_X"]
    PY = df[f"{particle}_PY"] - df[f"{particle}_TRUEP_Y"]
    PZ = df[f"{particle}_PZ"] - df[f"{particle}_TRUEP_Z"]

    return np.sqrt((PX**2 + PY**2 + PZ**2)) * 1e-3, np.sqrt((PX**2 + PY**2)) * 1e-3


# def compute_miss_mom(df, mother, particles, true_vars=True):
def compute_missing_momentum(df, mother, particles, true_vars=True):

    if true_vars:
        PX = df[f"{mother}_TRUEP_X"]
        PY = df[f"{mother}_TRUEP_Y"]
        PZ = df[f"{mother}_TRUEP_Z"]

        for particle in particles:
            PX += -1.0 * df[f"{particle}_TRUEP_X"]
            PY += -1.0 * df[f"{particle}_TRUEP_Y"]
            PZ += -1.0 * df[f"{particle}_TRUEP_Z"]
    else:
        PX = df[f"{mother}_PX"]
        PY = df[f"{mother}_PY"]
        PZ = df[f"{mother}_PZ"]

        for particle in particles:
            PX += -1.0 * df[f"{particle}_PX"]
            PY += -1.0 * df[f"{particle}_PY"]
            PZ += -1.0 * df[f"{particle}_PZ"]

    return np.sqrt((PX**2 + PY**2 + PZ**2)) * 1e-3, np.sqrt((PX**2 + PY**2)) #* 1e-3

def compute_reconstructed_intermediate_momenta(df, particles, true_vars=True):

    PX = 0
    PY = 0
    PZ = 0

    if true_vars:
        for particle in particles:
            PX += df[f"{particle}_TRUEP_X"]
            PY += df[f"{particle}_TRUEP_Y"]
            PZ += df[f"{particle}_TRUEP_Z"]
    else:
        for particle in particles:
            PX += df[f"{particle}_PX"]
            PY += df[f"{particle}_PY"]
            PZ += df[f"{particle}_PZ"]

    return np.sqrt((PX**2 + PY**2 + PZ**2)) * 1e-3, np.sqrt((PX**2 + PY**2)) #* 1e-3

# def compute_B_mom(df, mother, true_vars=True):
def compute_reconstructed_mother_momenta(df, mother, true_vars=True):

    if true_vars:
        PX = df[f"{mother}_TRUEP_X"]
        PY = df[f"{mother}_TRUEP_Y"]
        PZ = df[f"{mother}_TRUEP_Z"]
    else:
        PX = df[f"{mother}_PX"]
        PY = df[f"{mother}_PY"]
        PZ = df[f"{mother}_PZ"]

    return np.sqrt((PX**2 + PY**2 + PZ**2)) * 1e-3, np.sqrt((PX**2 + PY**2)) #* 1e-3


def compute_mass_3(df, i, j, k, mass_i, mass_j, mass_k, true_vars=True):

    if true_vars:
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
        ) + np.sqrt(
            mass_k**2
            + df[f"{k}_TRUEP_X"] ** 2
            + df[f"{k}_TRUEP_Y"] ** 2
            + df[f"{k}_TRUEP_Z"] ** 2
        )
        PX = df[f"{i}_TRUEP_X"] + df[f"{j}_TRUEP_X"] + df[f"{k}_TRUEP_X"]
        PY = df[f"{i}_TRUEP_Y"] + df[f"{j}_TRUEP_Y"] + df[f"{k}_TRUEP_Y"]
        PZ = df[f"{i}_TRUEP_Z"] + df[f"{j}_TRUEP_Z"] + df[f"{k}_TRUEP_Z"]
    else:
        PE = np.sqrt(
            mass_i**2 + df[f"{i}_PX"] ** 2 + df[f"{i}_PY"] ** 2 + df[f"{i}_PZ"] ** 2
        ) + np.sqrt(
            mass_j**2 + df[f"{j}_PX"] ** 2 + df[f"{j}_PY"] ** 2 + df[f"{j}_PZ"] ** 2
        ) + np.sqrt(
            mass_k**2 + df[f"{k}_PX"] ** 2 + df[f"{k}_PY"] ** 2 + df[f"{k}_PZ"] ** 2
        )
        PX = df[f"{i}_PX"] + df[f"{j}_PX"] + df[f"{k}_PX"]
        PY = df[f"{i}_PY"] + df[f"{j}_PY"] + df[f"{k}_PY"]
        PZ = df[f"{i}_PZ"] + df[f"{j}_PZ"] + df[f"{k}_PZ"]

    mass = np.sqrt((PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6)

    return mass



# def compute_mass(df, i, j, mass_i, mass_j, true_vars=True):
def compute_mass(df, i, j, mass_i, mass_j, true_vars=True):

    if true_vars:
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
    else:
        PE = np.sqrt(
            mass_i**2 + df[f"{i}_PX"] ** 2 + df[f"{i}_PY"] ** 2 + df[f"{i}_PZ"] ** 2
        ) + np.sqrt(
            mass_j**2 + df[f"{j}_PX"] ** 2 + df[f"{j}_PY"] ** 2 + df[f"{j}_PZ"] ** 2
        )
        PX = df[f"{i}_PX"] + df[f"{j}_PX"]
        PY = df[f"{i}_PY"] + df[f"{j}_PY"]
        PZ = df[f"{i}_PZ"] + df[f"{j}_PZ"]

    mass = np.sqrt((PE**2 - PX**2 - PY**2 - PZ**2) * 1e-6)

    return mass, (PE**2 - PX**2 - PY**2 - PZ**2) * 1e3