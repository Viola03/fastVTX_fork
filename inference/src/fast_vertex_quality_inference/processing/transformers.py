import numpy as np
import uproot
import uproot3
import matplotlib.pyplot as plt
import pandas as pd


def transform_df(data, transformers):

    branches = list(data.keys())

    for branch in branches:	
        convert_units = False
        for P in ["P","PT","PX","PY","PZ"]:
            if f"_{P}_" in branch or branch[-(len(P)+1):] == f"_{P}":
                convert_units = True
        if convert_units:
            data[branch] = transformers[branch].process(np.asarray(data[branch])*1000.)
        else:
            data[branch] = transformers[branch].process(np.asarray(data[branch]))

    return data

def untransform_df(data, transformers):

    branches = list(data.keys())

    for branch in branches:	
        convert_units = False
        for P in ["P","PT","PX","PY","PZ"]:
            if f"_{P}_" in branch or branch[-(len(P)+1):] == f"_{P}":
                convert_units = True
        if convert_units:
            data[branch] = transformers[branch].unprocess(np.asarray(data[branch]))/1000.
        else:
            data[branch] = transformers[branch].unprocess(np.asarray(data[branch]))

    return data