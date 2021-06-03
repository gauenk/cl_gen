"""

Create mesh to evaluate all search parameters

"""

# -- python imports --
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

def create_meshgrid(lists):
    # -- num lists --
    L = len(lists)

    # -- tokenize each list --
    codes,uniques = [],[]
    for l in lists:
        l_codes,l_uniques = pd.factorize(l)
        codes.append(l_codes)
        uniques.append(l_uniques)

    # -- meshgrid and flatten --
    lmesh = np.meshgrid(*codes)
    int_mesh = [grid.ravel() for grid in lmesh]

    # -- convert back to tokens --
    mesh = [uniques[i][int_mesh[i]] for i in range(L)]

    # -- "transpose" the axis to iter goes across original lists --
    mesh_T = []
    L,M = len(mesh),len(mesh[0])
    for m in range(M):
        mesh_m = []
        for l in range(L):
            elem = mesh[l][m]
            if isinstance(elem,np.int64):
                elem = int(elem)
            mesh_m.append(elem)
        mesh_T.append(mesh_m)

    return mesh_T


def apply_mesh_filters(mesh,filters):
    filtered_mesh = mesh
    for mfilter in filters:
        filtered_mesh = apply_mesh_filter(filtered_mesh,mfilter)
    return filtered_mesh

def apply_mesh_filter(mesh,mfilter):
    filtered_mesh = []
    fields_str = list(mfilter.keys())[0]
    values = mfilter[fields_str]
    field1,field2 = fields_str.split("-")
    for elem in mesh:
        match_any = False
        for val in values:
            eq1 = (elem[field1] == val[0])
            eq2 = (elem[field2] == val[1])
            if eq1 and eq2: match_any = True
        if match_any: filtered_mesh.append(elem)
    return filtered_mesh
