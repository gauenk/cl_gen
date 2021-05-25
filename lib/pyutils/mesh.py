import numpy as np
import pandas as pd
from collections import OrderedDict
from easydict import EasyDict as edict

def groupby_fields(named_fields,df):
    if len(named_fields) == 0:
        pairs = []
        tests = list(df.T.to_dict().values())
        for i,test_i in enumerate(tests):
            for j,test_j in enumerate(tests):
                if i >= j: continue
                pair = [edict(test_i),edict(test_j)]
                pairs.append(pair)
        return pairs
    else:
        fieldname = named_fields[0]
        if len(named_fields) > 1: new_fields = named_fields[1:]
        else: new_fields = []
        pairs = []
        for field,field_df in df.groupby(fieldname):
            field_fields = groupby_fields(new_fields,field_df)
            pairs.extend(field_fields)
        return pairs

def create_named_meshgrid(lists,names):
    named_mesh = []
    mesh = create_meshgrid(lists)
    for elem in mesh:
        named_elem = edict(OrderedDict(dict(zip(names,elem))))
        named_mesh.append(named_elem)
    return named_mesh

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
