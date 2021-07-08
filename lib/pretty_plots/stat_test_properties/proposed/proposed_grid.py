
# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- project imports --
from pyutils import create_named_meshgrid,np_log

# -- local imports --
from pretty_plots.stat_test_properties.misc import skip_with_endpoints

def create_proposed_parameter_grid():
    pgrid = edict()

    start,end,size,base = 1,4,10,10
    # [10000,1778,316,56,10]
    D_exp = np.linspace(start,end,size)
    D = np.power([base],D_exp).astype(np.long) # np.linspace(3,128,10)**2
    D_tickmarks = np.linspace(start,end,end-start+1)
    D_tickmarks = skip_with_endpoints(D_tickmarks,1)
    D_ticks = np.power([base],D_tickmarks)
    D_tickmarks_str = ["%d" % x for x in D_tickmarks]
    D = D[::-1]

    start,end,size,base = 0,100,21,10
    std_exp = np.linspace(start,end,size).astype(np.int)
    std = std_exp
    std_tickmarks = np.linspace(start,end,end - start + 1)
    std_tickmarks = skip_with_endpoints(std_tickmarks,50)
    std_ticks = std_tickmarks
    std_tickmarks_str = ["%d" % x for x in std_tickmarks]

    start,end,size,base = -3,1,4,10
    ub_exp = np.linspace(start,end,size)
    ub = np.power([base],ub_exp) # [0,1e-6,1e-3,1e-1,1,1e1]
    ub_tickmarks = np.linspace(start,end,end - start + 1)
    ub_tickmarks = skip_with_endpoints(ub_tickmarks,3)
    ub_ticks = np.power([base],ub_tickmarks)
    ub_tickmarks_str = ["%d" % x for x in ub_tickmarks]

    start,end = 35,95
    int_spacing = (end - start + 1)
    num = 3 # int_spacing
    pmis = np.linspace(start,end,num) # to percent
    pmis_tickmarks = np.linspace(start,end,num)
    pmis_ticks = pmis_tickmarks
    pmis_tickmarks_str = ["%d" % x for x in pmis_tickmarks]

    start,end = 3,50
    int_spacing = (end - start + 1)
    num = 5
    nframes = [3,10,50]
    #nframes = np.linspace(start,end,num).astype(np.long)
    # nframes = [3,18,34,50]
    #nframes_tickmarks = np.linspace(start,end,num)
    nframes_tickmarks = nframes
    nframes_ticks = np_log(nframes_tickmarks)/ np_log([10])
    nframes_tickmarks_str = ["%d" % x for x in nframes_tickmarks]



    start,end = 1000,1200
    int_spacing = (end - start + 1)
    num = 1
    B = np.linspace(start,end,num).astype(np.long)
    B_tickmarks = np.linspace(start,end,num)
    B_ticks = B_tickmarks
    B_tickmarks_str = ["%d" % x for x in B_tickmarks]

    size = [50]

    params = [D,ub,std,pmis,nframes,B,size]
    names = ['D','ub','std','pmis','T','B','size']
    named_params = {key:val for key,val in zip(names,params)}
    print("About to create mesh.")
    for n,p in zip(names,params): print(f"# of [{n}] terms is [{len(p)}]: {p}")
    pgrid = create_named_meshgrid(params,names)
    print(f"Mesh grid created of size [{len(pgrid)}]")

    logs = {'D':True,'ub':True,'std':False,'pmis':False,'T':False,'B':False,'size':False}
    ticks = edict({'D':D_ticks,'ub':ub_ticks,'std':std_ticks,
                   'pmis':pmis_ticks,'T':nframes_ticks,'B':B_ticks})
    tickmarks = edict({'D':D_tickmarks,'ub':ub_tickmarks,
                       'std':std_tickmarks,'pmis':pmis_tickmarks,
                       'T':nframes_tickmarks,'B':B_tickmarks})
    tickmarks_str = edict({'D':D_tickmarks_str,'ub':ub_tickmarks_str,
                           'std':std_tickmarks_str,'pmis':pmis_tickmarks_str,
                           'T':nframes_tickmarks_str,'B':B_tickmarks_str})
    lgrid = edict({'ticks':ticks,'tickmarks':tickmarks,
                   'tickmarks_str':tickmarks_str,'logs':logs,
                   'grids':named_params})
                   #'log_tickmarks':log_tickmarks})

    return pgrid,lgrid

