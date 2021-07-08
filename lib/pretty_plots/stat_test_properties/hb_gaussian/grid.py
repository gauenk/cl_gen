
# -- python --
import numpy as np
from easydict import EasyDict as edict

# -- pytorch --

# -- project --
from pyutils import create_named_meshgrid,np_log
from pretty_plots.stat_test_properties.misc import skip_with_endpoints


def get_grid():
    
    # -- PATCHSIZE --
    start,end,size,base = 1,2,5,10
    D_exp = np.linspace(start,end,size)
    D = np.power([base],D_exp).astype(np.long) # np.linspace(3,128,10)**2
    D_tickmarks = np.linspace(start,end,end-start+1)
    D_tickmarks = skip_with_endpoints(D_tickmarks,1)
    D_ticks = np.power([base],D_tickmarks)
    D_tickmarks_str = ["%d" % x for x in D_tickmarks]
    D = D[::-1]

    # -- NUM OF FRAMES --
    start,end = 3,50
    int_spacing = (end - start + 1)
    num = 5
    nframes = [3,10,20,35,45,50]
    nframes_tickmarks = nframes
    nframes_ticks = np_log(nframes_tickmarks)/ np_log([10])
    nframes_tickmarks_str = ["%d" % x for x in nframes_tickmarks]

    # -- NOISE LEVEL --
    start,end,size,base = 0,100,21,10
    std_exp = np.linspace(start,end,size).astype(np.int)
    std = std_exp
    std_tickmarks = np.linspace(start,end,end - start + 1)
    std_tickmarks = skip_with_endpoints(std_tickmarks,50)
    std_ticks = std_tickmarks
    std_tickmarks_str = ["%d" % x for x in std_tickmarks]

    # -- ERROR VAR --
    start,end,size,base = 0,100,21,10
    eps_exp = np.linspace(start,end,size)
    #eps = np.power([base],eps_exp) # [0,1e-6,1e-3,1e-1,1,1e1]
    eps = eps_exp
    eps_tickmarks = np.linspace(start,end,end - start + 1)
    eps_tickmarks = skip_with_endpoints(eps_tickmarks,3)
    eps_ticks = np.power([base],eps_tickmarks)
    eps_tickmarks_str = ["%d" % x for x in eps_tickmarks]

    # -- NUM OF BOOTSTRAP REPS --
    start,end = 1000,1200
    int_spacing = (end - start + 1)
    num = 1
    B = np.linspace(start,end,num).astype(np.long)
    B_tickmarks = np.linspace(start,end,num)
    B_ticks = B_tickmarks
    B_tickmarks_str = ["%d" % x for x in B_tickmarks]

    # -- NUM OF REPEAT EXPERIMENT --
    size = [50]

    # -- CREATE GRID --
    params = [D,eps,std,nframes,B,size]
    names = ['D','eps','std','T','B','size']
    named_params = {key:val for key,val in zip(names,params)}
    print("About to create mesh.")
    for n,p in zip(names,params): print(f"# of [{n}] terms is [{len(p)}]: {p}")
    grid = create_named_meshgrid(params,names)
    print(f"Mesh grid created of size [{len(grid)}]")

    logs = {'D':True,'eps':False,'std':False,'T':False,'B':False,'size':False}
    ticks = edict({'D':D_ticks,'eps':eps_ticks,'std':std_ticks,
                   'T':nframes_ticks,'B':B_ticks})
    tickmarks = edict({'D':D_tickmarks,'eps':eps_tickmarks,
                       'std':std_tickmarks,
                       'T':nframes_tickmarks,'B':B_tickmarks})
    tickmarks_str = edict({'D':D_tickmarks_str,'eps':eps_tickmarks_str,
                           'std':std_tickmarks_str,
                           'T':nframes_tickmarks_str,'B':B_tickmarks_str})
    lgrid = edict({'ticks':ticks,'tickmarks':tickmarks,
                   'tickmarks_str':tickmarks_str,'logs':logs,
                   'grids':named_params})

    return grid,lgrid


    
