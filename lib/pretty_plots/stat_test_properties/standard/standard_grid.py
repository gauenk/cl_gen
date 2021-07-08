
# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- project imports --
from pyutils import create_named_meshgrid,np_log
from pretty_plots.stat_test_properties.misc import skip_with_endpoints

def create_standard_parameter_grid():
    pgrid = edict()

    start,end,size,base = 1,4,15,10
    D_exp = np.linspace(start,end,size)
    D = np.power([base],D_exp).astype(np.long) # np.linspace(3,128,10)**2
    D_tickmarks = np.linspace(start,end,end-start+1)
    D_tickmarks = skip_with_endpoints(D_tickmarks,1)
    D_ticks = np.power([base],D_tickmarks)
    D_tickmarks_str = ["%d" % x for x in D_tickmarks]

    start,end,size,base = -1,2,5,10
    mu2_exp = np.linspace(start,end,size)
    mu2 = np.power([base],mu2_exp) # [0,1e-6,1e-3,1e-1,1,1e1]
    mu2_tickmarks = np.linspace(start,end,end - start + 1)
    mu2_tickmarks = skip_with_endpoints(mu2_tickmarks,3)
    mu2_ticks = np.power([base],mu2_tickmarks)
    mu2_tickmarks_str = ["%d" % x for x in mu2_tickmarks]

    # start,end,size,base = -5,2,50,10
    start,end,size,base = 0,100,21,10
    #std_exp = np.linspace(start,end,size).astype(np.int)
    #std = np.power([base],std_exp) # [1e-6,1e-3,1e-1,1,1e1]
    std_exp = np.linspace(start,end,end - start + 1)
    std_exp = skip_with_endpoints(std_exp,len(std_exp)//size)
    std = std_exp
    std_tickmarks = np.linspace(start,end,end - start + 1)
    std_tickmarks = skip_with_endpoints(std_tickmarks,50)
    #std_ticks = np.power([base],std_tickmarks)
    std_ticks = std_tickmarks
    std_tickmarks_str = ["%d" % x for x in std_tickmarks]

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

    size = [5000]

    params = [D,mu2,std,nframes,size]
    names = ['D','mu2','std','T','size']
    pgrid = create_named_meshgrid(params,names)
    print(f"Mesh grid created of size [{len(pgrid)}]")

    ticks = edict({'D':D_ticks,'mu2':mu2_ticks,
                   'std':std_ticks,'T':nframes_ticks})
    tickmarks = edict({'D':D_tickmarks,'mu2':mu2_tickmarks,
                       'std':std_tickmarks,'T':nframes_tickmarks})
    tickmarks_str = edict({'D':D_tickmarks_str,'mu2':mu2_tickmarks_str,
                           'std':std_tickmarks_str,
                           'T':nframes_tickmarks_str})
    logs = {'D':True,'mu2':True,'std':False,'T':False}
    # log_tickmarks = edict({'D':np_log(D_ticks),
    #                        'mu2':np_log(mu2_ticks),
    #                        'std':np_log(std_ticks)})
    lgrid = edict({'ticks':ticks,'tickmarks':tickmarks,
                   'tickmarks_str':tickmarks_str,'logs':logs})
                   #'log_tickmarks':log_tickmarks})

    return pgrid,lgrid


