# -- python imports --
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

# -- project imports --
from pyutils import add_legend

def add_jitter(ndarray,std=0.05):
    return np.random.normal(ndarray,scale=std)

def get_local_motion_fake_data():

    fake = edict()

    fake.raft = add_jitter([1., .8, .5, .2, .1, 0.05])
    fake.raft_std = add_jitter([0.05, 0.06, 0.08, 0.08, 0.08, 0.09],0.01)
    fake.lsrmtf = add_jitter([1., .8, .5, .2, .1, 0.05])
    fake.lsrmtf_std = add_jitter([0.05, 0.06, 0.08, 0.08, 0.08, 0.09],0.01)
    fake.raft3d = add_jitter([1., .8, .5, .2, .1, 0.05])
    fake.raft3d_std = add_jitter([0.05, 0.06, 0.08, 0.08, 0.08, 0.09],0.01)
    fake.drisf = add_jitter([1., .8, .5, .2, .1, 0.05])
    fake.drisf_std = add_jitter([0.05, 0.06, 0.08, 0.08, 0.08, 0.09],0.01)
    fake.asvof = add_jitter([1., .8, .5, .2, .1, 0.05])
    fake.asvof_std = add_jitter([0.05, 0.06, 0.08, 0.08, 0.08, 0.09],0.01)
    
    fake.nnf = add_jitter([.8, .75, .5, .25, .2, 0.15])
    fake.nnf_std = add_jitter([0.05, 0.06, .08, 0.08, 0.08, 0.09],0.01)
    fake.ldof = add_jitter([.8, .75, .5, .25, .2, 0.15])
    fake.ldof_std = add_jitter([0.05, 0.06, .08, 0.08, 0.08, 0.09],0.01)
    
    fake.sup = add_jitter([1., .98, .93, .90, .86, 0.84])
    fake.sup_std = add_jitter([0.05, 0.06, .055, 0.055, 0.06, 0.07],0.01)
    fake.n2n = add_jitter([1., .98, .93, .90, .86, 0.84])
    fake.n2n_std = add_jitter([0.05, .05, .05, 0.05, 0.05, 0.06],0.01)

    fake.ours = add_jitter([.8, .75, .65, .55, .50, 0.45])
    fake.ours_std = add_jitter([0.05, 0.06, .055, 0.055, 0.06, 0.07],0.01)
    fake.ours_seg = add_jitter([.8, .75, .65, .55, .50, 0.45])
    fake.ours_seg_std = add_jitter([0.05, .05, .05, 0.05, 0.05, 0.06],0.01)

    fake.noise_label = ['none','g10','g25','g50','g75','g100']
    fake.noise_level = [0., 10., 25., 50., 75., 100.]

    fake = pd.DataFrame(fake)

    for field in fake.keys():
        if "_std" in field: fake[field] *= 15

    methods = OrderedDict({'raft':{'cat':'nn','line':'-'},
               'lsrmtf':{'cat':'nn','line':'--'},
               'raft3d':{'cat':'nn','line':'-.'},
               'drisf':{'cat':'nn','line':':'},
               #'asvof':{'cat':'nn','line':' '},
               'nnf':{'cat':'classic','line':'-'},
               'ldof':{'cat':'classic','line':'--'},
               'n2n':{'cat':'ref','line':'-'},
               'sup':{'cat':'ref','line':'--'},
               'ours':{'cat':'ours','line':'-'},
               'ours_seg':{'cat':'ours','line':'--'},
               })
    methods = methods

    colors = {'nn':'y','classic':'g','ours':'b','ref':'k'}

    return fake,methods,colors

def get_global_motion_fake_data():
    fake = edict()

    fake.raft = add_jitter([1., .95, .93, .88, .82, 0.75],0.01)
    fake.raft_std = add_jitter([0.05, 0.05, 0.05, 0.05, 0.07, 0.08],0.005)
    fake.lsrmtf = add_jitter([1., .95, .93, .88, .82, 0.75],0.01)
    fake.lsrmtf_std = add_jitter([0.05, 0.05, 0.05, 0.05, 0.07, 0.08],0.005)
    fake.raft3d = add_jitter([1., .95, .93, .88, .82, 0.75],0.01)
    fake.raft3d_std = add_jitter([0.05, 0.05, 0.05, 0.05, 0.07, 0.08],0.005)
    fake.drisf = add_jitter([1., .95, .93, .88, .82, 0.75],0.01)
    fake.drisf_std = add_jitter([0.05, 0.05, 0.05, 0.05, 0.07, 0.08],0.005)
    fake.asvof = add_jitter([1., .95, .93, .88, .82, 0.75],0.01)
    fake.asvof_std = add_jitter([0.05, 0.05, 0.05, 0.05, 0.07, 0.08],0.005)
    
    fake.nnf = add_jitter([1., .98, .93, .90, .86, 0.84],0.01)
    fake.nnf_std = add_jitter([0.02, 0.02, 0.03, 0.04, 0.05, 0.05],0.005)
    fake.ldof = add_jitter([1., .98, .93, .90, .86, 0.84],0.01)
    fake.ldof_std = add_jitter([0.02, 0.02, 0.03, 0.04, 0.05, 0.05],0.005)
    
    fake.sup = add_jitter([1., .98, .93, .90, .86, 0.84],0.01)
    fake.sup_std = add_jitter([0.02, 0.02, 0.02, 0.03, 0.03, 0.03],0.005)
    fake.n2n = add_jitter([1., .98, .93, .90, .86, 0.84],0.01)
    fake.n2n_std = add_jitter([0.02, 0.02, 0.02, 0.03, 0.03, 0.03],0.005)

    fake.ours = add_jitter([1., .98, .93, .90, .86, 0.84],0.01)
    fake.ours_std = add_jitter([0.02, 0.02, 0.03, 0.03, 0.04, 0.04],0.005)
    fake.ours_seg = add_jitter([1., .98, .93, .90, .86, 0.84],0.01)
    fake.ours_seg_std = add_jitter([0.02, 0.02, 0.02, 0.03, 0.03, 0.03],0.005)

    fake.noise_label = ['none','g10','g25','g50','g75','g100']
    fake.noise_level = [0., 10., 25., 50., 75., 100.]

    for field in fake.keys():
        if "_std" in field: fake[field] *= 10
        
    fake = pd.DataFrame(fake)

    methods = OrderedDict({'raft':{'cat':'nn','line':'-'},
               'lsrmtf':{'cat':'nn','line':'--'},
               'raft3d':{'cat':'nn','line':'-.'},
               'drisf':{'cat':'nn','line':':'},
               #'asvof':{'cat':'nn','line':' '},
               'nnf':{'cat':'classic','line':'-'},
               'ldof':{'cat':'classic','line':'--'},
               'n2n':{'cat':'ref','line':'-'},
               'sup':{'cat':'ref','line':'--'},
               'ours':{'cat':'ours','line':'-'},
               'ours_seg':{'cat':'ours','line':'--'},
               })
    methods = methods

    colors = {'nn':'y','classic':'g','ours':'b','ref':'k'}

    return fake,methods,colors

def run():
    plot_local_motion()
    plot_global_motion()

def plot_global_motion():
    fake,methods,colors = get_global_motion_fake_data()
    title = "Simple Global Motion"
    fname = "gm"
    plot_motion(fake,methods,colors,title,fname)

def plot_local_motion():
    fake,methods,colors = get_local_motion_fake_data()
    title = "Rigid Body Local Motion"
    fname = "lm"
    plot_motion(fake,methods,colors,title,fname)

def plot_motion(fake,methods,colors,title,fname):
    method_names = list(methods.keys())

    fig,ax = plt.subplots(figsize=(8,4))
    cols = fake.columns.to_list()
    noise_level = fake['noise_level'].to_list()
    for method in methods:
        cat = methods[method]['cat']
        line = methods[method]['line']
        color = colors[cat]
        samples = 36.*np.clip(fake[method].to_numpy(),0.,1.)
        error = fake[method+"_std"].to_numpy()
        print(color,line)
        handle = ax.errorbar(noise_level,samples,yerr=error,color=color,ls=line,lw=2)
    noise_label = fake['noise_label'].to_list() 
    ax.set_xticks(noise_level)
    ax.set_xticklabels(noise_label,fontsize=15,rotation=45,ha="right")
    ax.set_xlabel("Noise Level",fontsize=15)
    ax.set_ylabel("Denoiser Quality (PSNR)",fontsize=15)
    ax.set_title(title,fontsize=18)
    ax = add_legend(ax,'Alignment Methods',methods,shrink_perc=.9,framealpha=0.0)

    DIR = Path("./output/pretty_plots")
    if not DIR.exists(): DIR.mkdir()
    fn =  DIR / f"./comparing_denoiser_quality_{fname}.png"
    plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")
