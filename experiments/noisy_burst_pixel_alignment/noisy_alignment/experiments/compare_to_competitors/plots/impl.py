
# -- python --
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict

# -- python plotting --
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

# -- project imports --
from pyutils import create_list_pairs

# -- project imports --
from pyplots.misc import filter_sim_fields_range
from pyplots.contours import plot_contour_pairs
from pyplots.contours import stratify_contour_plots
from pyplots.lines import stratify_line_plots,plot_single_sim_group
from pyplots.legend import add_legend
#from pyplots.contours import plot_sim_test_pairs

# -- local imports --
from .example_images import plot_example_images
from .quality_v_noise import create_quality_v_noise_plot,create_ideal_v_noise_plot
from .quality_v_runtime import create_quality_v_runtime_plot
from .runtime_hist import runtime_hist

def plot_experiment(records,egrids,exp_cfgs):
    """

    records: pandas Dataframe with fields
    "nnf_acc","of_acc","method","noise_level","patchsize","image_id",...

    """
    # for iindex,mgroup in records.groupby('image_index'):
    #     print(iindex)
    #     print(mgroup)


    # for ntype,mgroup in records.groupby('noise_type'):
    #     print(ntype)
    #     print(mgroup)

    # title = "all"
    # fname = "all"
    # plot_results(records,egrids,title,fname)

    # plot_example_images(records,exp_cfgs)
    fname = "all"
    print("plot exp.")
    records = records.astype({'methods':'string'})
    records = records.astype({'nframes':'int'})

    # -- quality v noise --
    # create_quality_v_noise_plot(records,egrids,exp_cfgs)
    # create_ideal_v_noise_plot(records,egrids,exp_cfgs)
    # frecords = records[records['image_xform'].isin(['none'])]
    print(records[records['methods'].isin(['ave'])])
    print(records[records['methods'].isin(['nnf'])])
    print(records['image_xform'])
    
    dsname = records['dataset'].to_numpy()[0]
    # records = records[records['patchsize'].isin([11])]
    # runtime_hist(records,egrids,exp_cfgs,dsname)

    frecords = records[records['image_xform'].isin(['none'])]
    for std,std_group in frecords.groupby('std'):
        print(std)
        # if not(std == 100.): continue
        for dsname,ds_group in std_group.groupby('dataset'):
            create_quality_v_runtime_plot(ds_group,egrids,exp_cfgs,dsname)

    # -- plot accuracy of methods  --
    import numpy as np
    skips = ['ave','of','nnf_local','nnf']
    for nframes,fgroup in records.groupby('nframes'):
        handles,labels = [],[]
        fig,ax = plt.subplots(figsize=(8,4))
        for method,mgroup in fgroup.groupby('methods'):
            if method in skips: continue
            mdata = []
            for std,std_group in mgroup.groupby('std'):
                psnrs = []
                for mpsnrs in std_group['psnrs']:
                    psnrs.extend(list(mpsnrs))
                psnrs = np.nan_to_num(psnrs,neginf=0)
                ave_psnr = np.mean(psnrs)
                mdata.append([std,ave_psnr])
            mdata = np.array(mdata)
            x,y = mdata[:,0],mdata[:,1]
            print(method,x,y)
            # -- order by y --
            order = np.argsort(x)
            y = y[order]
            x = x[order]
            smethod = method.replace("_","-")
            handle = ax.plot(x,y,'x-',label=smethod)
            labels.append(smethod)
            handles.append(handle)
        title = "Method"
        add_legend(ax,title,labels,None,shrink = True,
                   fontsize=15,framealpha=1.0,ncol=1,shrink_perc=.80)
        save_dir = Path("./")
        fn =  save_dir / f"./psnr_v_noise_{nframes}.png"
        plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
        print(f"Saved plot to [{fn}]")
        plt.close('all')

    print(records)
    print(records['methods'])
    sims = records[records['methods'].isin(['ave'])]
    sims = sims[sims['patchsize'].isin([7])]
    sims = sims[sims['nframes'].isin([10])]
    fig,ax = plt.subplots(figsize=(8,4))
    skips = ['ave']
    for method,mgroup in sims.groupby('methods'):
        if method in skips: continue
        tmethod = method.replace("_"," ")
        title = f"Method All"#[{tmethod}]"
        fname = f"method_all"#{method}"
        print(mgroup)
        plot_results(ax,mgroup,egrids,title,fname)
    psnr_info = edict({'group':'psnrs','title':'PSNRS'})
    save_dir = Path("./")
    if not save_dir.exists(): save_dir.mkdir()
    fn =  save_dir / f"./{fname}_group-{psnr_info.group}.png"
    plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")

def plot_results(input_ax,sims,egrids,title,fname):

    # -- 2d plots --
    plot_2d = True
    if plot_2d:
        psnr_info = edict({'group':'psnrs','title':'PSNRS'})
        # fields = ["patchsize","std","nframes"]
        fields = ["std"]
        for field in fields:
            plot_single_sim_group(input_ax,sims,egrids,title,fname,field,psnr_info,
                                  logx=True,scatter=False)
            # plot_single_sim_group(sims,egrids,title,fname,field,mu2_info,
            #                       logx=True,scatter=True)

        return
        # -- stratify line across ub --
        fields,sfield = ["patchsize","pmis","std","T"],["ub"]
        stratify_line_plots(None,fields,sfield,sims,egrids,title,fname)
    
        # -- stratify line across frames --
        fields,sfield = ["patchsize","pmis","std","ub"],["T"]
        stratify_line_plots(None,fields,sfield,sims,egrids,title,fname)

    # -- create pairs for contour plots --
    fields = ["patchsize","std","nframes"]
    pairs = create_list_pairs(fields)
    
    # -- setup zaxis vars --
    xform = None
    title = f"Approx. Probability of Correct Alignment [{fname.capitalize()}]"
    label = "Approx. Prob."
    zinfo = edict({'mean':'est_mean',
                   'std':'est_std',
                   'title':title,
                   'label':label,
                   'xform':xform})
    title = f"Approx. Mean of Subset-MSE"
    label = "Approx. Mean"
    z_mu2 = edict({'mean':'est_mu2_mean',
                       'std':'est_mu2_std',
                       'title':title,
                       'label':label,
                       'xform':xform})

    # -- create contour plots --
    ax,mlevels,legend = None,None,True
    for pair in pairs:
        f1,f2 = pair
        cs = plot_contour_pairs(ax,sims,egrids,mlevels,title,fname,f1,f2,legend,zinfo)
        cs = plot_contour_pairs(ax,sims,egrids,mlevels,title,fname,f1,f2,legend,z_mu2)

    # -- stratify contours across T --
    fields,sfield = ["patchsize","pmis","std","ub"],["T"]
    stratify_contour_plots(None,fields,sfield,sims,egrids,mlevels,title,fname,zinfo)

    # -- stratify contours across ub --
    fields,sfield = ["patchsize","pmis","std","T"],["ub"]
    stratify_contour_plots(None,fields,sfield,sims,egrids,mlevels,title,fname,zinfo)

    # -- stratify contours across ub & D --
    fields,sfield = ["std","T"],["ub","patchsize"]
    stratify_contour_plots(None,fields,sfield,sims,egrids,mlevels,title,fname,zinfo)

    # -- stratify contours across ub & T --
    fields,sfield = ["std","patchsize"],["ub","T"]
    stratify_contour_plots(None,fields,sfield,sims,egrids,mlevels,title,fname,zinfo)

