
# -- python --
from pathlib import Path
from easydict import EasyDict as edict

# -- python plotting --
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

# -- project imports --
from pyutils import create_list_pairs

# -- local imports --
from pyplots.misc import filter_sim_fields_range
from pyplots.contours import plot_contour_pairs
from pyplots.contours import stratify_contour_plots
from pyplots.lines import stratify_line_plots,plot_single_sim_group
#from pyplots.contours import plot_sim_test_pairs

def plot_experiment(records,egrids):
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

    print("plot exp.")
    records = records.astype({'methods':'string'})
    print(records['methods'])
    sims = records[records['methods'].isin(['ave','est'])]
    fig,ax = plt.subplots(figsize=(8,4))
    for method,mgroup in sims.groupby('methods'):
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

