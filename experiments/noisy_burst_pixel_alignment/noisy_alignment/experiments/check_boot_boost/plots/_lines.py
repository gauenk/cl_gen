
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from ._settings import FONTSIZE,SAVE_PATH

def plot_stats(stats,field_x,field_y,plt_fmt,title,fname,input_ax=None):

    # -- get axis --
    if input_ax is None: fig,ax = plt.subplots(figsize=(8,4))
    else: ax = input_ax

    ax.errorbar(stats[field_x],stats.mean,yerr=stats.std/stats.n)
    ax.set_xticks(plt_fmt.ticks[field_x])
    ax.set_xticklabels(plt_fmt.tickmarks_str[field_x],fontsize=FONTSIZE)
    label_x = xfer_field_to_label(field_x)
    label_y = xfer_field_to_label(field_y)
    ax.set_xlabel(label_x,fontsize=FONTSIZE)
    ax.set_ylabel(label_y,fontsize=FONTSIZE)
    ax.set_title(title,fontsize=18)
    ax.minorticks_off()

    # -- save --
    if input_ax is None:
        if not SAVE_PATH.exists(): SAVE_PATH.mkdir()
        fn =  SAVE_PATH / f"./{fname}_x-{field_x}_y-{field_y}.png"
        plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
        plt.close('all')
        print(f"Wrote plot to [{fn}]")
    plt.xscale("linear")
    

