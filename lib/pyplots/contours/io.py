
def save_pair_contours(axes,mlevels,fname,cs,postfix):
    if axes[0] is None: return
    
    # -- add legend --
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
             for pc in cs.collections]
    mlevels_fmt = ["%0.2f" % x for x in mlevels]
    N,num = len(proxy),5
    skip = N//num
    if skip < 1: skip = 1
    skim_proxy = proxy[::skip]
    skim_fmt = mlevels_fmt[1::skip]
    if mlevels_fmt[-1] != skim_fmt[-1]:
        skim_proxy[-1] = proxy[-1]
        skim_fmt[-1] = mlevels_fmt[-1]
    add_legend(axes[-1],"Approx. Prob.",skim_fmt,
               skim_proxy,framealpha=0.,shrink_perc=1.0,
               fontsize=FONTSIZE)

    # -- create plots --
    plt.subplots_adjust(right=.85)
    title = "Contour Maps of the Approximate Probability of Correct Alignment"
    # make_space_above(axes, topmargin=0.7)
    # plt.suptitle(title,fontsize=18,y=1.0)

    DIR = Path("./output/pretty_plots")
    if not DIR.exists(): DIR.mkdir()
    fn =  DIR / f"./stat_test_properties_{fname}_contours-agg_{postfix}.png"
    plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    plt.close('all')
    print(f"Wrote plot to [{fn}]")
