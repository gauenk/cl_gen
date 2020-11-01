# 2d scatter plot
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def add_legend(ax,legend_title,legend_str,legend_handles=None,shrink = True,fontsize=15,framealpha=1.0,ncol=1,shrink_perc=.90):
    box = ax.get_position()
    if shrink:
        ax.set_position([box.x0, box.y0, box.width * shrink_perc, box.height])
    if legend_handles is not None:
        ax.legend(legend_handles,legend_str,
                  title = legend_title,
                  title_fontsize=fontsize,
                  fontsize=fontsize,
                  ncol=ncol,
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  framealpha=framealpha)
    else:
        ax.legend(legend_str,
                  title = legend_title,
                  title_fontsize=fontsize,
                  fontsize=fontsize,
                  ncol=ncol,
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  framealpha=framealpha)
        
    return ax


def proj2d(features,labels,ax=None,method='TSNE'):
    if method == 'TSNE':
        xfer = TSNE(n_components=2)
    elif method == 'PCA':
        xfer = PCA(n_components=2)        
    else:
        raise ValueError(f"Uknown dim reduction method {method}")
    ftrs_2d = xfer.fit_transform(features)
    _labels = np.unique(labels)
    df = pd.DataFrame({
        'x_1': ftrs_2d[:,0],
        'x_2': ftrs_2d[:,1],
        'labels': labels,
    }) 
    fg = sns.FacetGrid(data=df, hue='labels', hue_order=_labels, aspect=1.61)
    if ax is None:
        sns_ax = fg.map(plt.scatter, 'x_1', 'x_2').add_legend()
    else:
        sns_ax = fg.map(ax.scatter, 'x_1', 'x_2').add_legend()
        # add_legend(ax,"Classes",_labels)
        plt.close(fg.fig)
    return sns_ax

    
