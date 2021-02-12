
# -- python imports --
import uuid
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange

# -- project imports --
from pyutils.plot import add_legend

def plot_historgram_batch(batch,index,prefix,rand_name=True):
    # -- init --
    uuid_str = str( uuid.uuid4() )
    B = batch.shape[0]

    # -- directory --
    path = Path(prefix) / Path(f"./histograms/{index}")
    if rand_name: path = path / Path(uuid_str)
    else: path = path / Path("default")
    if not path.exists(): path.mkdir(parents=True)

    # -- iterate each element --
    for batch_index in range(B):

        # -- filename --
        batch_path = path / Path( "burst_{}.png".format(batch_index) )

        # -- plot for each burst --
        burst = batch[batch_index]
        plot_historgram_burst(burst,batch_path)
                                                               
def plot_historgram_burst(burst,filename):

    # -- init --
    burst = rearrange(burst,'n c h w -> n (c h w)')
    N,D = burst.shape
    labels = [str(i) for i in range(N)]
    xgrid = np.arange(D)
    fig,ax = plt.subplots()
    
    # -- frame index --
    for frame_index in range(N):
        ax.plot(xgrid,frame,label=f"{frame_index}")
    ax = add_legend(ax,"Frame Index",labels)

    # -- save filename --
    plt.savefig(filename,dpi=300,bbox_inches='tight')
    
