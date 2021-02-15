
# -- python imports --
import uuid
import numpy as np
from pathlib import Path
from einops import rearrange
import matplotlib.pyplot as plt

# -- project imports --
from pyutils.plot import add_legend

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Create a Plot of Gradient Norm v Layer
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def plot_histogram_gradient_norms(model,global_step,prefix,rand_name=False):

    # -- directory --
    path = Path(prefix) / Path("histogram")
    if rand_name: path = path / Path(uuid_str)
    else: path = path / Path("default")
    if not path.exists(): path.mkdir(parents=True)

    # -- filename --
    grad_path = path / Path( "gradient_norms_{}.png".format(global_step) )

    # -- init matplotlibs --
    fig,ax = plt.subplots()

    # -- compute and plot norms --
    grad_norms,idx,eps = [],0,1e-15
    for name,params in model.named_parameters():
        grad_norm = np.log(params.grad.norm(2).item() + eps)
        grad_norms.append(grad_norm)

    # -- create a plot --
    xgrid = np.arange(len(grad_norms))
    ax.plot(xgrid,grad_norms,'kx-')
    ax.set_title(f"Plot of Gradient Norms: [{global_step}]")
    ax.set_xlabel("Layer Number")
    ax.set_ylabel("Gradient Norm Value")    

    # -- save and close --
    plt.savefig(grad_path,dpi=300)
    plt.close("all")

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Create a Histogram Plot of Gradient Values
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def plot_histogram_gradients(model,global_step,prefix,rand_name=False):
    
    # -- directory --
    path = Path(prefix) / Path("histogram")
    if rand_name: path = path / Path(uuid_str)
    else: path = path / Path("default")
    if not path.exists(): path.mkdir(parents=True)

    # -- aggregate gradient values --
    parameters = np.array([])
    for name,params in model.named_parameters():
        grad = params.grad.view(-1).cpu().numpy()
        parameters = np.r_[parameters,grad]
    
    # -- init matplotlibs --
    fig,ax = plt.subplots()

    # -- filename & range --
    grad_path = path / Path( "gradients_{}.png".format(global_step) )
    amin,amax = parameters.min(),parameters.max()

    # -- create a plot --
    ax.hist( parameters , range=(amin,amax), log=True, bins=20 )

    # -- save and close --
    ax.set_title(f"Histogram of Gradient Values: [{global_step}]")
    plt.savefig(grad_path,dpi=300)
    plt.close("all")

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Create a Histogram Plot of Residuals
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def plot_histogram_residuals_batch(batch,global_step,prefix,rand_name=True):
    # -- init --
    uuid_str = str( uuid.uuid4() )
    B = batch.shape[0]

    # -- directory --
    path = Path(prefix) / Path("histogram")
    if rand_name: path = path / Path(uuid_str)
    else: path = path / Path("default")
    if not path.exists(): path.mkdir(parents=True)

    # -- iterate each element --
    for batch_index in range(B):

        # -- filename --
        batch_path = path / Path( "residuals_{}_{}.png".format(global_step,batch_index) )

        # -- plot for each burst --
        burst = batch[batch_index]
        plot_histogram_residual_burst(burst,batch_path,global_step)
                                                               
def plot_histogram_residual_burst(burst,filename,global_step):

    # -- init --
    burst = rearrange(burst,'n c h w -> n (c h w)')
    N,D = burst.shape
    labels = [str(i) for i in range(N)]
    fig,ax = plt.subplots()
    
    # -- frame index --
    amin,amax = burst.min(),burst.max()
    for frame_index in range(N):
        frame = burst[frame_index]
        ax.hist(frame,label=f"{frame_index}",bins=30,range=(amin,amax),alpha=0.5,lw=1,edgecolor='k')
    ax = add_legend(ax,"Frame Index",labels)
    ax.set_title(f"Residuals at Iteration [{global_step}]")

    # -- save filename --
    plt.savefig(filename,dpi=300,bbox_inches='tight')
    
