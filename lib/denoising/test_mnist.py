"""
Verify the mnist data is working for the denoising experiment

"""

# python code
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# project code
import settings
from datasets import load_dataset
from denoising.config import load_cfg,save_cfg,get_cfg,get_args

def run_test():
    args = get_args()
    cfg = get_cfg(args)
    cfg.use_ddp = False
    data,loader = load_dataset(cfg,'denoising')
    pic_set,raw_pic = next(iter(loader.tr))
    fig,ax = plt.subplots(cfg.N,2,figsize=(8,8))
    for i,pic in enumerate(pic_set):
        title = str(i)
        plot_th_tensor(ax,i,0,pic,title)
        title = "raw"
        plot_th_tensor(ax,i,1,raw_pic,title)

    root = Path(settings.ROOT_PATH)
    save_dir = root / Path("test/denoising/")
    if not save_dir.exists(): save_dir.mkdir(parents=True)
    path = save_dir / Path("loading_mnist_noise_g.png")

    plt.savefig(path)


def plot_th_tensor(ax,i,j,pic,title):
    pic = pic.to('cpu').detach().numpy()[0,0]
    pic += np.abs(np.min(pic))
    pic = pic / pic.max()
    ax[i,j].imshow(pic,  cmap='Greys_r',  interpolation=None)
    ax[i,j].set_xticks([])
    ax[i,j].set_yticks([])
    ax[i,j].set_title(title)

if __name__ == "__main__":
    print("what is this?")
