# -- python imports --
import pickle,lmdb
import numpy as np
from pathlib import Path
import numpy.random as npr
from easydict import EasyDict as edict
from einops import rearrange, repeat, reduce

# -- faiss imports --
import faiss_mod
import faiss
import faiss.contrib.torch_utils

# -- pytorch imports --
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as tv_utils
import torchvision.transforms.functional as tvF




def search_nlm_images_gpu(img, patches, num_select):
    [H, W, D, C] = patches.shape
    patches = patches.reshape([H * W, D * C]).astype(np.float32)
    patches = torch.from_numpy(patches).to("cuda:0")
    dist, ind_y = search_raw_array_pytorch(res, patches, patches, num_select, metric=faiss.METRIC_L2)

    images_sim = np.zeros([num_select, H * W, C]).astype(np.float32)

    for s in range(num_select):
        images_sim[s, :, :] = img.reshape([H * W, C])[ind_y[:, s].cpu().numpy(), :]

    images_sim = images_sim.reshape([num_select, H, W, C])
    return images_sim


def compute_sim_images(img, patch_size, num_select, img_ori=None):
    if img_ori is not None:
        patches = shift_concat_image(img_ori, patch_size)
    else:
        patches = shift_concat_image(img, patch_size)
    images_sim = search_nlm_images_gpu(img, patches, num_select+1)
    return images_sim[1::, ...]


