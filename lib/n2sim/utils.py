
# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

def crop_center_patch(image_list,M,FS):
    """
    M == Margin
    FS == Frame Size
    """
    cropped = []
    L = len(image_list)
    for l in range(L):
        crop = tvF.crop(image_list[l],M,M,FS,FS)
        cropped.append(crop)
    return cropped
    
