
import sys
sys.path.append("/home/gauenk/Documents/experiments/cl_gen/lib")

import cv2

import numpy as np
import numpy.random as npr

import pywt
import pywt.data

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from pyutils import images_to_psnrs


def checkout_haar():
    cfg = None
    std = 50.
    alpha = 20.
    # filter_name = "bior1.3"
    filter_name = "haar"

    image = pywt.data.camera()
    noisy = npr.normal(image,scale=std)
    noisy = npr.poisson(alpha*image/255.) / alpha * 255.

    haar_image = pywt.dwt2(np.copy(noisy), filter_name)[0]
    sratio = haar_image.shape[-1] / image.shape[-1]

    image_shrink = cv2.resize(image, None, fx = sratio, fy = sratio, 
                              interpolation = cv2.INTER_NEAREST)
    noisy_shrink = cv2.resize(noisy, None, fx = sratio, fy = sratio,
                              interpolation = cv2.INTER_NEAREST)
    
    # haar_image -= haar_image.min()
    haar_image /= 2.

    print(image.min(),image.max(),image.mean())
    print(haar_image.min(),haar_image.max(),haar_image.mean())

    # -- plot result --
    fig,axes = plt.subplots(1,3,figsize=(3*4,4))
    images = [haar_image,noisy_shrink,image_shrink]
    nmlz_const = [haar_image.max(),255.,255.]
    titles = ["Haar Image","Noisy","Original Image"]
    shift_x,shift_y = 1,1

    for idx,ax in enumerate(axes):
        nmlz_img = images[idx]/255.#nmlz_const[idx]
        ref_img = image_shrink/255.

        # diff = nmlz_img[shift_y:,shift_x:] - ref_img[:-shift_y,:-shift_x]
        # diff = nmlz_img[:-shift_y,:-shift_x:] - ref_img[shift_y:,shift_x:]
        diff = nmlz_img# - ref_img

        ax_image = 255.*diff
        ax_title = titles[idx]
        ax.imshow(ax_image)
        ax.set_title(ax_title,fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.savefig("./tmp.png",dpi=300)

    print(haar_image)
    haar_image = haar_image / 255.
    noisy_shrink = noisy_shrink / 255.
    image_shrink = image_shrink / 255.

    haar_psnrs = images_to_psnrs(haar_image,image_shrink)
    noisy_psnrs = images_to_psnrs(noisy_shrink,image_shrink)
    print(haar_psnrs,noisy_psnrs,sratio,haar_image.shape,image.shape)


if __name__ == "__main__":
    checkout_haar()
