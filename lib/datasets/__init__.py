from .cifar10 import ClCIFAR10,ImgRecCIFAR10,get_cifar10_dataset
from .mnist import get_mnist_dataset
from .celeba import get_celeba_dataset
from .imagenet import get_imagenet_dataset
from .pascal_voc import get_voc_dataset
from .cbsd68 import get_cbsd68_dataset
from .transform import TransformsSimCLR,LowLight,BlockGaussian


def load_dataset(cfg,cfg_type):
    return get_dataset(cfg,cfg_type)

def get_dataset(cfg,cfg_type):

    # added for backward compatibility 09-14-20
    ds_dict = cfg
    if cfg_type != "denoising" and cfg_type != "simcl" and cfg_type != "simcl_cls" and cfg_type != 'cls_3c' and cfg_type != "dynamic" and cfg_type != "single_denoising": 
        ds_dict = cfg[cfg_type]

    if ds_dict.dataset.name.lower() == "mnist":
        return get_mnist_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "cifar10":
        return get_cifar10_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "celeba":
        return get_celeba_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "imagenet":
        return get_imagenet_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "voc":
        return get_voc_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "cbsd68":
        return get_cbsd68_dataset(cfg,cfg_type)
    else:
        raise ValueError(f"Uknown dataset name {ds_dict.dataset.name}")

