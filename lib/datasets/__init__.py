from .cifar10 import ClCIFAR10,ImgRecCIFAR10,get_cifar10_dataset
from .mnist import get_mnist_dataset
from .celeba import get_celeba_dataset
from .imagenet import get_imagenet_dataset
from .pascal_voc import get_voc_dataset
from .cbsd68 import get_cbsd68_dataset
from .sun2009 import get_sun2009_dataset
from .yiheng import get_eccv2020_dataset
from .rebel2021 import get_rebel2021_dataset
from .rots import get_rots_dataset
from .kitti import get_kitti_dataset,get_burst_kitti_dataset,get_burst_with_flow_kitti_dataset

def load_dataset(cfg,cfg_type):
    return get_dataset(cfg,cfg_type)

def get_dataset(cfg,cfg_type):

    # added for backward compatibility 09-14-20
    ds_dict = cfg
    exempt_types = ["denoising","simcl","simcl_cls","cls_3c","dynamic","single_denoising","dynamic-lmdb","default","rebel2021","dynamic-lmdb-burst","dynamic-lmdb-all","kitti"]
    if not (cfg_type in exempt_types): ds_dict = cfg[cfg_type]

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
    elif ds_dict.dataset.name.lower() == "sun2009":
        return get_sun2009_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "eccv2020":
        return get_eccv2020_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "rebel2021":
        return get_rebel2021_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "rots":
        return get_rots_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "kitti":
        return get_kitti_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "burst_kitti":
        return get_burst_kitti_dataset(cfg,cfg_type)
    elif ds_dict.dataset.name.lower() == "burst_with_flow_kitti":
        return get_burst_with_flow_kitti_dataset(cfg,cfg_type)
    else:
        raise ValueError(f"Uknown dataset name {ds_dict.dataset.name}")

