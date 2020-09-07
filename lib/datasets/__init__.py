from .cifar10 import ClCIFAR10,ImgRecCIFAR10,get_cifar10_dataset
from .mnist import get_mnist_dataset
from .celeba import get_celeba_dataset
from .imagenet import get_imagenet_dataset
from .transform import TransformsSimCLR,LowLight,BlockGaussian


def get_dataset(cfg,cfg_type):
    if cfg[cfg_type].dataset.name.lower() == "mnist":
        return get_mnist_dataset(cfg,cfg_type)
    elif cfg[cfg_type].dataset.name.lower() == "cifar10":
        return get_cifar10_dataset(cfg,cfg_type)
    elif cfg[cfg_type].dataset.name.lower() == "celeba":
        return get_celeba_dataset(cfg,cfg_type)
    elif cfg[cfg_type].dataset.name.lower() == "imagenet":
        return get_imagenet_dataset(cfg,cfg_type)
    else:
        raise ValueError(f"Uknown dataset name {cfg[cfg_type].dataset.name}")

