
# python imports
import sys,os
sys.path.append("./lib")
import numpy as np

# pytorch imports
import torch

# project imports
from layers.resnet import ResNetWithSkip,get_resnet
from layers.denoising import Decoder
from layers.denoising.decoder_simple import Decoder as DecoderSimple


def main():
    name = "resnet50"
    dataset = "cifar10"
    resnet = get_resnet(name, dataset, pretrained=False)
    enc = ResNetWithSkip(resnet)
    data = torch.rand((10,3,32,32))
    output = enc(data)
    print(output[0].shape)
    for x in output[1]:
        print(x.shape)
    decoder = Decoder()
    dec = decoder(output)
    print(dec.shape)

    # dec_simp = DecoderSimple()
    # dec_ex = torch.rand((10,768))
    # skips = [torch.rand((10,16,8,8))]
    # dec_simp([dec_ex,skips])
    


if __name__ == "__main__":
    print("HI")
    main()
