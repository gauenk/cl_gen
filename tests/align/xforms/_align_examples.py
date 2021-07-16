
# -- python imports --
import torch
import numpy as np
from easydict import EasyDict as edict
from einops import repeat

def get_example(name):
    if name == "example_1":
        return example_1()
    elif name == "example_2":
        return example_2()
    elif name == "example_3":
        return example_3()
    elif name == "example_4":
        return example_4()
    else:
        raise ValueError(f"Uknown example name [{name}]")

def create_pix_ex1(flow,centers,npix):
    pix = []
    # pix = torch.zeros((1,npix,3,2))
    for p in range(npix):
        c = centers[p]
        pix_p = torch.LongTensor([[[
            [c-(1),c-(-1)],
            [c,c],
            [c+(0),c+(-1)]
        ]]]) # "-" before and "+" after the reference frame using cumulative sums
        # pix[p] = pix_p
        pix.append(p)
    pix = torch.cat(pix,dim=0).type(torch.long)
    return pix

def example_1():
    burst = torch.Tensor([
        [[[
            [0,1,0,1,2],            
            [1,0,1,0,2],
            [1,0,1,1,2],
            [1,1,1,0,2],
            [2,2,2,2,2]
        ]]],
        [[[
            [1,1,0,1,1],
            [1,0,1,0,1],            
            [0,1,0,1,0],
            [1,1,0,1,1],
            [0,1,1,1,0]
        ]]],
        [[[
            [2,2,2,2,2],
            [1,1,0,1,1],
            [1,0,1,0,1],            
            [0,1,0,1,0],
            [1,1,0,1,1],
        ]]]
    ])
    burst = burst.reshape((3,1,1,5,5))
    patchsize = 3
    isize = edict({'h':5,'w':5})
    npix = isize.h * isize.w

    flow = torch.LongTensor([[[
        [1,-1],
        [0,-1]
    ]]])
    flow = repeat(flow,'i 1 tm1 two -> i p tm1 two',p=npix)

    blocks = torch.LongTensor([[
        [8,4,1]
    ]])
    blocks = repeat(blocks,'i 1 t -> i p t',p=npix)

    centers = torch.LongTensor([
        np.c_[np.unravel_index(np.arange(npix),(isize.h,isize.w))]
    ])

    # -- create pix, yes it is gross but verbose (good for testing) --

    # a.) convert flow axis (y = 0 @ bottom) to object axis (y = 0 @ top)
    # b.) "-" before and "+" after the reference frame using cumulative sums
    w = 5
    pix = []
    for p in range(npix):
        c = centers[0,p]
        p_row = p // w
        p_col = p % w
        row,col = c[0],c[1]
        # print("p,row,col: ",p,row,col,p_row,p_col)
        pix_p = torch.LongTensor([[[
            [col-(1),row-(-(-1))],
            [col,row],
            [col+(0),row+(-(-1))]
        ]]])
        pix.append(pix_p)
    pix = torch.cat(pix,dim=1).type(torch.long)
    
    # print("pix")
    # print(pix)
    # print("-"*10)

    nblocks = 3
    patchsize = 3
    sample = edict({'burst':burst,'flow':flow,'blocks':blocks,
                    'pix':pix,'centers':centers,
                    'nblocks':nblocks,'isize':isize,
                    'patchsize':patchsize})
    return sample

def example_2():

    w = 5
    isize = edict({'h':5,'w':5})
    npix = isize.h * isize.w

    burst = torch.Tensor([
        [[[
            [2,2,2,2,2],
            [1,0,1,1,2],
            [0,1,0,1,2],            
            [1,0,1,0,2],
            [1,0,1,1,2],
        ]]],
        [[[
            [1,1,0,1,1],
            [1,0,1,0,1],            
            [0,1,0,1,0],
            [1,1,0,1,1],
            [0,1,1,1,0]
        ]]],
        [[[
            [2,2,2,2,2],
            [1,1,0,1,1],
            [1,0,1,0,1],            
            [0,1,0,1,0],
            [1,1,0,1,1],
        ]]]
    ])

    flow = torch.LongTensor([[[
        [1,1],[0,-1]
    ]]])
    flow = repeat(flow,'i 1 tm1 two -> i p tm1 two',p=npix)

    blocks = torch.LongTensor([[
        [2,4,1]
    ]])
    blocks = repeat(blocks,'i 1 t -> i p t',p=npix)

    centers = torch.LongTensor([
        np.c_[np.unravel_index(np.arange(npix),(isize.h,isize.w))]
    ])

    # -- create pix, yes it is gross but verbose (good for testing) --
    pix = []
    for p in range(npix):
        c = centers[0,p]
        p_row = p // w
        p_col = p % w
        row,col = c[0],c[1]
        # print("p,row,col: ",p,row,col,p_row,p_col)
        pix_p = torch.LongTensor([[[
            [col-(1),row-(-(1))],
            [col,row],
            [col+(0),row+(-(-1))]
        ]]])
        pix.append(pix_p)
    pix = torch.cat(pix,dim=1).type(torch.long)

    nblocks = 3
    patchsize = 3
    sample = edict({'burst':burst,'flow':flow,'blocks':blocks,
                    'pix':pix,'centers':centers,
                    'nblocks':nblocks,'isize':isize,
                    'patchsize':patchsize})
    return sample

def example_3():
    flow = torch.LongTensor([[[
        [1,-1],[0,-1],[1,0]
    ]]])
    flow = repeat(flow,'i 1 tm1 two -> i p tm1 two',p=npix)

    blocks = torch.LongTensor([[
        [23,17,12,11]
    ]])
    blocks = repeat(blocks,'i 1 t -> i p t',p=npix)

    centers = torch.LongTensor([
        np.c_[np.unravel_index(np.arange(npix),(isize.h,isize.w))]
    ])

    # -- create pix, yes it is gross but verbose (good for testing) --
    pix = []
    for p in range(npix):
        c = centers[0,p]
        x,y = c[0],c[1]
        pix_p = torch.LongTensor([[[
            [x-(0)-(1)  ,y-(-1)-(-1)],
            [x-(0)      ,y-(-1)],
            [x,y],
            [x+(1),y+(0)]
        ]]]) # "-" before and "+" after the reference frame using cumulative sums
        pix.append(pix_p)
    pix = torch.cat(pix,dim=0).type(torch.long)

    nblocks = 5
    isize = edict({'h':1,'w':1})
    sample = edict({'flow':flow,'blocks':blocks,
                    'pix':pix,'centers':centers,
                    'nblocks':nblocks,'isize':isize})
    return sample


def example_4():
    flow = torch.LongTensor([[[
        [1,0],[1,-1],[0,-1],[1,0],[-1,0],[1,0]
    ]]])
    flow = repeat(flow,'i 1 tm1 two -> i p tm1 two',p=npix)

    blocks = torch.LongTensor([[
        [24,23,17,12,11,12,11]
    ]])
    blocks = repeat(blocks,'i 1 t -> i p t',p=npix)

    centers = torch.LongTensor([
        np.c_[np.unravel_index(np.arange(npix),(isize.h,isize.w))]
    ])

    # -- create pix, yes it is gross but verbose (good for testing) --
    pix = []
    for p in range(npix):
        c = centers[0,p]
        x,y = c[0],c[1]
        pix_p = torch.LongTensor([[[
            [x-(0)-(1)-(1), y-(-1)-(-1)-(0)],
            [x-(0)-(1),     y-(-1)-(-1)],
            [x-(0),         y-(-1)],
            [x,y],
            [x+(1),          y+(0)],
            [x+(1)+(-1),     y+(0)+(0)],
            [x+(1)+(-1)+(1), y+(0)+(0)+(0)]
        ]]]) # "-" before and "+" after the reference frame using cumulative sums
        pix.append(pix_p)
    pix = torch.cat(pix,dim=0).type(torch.long)


    nblocks = 5
    isize = edict({'h':1,'w':1})
    sample = edict({'flow':flow,'blocks':blocks,
                    'pix':pix,'centers':centers,
                    'nblocks':nblocks,'isize':isize})
    return sample

