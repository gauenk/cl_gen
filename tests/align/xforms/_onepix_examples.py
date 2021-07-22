
# -- python imports --
import torch
from easydict import EasyDict as edict

def get_example(name):
    if name == "example_1":
        return example_1()
    elif name == "example_2":
        return example_2()
    elif name == "example_3":
        return example_3()
    elif name == "example_4":
        return example_4()
    elif name == "example_5":
        return example_5()
    else:
        raise ValueError(f"Uknown example name [{name}]")

def example_1():
    patchsize = 3
    flow = torch.LongTensor([[[
        [1,-1],
        [0,-1]
    ]]])
    blocks = torch.LongTensor([[
        [8,4,1]
    ]])
    centers = torch.LongTensor([[
        [2,3]
    ]])
    # a.) convert flow axis (y = 0 @ bottom) to object axis (y = 0 @ top)
    # b.) "-" before and "+" after the reference frame using cumulative sums
    pix = torch.LongTensor([[[ 
        [2-(1),3-(-(-1))],
        [2,3],
        [2+(0),3+(-(-1))]
    ]]]) 
    nblocks = 3
    isize = edict({'h':1,'w':1})
    sample = edict({'flow':flow,'blocks':blocks,
                    'pix':pix,'centers':centers,
                    'nblocks':nblocks,'isize':isize})
    return sample

def example_2():
    flow = torch.LongTensor([[[
        [1,1],[0,-1]
    ]]])
    blocks = torch.LongTensor([[
        [2,4,1]
    ]])
    centers = torch.LongTensor([[
        [100,110]
    ]])
    # a.) convert flow axis (y = 0 @ bottom) to object axis (y = 0 @ top)
    # b.) "-" before and "+" after the reference frame using cumulative sums
    pix = torch.LongTensor([[[
        [100-(1),110-(-(1))],
        [100,110],
        [100+(0),110+(-(-1))]
    ]]]) # "-" before and "+" after the reference frame using cumulative sums
    print(flow)
    print(pix)
    nblocks = 3
    isize = edict({'h':1,'w':1})
    sample = edict({'flow':flow,'blocks':blocks,
                    'pix':pix,'centers':centers,
                    'nblocks':nblocks,'isize':isize})
    return sample

def example_3():
    flow = torch.LongTensor([[[
        [1,-1],[0,-1],[1,0]
    ]]])
    blocks = torch.LongTensor([[
        [23,17,12,11]
    ]])
    centers = torch.LongTensor([[
        [100,110]
    ]])
    # a.) convert flow axis (y = 0 @ bottom) to object axis (y = 0 @ top)
    # b.) "-" before and "+" after the reference frame using cumulative sums
    pix = torch.LongTensor([[[
        [100-(0)-(1)  ,110-(-(-1))-(-(-1))],
        [100-(0)      ,110-(-(-1))],
        [100,110],
        [100+(1),110+(-(0))]
    ]]]) # "-" before and "+" after the reference frame using cumulative sums
    nblocks = 5
    isize = edict({'h':1,'w':1})
    sample = edict({'flow':flow,'blocks':blocks,
                    'pix':pix,'centers':centers,
                    'nblocks':nblocks,'isize':isize})
    return sample


def example_4():
    flow = torch.LongTensor([[[
        [1,0],[1,-1],[0,-1],[1,0],[-1,0],[1,1]
    ]]])
    blocks = torch.LongTensor([[
        [24,23,17,12,11,12,16]
    ]])
    centers = torch.LongTensor([[
        [100,110]
    ]])
    # a.) convert flow axis (y = 0 @ bottom) to object axis (y = 0 @ top)
    # b.) "-" before and "+" after the reference frame using cumulative sums
    pix = torch.LongTensor([[[
        [100-(-0)-(1)-(1), 110-(-(-1))-(-(-1))-(-(0))],
        [100-(-0)-(1),     110-(-(-1))-(-(-1))],
        [100-(0),         110-(-(-1))],
        [100,110],
        [100+(1),          110+(-(0))],
        [100+(1)+(-1),     110+(-(0))+(-(0))],
        [100+(1)+(-1)+(1), 110+(-(0))+(-(0))+(-(1))]
    ]]]) # "-" before and "+" after the reference frame using cumulative sums
    nblocks = 5
    isize = edict({'h':1,'w':1})
    sample = edict({'flow':flow,'blocks':blocks,
                    'pix':pix,'centers':centers,
                    'nblocks':nblocks,'isize':isize})
    return sample


def example_5():
    flow = torch.LongTensor([[[
        [1,0],[1,-1],[1,0],[1,0],[1,0],[1,-1],[1,0],[1,0],[1,0]]
    ]])
    # blocks = torch.LongTensor([[
    #     [44,45,57,58,59,55,72,73,74,75]
    # ]])
    blocks = torch.LongTensor([[
        [76,75,63,62,61,60,48,47,46,45]
    ]])
    centers = torch.LongTensor([[
        [100,110]
    ]])
    # a.) convert flow axis (y = 0 @ bottom) to object axis (y = 0 @ top)
    # b.) "-" before and "+" after the reference frame using cumulative sums
    pix = torch.LongTensor([[[
        [100-(1)-(1)-(1)-(1)-(1), 110],
        [100-(1)-(1)-(1)-(1),     110-(-(-1))],
        [100-(1)-(1)-(1),         110],
        [100-(1)-(1),             110],
        [100-(1),                 110],
        [100,                     110],
        [100+(1),                 110+(-(-1))],
        [100+(1)+(1),             110+(-(-1))+(0)],
        [100+(1)+(1)+(1),         110+(-(-1))+(0)],
        [100+(1)+(1)+(1)+(1),     110+(-(-1))+(0)],
    ]]]) # "-" before and "+" after the reference frame using cumulative sums
    nblocks = 11
    isize = edict({'h':1,'w':1})
    sample = edict({'flow':flow,'blocks':blocks,
                    'pix':pix,'centers':centers,
                    'nblocks':nblocks,'isize':isize})
    return sample

