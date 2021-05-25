
# -- python imports --
import numpy as np
from einops import rearrange

# -- pytorch imports --
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

def align_burst_from_block(burst,block,nblocks,mtype):
    if mtype == "global":
        return align_burst_from_block_global(burst,block,nblocks)
    elif mtype == "local":
        return align_burst_from_block_local(burst,block,nblocks)
    else:
        raise ValueError(f"Uknown motion type [{mtype}]")

def align_burst_from_flow(burst,flow,nblocks,mtype):
    if mtype == "global":
        return align_burst_from_flow_global(burst,flow,nblocks)
    elif mtype == "local":
        raise NotImplemented("No local flow supported yet.")
    else:
        raise ValueError(f"Uknown motion type [{mtype}]")

def align_burst_from_block_global(bursts,blocks,nblocks):
    T,B,FS = bursts.shape[0],bursts.shape[1],bursts.shape[-1]
    tiles = tile_across_blocks(bursts,nblocks)
    crops = []
    for b in range(B):
        block = blocks[b]
        for t in range(T):
            index = block[t]
            crops.append(tiles[t,b,index])
    crops = rearrange(crops,'(b t) c h w -> t b c h w',b=B)
    return crops

def align_burst_from_flow_global(bursts,flow,nblocks):
    blocks = global_flow_to_blocks(flow,nblocks)
    return align_burst_from_block_global(bursts,blocks,nblocks)

def global_flow_to_blocks(_flow,nblocks):
    """
    "flow" is a integer direction of motion between two frames
       flow[b,t] is a vector of direction [dx, dy] wrt previous frame
    "nblocks" is the maximum number of pixels changed between adj. frames
    "indices" are the 
       indices[b,t] is the integer index representing the specific
       neighbor for a given flow
    """
    flow = _flow.clone()
    B,Tm1 = flow.shape[:2]
    T = Tm1 + 1
    grid = np.arange(nblocks**2).reshape(nblocks,nblocks)
    ref_t,ref_bl = T//2,nblocks//2
    indices,coord_ref = [],torch.IntTensor([[ref_bl,ref_bl]])
    for b in range(B):
        flow_b = flow[b]
        flow_b[:,1] *= -1 # -- spatial direction to matrix direction --
        left = ref_bl - rcumsum(flow_b[:ref_t]) # -- moving backward
        right = torch.cumsum(flow_b[ref_t:],0) + ref_bl # -- moving forward
        coords = torch.cat([left,coord_ref,right],dim=0)
        for t in range(T):
            x,y = coords[t][0].item(),coords[t][1].item()
            index = grid[y,x] # up&down == rows, left&right == cols
            indices.append(index)
    indices = torch.LongTensor(indices)
    indices = rearrange(indices,'(b t) -> b t',b=B)
    return indices

def global_blocks_to_flow(blocks,nblocks):
    """
    flow 
    """
    B,T = blocks.shape[:2]
    grid = np.arange(nblocks**2).reshape(nblocks,nblocks)
    flow = []
    ref_t,ref_bl = T//2,nblocks//2
    for b in range(B):
        block_b = blocks[b]
        coords = []
        for t in range(T):
            coord = np.r_[np.where(grid == block_b[t].item())]
            coords.append(coord)
        coords = np.stack(coords)
        coords = coords[:,::-1] # -- x <-> y swap --
        coords[:,1] *= -1 # -- Matrix_dir -> Spatial_dir --
        # -- compute the flow --
        coords[:,0] = np.ediff1d(coords[:,0],0)
        coords[:,1] = np.ediff1d(coords[:,1],0)
        flow_b = coords[:-1] # -- last one is [0,0] --
        flow.append(flow_b)
    flow = np.stack(flow)
    flow = torch.LongTensor(flow)
    return flow

#
# Supporting
#

def rcumsum(tensor,dim=0):
    return torch.flip(torch.cumsum(torch.flip(tensor,(dim,)),dim),(dim,))

def reshape_and_pad(images,nblocks):
    T,B,C,H,W = images.shape
    images = rearrange(images,'t b c h w -> (t b) c h w')
    padded = F.pad(images, [nblocks//2,]*4, mode="reflect")
    padded = rearrange(padded,'(t b) c h w -> t b c h w',b=B)
    return padded

def tile_across_blocks(batches,nblocks):
    B = batches.shape[1]
    H = nblocks
    FS = batches.shape[-1]
    crops,tiles = [],[]
    grid = np.arange(nblocks**2).reshape(nblocks,nblocks)
    blocks = []
    center = H//2
    padded = reshape_and_pad(batches,nblocks)
    for i in range(-H//2+1,H//2+1):
        for j in reversed(range(-H//2+1,H//2+1)):
            # grid[i+H//2,j+H//2] = i*j + j + center
            # print(grid[i+center,j+center],(i+center,j+center))
            crops.append(tvF.crop(padded,i+center,j+center,FS,FS))
    # print(blocks)
    crops = torch.stack(crops,dim=0)
    crops = rearrange(crops,'g t b c h w -> t b g c h w')
    return crops

#
# Testing
# 

def test_global_flow_to_block():
    def run_check(flow,blocks,nblocks):
        gt_blocks = global_flow_to_blocks(flow,nblocks)
        delta = torch.sum(torch.abs(gt_blocks-blocks)).item()
        assert torch.isclose(delta,0),"No error"

    def test1():
        flow = torch.LongTensor([[[1,-1],[0,-1],[1,0]]])
        blocks = [[0,4,7]]
        nblocks = 3
        run_check(flow,blocks,nblocks)

    def test2():
        flow = torch.LongTensor([[[1,1],[0,-1]]])
        blocks = [[6,4,7]]
        nblocks = 3
        run_check(flow,blocks,nblocks)

    def test3():
        flow = torch.LongTensor([[[1,-1],[0,-1],[1,0]]])
        blocks = [[1,7,12,13]]
        nblocks = 5
        run_check(flow,blocks,nblocks)

    test1()
    test2()
    test3()
    
def test_global_block_to_flow():
    def run_check(flow,blocks,nblocks):
        est_flow = global_blocks_to_flow(blocks,nblocks)
        delta = torch.sum(torch.abs(est_flow-flow)).item()
        assert torch.isclose(delta,0),"No error"

    def test1():
        flow = torch.LongTensor([[[1,-1],[0,-1],[1,0]]])
        blocks = [[0,4,7]]
        nblocks = 3
        run_check(flow,blocks,nblocks)

    def test2():
        flow = torch.LongTensor([[[1,1],[0,-1]]])
        blocks = [[6,4,7]]
        nblocks = 3
        run_check(flow,blocks,nblocks)

    def test3():
        flow = torch.LongTensor([[[1,-1],[0,-1],[1,0]]])
        blocks = [[1,7,12,13]]
        nblocks = 5
        run_check(flow,blocks,nblocks)

    test1()
    test2()
    test3()




