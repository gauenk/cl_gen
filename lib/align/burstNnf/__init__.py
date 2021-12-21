


# -- python imports --
import numpy as np
from einops import rearrange,repeat
import scipy.linalg as scl

# -- pytorch imports --
import torch
from torch.nn import functional as F

# -- project imports --
from pyutils import tile_patches_with_nblocks
#from scipy import gaussian_filter as gfilter


VERBOSE = False
def vprint(*args,**kwargs):
    if VERBOSE:
        print(*args,**kwargs)

def get_filter_x():
    filter_x = torch.Tensor([
                             [1, -1 ],
                             ])/1.
    filter_x = filter_x.view((1,1,1,2))

    # filter_x = torch.Tensor([[1, 0, -1],
    #                          [2, 0, -2],
    #                          [1, 0, -1]])/8.
    # filter_x = filter_x.view((1,1,3,3))
    # filter_x = filter_x.repeat(3,3,1,1)
    return filter_x

def get_filter_y():
    filter_y = torch.Tensor([
                             [1, -1 ],
                             ])/1.
    filter_y = filter_y.view((1,1,2,1))
    # filter_y = torch.Tensor([[1, 2, 1],
    #                          [0, 0, 0],
    #                          [-1, -2, -1]])/8.
    # filter_y = filter_y.view((1,1,3,3))
    # filter_y = filter_y.repeat(3,3,1,1)
    return filter_y

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussian_weights(nftrs,color):
    W = int(np.sqrt(nftrs/color))
    assert W**2*color == nftrs,"Square patches."
    shape = (W,W)
    gW = matlab_style_gauss2D(shape,sigma=0.5)
    gW = repeat(gW,'w1 w2 -> (w1 w2 r)',r=color)
    gW = torch.FloatTensor(gW)
    return gW

def compute_space_derivatives(patches,nblocks,patchsize,color):

    # -- reshape input --
    nimages,nframes,npix,nftrs = patches.shape
    ps = int(np.sqrt(nftrs/color))
    patches = rearrange(patches,'i t p (p1 p2 c) -> (i t p) c p1 p2',p1=ps,p2=ps)

    # -- compute filter --
    fX = get_filter_x().to(patches.device)
    fY = get_filter_y().to(patches.device)
    G_x = F.conv2d(patches, fX, padding='valid')
    G_y = F.conv2d(patches, fY, padding='valid')
    # print("patches.shape ", patches.shape,patchsize)
    # print("G_x.shape ",G_x.shape)
    # print("G_y.shape ",G_y.shape)

    # -- crop to actual patchsize --
    dpad = ps - patchsize
    start = nblocks//2
    # print("nblocks ",nblocks,nblocks//2)
    pslice = slice(start,start+patchsize)
    # print("[before] patches.shape ",patches.shape)
    # print("[after] patches.shape ",patches.shape)

    # -- good y for [0 1] --
    # psliceA = slice(start,start+patchsize)
    # psliceB = slice(start-1,start-1+patchsize)
    # psliceC = slice(start+1,start+1+patchsize)
    # patches = patches[...,psliceC,pslice]
    # G_x = G_x[...,psliceA,pslice]
    # G_y = G_y[...,pslice,psliceA]

    psliceA = slice(0,patchsize)
    psliceB = slice(start-1,start-1+patchsize)
    psliceC = slice(start+1,start+1+patchsize)
    # patches = patches[...,5:12,5:12]
    pix = 32*64+16
    patches = patches[...,4:11,4:11]
    G_x = G_x[...,4:11,3:10]
    G_y = G_y[...,3:10,4:11]

    # -- reshape back --
    shape_str = '(i t p) c p1 p2 -> i t p p1 p2 c'
    patches_img = rearrange(patches,shape_str,t=nframes,i=nimages)

    shape_str = '(i t p) c p1 p2 -> i t p (p1 p2 c)'
    patches = rearrange(patches,shape_str,t=nframes,i=nimages)
    G_x = rearrange(G_x,shape_str,t=nframes,i=nimages)
    G_y = rearrange(G_y,shape_str,t=nframes,i=nimages)


    # p0 = patches_img[0,0,pix,:,:,0].cpu().numpy()
    # p1 = patches_img[0,1,pix,:,:,0].cpu().numpy()
    # p2 = patches_img[0,2,pix,:,:,0].cpu().numpy()
    # print(np.around(p1 - p0,3))
    # print(np.around(p2 - p1,3))

    return patches,G_x,G_y

def compute_time_derivative(patches,nframes):
    """
    shaped like:

    [ (t0 - t0)  (t1 - t0) (t1 - t2) ..
      (t1 - t0)  (t1 - t1) ...
    ...]
    
    """
    dtime = []
    pix = 32*64+16

    for t1 in range(nframes):
        dt_t1 = []
        for t2 in range(nframes):
            delta = patches[:,t2] - patches[:,t1]
            dt_t1.append(delta)
            # print("t2 - t1: ",t2,t1,np.around(delta[0,pix].cpu().numpy(),3))

        dt_t1 = torch.stack(dt_t1,dim=1)
        dtime.append(dt_t1)
    dtime = torch.stack(dtime,dim=2)
    # print(np.around(dtime[0,1,0,pix].cpu().numpy(),3))
    # print("patches.shape ",patches.shape)
    # print("dtime.shape ",dtime.shape)
    return dtime

def not_t_index(t,nframes):
    return torch.LongTensor([i for i in range(nframes) if i != t])

def update_v_kl(patches,dX,dY,dtime,V,t,ref,color):

    # -- get vars --
    dt = dtime[:,t,ref]
    dx = dX[:,t]
    dy = dY[:,t]

    # -- get gaussian weights --
    w = gaussian_weights(patches.shape[-1],color)
    w = gaussian_weights(patches.shape[-1],color)
    w = w.to(patches.device)

    # -- report shapes --
    VERBOSE=False
    # print("dtime.shape ",dtime.shape)
    # print("dt.shape ",dt.shape)
    # print("dx.shape ",dx.shape)
    # print("dy.shape ",dy.shape)
    # print("w.shape ",w.shape)
    VERBOSE=False

    # -- ave over features --
    sum_dx2 = torch.sum(torch.pow(dx,2),dim=-1)
    sum_dy2 = torch.sum(torch.pow(dy,2),dim=-1)
    sum_dxdy = torch.sum(dx*dy,dim=-1)
    sum_dxdt = torch.sum(dx*dt,dim=-1)
    sum_dydt = torch.sum(dy*dt,dim=-1)

    # -- ave over features --
    # sum_dx2 = torch.sum(w*torch.pow(dx,2),dim=-1)
    # sum_dy2 = torch.sum(w*torch.pow(dy,2),dim=-1)
    # sum_dxdy = torch.sum(w*dx*dy,dim=-1)
    # sum_dxdt = torch.sum(w*dx*dt,dim=-1)
    # sum_dydt = torch.sum(w*dy*dt,dim=-1)
    vprint("sum_dx2.shape ",sum_dx2.shape)
    vprint("sum_dy2.shape ",sum_dy2.shape)
    vprint("sum_dxdy.shape ",sum_dxdy.shape)
    vprint("sum_dxdt.shape ",sum_dxdt.shape)
    vprint("sum_dydt.shape ",sum_dydt.shape)

    # -- solve using "A" --
    """
    A = 
    [sum_dx2, sum_dxdy;
    sum_dxdy, sum_dy2]

    A{-1} = 
    1/det(A)
    [sum_dy2, -sum_dxdy;
    -sum_dxdy, sum_dx2]

    b = [-sum_dxdt;
         -sum_dydt]

    """
    # -- standard --
    Adet_inv = 1./(sum_dx2 * sum_dy2 - sum_dxdy ** 2)
    vX = Adet_inv * ( sum_dxdy * sum_dydt - sum_dy2 * sum_dxdt )
    vY = Adet_inv * ( sum_dxdy * sum_dxdt - sum_dx2 * sum_dydt )
    v = torch.stack([vX,vY],dim=-1)

    # -- check with A --
    Atop = torch.stack([sum_dx2,sum_dxdy],dim=0)
    Abtm = torch.stack([sum_dxdy,sum_dy2],dim=0)
    A = torch.stack([Atop,Abtm],dim=0)
    pix = 64*32+16
    # print(scl.eig(A[:,:,0,pix].cpu().numpy()))
    # b = torch.stack([-sum_dxdt,-sum_dydt],dim=0)
    # delta = torch.sum(torch.abs(torch.matmul(A[:,:,0,pidx],v[0,pidx,:]) - b[:,0,pidx]))
    # print(f"Inv Error {delta}")

    
    return v



def update_v(patches,dX,dY,dtime,V,t,ref):

    # -- get vars --
    tidx = t if t < ref else t-1
    nframes = dX.shape[1]
    not_t = not_t_index(t,nframes)
    dt = dtime[:,tidx,not_t]
    dt_not_t = dtime[:,not_t,not_t]
    dx = dX[:,not_t]
    dy = dY[:,not_t]
    nV = V[not_t]

    # -- report shapes --
    vprint("dt.shape ",dt.shape)
    vprint("dx.shape ",dx.shape)
    vprint("dy.shape ",dy.shape)
    vprint("nV.shape ",nV.shape)

    # -- get gaussian weights --
    gweights = gaussian_weights(patches.shape[-1],color)
    gweights = gweights.to(patches.device)

    # -- ave over features --
    sum_dx2 = torch.sum(torch.pow(gweights*dx,2),dim=-1)
    sum_dy2 = torch.sum(torch.pow(gweights*dy,2),dim=-1)
    sum_dxy = torch.sum(gweights*dy*dx,dim=-1)
    sum_dxdt = torch.sum(gweights*dx*dt,dim=-1)
    sum_dydt = torch.sum(gweights*dy*dt,dim=-1)
    vprint("sum_dx2.shape ",sum_dx2.shape)

    # -- select specific frames --
    # sum_dx2 = sum_dx2[:,t]
    # sum_dy2 = sum_dy2[:,t]
    # sum_dxy = sum_dxy[:,t]
    # sum_dxdt = sum_dxdt[:,t]
    # sum_dydt = sum_dydt[:,t]

    # -- ave over frames --
    # sum_dx2 = torch.sum(torch.pow(sum_dx2,2),dim=1)
    # sum_dy2 = torch.sum(torch.pow(sum_dy2,2),dim=1)
    # sum_dxy = torch.sum(sum_dxy,dim=1)
    # sum_dxdt = torch.sum(sum_dxdt,dim=1)
    # sum_dydt = torch.sum(sum_dydt,dim=1)
    # vprint("sum_dxdt.shape ",sum_dxdt.shape)


    # -- construct adjusted "b" --
    # dt_pairs = patches[:,not_t] - patches[:,t]
    # 2*dt_pairs + dx * nV[:,:,:,0] + dy * nV[:,:,:,1]

    # -- solve using "A" --
    """
    A = 
    [sum_dx2, sum_dxy;
    sum_dxy, sum_dy2]

    A{-1} = 
    1/det(A)
    [sum_dy2, -sum_dxy;
    -sum_dxy, sum_dx2]

    b = 

    """
    # -- proposed --
    # Adet_inv = 1./(sum_dx2 * sum_dy2 - sum_dxy * sum_dxy)
    # vX = Adet_inv * ( -sum_dy2 * sum_dxdt + sum_dxy * sum_dydt )
    # vY = Adet_inv * ( sum_dxy * sum_dxdt - sum_dx2 * sum_dydt )

    # -- standard --
    Adet_inv = 1./(sum_dx2 * sum_dy2 - sum_dxy * sum_dxy)
    vX = Adet_inv * ( -sum_dy2 * sum_dxdt + sum_dxy * sum_dydt )
    vY = Adet_inv * ( sum_dxy * sum_dxdt - sum_dx2 * sum_dydt )

    v = torch.stack([vX,vY],dim=-1)

    return v

def sanity_check(dX,dY,dtime,V,t,ref):


    # -- get vars --
    dt = dtime[:,t,ref]
    dx = dX[:,t]
    dy = dY[:,t]
    vX = V[t,:,:,0,None]
    vY = V[t,:,:,1,None]

    # print(dt.shape)
    # print(dx.shape)
    # print(dy.shape)
    # print(vX.shape)
    # print(vY.shape)

    # -- compute should-be-zero --
    # eq_zero = dx * vX + dy * vY + dt
    if t == 0:
        vX = torch.zeros_like(vX)
        vY = torch.ones_like(vY)
    elif t == 2:
        vX = -torch.ones_like(vX)
        vY = -torch.ones_like(vY)
    pix = 32*64+16
    def rs(array):
        h = int(np.sqrt(len(array)))
        return np.around(rearrange(array,'(h w) -> h w',h=h).cpu().numpy(),3)
    # print("V.shape ",V.shape)
    # print("dX.shape ",dX.shape)
    # print(rs(V[t,0,pix])) # (vX,vY)
    # print(rs(dX[0,t,pix])) # (dX)
    # # print(rs(dY[0,t,pix])) # (dY)
    # # print(rs(dtime[0,t,ref,pix]))
    # print("dtime")
    # print(rs(dtime[0,t,ref,pix]))
    # print("dY")
    # print(rs(dY[0,t,pix]))
    # print("dtime - dY")
    # print(rs(dtime[0,t,ref,pix] - dY[0,t,pix]))
    # print("dtime - dX")
    # print(rs(dtime[0,t,ref,pix] - dX[0,t,pix]))
    # print("dY + dX")
    # print(rs(dY[0,t,pix] + dX[0,t,pix]))
    # print("dY + dX - dt")
    # print(rs(dY[0,t,pix] + dX[0,t,pix] - dtime[0,t,ref,pix]))
    # # print(rs((dX[0,t,pix] + dY[0,t,pix]) / dtime[0,t,ref,pix]))
    # print("\n"*20)
    # print(rs(dX[0,t,pix] / (dtime[0,t,ref,pix] + dY[0,t,pix])))
    # print("diff: ",rs((dX[0,t,pix] + dY[0,t,pix]) - dtime[0,t,ref,pix]))

    # eq_zero = dx * vX + dy * vY + dt
    eq_zero = dx * vX + dy * vY + dt
    error = torch.mean(torch.abs(eq_zero))
    # print("Sanity Check: %2.3e" % error)


def run(burst,patchsize,nblocks):

    # -- unpack some shapes --
    ps = 7#patchsize
    nblocks = 5
    nframes,nimages,c,h,w = burst.shape
    burst,c = burst[:,:,[0],:,:],1

    patches = tile_patches_with_nblocks(burst,ps,nblocks).pix
    nimages,nframes,npix,nftrs = patches.shape
    pix = 64*32+16
    def rs(array):
        h = int(np.sqrt(len(array)))
        return rearrange(array,'(h w) -> h w',h=h)
    # print(rs(patches[0,1,pix,:] - patches[0,0,pix,:]))
    # exit()
    
    # -- compute space derivates --
    cropped,dX,dY = compute_space_derivatives(patches,nblocks,ps,c)
    # -- compute time derivatives --
    dtime = compute_time_derivative(cropped,nframes)
    vprint("dX.shape ",dX.shape)
    vprint("dY.shape ",dY.shape)
    vprint("dtime.shape ",dtime.shape)
    vprint("dX(min,max) ",dX.min(),dX.max())
    vprint("dY(min,max) ",dY.min(),dY.max())
    vprint("patches(min,max) ",patches.min(),patches.max())

    # -- update using LK-type method --
    ref = nframes//2
    niters = 1
    V = torch.zeros((nframes,nimages,npix,2)).to(patches.device)
    for i in range(niters):

        # -- update each v --
        for t in range(nframes):
            if t == ref: continue
            # V[t] = update_v(patches,dX,dY,dtime,V,t,ref)
            V[t] = update_v_kl(cropped,dX,dY,dtime,V,t,ref,c)
            if t > ref:
                V[t] = -V[t]
            sanity_check(dX,dY,dtime,V,t,ref)
    # print(V[:,0,64*32+16])
    # print(torch.mean(V,dim=2))

    # -- api interface --
    V = torch.round(rearrange(V,'t i p two -> i p t two'))
            
    return V
