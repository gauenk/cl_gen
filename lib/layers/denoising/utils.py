# pytorch imports
import torch
import numpy as np
import matplotlib.pyplot as plt

def share_encoding_mean(agg_type,h,skip,N,BS):

    if agg_type == 'h' or agg_type == 'full':
        h = share_encoding_mean_h(agg_type,h,N,BS)
    elif agg_type == 'skip' or agg_type == 'full':
        skip = share_encoding_mean_skip(agg_type,skip,N,BS)
    else:
        raise ValueError(f"Uknown agg_type [{agg_type}]")
    return h,skip

def share_encoding_mean_h(agg_type,h,N,BS):
    # share h, (N*BS,-1) or (N,BS,-1)
    h = normalize_nd(h)
    h = h.reshape(N,BS,-1)
    h = torch.mean(h,dim=0)
    h = h.expand((N,) + h.shape)
    h = h.reshape(N*BS,h.shape[-1]) # output is (N*BS,-1)
    return h

def share_encoding_mean_skip(agg_type,skip,N,BS):
    # share skip, [ (N*BS,...), (N*BS,...), ...]
    skip_mean = []
    for skip_layer in skip:
        a_shape = skip_layer.shape # (N*BS,...)
        m_shape = (N,BS,) + a_shape[1:] # (N,BS,...)
        skip_layer = normalize_nd(skip_layer)
        skip_layer = skip_layer.reshape(m_shape)
        skip_layer = torch.mean(skip_layer,dim=0)
        skip_layer = skip_layer.expand(m_shape)
        skip_layer = skip_layer.reshape(a_shape) # output is (N*BS,...)
        skip_mean.append(skip_layer)
    skip = skip_mean
    return skip
    

def share_encoding_mean_check(h,aux,N,BS):

    # share h, (N*BS,-1) or (N,BS,-1)
    h = h.reshape(N,BS,-1)
    h = torch.mean(h,dim=0)
    h = h.expand((N,) + h.shape)
    h = h.reshape(N*BS,h.shape[-1]) # output is (N*BS,-1)

    # share aux, [ (N*BS,...), (N*BS,...), ...]
    bsidx = 0
    fig,ax = plt.subplots(N+1,2,figsize=(8,8))
    print(aux[0].shape)
    plot_aux = aux[0].reshape((N,BS,)+aux[0].shape[1:])
    for i,img in enumerate(plot_aux[:,bsidx]):
        plot_th_tensor(ax,i,0,img)

    aux_mean = []
    for aux_layer in aux:
        a_shape = aux_layer.shape # (N*BS,...)
        m_shape = (N,BS,) + a_shape[1:] # (N,BS,...)
        aux_layer = aux_layer.reshape(m_shape)
        aux_layer = torch.mean(aux_layer,dim=0)
        aux_layer = aux_layer.expand(m_shape)
        aux_layer = aux_layer.reshape(a_shape) # output is (N*BS,...)
        aux_mean.append(aux_layer)
    aux = aux_mean

    plot_aux = aux[0].reshape((N,BS,)+aux[0].shape[1:])
    for i,img in enumerate(plot_aux[:,bsidx]):
        plot_th_tensor(ax,i,1,img)

    plt.savefig("check_enc_sharing.png")

    return h,aux
    
def plot_th_tensor(ax,i,j,img):
    img = img.to('cpu').detach().numpy()[0]
    img += np.abs(np.min(img))
    img = img / img.max()
    ax[i,j].imshow(img,  cmap='Greys_r',  interpolation=None)
    ax[i,j].set_xticks([])
    ax[i,j].set_yticks([])

def normalize_nd(tensor):
    t_shape = tensor.shape
    n_shape = (t_shape[0],-1)
    nt = tensor.reshape(n_shape)
    nt = (tensor.T / torch.norm(tensor,dim=1)).T
    nt = nt.reshape(t_shape)
    return nt

def reconstruct_set(pic_set,encoder,decoder,_share_enc=False):
    
    # shape
    N = len(pic_set)
    BS = len(pic_set[0])
    pic_shape = pic_set[0][0].shape

    # encode
    if isinstance(pic_set,list):
        pic_set = torch.cat(pic_set,dim=0)
    else:
        pic_set = pic_set.reshape((N*BS,)+pic_shape)
    h,aux = encoder(pic_set)

    # aggregate encodings
    if _share_enc:
        h,aux = share_encoding_mean(h,aux,N,BS)

    # decode
    input_i = [h,aux]
    dec_pics = decoder(input_i)
    dec_pics = dec_pics.reshape((N,BS,) + pic_shape)

    return dec_pics
