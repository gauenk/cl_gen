
# -- python --

# -- pytorch --

# -- project --

def run_pixel_diffs(cfg,clean,noisy,directions,results):
    results['dpixClean'] = compute_pixel_difference(clean_blocks,
                                                    block_search_space)
    results['dpixNoisy'] = compute_pixel_difference(noisy_blocks,
                                                    block_search_space)


def compute_pixel_difference(blocks,block_search_space):
    # -- vectorize search since single patch --
    R,B,T,N,C,PS1,PS2 = blocks.shape
    REF_N = get_ref_block_index(int(np.sqrt(N)))
    #print(cfg.nframes,T,cfg.nblocks,N,block_search_space.shape)
    assert (R == 1) and (B == 1), "single pixel's block and single sample please."
    expanded = blocks[:,:,np.arange(T),block_search_space]
    E = expanded.shape[2]
    R,B,E,T,C,H,W = expanded.shape
    PS = PS1

    ref = repeat(expanded[:,:,:,T//2],'r b e c h w -> r b e tile c h w',tile=T-1)
    neighbors = torch.cat([expanded[:,:,:,:T//2],expanded[:,:,:,T//2+1:]],dim=3)
    delta = F.mse_loss(ref[...,PS//2,PS//2],neighbors[...,PS//2,PS//2],reduction='none')
    delta = delta.view(R,B,E,-1)
    delta = torch.mean(delta,dim=3)
    pix_diff = delta[0,0]
    return pix_diff

