
def spoof(burst,motion,nblocks,mtype,acc):
    T = burst.shape[0]
    ref_block = get_ref_block_index(nblocks)
    gt_blocks = global_flow_frame_blocks(motion,nblocks)
    # gt_blocks = global_flow_to_blocks(motion,nblocks) -- this returns frame alignment
    rands = npr.uniform(0,1,size=motion.shape[0])
    scores,blocks = [],[]
    for idx,rand in enumerate(rands):
        if rand > acc:
            fake = torch.randint(0,nblocks**2,(T,))
            fake[T//2] = ref_block
            blocks.append(fake)
        else: blocks.append(gt_blocks[idx])
        scores.append(0)
    blocks = torch.stack(blocks)
    burst_clone = burst.clone()
    aligned = align_burst_from_block(burst_clone,blocks,nblocks,"global")
    # print_tensor_stats("[lpas]: burst0 - burst1",burst[0] - burst[1])
    # print_tensor_stats("[lpas]: aligned0 - aligned1",aligned[0] - aligned[1])
    # print_tensor_stats("[lpas]: aligned[T/2] - burst[T/2]",aligned[T//2] - burst[T//2])
    return scores,aligned
