"""
Given the the OT losses and the MSE losses 
between clean and noisy frames in a burst
which combinations gives a stable point of loss?

"""



def stability_check(cfg,model,optimizer,criterion,train_loader,epoch):
    model.train()
    model = model.to(cfg.device)
    N = cfg.N
    total_loss = 0
    running_loss = 0
    szm = ScaleZeroMean()
    # record = init_record()
    comparisons = {}

    for batch_idx, (burst, res_imgs, raw_img) in enumerate(train_loader):

        # -- zero grad --
        optimizer.zero_grad()
        model.zero_grad()

        
        # -- cuda --
        burst = burst.cuda(non_blocking=True)


        # -- zero mean --
        raw_img = raw_img - 0.5
        
        # -- forward pass model --
        cat_burst = rearrange(burst,'n b c h w -> b n c h w')
        stacked_burst = rearrange(burst,'n b c h w -> (b n) c h w')
        rec_img_i,rec_img = model(cat_burst,stacked_burst)

        # -- compute KPN loss --
        kpn_loss = criterion(rec_img_i, rec_img, t_img, cfg.global_step)
        kpn_grads = get_loss_grads(kpn_loss,model)

        # -- compute MSE loss --
        mse_loss = F.mse_loss(rec_img,raw_img)
        mse_grads = get_loss_grads(mse_loss,model)

        # -- compute OT loss --
        residuals = rec_img_i - rec_img.unsqueeze(1).repeat(1,N,1,1,1)
        residuals = rearrange(residuals,'b n c h w -> b n (h w) c')
        ot_loss = ot_frame_pairwise_bp(residuals,reg=reg,K=3)
        ot_grads = get_loss_grads(mse_loss,model)


        # -- compare gradients --
        comparisons[batch_idx] = {}
        for kpn_l,mse_l,ot_l in zip(kpn_grads,mse_grads,ot_grads):
            kpn_mse = compare_grads(kpn_l,mse_l)
            comparisons[batch_idx]['kpn_mse'] = kpn_mse
            
            kpn_ot = compare_grads(kpn_l,ot_l)
            comparisons[batch_idx]['kpn_ot'] = kpn_ot

            mse_ot = compare_grads(mse,ot_l)
            comparisons[batch_idx]['mse_ot'] = mse_ot


    return comparisons

