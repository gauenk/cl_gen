
def main():

    # load the data
    data,loader = get_dataset(cfg,'disent')

    # load the model and set criterion
    models = load_static_models(cfg)
    for name,model in models.items(): model.eval()

    # run a test image

    numOfExamples = 4
    fig,ax = plt.subplots(4,numOfExamples,figsize=(8,8))
    for num_ex in range(numOfExamples):
        pic_set,raw_img = next(iter(loader.te))
        pic_set = pic_set.to(cfg.disent.device)
        raw_img = raw_img.to(cfg.disent.device)

        N = len(pic_set)
        BS = len(pic_set[0])
        pshape = pic_set[0][0].shape
        shape = (N,BS,) + pshape

        rec_set = reconstruct_set(pic_set,enc,dec,cfg.disent.share_enc)
        rec_set = rescale_noisy_image(rec_set)
        rec_set_i = rec_set[0]

        # Plot Decoded Image
        mse = F.mse_loss(rec_set_i,raw_img).item()


    fn = "denoising_check_encoder_sharing.png"
    plt.savefig(fn)
