"""
This file is used to train and test a logistic classification model 
using an encoded representation of the input.

"""


def train_logit(cfg):
    pass


def test_logit(cfg):
    pass

def load_encoder(enc_path):
    model = Encoder()
    if cfg.disent.load:
        fn = Path("checkpoint_{}.tar".format(cfg.disent.epoch_num))
        model_fp = Path(cfg.disent.model_path) / Path(enc_str) / fn
        model.load_state_dict(torch.load(model_fp, map_location=cfg.disent.device.type))
    model = model.to(cfg.disent.device)
    return model

    


if __name__ == "__main__":
    print("HI")

    train_logit(cfg)
    test_logit(cfg)
