"""
Compare L2 with Boostrapping for 3 frames.

"""


def l2_loss_landscape():
    pass


def main():
    seed = 234
    np.random.seed(seed)
    torch.manual_seed(seed)
    dynamic()

if __name__ == "__main__":
    main()

