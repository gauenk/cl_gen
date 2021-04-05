
def print_tensor_stats(name,tensor):
    stats_fmt = (tensor.min().item(),tensor.max().item(),tensor.mean().item())
    stats_str = "%2.2e,%2.2e,%2.2e" % stats_fmt
    print(f"[{name}]",stats_str)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, model_target, model_online):
    """
    ema_updater: Exponential moving average updater. 
    model_target: The model with ema parameters. 
    model_online: The model online uses sgd.
    """
    for params_online, params_target in zip(model_online.parameters(), model_target.parameters()):
        online_weights, target_weights = params_online.data, params_target.data
        params_target.data = ema_updater.update_average(online_weights, target_weights)

