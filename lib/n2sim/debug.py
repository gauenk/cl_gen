def print_tensor_stats(prefix,tensor):
    stats_fmt = (tensor.min().item(),tensor.max().item(),
                 tensor.mean().item(),tensor.std().item())
    stats_str = "%2.2e,%2.2e,%2.2e,%2.2e" % stats_fmt
    print(prefix,stats_str)
