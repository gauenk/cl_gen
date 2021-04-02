def print_tensor_stats(name,tensor):
    print(name,tensor.min().item(),tensor.max().item(),tensor.mean().item())
