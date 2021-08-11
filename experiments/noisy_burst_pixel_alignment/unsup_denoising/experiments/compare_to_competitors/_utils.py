class GradientHooks():

    def __init__(self,model):
        self.hooks = {}
        self.samples = {}
        self,add_gradient_hooks(model)

    def add_gradient_hooks(self,model):
        def hook_fn(module,grad_inputs,grad_outputs):
            name = module.name
            if name in self.samples:
                self.samples[name].append(grad_inputs)
            else:
                self.samples[name].append(grad_outputs)

        hooks = {}
        for name, module in model.named_modules():
            hooks[name] = module.register_forward_hook(hook_fn)
        return hooks
