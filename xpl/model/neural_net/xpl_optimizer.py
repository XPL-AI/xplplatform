import torch
import torch.optim._functional as F

# This is a copy paste from SGD. I'll fix it later


class XPLOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(XPLOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(XPLOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    alpha = 1
                    eps = 1e-8
                    scale = (alpha * torch.abs(p.data) + eps)
                    # 1/ ln(2) = 0.693
                    scaled_grad = 0.693 * scale * torch.sign(p.grad) * torch.log(torch.abs(p.grad) / scale + 1)
                    d_p_list.append(scaled_grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            F.sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay,
                  momentum,
                  lr,
                  dampening,
                  nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
