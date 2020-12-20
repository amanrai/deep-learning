import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer
import math

mnist_data = torchvision.datasets.MNIST("./MNIST/", 
            download=True, 
            train=True,
            transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
)
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size = 64, shuffle=True, num_workers=1)

examples = enumerate(data_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)
print(example_data.shape)

"""
OPTIMIZER IMPLEMENTATIONS
"""
class SGD(Optimizer):
    def __init__(self, params, lr=1e-2):
        defaults = dict(lr=lr)
        super (SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    dP = p.grad
                    p.add_(-dP, alpha=group['lr']) 
                    #the _ ahead of the function name implies that the operation will be carried out in place. 
                    #<tensor>.add_ will result in <tensor> + arg*alpha

class SGDWithMomentum(Optimizer):
    def __init__(self, params, lr=1e-2, beta=0.99):
        defaults = dict(lr=lr, beta=beta)
        super (SGDWithMomentum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDWithMomentum, self).__setstate__(state)

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                beta = group['beta']
                lr = group['lr']
                for p in group["params"]:
                    dP = p.grad
                    if (beta != 0):
                        #since we need to track v, we define a new variable in the tensor state
                        param_state = self.state[p]
                        if 'v' not in param_state:
                            param_state['v'] = torch.clone(dP).detach() * lr
                            v = param_state['v'] 
                        else:
                            v = param_state['v']
                            v.mul_(beta)
                            v.add_(dP*lr) 
                        p.add_(-v)
                    else:
                        p.add_(-dP, alpha=group['lr']) 

class Adagrad(Optimizer):
    def __init__(self, params, eta=0.01, eps=1e-8):
        defaults = dict(lr=eta, eps=eps)
        super (Adagrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adagrad, self).__setstate__(state)

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                eta = group['lr']
                eps = group['eps']
                for p in group["params"]:
                    dP = p.grad
                    param_state = self.state[p]
                    if 'v' not in param_state:
                        param_state['v'] = torch.clone(dP).detach()
                        v = param_state['v'] 
                        v.mul_(v)
                    else:
                        """
                            v = vt-1 + grad^2
                            w = wt-1 - (eta/sqrt(vt + eps)) * grad
                            the term (eta/sqrt(vt + eps)) is effectively an individual learning rate per parameter 
                        """
                        v = param_state['v']
                        v.add_(dP*dP)
                    update = (eta/torch.sqrt(v + eps))*dP
                    p.add_(-update)

class RMSProp(Optimizer):
    #RMSProp is basically Adagrad with v being computed as an exponential moving average
    #this implementation is not converging at the moment
    def __init__(self, params, eta=0.01, eps=1e-8, beta=0.99):
        defaults = dict(lr=eta, eps=eps, beta=beta)
        super (RMSProp, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSProp, self).__setstate__(state)

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                eta = group['lr']
                eps = group['eps']
                beta = group['beta']
                for p in group["params"]:
                    dP = p.grad
                    param_state = self.state[p]
                    if 'v' not in param_state:
                        param_state['v'] = torch.clone(dP).detach()
                        v = param_state['v'] 
                        v.mul_(v*(1-beta))
                    else:
                        v = param_state['v']
                        v.mul_(beta).add(dP*dP, alpha=(1-beta))
                    update = (eta/torch.sqrt(v + eps))*dP
                    p.add_(-update)

class Adam(Optimizer):
    def __init__(self, params, beta_m = 0.99, beta_v = 0.99, lr=1e-4, eps=1e-8):
        defaults = dict(lr=lr, eps=eps, beta_m=beta_m, beta_v=beta_v)
        super (Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                eta = group['lr']
                eps = group['eps']
                beta_m = group['beta_m']
                beta_v = group['beta_v']
                for p in group["params"]:
                    dP = p.grad
                    param_state = self.state[p]
                    if ('step' not in param_state):
                        param_state['step'] = 1
                        step = param_state['step']
                    else:
                        step = param_state['step']
                        step += 1
                        param_state['step'] = step

                    if 'm' not in param_state:
                        param_state['m'] = torch.clone(dP).detach() * (1-beta_m)
                        m = param_state['m']                        
                    else:
                        m = param_state['m']
                        m.mul_(beta_m).add_(dP, alpha=1-beta_m)
                    #m.mul_(1/(1-math.pow(beta_m, step)))
                    
                    if 'v' not in param_state:
                        param_state['v'] = torch.clone(dP).detach()
                        v = param_state['v'] 
                        v.mul_(v).mul_(1-beta_v)
                    else:
                        v = param_state['v']
                        v.mul_(beta_v).add_(dP*dP, alpha=1-beta_v)
                    #v.mul_(1/(1-math.pow(beta_v, step)))

                    update = (eta/torch.sqrt(v + eps))*m 
                    p.add_(-update)

class AdamW(Optimizer):
    def __init__(self, params, beta_m = 0.99, beta_v = 0.99, lr=1e-4, eps=1e-8, wd=1e-4):
        defaults = dict(lr=lr, eps=eps, beta_m=beta_m, beta_v=beta_v, wd=wd)
        super (AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                eta = group['lr']
                eps = group['eps']
                beta_m = group['beta_m']
                beta_v = group['beta_v']
                wd = group['wd']
                for p in group["params"]:
                    dP = p.grad
                    param_state = self.state[p]
                    #this is the part that makes it Adam - W. just decay the weight separately. 
                    p.add_(-p*wd*eta)
                    #and thats it.
                    if ('step' not in param_state):
                        param_state['step'] = 1
                        step = param_state['step']
                    else:
                        step = param_state['step']
                        step += 1
                        param_state['step'] = step

                    if 'm' not in param_state:
                        param_state['m'] = torch.clone(dP).detach() * (1-beta_m)
                        m = param_state['m']                        
                    else:
                        m = param_state['m']
                        m.mul_(beta_m).add_(dP, alpha=1-beta_m)
                    #m.mul_(1/(1-math.pow(beta_m, step)))
                    
                    if 'v' not in param_state:
                        param_state['v'] = torch.clone(dP).detach()
                        v = param_state['v'] 
                        v.mul_(v).mul_(1-beta_v)
                    else:
                        v = param_state['v']
                        v.mul_(beta_v).add_(dP*dP, alpha=1-beta_v)
                    #v.mul_(1/(1-math.pow(beta_v, step)))

                    update = (eta/torch.sqrt(v + eps))*m 
                    p.add_(-update)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(0.25)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        int_ = F.tanh(self.layer1(x))
        int_ = self.dropout(int_)
        final = self.layer2(int_)
        return F.log_softmax(final)

network = Network()
#optimizer = AdamW(network.parameters(), beta_m=0.9, beta_v=0.999, lr=1e-4)
#optimizer = SGDWithMomentum(network.parameters(), lr=1e-4)
#optimizer = RMSProp(network.parameters(), eta=1)
optimizer = Adagrad(network.parameters())

network.train()

epochs = 100
for k in range(epochs):
    losses = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.squeeze(1)
        data = data.view(data.size()[0], -1)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        losses.append(loss.item())
        _str = "loss: " + str(np.round(np.mean(losses[-1000:]), 5))
        print(_str , end="\r")
        loss.backward()
        optimizer.step()
    print("Loss at end of epoch ", str(k+1), ": ", np.round(np.mean(losses[-1000:]), 5))