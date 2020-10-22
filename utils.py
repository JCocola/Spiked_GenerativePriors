import numpy as np

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients

from mpl_toolkits.mplot3d import Axes3D
import random

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F



# Freeze weigths 
def freeze_weights(net):
    for param in net.parameters():
        param.requires_grad = False
        
    
# Count number of trainable parameters
def count_parameters(model, verbosity = 0):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if(verbosity):
                if param.dim() > 1:
                    print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                else:
                    print(name, ':', num_param)
            total_param += num_param
    return total_param

# Normalize
def normalize(X, mean = torch.empty(0,0), std = torch.empty(0,0)):
    if(mean.nelement() == 0):
        mean = X.mean(0)    
    if(std.nelement() == 0):
        std = X.std(0) 
        
    Y = (X - mean)/std
    Y[Y != Y] = 0 
    return [Y,mean,std]
