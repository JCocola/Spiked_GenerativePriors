########### Modules  ################
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import utils



######### Deep Random Nets ################

# 1 layer Generative  network
class Generative_1L(nn.Module):
    def __init__(self, input_dim, output_dim, bias = False):
        super(Generative_1L, self).__init__() 
        self.W1 = nn.Linear(input_dim, output_dim, bias = bias)
        self.num_parameters = utils.count_parameters(self, verbosity=0)
                    
    def forward(self, x):        
        x = F.relu(self.W1(x));
        return x
        
    
# 2 layer Generative network
class Generative_2L(nn.Module):
    def __init__(self, input_dim, hid_dim1, output_dim, bias = False):

        super(Generative_2L, self).__init__() 
        self.W1 = nn.Linear(input_dim, hid_dim1, bias = bias)
        self.W2 = nn.Linear(hid_dim1, output_dim, bias = bias)
        self.num_parameters = utils.count_parameters(self, verbosity=0)
                
    def forward(self, x):        
        x = F.relu(self.W1(x));
        x = F.relu(self.W2(x));
        
        return x
        
# 3 layer Generative network
class Generative_3L(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, output_dim, bias = False):

        super(Generative_3L, self).__init__() 
        self.W1 = nn.Linear(input_dim, hid_dim1, bias = bias)
        self.W2 = nn.Linear(hid_dim1, hid_dim2, bias = bias)
        self.W3 = nn.Linear(hid_dim2, output_dim, bias = bias)
        self.num_parameters = utils.count_parameters(self, verbosity=0)
                
    def forward(self, x):        
        x = F.relu(self.W1(x));
        x = F.relu(self.W2(x));
        x = F.relu(self.W3(x));
        
        return x
    
# 3 layer Generative network
class Generative_4L(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, hid_dim3, output_dim, bias = False):

        super(Generative_4L, self).__init__() 
        self.W1 = nn.Linear(input_dim, hid_dim1, bias = bias)
        self.W2 = nn.Linear(hid_dim1, hid_dim2, bias = bias)
        self.W3 = nn.Linear(hid_dim2, hid_dim3, bias = bias)
        self.W4 = nn.Linear(hid_dim3, output_dim, bias = bias)
        self.num_parameters = utils.count_parameters(self, verbosity=0)
                
    def forward(self, x):        
        x = F.relu(self.W1(x));
        x = F.relu(self.W2(x));
        x = F.relu(self.W3(x));
        x = F.relu(self.W4(x));
        
        return x
    
    
    
    
# Initialize weigths as paper
def init_weights(m):
    if type(m) == nn.Linear:
        sigma = 2./np.sqrt(m.weight.data.shape[0]);
        m.weight.data.data.normal_(0, sigma)
        
        
        
        
        
        
# Create Random Generative Networks with multiple layers
def createRandGenNet(layers, bias = False, printNparam = False, Freeze = True):
    input_dim  = layers[0];
    output_dim = layers[-1];
    
    # create net
    if (len(layers) == 2):
        GenNet = Generative_1L(input_dim, output_dim, bias);
    elif (len(layers) == 3):
        GenNet = Generative_2L(input_dim, layers[1], output_dim, bias);
    elif (len(layers) == 4):
        GenNet = Generative_3L(input_dim, layers[1], layers[2], output_dim, bias);
    elif (len(layers) == 5):
        GenNet = Generative_4L(input_dim, layers[1], layers[2], layers[3], output_dim, bias);
    else:
        raise Exception('Code for at-most-four-layer generative networks')

    GenNet.apply(init_weights); #initialize weigths
            
    if(printNparam):
        print(GenNet)
        print('Number of parameters ', utils.count_parameters(GenNet, verbosity=0))
    
    if(Freeze):
        utils.freeze_weights(GenNet); #freeze weigths
    
    return GenNet
