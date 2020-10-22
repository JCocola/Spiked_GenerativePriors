import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import math
import numpy as np
import generative as gnl


### Functions for the definition of the spiked model


# Generate data matrix X for the spiked model N(0, G(z)G(z)^t + \sigma^2*I)
def gen_spikeX(net, orig_z, N = 50, sigma = 1.):
    '''
    sigma = noise level
    N    = number of samples
    net  = random generative model
    z0   = latent code
    '''
    u = torch.randn([N]).cuda()
    v = net(orig_z).flatten().cuda()
    Z = torch.randn([N,len(v)]).cuda()
    return torch.ger(u,v)+ sigma*Z

# Compute covariance matrix from the data matrix X
def covMatr(X):
    return torch.mm(X.t(),X)/X.shape[0]

# Compute the empirical covariance matrix of a spiked model
def spike_Covn(net, orig_z, N = 50, sigma = 1.):
    X = gen_spikeX(net, orig_z, N, sigma)
    return covMatr(X)
    

### Loss functions
def angle_loss(x,y):
    return torch.abs(torch.dot(x.flatten(),y.flatten()))
    
### Loss functions
def angle_loss2(x,y):
    return torch.abs(torch.dot(x,y))

def l2_loss(x,y):
    return (x-y).norm(2)

def frob_loss(x,Y):
    X = torch.ger(x,x)
    return (torch.norm(X-Y, p = 'fro')**2)/4



### Functions for optimization and recovery 
# single run used for tuning params
def latent_GD(net, Ytr, Yts, latent_param, loss_func, ystar,  learning_rate=0.1, num_steps=500):
    # We fit Ytr and check performance on Yts
    # solve min_z loss_func(net(z),Ytr)) using GD

    optimizer = torch.optim.SGD([latent_param], lr=learning_rate) 
    iter_idx = 0
    LOGS = {}; 
    LOGS['train_loss'] = np.zeros(num_steps); 
    LOGS['test_loss'] = np.zeros(num_steps); 
    LOGS['angle_loss'] = np.zeros(num_steps);
    LOGS['l2_loss'] = np.zeros(num_steps);
    ystar_nrm = ystar/ystar.norm(2)
        
    while iter_idx < num_steps:
        # GD steps
        optimizer.zero_grad()
        out = net(latent_param)
        loss_tr = loss_func(out, Ytr)
        loss_tr.backward()
        optimizer.step()
                           
        # Logs
        loss_ts = loss_func(out, Yts)
        LOGS['train_loss'][iter_idx]=(loss_tr.cpu().data.numpy().item())
        LOGS['test_loss'][iter_idx]=(loss_ts.cpu().data.numpy().item())
        LOGS['l2_loss'][iter_idx]=(l2_loss(ystar,out).cpu().data.numpy().item())
        Gzk_nrm = out/out.norm()
        LOGS['angle_loss'][iter_idx]=(angle_loss(Gzk_nrm, ystar_nrm).cpu().data.numpy().item())
        
        if(math.isnan(LOGS['train_loss'][-1])):
            print('NaN value encountered')
            break
        
        iter_idx += 1
        
    return [latent_param, LOGS]

    
##########################################################################
##########################################################################
### Functions for optimization and recovery 

def single_GD_expr(net, Ytr, latent_param, loss_func, ystar, learning_rate=0.1, num_steps=500):

    optimizer = torch.optim.SGD([latent_param], lr=learning_rate) 
    iter_idx = 0
    ystar_nrm = ystar.norm(2)

    while iter_idx < num_steps:
        # GD steps
        optimizer.zero_grad()
        out = net(latent_param)
        loss_tr = loss_func(out, Ytr)        
        loss_tr.backward()
        optimizer.step()
        iter_idx += 1
        
        
    # vector of logs
    LOGS = np.zeros(4); 
    #train_loss 
    LOGS[0] = loss_tr.cpu().data.numpy().item(); 
    # l2_loss
    LOGS[1] = l2_loss(ystar,out).cpu().data.numpy().item();
    #angle_loss 
    Gzk_nrm = out/out.norm(2)
    LOGS[2] = angle_loss(Gzk_nrm, ystar/ystar_nrm).cpu().data.numpy().item();
    # l2_normalized
    LOGS[3] = (l2_loss(ystar,out)/ystar_nrm).cpu().data.numpy().item()

    return LOGS, latent_param
    

def GD_expr(net, Ytr, latent_dim, loss_func, ystar,  learning_rate=0.1, num_steps=500, sigma0 = .1):

	# defining the two starting points
    z0 = sigma0*torch.randn(latent_dim).cuda()
    z01 = Variable(z0, requires_grad=True);
    z02 = Variable(-z0[:], requires_grad=True);
        
    LOGS1, latent_param1 = single_GD_expr(net, Ytr, z01, loss_func, ystar,  learning_rate=learning_rate, num_steps=num_steps)
    
    LOGS2, latent_param2 = single_GD_expr(net, Ytr, z02, loss_func, ystar,  learning_rate=learning_rate, num_steps=num_steps)
    
    if(LOGS1[0] < LOGS2[0]):
        return LOGS1
    else:
        return LOGS2


# Perform experiments for the Wigner model
def MC_expr_Wigner(NMC, zstar, layers, nu, params, normalize = True):

    LOGS = np.zeros([NMC, 6]);
    # we record 0) Train loss 1) l2_loss 3) angle_loss 4) ystar.norm 5) M.norm
    
    [lr, nsteps, sigma0] = params;

    for i in range(NMC):
        GenNet = gnl.createRandGenNet(layers = layers, bias = False, Freeze = True).cuda()
        ystar = GenNet(zstar);
        if(normalize):
            alpha = ystar.norm(2);
            zhat  = zstar/alpha;
            ystar = GenNet(zhat);

		#Wigner model
        yyT = torch.ger(ystar,ystar)
        W   = torch.randn([layers[-1],layers[-1]])
        H   = (W.t() + W.t())/np.sqrt(2*layers[-1])
        Ytr = yyT + nu*H.cuda()
        
        # train loss, l2 loss, angle loss, normalized MSE
        LOGS[i,:4] = GD_expr(GenNet, Ytr, layers[0], frob_loss, ystar, learning_rate=lr, num_steps=nsteps, sigma0 = sigma0)
        # ystar.norm
        LOGS[i,4] = ystar.norm().cpu().item()
        # M.norm
        LOGS[i,5] = Ytr.norm().cpu().item()
        
        LOGS_mean = LOGS.mean(axis = 0)
        LOGS_std  = LOGS.std(axis = 0)

    return [LOGS_mean, LOGS_std]


        
# perform experiments for the Wishart model
def MC_expr_PCA(NMC, zstar, layers, Nsamples, params, normalize = True):

    LOGS = np.zeros([NMC, 6]);
    # we record 0) Train loss 1) l2_loss 3) angle_loss 4) ystar.norm 5) M.norm
    
    [lr, nsteps, sigma, sigma0] = params;

    for i in range(NMC):
        GenNet = gnl.createRandGenNet(layers = layers, bias = False, Freeze = True).cuda()
        ystar = GenNet(zstar);
        if(normalize):
            zstar  = zstar/ystar.norm(2);
            ystar = GenNet(zstar);
            
        # empirical covariance matrix
        SigmaN = spike_Covn(GenNet, zstar, Nsamples, sigma);
        # empirical covariance minus noise part
        M = SigmaN - (sigma**2)*torch.eye(len(ystar)).cuda();
        # train loss, l2 loss, angle loss, normalized MSE
        LOGS[i,:4] = GD_expr(GenNet, M, layers[0], frob_loss, ystar, learning_rate=lr, num_steps=nsteps)
        # ystar.norm
        LOGS[i,4] = ystar.norm().cpu().item()
        # M.norm
        LOGS[i,5] = M.norm().cpu().item()
        
        LOGS_mean = LOGS.mean(axis = 0)
        LOGS_std  = LOGS.std(axis = 0)

    return [LOGS_mean, LOGS_std]
   