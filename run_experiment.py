from diffusion import Diffusion
from data import Data
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from model import AE_4, AE_3, AE_2, AE_1, AE_0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(device)

ntot = 64     # number of samples
d = 512     # dimensions
num_batches = 4_000 # training
num_num_batches = 12
p = 0.8       # imbalance prob
N = 3_000      # generation size
std = d ** (1/2) # std of initial sample
N_steps=100   # number of "discretization" steps for ODE
# data
nrm = 1
σ = 1
μ = torch.ones(d)
model_class = AE_2

weight_decay_dict = {
    'w': 1e-1,
    #'u': 1e-1,
    #'b': 1e-2,
    #'c': 0.
}

def opt_gen(model):
    return torch.optim.Adam([{'params': [getattr(model,k)], 'weight_decay': v} for k, v in weight_decay_dict.items()], lr=.01)

α  = lambda t: (1-t) * std
β  = lambda t: t

X_train = Data(ntot,std,μ,σ,d,p,α,β,device, num_batches)
np.random.seed(1)
X_gen   = np.random.randn(N,d) * std

diffusion = Diffusion(α, β, model_class, opt_gen, X_train, copy.deepcopy(X_gen), N_steps, ntot, d, device, num_batches, num_num_batches)
diffusion.run()

import datetime
import pickle 
date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
with open(f'{date}.pkl', 'wb') as outp:
    pickle.dump((ntot, d, num_batches * num_num_batches, diffusion.summary), outp, pickle.HIGHEST_PROTOCOL)
