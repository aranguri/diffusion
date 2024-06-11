# work in progress
from torch.func import grad
from torch.utils.data import DataLoader
import numpy as np
import torch

def quad_loss(y_pred, y):
    # Loss function
    # The regularization enters at the level of the optimizer as the weight decay
    return torch.sum((y_pred-y)**2)/2

# weight_decay_dict = {
#     'w': 1e-4,
#     'b': 1e-0,
#     'c': 0.
# }
# lr

class Diffusion:
    def __init__(self, α, β, model_class, X_train, X_gen, N_steps, epochs, ntot, d, device):
        self.α = α
        self.β = β
        self.α_dot = grad(α)
        self.β_dot = grad(β)
        self.X_train = X_train 
        self.X_gen = X_gen
        self.t = 0.
        self.N_steps = N_steps
        self.dt = 1./N_steps
        self.epochs = epochs
        self.ntot = ntot
        self.model_class = model_class
        self.d = d
        self.device = device
        self.losses = []

    def generate_data(self):
        self.X_train.set_t(self.t)
        self.X_train.gen_x1_xt()
        self.train_loader = DataLoader(self.X_train, batch_size = self.ntot)

    def run(self):
        self.init_stats()
        self.stats()
        for _ in range(self.N_steps):
            print(self.t)
            self.generate_data()
            self.train_step()
            self.generate_step()
            self.t_step()
            self.stats()

    def t_step(self):
        self.t += self.dt

    def train_step(self):
        self.model = self.model_class(self.d).to(self.device)
        opt = torch.optim.Adam([
            {'params': [self.model.w], 'weight_decay':1e-0},
            #{'params': [model.b], 'weight_decay':1e-0}, 
            {'params': [self.model.c], 'weight_decay':0.}]
        , lr=.04)

        for _ in range(self.epochs):
            for x_t, x_1 in self.train_loader:   # Optimization steps
                x1_pred = self.model(x_t)
                loss = quad_loss(x1_pred, x_1)
                self.losses.append(loss.detach().cpu().numpy())
                opt.zero_grad()
                loss.backward()
                opt.step()

    def init_stats(self):
        self.summary = {"Mag":[], "Mag_std":[],"t":[],"Mag_ξ":[],"Mag_η":[], "Cosine":[],"Norm":[], "p": [], "M_t": [], "b":[], "Cos w":[]}

    def stats(self):
        μ, σ, d = self.X_train.μ.numpy(), self.X_train.σ, self.X_train.d
        ξ_tot, η_tot = self.X_train.ξ_tot, self.X_train.η_tot
        X = self.X_gen

        p    = np.mean(X@μ > 0)
        M_t  = X@μ
        Mt   = ((X.T*np.sign(X@μ)).T)@μ/d
        M_ξ  = ((X.T*np.sign(X@μ)).T)@ξ_tot/d
        M_η  = ((X.T*np.sign(X@μ)).T)@η_tot/d/σ
        X_   = (X.T*np.sign(X@μ)).T
        Simi = X_ @ μ/np.sqrt(d)/np.sqrt(np.sum(X_**2, 1))

        self.summary["Mag"].append(Mt.mean())
        self.summary["Mag_std"].append(Mt.std())
        self.summary["t"].append(self.t)
        self.summary["Mag_ξ"].append(M_ξ.mean())
        self.summary["Mag_η"].append(M_η.mean())
        self.summary["Cosine"].append(Simi.mean())
        self.summary["Norm"].append(np.sum(X_**2)/X_.shape[0]/d)
        if hasattr(self, "model"):
            if hasattr(self.model, "w"):
                w  = self.model.w.detach().cpu().numpy()
                self.summary["Cos w"].append( (w@μ)/(μ@μ)**0.5/(w@w)**0.5 )
            if hasattr(self.model, "b"):
                b = self.model.b.detach().cpu().numpy()
                self.summary["b"].append(b)

    def generate_step(self):
        α_t = self.α(self.t)
        α_dot_t= self.α_dot(torch.tensor(self.t)).item()
        β_t = self.β(self.t)
        β_dot_t=self.β_dot(torch.tensor(self.t)).item()

        with torch.no_grad():
            X_1_pred = self.model(torch.tensor(self.X_gen.astype(np.float32))).numpy()
            r = 0 if α_t == 0 else α_dot_t/α_t
            v = (β_dot_t -  β_t * r) * X_1_pred + r * self.X_gen

        self.X_gen += v * self.dt
