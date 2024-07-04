from torch.func import grad
from torch.utils.data import DataLoader
import numpy as np
import torch

def quad_loss(y_pred, y):
    # Loss function
    # The regularization enters at the level of the optimizer as the weight decay
    return torch.sum((y_pred-y)**2)/2


class Diffusion:
    def __init__(self, α, β, model_class, opt_gen, X_train, X_gen, N_steps, ntot, d, device, num_batches):
        '''
        model_class: d -> model
        opt_gen: model -> opt
        '''
        self.α = α
        self.β = β
        self.α_dot = grad(α)
        self.β_dot = grad(β)
        self.X_train = X_train 
        self.X_gen = X_gen
        self.t = 0.
        self.N_steps = N_steps
        self.dt = 1./N_steps
        self.num_batches = num_batches
        self.ntot = ntot
        self.model_class = model_class
        self.opt_gen = opt_gen
        self.d = d
        self.device = device
        self.losses = []

    def generate_data(self):
        self.X_train.gen_rand()
        self.X_train.set_t(self.t)
        self.X_train.gen_x1_xt()
        self.train_loader = DataLoader(self.X_train, batch_size = int(self.ntot))

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
        #if not hasattr(self, "model"):
        self.model = self.model_class(self.d).to(self.device)
        opt = self.opt_gen(self.model)
        for x_t, x_1 in self.train_loader:
            x_t, x_1 = x_t.to(self.device), x_1.to(self.device)
            x1_pred = self.model(x_t, self.t)
            loss = quad_loss(x1_pred, x_1)
            self.losses.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()

    def init_stats(self):
        self.summary = {"p": [], "M_t": [], "Mag":[], "Mag_std":[],"t":[],"Mag_ξ":[],"Mag_η":[], "Cosine":[],"Norm":[], "p": [], "M_t": [], "b":[], "Cos w":[], "Cos u": [], "Norm w": [], "Norm u": [], "Grad w": [], "Grad u": []}

    def stats(self):
        μ, σ, d = self.X_train.μ.numpy(), self.X_train.σ, self.X_train.d
        #ξ_tot, η_tot = self.X_train.ξ_tot, self.X_train.η_tot
        X = self.X_gen

        p    = np.mean(X@μ > 0)
        M_t  = X@μ/d
        Mt   = ((X.T*np.sign(X@μ)).T)@μ/d
        #M_ξ  = ((X.T*np.sign(X@μ)).T)@ξ_tot/d
        #M_η  = ((X.T*np.sign(X@μ)).T)@η_tot/d/σ
        X_   = (X.T*np.sign(X@μ)).T
        Simi = X_ @ μ/np.sqrt(d)/np.sqrt(np.sum(X_**2, 1))

        self.summary["p"].append(p)
        self.summary["M_t"].append(M_t)
        self.summary["Mag"].append(Mt.mean())
        self.summary["Mag_std"].append(Mt.std())
        self.summary["t"].append(self.t)
        #self.summary["Mag_ξ"].append(M_ξ.mean())
        #self.summary["Mag_η"].append(M_η.mean())
        self.summary["Cosine"].append(Simi.mean())
        self.summary["Norm"].append(np.sum(X_**2)/X_.shape[0]/d)
        if hasattr(self, "model"):
            if hasattr(self.model, "u"):
                u  = self.model.u.detach().cpu().numpy()
                #self.summary["Grad u"].append(self.model.u.grad.cpu().numpy())
                self.summary["Norm u"].append(torch.norm(self.model.u).detach().cpu().numpy())
                self.summary["Cos u"].append( (u@μ)/(μ@μ)**0.5/(u@u)**0.5 )
            if hasattr(self.model, "w"):
                w  = self.model.w.detach().cpu().numpy()
                #self.summary["Grad w"].append(self.model.w.grad.cpu().numpy())
                self.summary["Norm w"].append(torch.norm(self.model.w).detach().cpu().numpy())
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
            X_1_pred = self.model(torch.tensor(self.X_gen.astype(np.float32), device=self.device), self.t).cpu().numpy()
            r = 0 if α_t == 0 else α_dot_t/α_t
            v = (β_dot_t -  β_t * r) * X_1_pred + r * self.X_gen

        self.X_gen += v * self.dt
