from torch.utils.data import Dataset
import torch

class Data(Dataset):
    # data loader object
    def __init__(self,n,std,μ,σ,d,p,α,β,device,num_batches):
        self.μ=μ
        self.std=std
        self.σ=σ
        self.n=n
        self.d=d
        self.p=p
        self.device=device
        self.β = β
        self.α = α
        self.t=0
        self.num_batches=num_batches

    def gen_rand(self):
        # x_0^μ. 
        self.ξ=torch.randn(self.n * self.num_batches, self.d)
        
        # z^μ
        gen1 = torch.Generator().manual_seed(1)
        self.η=torch.randn(self.n,self.d,generator=gen1)
        
        # s^μ
        gen3 = torch.Generator().manual_seed(2)
        self.s=torch.sign(2.*(torch.rand(self.n,generator=gen3)[:self.n] < self.p) -1 ) 

        # useful quantities
        #self.ξ_tot=torch.sum(self.ξ.T*self.s,1).flatten()/self.n # ξ vector
        #self.η_tot=torch.sum(self.η.T*self.s,1).flatten()/self.n # η vector
        #self.ξ_tot=self.ξ_tot.numpy()
        #self.η_tot=self.η_tot.numpy()
        self.mu=self.μ.numpy()

    def set_t(self, t):
        self.t = t

    def gen_x1_xt(self):
        # constructs x(1)
        x_1 = self.s.reshape(self.n,1)@self.μ.reshape(1,self.d)
        x_1 += self.η*self.σ
        x_1 = torch.tile(x_1, (self.num_batches, 1))
        # constructs x(t)
        x_t = self.ξ * self.α(self.t) + x_1 * self.β(self.t)
        self.X_t, self.X_1 = x_t.to(self.device), x_1.to(self.device)

    def __getitem__(self,idx):
        return self.X_t[idx].to(self.device),self.X_1[idx].to(self.device)

    def __len__(self):
        return self.n