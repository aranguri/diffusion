from torch.utils.data import Dataset
import torch

class Data(Dataset):
    # data loader object
    def __init__(self,n,μ,σ,d,p,α,β,device):
        self.μ=μ
        self.σ=σ
        self.n=n
        self.d=d
        self.p=p
        self.device=device
        self.β = β
        self.α = α
        self.t=0

    def gen_rand(self):
        gen0 = torch.Generator().manual_seed(42)
        # x_0^μ. 
        self.ξ=torch.randn(self.n,self.d,generator=gen0)
        gen1 = torch.Generator().manual_seed(67)
        # z^μ
        self.η=torch.randn(self.n,self.d,generator=gen1)
        gen3 = torch.Generator().manual_seed(4122)
        # s^μ
        self.s=torch.sign(2.*(torch.rand(self.n,generator=gen3)[:self.n] < self.p) -1 ) 

        # useful quantities
        self.ξ_tot=torch.sum(self.ξ.T*self.s,1).flatten()/self.n # ξ vector
        self.η_tot=torch.sum(self.η.T*self.s,1).flatten()/self.n # η vector
        self.ξ_tot=self.ξ_tot.numpy()
        self.η_tot=self.η_tot.numpy()
        self.mu=self.μ.numpy()

    def set_t(self, t):
        self.t = t

    def gen_x1_xt(self):
        # constructs x(1)
        x_1=self.s.reshape(self.n,1)@self.μ.reshape(1,self.d)
        x_1+=self.η*self.σ
        # constructs x(t)
        x_t=self.ξ* self.α(self.t)+ x_1*self.β(self.t)
        self.X_t,self.X_1 = x_t.to(self.device), x_1.to(self.device)

    def __getitem__(self,idx):
        return self.X_t[idx].to(self.device),self.X_1[idx].to(self.device)

    def __len__(self):
        return self.n