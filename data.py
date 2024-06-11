def get_x1(n,μ,σ):
    # constructs x(1)
    x_1=s.reshape(n,1)@μ.reshape(1,d)
    x_1+=η*σ
    return x_1

def get_x1_xt(n,μ,σ,t):
    # constructs x(1), x(t)
    x_1=get_x1(n,μ,σ)
    x_t=ξ*α(t)+ x_1*β(t)
    return x_t.to(device), x_1.to(device)

class GenerateData(Dataset):
    # data loader object
    def __init__(self,n,μ,σ,t):
        self.μ=μ
        self.σ=σ
        self.t=t
        self.n=n
        self.X_t,self.X_1=get_x1_xt(n,μ,σ,t)

    def __getitem__(self,idx):
        return self.X_t[idx].to(device),self.X_1[idx].to(device)

    def __len__(self):
        return self.n