import torch

class AE(torch.nn.Module):
    """
    f(x) = c*x + w * tanh( x'w/d + b )
    """

    def __init__(self, d):
        super(AE, self).__init__()
        self.d = d
        self.c=torch.nn.Parameter(torch.Tensor([1]))   # skip connection
        self.u=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.w=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.b=torch.nn.Parameter(torch.Tensor([0]))   # bias

    def forward(self, x):
        h=torch.tanh((x@self.w)/(self.d) + self.b)
        y_hat = h.reshape(x.shape[0],1)@self.u.reshape(1,self.d)
        y_hat += self.c*x
        return y_hat

class AE_0(torch.nn.Module):
    """
    f(x) = c*x + w*sign(x'w/sqrt(d))
    """

    def __init__(self, d):
        super(AE_0, self).__init__()
        self.d = d
        self.c=torch.nn.Parameter(torch.Tensor([1]))   # skip connection
        self.w=torch.nn.Parameter(torch.randn(self.d)) # network weight

    def forward(self, x):
        h = torch.sign(x@self.w/(self.d)**0.5)
        y_hat = h.reshape(x.shape[0],1)@self.w.reshape(1,self.d)
        y_hat += self.c*x
        return y_hat