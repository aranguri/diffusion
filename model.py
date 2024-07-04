import torch

class AE_4(torch.nn.Module):
    """
    f(x) = c*x + u * tanh( x'w/d + b )
    """

    def __init__(self, d):
        super(AE_4, self).__init__()
        self.d = d
        self.c=torch.nn.Parameter(torch.Tensor([1]))   # skip connection
        self.w=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.b=torch.nn.Parameter(torch.Tensor([0]))   # bias

    def forward(self, x, t):
        c = t / (self.d * (1-t) ** 2 + t ** 2)
        coef1 = t / (self.d * (1 - t) ** 2 + t ** 2)
        h = torch.tanh(coef1 * torch.sum(x, axis=1) + self.b)#.693)
        coef2 = self.d * (1 - t)**2 / (self.d * (1 - t)**2 + t**2)
        
        mu = torch.ones((self.d, 1), device=h.get_device())
        y_hat = (coef2 * mu @ h.reshape(1, h.shape[0])).T
        y_hat += c*x + 0*torch.sum(self.w)
        return y_hat

class AE_3(torch.nn.Module):
    """
    learns u and b
    f(x) = c*x + u * tanh( x'w/d + b )
    """

    def __init__(self, d):
        super(AE_3, self).__init__()
        self.d = d
        self.c=torch.nn.Parameter(torch.Tensor([1]))   # skip connection
        self.u=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.b=torch.nn.Parameter(torch.Tensor([0]))   # bias

    def forward(self, x, t):
        c = t / (self.d * (1-t) ** 2 + t ** 2)
        coef1 = t / (self.d * (1 - t) ** 2 + t ** 2)
        h = torch.tanh(coef1 * torch.sum(x, axis=1) + self.b)
        
        mu = torch.ones((self.d, 1), device=h.get_device())
        y_hat = h.reshape(x.shape[0],1)@self.u.reshape(1,self.d)
        y_hat += c*x 
        return y_hat

class AE_2(torch.nn.Module):
    def __init__(self, d):
        super(AE_2, self).__init__()
        self.d = d
        #self.c=torch.nn.Parameter(torch.Tensor([1]))   # skip connection
        self.w=torch.nn.Parameter(torch.randn(self.d)) # network weight
        #self.u=torch.nn.Parameter(torch.randn(self.d)) # network weight
        #self.b=torch.nn.Parameter(torch.Tensor([0]))   # bias

    def forward(self, x, t):
        c = t / (self.d * (1-t) ** 2 + t ** 2)
        h = torch.tanh(x@self.w/self.d + .693)
        
        coef2 = self.d * (1 - t) ** 2 / (self.d * (1 - t)**2 + t**2)
        mu = torch.ones((self.d, 1), device=h.get_device())
        y_hat = (coef2 * mu @ h.reshape(1, h.shape[0])).T
        y_hat += c*x 
        return y_hat

class AE_1(torch.nn.Module):
    """
    f(x) = c*x + u * tanh( x'w/d + b )
    """

    def __init__(self, d):
        super(AE_1, self).__init__()
        self.d = d
        self.c=torch.nn.Parameter(torch.Tensor([1]))   # skip connection
        self.u=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.w=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.b=torch.nn.Parameter(torch.Tensor([0]))   # bias

    def forward(self, x):
        h=torch.tanh((x@self.w)/(self.d) + .693)#self.b)
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