import torch

class AE_4(torch.nn.Module):
    """
    f(x) = c*x + u * tanh( x'w/d + b )
    """

    def __init__(self, d, t):
        super(AE_4, self).__init__()
        self.d = d
        self.c=torch.nn.Parameter(torch.Tensor([1]))   # skip connection
        self.w=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.b=torch.nn.Parameter(torch.Tensor([0]))   # bias
        self.t=t

    def forward(self, x):
        #mu = [1] * self.d
        coef = self.d * (1 - self.t**2) / (self.d * (1 - self.t)**2 + self.t**2)
        h = torch.tanh((x@self.w)/(self.d)  + self.b)
        #print(h.shape)
        mu = torch.ones((self.d, 1), device=h.get_device())
        y_hat = (coef * mu @ h.reshape(1, h.shape[0])).T
        #print(x.shape)
        y_hat += self.c*x
        return y_hat

class AE_3(torch.nn.Module):
    """
    f(x) = c*x + w k tanh( x'w/d + b )
    """

    def __init__(self, d, t):
        super(AE_3, self).__init__()
        self.d = d
        self.c=torch.nn.Parameter(torch.Tensor([1]))   # skip connection
        self.k=torch.nn.Parameter(torch.Tensor([1])) # network weight
        self.w=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.b=torch.nn.Parameter(torch.Tensor([0]))   # bias
        self.t=t

    def forward(self, x):
        h = self.k * torch.tanh((x@self.w)/(self.d) + self.b)
        y_hat = h.reshape(x.shape[0],1) @ self.w.reshape(1,self.d)
        y_hat += self.c*x
        return y_hat

class AE_2(torch.nn.Module):
    """
    f(x) = c*x + u * tanh( x'w/d + b )
    """

    def __init__(self, d, t):
        super(AE_2, self).__init__()
        self.d = d
        self.c=torch.nn.Parameter(torch.Tensor([1]))   # skip connection
        #self.u=torch.nn.Parameter(torch.Tensor([1])) # network weight
        self.u=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.w=torch.nn.Parameter(torch.randn(self.d)) # network weight
        self.b=torch.nn.Parameter(torch.Tensor([0]))   # bias
        self.t=t
        #self.b=torch.nn.Parameter(torch.randn(1) * .01)   # bias

    def forward(self, x):
        #mu = [1] * self.d
        coef = self.t / (self.d * (1 - self.t)**2 + self.t**2)
        #h=torch.tanh((x@self.w)/(self.d) + self.b)
        #print(x.shape)
        h=torch.tanh(coef * (torch.sum(x, axis=1)) + self.b)
        y_hat = h.reshape(x.shape[0],1)@self.w.reshape(1,self.d)
        y_hat += self.c*x
        return y_hat

class AE_1(torch.nn.Module):
    """
    f(x) = c*x + u * tanh( x'w/d + b )
    """

    def __init__(self, d, t):
        super(AE_1, self).__init__()
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