# work in progress


def quad_loss(y_pred, y):
    # Loss function
    # The regularization enters at the level of the optimizer as the weight decay
    return torch.sum((y_pred-y)**2)/2

class Diffusion:
    def __init__(self, α, β, generate_train_data, model, optimizer, X):
        self.α = α
        self.β = β
        self.α_dot = grad(α)
        self.β_dot = grad(β)
        self.generate_train_data = generate_train_data
        self.model = model
        self.optimizer = optimizer
        self.X = X
        self.t = 0

    def generate_data(self, ntot):
        X_train = generate_train_data(ntot, self.t)
        self.train_loader=DataLoader(X_train, batch_size = ntot)

    def train(self):
        ae=model().to(device)
        opt=optimizer(model)

        losses = []
        for _ in range(epochs):
            for x_t,x_1 in self.train_loader:   # Optimization steps
                x1_pred = ae(x_t)
                loss = quad_loss(x1_pred,x_1)
                losses.append(loss.detach().cpu().numpy())
                opt.zero_grad()
                loss.backward()
                opt.step()

    def velocity(self):
        α_t = self.α(t)
        α_dot_t= self.α_dot(torch.tensor(self.t)).item()
        β_t = self.β(self.t)
        β_dot_t=self.β_dot(torch.tensor(self.t)).item()

        with torch.no_grad():
            X_1_pred = ae(torch.tensor(self.X.astype(np.float32))).numpy()
            r = 0 if α_t == 0 else α_dot_t/α_t
            v = (β_dot_t -  β_t * r ) * X_1_pred + r * self.X

        return v

    def step(self):
        self.X += self.velocity() * dt