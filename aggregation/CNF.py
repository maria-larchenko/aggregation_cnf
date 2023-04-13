import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

seed = 2346  # np.random.randint(10_000)
torch.manual_seed(seed)


class ConditionalAffineTransformer(nn.Module):

    def __init__(self, input_dim, condition_dim, split, hidden=512, T=1):
        super().__init__()
        self.split = split
        self.T = T
        # self.constrain = torch.nn.functional.softplus
        self.constrain = torch.exp
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, input_dim),
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, input_dim),
        )

    # masks zeroing (x, y)-input for 2-input NNs seemingly lead to vanishing grads.
    # replacing 2-input 2-output NNs with 1-input/output solved the problem.

    # def forward(self, x, mask=None):
    #     if mask is None:
    #         mask = self.mask
    #     scale = self.scale_net(mask * x)
    #     trans = self.translation_net(mask * x)
    #     y = mask * x + (1 - mask) * (x * torch.exp(scale) + trans)  # affine
    #     return y, scale, trans
    #
    # def inverse(self, y, mask=None):
    #     if mask is None:
    #         mask = self.mask
    #     scale = self.scale_net(mask * y)
    #     trans = self.translation_net(mask * y)
    #     x = mask * y + (1 - mask) * (y - trans) * torch.exp(-scale)  # affine
    #     return x, scale, trans

    def forward(self, x, condition, alternate=False):
        x1 = x[:, :self.split]
        x2 = x[:, self.split:]
        if alternate:
            xc = torch.cat((x1, condition), 1)
            scale = self.constrain(self.scale_net(xc) / self.T)
            trans = self.translation_net(xc)
            y1 = x1
            y2 = x2 * scale + trans
        else:
            xc = torch.cat((x2, condition), 1)
            scale = self.constrain(self.scale_net(xc) / self.T)
            trans = self.translation_net(xc)
            y1 = x1 * scale + trans
            y2 = x2
        y = torch.cat([y1, y2], 1)
        return y, scale, trans

    def inverse(self, y, condition, alternate=False):
        y1 = y[:, :self.split]
        y2 = y[:, self.split:]
        if alternate:
            yc = torch.cat((y1, condition), 1)
            scale = self.constrain(self.scale_net(yc) / self.T)
            trans = self.translation_net(yc)
            x1 = y1
            x2 = (y2 - trans) / scale
        else:
            yc = torch.cat((y2, condition), 1)
            scale = self.constrain(self.scale_net(yc) / self.T)
            trans = self.translation_net(yc)
            x1 = (y1 - trans) / scale
            x2 = y2
        x = torch.cat([x1, x2], 1)
        return x, scale, trans


class ConditionalRealNVP(nn.Module):
    def __init__(self, in_dim, cond_dim, layers=4, hidden=512, T=1, cond_prior=False, device=None, replace_nan=False):
        super().__init__()
        self.layers = layers
        self.hidden = hidden
        self.input_dim = in_dim
        self.condition_dim = cond_dim
        self.split = 1  # hardcoded for in, out = 1, 1
        self.T=T
        ## masks zeroing (x, y)-input for 2-input NNs seemingly lead to vanishing grads.
        ## replacing 2-input 2-output NNs with 1-input/output solved the problem.
        # self.masks = [
        #     torch.Tensor([1, 0]) if i % 2 == 0 else torch.Tensor([0, 1]) for i in range(0, layers)
        # ]
        self.alternate = [ i % 2 == 0 for i in range(0, layers)
        ]
        # to register submodules params ModuleList is required instead of list
        self.transformers = nn.ModuleList([
            ConditionalAffineTransformer(in_dim, cond_dim, self.split, hidden, T=T)
            for _ in range(0, layers)
        ])
        self.replace_nan = replace_nan
        self.cond_prior = cond_prior
        if cond_prior:
            self.prior_mean_net = nn.Sequential(
                nn.Linear(1, hidden),
                nn.LeakyReLU(),
                nn.Linear(hidden, self.input_dim),
            )
            self.prior_scale_net = nn.Sequential(
                nn.Linear(1, hidden),
                nn.LeakyReLU(),
                nn.Linear(hidden, 1),
            )
        self.noise = None
        if device is not None:
            self.device = device
            self.to(device)

    def cond_base_dist(self, cond):
        base_mu = torch.zeros(2, device=self.device)  # hardcoded for 2D
        base_cov = torch.eye(2, device=self.device)
        if self.cond_prior:
            mean = self.prior_mean_net(cond)[0]
            scale = torch.exp(self.prior_scale_net(cond))[0]
            base_mu = base_mu + mean
            base_cov = base_cov * scale
        base_dist = MultivariateNormal(base_mu, base_cov)
        return base_dist

    def forward(self, x, cond):
        log_determinant = 0
        for i in range(0, self.layers):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            x, scale, _ = transformer(x, cond, alternate)
            log_determinant += torch.log(scale).sum(1)
        log_likelihood = self.cond_base_dist(cond).log_prob(x)  # log likelihood of sample under the base measure
        model_loss = - (log_determinant + log_likelihood).mean()
        return x, model_loss

    def forward_x(self, x, cond):
        log_determinant = 0
        for i in range(0, self.layers):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            x, scale, _ = transformer(x, cond, alternate)
            log_determinant += torch.log(scale).sum(1)
        nans = torch.isnan(x)
        infs = torch.isinf(x)
        log_pdf = torch.zeros((len(x)))
        if not torch.any(nans):
            log_likelihood = self.cond_base_dist(cond).log_prob(x)  # log likelihood of sample under the base measure
            log_pdf = log_determinant + log_likelihood
        elif self.replace_nan:
            print(f"model.forward_x:  NAN -> INF for {torch.count_nonzero(infs)} points")
            inf = torch.tensor(torch.inf, dtype=torch.float32)
            x = torch.where(torch.isnan(x), inf, x)
            log_likelihood = self.cond_base_dist(cond).log_prob(x)  # log likelihood of sample under the base measure
            log_pdf = log_determinant + log_likelihood
        return x, log_pdf, nans, infs

    def inverse(self, batch_size, cond):
        base_dist = self.cond_base_dist(cond)
        y = base_dist.rsample((batch_size,))
        self.noise = y
        log_likelihood = base_dist.log_prob(y)  # log likelihood of noise under the base measure
        log_determinant = 0
        for i in reversed(range(0, self.layers)):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            y, scale, _ = transformer.inverse(y, cond, alternate)
            log_determinant += torch.log(scale).sum(1)
        model_loss = - (log_determinant - log_likelihood).mean()
        return y, model_loss

    def inverse_z(self, z, cond):
        base_dist = self.cond_base_dist(cond)
        log_likelihood = base_dist.log_prob(z)  # log likelihood of noise under the base measure
        log_determinant = 0
        for i in reversed(range(0, self.layers)):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            z, scale, _ = transformer.inverse(z, cond, alternate)
            log_determinant += torch.log(scale).sum(1)
        log_pdf = log_determinant - log_likelihood
        return z, log_pdf

