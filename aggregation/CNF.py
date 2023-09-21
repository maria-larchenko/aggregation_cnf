import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from aggregation.Static import external_prior_llk

seed = 2346  # np.random.randint(10_000)
torch.manual_seed(seed)


class ConditionalAffineTransformer(nn.Module):

    def __init__(self, input_dim, condition_dim, split, hidden=512, T=1, activation=nn.ReLU):
        super().__init__()
        self.split = split
        self.T = T
        self.hidden = hidden
        self.activation = activation
        self.constrain = torch.exp  #torch.nn.functional.softplus
        # # tried for 512 and 2024 hidden and 6-10 trans. layers, not ok
        # self.scale_net = nn.Sequential(
        #     nn.Linear(input_dim + condition_dim, hidden),
        #     self.activation(),
        #     nn.Linear(hidden, hidden),
        #     self.activation(),
        #     nn.Linear(hidden, input_dim),
        # )
        # self.translation_net = nn.Sequential(
        #     nn.Linear(input_dim + condition_dim, hidden),
        #     self.activation(),
        #     nn.Linear(hidden, hidden),
        #     self.activation(),
        #     nn.Linear(hidden, input_dim),
        # )
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden),   # for 32 layers and 20 hidden
            self.activation(),
            nn.Linear(hidden, hidden),
            self.activation(),
            nn.Linear(hidden, hidden),
            self.activation(),
            nn.Linear(hidden, input_dim),
            nn.Tanh(),
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden),
            self.activation(),
            nn.Linear(hidden, hidden),
            self.activation(),
            nn.Linear(hidden, hidden),
            self.activation(),
            nn.Linear(hidden, input_dim),
            nn.Tanh(),
        )
        self.layers = int(len(self.scale_net) / 2)
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


class ConditionalRealNVP2D(nn.Module):
    def __init__(self, cond_dim, layers=4, hidden=512, activation=nn.ReLU, T=1, external_prior=False, device=None, replace_nan=False, ):
        super().__init__()
        self.layers = layers
        self.hidden = hidden
        self.input_dim = 1  # hardcoded for 2D
        self.split = 1  # hardcoded for 2D
        self.condition_dim = cond_dim
        self.external_prior = external_prior
        self.T=T
        self.alternate = [ i % 2 == 0 for i in range(0, layers)
        ]
        # to register submodules params ModuleList is required instead of list
        self.transformers = nn.ModuleList([
            ConditionalAffineTransformer(self.input_dim, cond_dim, self.split, hidden, T=T, activation=activation)
            for _ in range(0, layers)
        ])
        self.hidden_layers = int( len(self.transformers[0].scale_net) / 2)
        self.replace_nan = replace_nan
        self.noise = None
        if device is not None:
            self.device = device
            self.to(device)
            base_mu = torch.ones(2, device=device) # hardcoded for 2D
            base_cov = torch.eye(2, device=device)
        else:
            base_mu = torch.zeros(2)
            base_cov = torch.eye(2)
        self.base_dist = MultivariateNormal(base_mu, base_cov)
        i = str(activation).rfind('activation.') + len('activation.')
        self.activation_str = str(activation)[i:-1]

    def set_temperature(self, T):
        for transformer in self.transformers:
            transformer.T = T
        self.T = T

    def forward(self, x, cond, base_llk_lst=None, xedges=None):
        log_determinant = 0
        for i in range(0, self.layers):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            x, scale, _ = transformer(x, cond, alternate)
            log_determinant += torch.log(scale).sum(1)
        if base_llk_lst is None:
            assert not self.external_prior
            log_likelihood = self.base_dist.log_prob(x).mean()  # log likelihood of sample under the base measure
        else:
            assert self.external_prior
            xx = x.detach().cpu().numpy()
            prior_llk = external_prior_llk(xx, base_llk_lst, xedges)
            log_likelihood = np.mean(prior_llk)
        model_loss = - (log_determinant.mean() + log_likelihood)
        return x, model_loss

    def forward_x(self, x, cond, base_llk_lst=None, xedges=None):
        """returns log pdf for every (x, cond) input, for visualization"""
        log_determinant = 0
        for i in range(0, self.layers):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            x, scale, _ = transformer(x, cond, alternate)
            log_determinant += torch.log(scale).sum(1)
        nans = torch.isnan(x)
        infs = torch.isinf(x)
        if torch.any(nans) and self.replace_nan:
            print(f"model.forward_x:  NAN -> INF for {torch.count_nonzero(infs)} points")
            inf = torch.tensor(torch.inf, dtype=torch.float32, device=self.device)
            x = torch.where(torch.isnan(x), inf, x)
        if base_llk_lst is None:
            assert not self.external_prior
            log_likelihood = self.base_dist.log_prob(x)  # log likelihood of sample under the base measure
        else:
            assert self.external_prior
            xx = x.detach().cpu().numpy()
            log_likelihood = external_prior_llk(xx, base_llk_lst, xedges)
            log_likelihood = torch.tensor(log_likelihood, dtype=torch.float32, device=self.device)
        log_pdf = log_determinant + log_likelihood
        return x, log_pdf, nans, infs

    def inverse(self, sample_size, cond, compute_pdf=False):
        assert not self.external_prior
        z = self.base_dist.rsample((sample_size,))
        log_likelihood = self.base_dist.log_prob(z) if compute_pdf else 0
        self.noise = z.detach().cpu()
        log_determinant = 0
        for i in reversed(range(0, self.layers)):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            z, scale, _ = transformer.inverse(z, cond, alternate)
            log_determinant += torch.log(scale).sum(1) if compute_pdf else 0
        model_loss = - (log_determinant - log_likelihood).mean() if compute_pdf else 0
        return z, model_loss

    def inverse_z(self, z, cond, base_llk_lst=None, xedges=None, compute_pdf=False):
        log_likelihood = 0
        log_determinant = 0
        self.noise = z
        if compute_pdf and self.external_prior:
            zz = z.detach().cpu().numpy()
            log_likelihood = np.sum(external_prior_llk(zz, base_llk_lst, xedges))
        elif compute_pdf:
            log_likelihood = self.base_dist.log_prob(z)
        for i in reversed(range(0, self.layers)):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            z, scale, _ = transformer.inverse(z, cond, alternate)
            log_determinant += torch.log(scale).sum(1) if compute_pdf else 0
        log_pdf = log_determinant - log_likelihood
        return z, log_pdf

