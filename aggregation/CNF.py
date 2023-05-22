import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from aggregation.Static import external_prior_llk

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
            # nn.Linear(hidden, hidden),
            # nn.LeakyReLU(),
            nn.Linear(hidden, input_dim),
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden),
            nn.LeakyReLU(),
            # nn.Linear(hidden, hidden),
            # nn.LeakyReLU(),
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
    def __init__(self, in_dim, cond_dim, layers=4, hidden=512, T=1, external_prior=True, device=None, replace_nan=False):
        super().__init__()
        self.layers = layers
        self.hidden = hidden
        self.input_dim = in_dim
        self.condition_dim = cond_dim
        self.external_prior = external_prior
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
        self.noise = None
        if device is not None:
            self.device = device
            self.to(device)

    def base_dist(self, cond):
        base_mu = torch.zeros(2, device=self.device)  # hardcoded for 2D
        base_cov = torch.eye(2, device=self.device)
        # if cond_prior_pdf:
        #     mean = self.prior_mean_net(cond)[0]
        #     scale = torch.exp(self.prior_scale_net(cond))[0]
        #     base_mu = base_mu + mean
        #     base_cov = base_cov * scale
        base_dist = MultivariateNormal(base_mu, base_cov)
        return base_dist

    def forward(self, x, cond, base_llk_lst=None, xedges=None):
        log_determinant = 0
        for i in range(0, self.layers):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            x, scale, _ = transformer(x, cond, alternate)
            log_determinant += torch.log(scale).sum(1)
        if base_llk_lst is None:
            assert not self.external_prior
            log_likelihood = self.base_dist(cond).log_prob(x).mean()  # log likelihood of sample under the base measure
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
            inf = torch.tensor(torch.inf, dtype=torch.float32)
            x = torch.where(torch.isnan(x), inf, x)
        if base_llk_lst is None:
            assert not self.external_prior
            log_likelihood = self.base_dist(cond).log_prob(x)  # log likelihood of sample under the base measure
        else:
            assert self.external_prior
            xx = x.detach().cpu().numpy()
            log_likelihood = external_prior_llk(xx, base_llk_lst, xedges)
            log_likelihood = torch.tensor(log_likelihood, dtype=torch.float32, device=self.device)
        log_pdf = log_determinant + log_likelihood
        return x, log_pdf, nans, infs

    def inverse(self, sample_size, cond, compute_pdf=False):
        assert not self.external_prior
        base_dist = self.base_dist(cond)
        z = base_dist.rsample((sample_size,))
        log_likelihood = base_dist.log_prob(z) if compute_pdf else 0
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
            log_likelihood = self.base_dist(cond).log_prob(z)
        for i in reversed(range(0, self.layers)):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            z, scale, _ = transformer.inverse(z, cond, alternate)
            log_determinant += torch.log(scale).sum(1) if compute_pdf else 0
        log_pdf = log_determinant - log_likelihood
        return z, log_pdf

