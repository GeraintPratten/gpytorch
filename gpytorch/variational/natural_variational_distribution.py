#!/usr/bin/env python3

import torch
from ..lazy import CholLazyTensor
from ..distributions import MultivariateNormal
from .variational_distribution import VariationalDistribution


class NaturalVariationalDistribution(VariationalDistribution):
    """
    VariationalDistribution objects represent the variational distribution q(u) over a set of inducing points for GPs.

    The most common way this distribution is defined is to parameterize it in terms of a mean vector and a covariance
    matrix. In order to ensure that the covariance matrix remains positive definite, we only consider the lower triangle
    and we manually ensure that the diagonal remains positive.
    """

    def __init__(self, num_inducing_points, batch_size=None):
        """
        Args:
            num_inducing_points (int): Size of the variational distribution. This implies that the variational mean
                should be this size, and the variational covariance matrix should have this many rows and columns.
            batch_size (int, optional): Specifies an optional batch size for the variational parameters. This is useful
                for example when doing additive variational inference.
        """
        super().__init__()
        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)
        if batch_size is not None:
            mean_init = mean_init.repeat(batch_size, 1)
            covar_init = covar_init.repeat(batch_size, 1, 1)

        self.register_parameter(name="natural_variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="natural_variational_covar", parameter=torch.nn.Parameter(covar_init))

        self.has_buffer = False

        # convert from normal expectations eta=(eta1, eta2) to mu, L representation
        def _dist_from_natural(nat_mean, nat_L):
            mumu = torch.matmul(nat_mean.unsqueeze(-1), nat_mean.unsqueeze(-2))
            L = torch.cholesky(nat_L - mumu, upper=False)
            return nat_mean, L

        # convert from eta representation to mu, L representation
        def _natural_from_dist(mu, L):
            mumu = torch.matmul(mu.unsqueeze(-1), mu.unsqueeze(-2))
            nat_L = mumu + torch.matmul(L, torch.transpose(L, -1, -2))
            return mu, nat_L

    def initialize_variational_distribution(self, prior_dist):
        prior_mean = prior_dist.mean
        prior_L = torch.cholesky(prior_dist.covariance_matrix, upper=True)

        nat_mean, nat_L = self._natural_from_dist(prior_mean, prior_L)

        self.variational_mean.data.copy_(nat_mean)
        self.chol_variational_covar.data.copy_(nat_L)

    @property
    def variational_distribution(self):
        if self.has_buffer:
            variatonal_mean, chol_variational_covar = self.buffer
        else:
            variational_mean, chol_variational_covar = self._dist_from_natural(
                self.natural_variational_mean,
                self.natural_variational_covar
            )
            self.buffer = (variational_mean, chol_variational_covar)
            self.has_buffer = True

        variational_covar = CholLazyTensor(chol_variational_covar.transpose(-1, -2))
        return MultivariateNormal(variational_mean, variational_covar)
