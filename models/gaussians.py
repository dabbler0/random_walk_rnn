import torch
from torch.distributions.multivariate_normal import MultivariateNormal

epsilon = 1e-4

class MVGDistribution:
    """
    Class representing a single mutlvariate Gaussian distribution.

    Args:
        mean (1d tensor): mean
        cov (2d tensor): covariance

    Attributes:
        mean (1d tensor): mean
        cov (2d tensor): covariance
        distrubition (Distribution): pytorch Distribution representing this distribution
    """
    def __init__(self, mean, cov):
        self.mean = torch.Tensor(mean).cuda()

        cov = torch.Tensor(cov).cuda()

        # For estimating degrees of freedom
        self.rank = torch.matrix_rank(cov)

        self.cov = cov + torch.eye(cov.shape[0]).cuda() * epsilon

        self.inv = torch.inverse(self.cov)

        self.distribution = MultivariateNormal(self.mean, self.cov)

    def mahalanobis(self, hidden_concat):
        """
        Get the normalized Mahalanobis distance of each point in a given batch

        Args:
            hidden_concat (batch_size x dims tensor): batch of points to get Mahalanobis distance for
        """
        diff = hidden_concat - self.mean
        batch_size = diff.shape[0]

        return torch.bmm(
                torch.bmm(diff.unsqueeze(1),
                    self.inv.unsqueeze(0).expand(batch_size, -1, -1)
                ),
                diff.unsqueeze(2)
        ) / self.rank

class DistributionsRecord:
    """
    Record of several MVGDistributions for simultaneous evaluation

    Args:
        params (dict): output of `estimate_gaussians.py`.
        graph (WalkGraph): WalkGraph whose states this is associated with

    Attributes:
        graph (WalkGraph): WalkGraph whose states this is associated with
        distributions (array of MVGDistribution): MVGDistribution, one for each state
    """
    def __init__(self, params, graph):
        self.graph = graph
        self.distributions = []
        for state in range(graph.states):
            mean, cov = params[str(state)]
            self.distributions.append(MVGDistribution(mean, cov))

    def probs(self, hidden_concat):
        """
        Get log probabilities that each point in the given batch is in
        each distribution in the record.

        Args:
            hidden_concat (batch_size x dims tensor): batch of points to get log probs for

        Returns:
            batch_size x states tensor of log probs
        """
        return torch.stack([
            dist.distribution.log_prob(hidden_concat).squeeze()
            for dist in self.distributions
        ], dim=1)

    def mahalanobis(self, hidden_concat):
        """
        Get normalized Mahalanobis distance for each point in the given batch is in
        each distribution in the record.

        Args:
            hidden_concat (batch_size x dims tensor): batch of points to get Mahalanobis distance for

        Returns:
            batch_size x states tensor of Mhalanobis distances
        """
        return torch.stack([
            dist.mahalanobis(hidden_concat).squeeze()
            for dist in self.distributions
        ], dim=1)

    def evaluate(self, hidden_concat):
        """
        Get both log probabilities and normalized Mahalanobis distance for each point in the given batch is in
        each distribution in the record.

        Args:
            hidden_concat (batch_size x dims tensor): batch of points to get Mahalanobis distance and log probs for

        Returns:
            batch_size x states x 2 tensor of log probs and Mahalanobis distances
        """
        return torch.stack([
            self.probs(hidden_concat),
            self.mahalanobis(hidden_concat)
        ], dim=2)
