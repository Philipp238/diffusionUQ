# Implements the different losses used in the training of the models.

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.lowrank_multivariate_normal import (
    LowRankMultivariateNormal,
    _batch_lowrank_logdet,
    _batch_lowrank_mahalanobis,
)
from torch.distributions.multivariate_normal import (
    MultivariateNormal,
    _batch_mahalanobis,
)

EPS = 1e-9


class GaussianKernelScore(nn.Module):
    """Computes the Gaussian kernel score for a predictive normal distribution and corresponding observations."""

    def __init__(
        self,
        reduction="mean",
        dimension="univariate",
        gamma: float = 1.0,
        method: str = "lora",
    ) -> None:
        """Initialize class

        Args:
            reduction (str, optional): Reduction to aplly. Defaults to "mean".
            dimension (str, optional): Whether to calculate uni- or multivariate score. Defaults to "univariate".
            gamma (float, optional): Kernel bandwidth. Defaults to 1.0.
            method (str, optional): Covariance approximation method. Defaults to "lora".
        """
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.dimension = dimension
        self.method = method

    def univariate(
        self, observation: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        # Sizes are [B x c x d0 x ... x dn] for mu and sigma and observation
        mu, sigma = torch.split(prediction, 1, dim=-1)
        # Use power of sigma
        sigma2 = torch.pow(sigma, 2)
        # Flatten values
        mu = torch.flatten(mu, start_dim=1)
        sigma2 = torch.flatten(sigma2, start_dim=1)
        observation = torch.flatten(observation, start_dim=1)
        gamma = torch.tensor(self.gamma, device=mu.device)
        gamma2 = torch.pow(gamma, 2)
        # Calculate the Gaussian kernel score
        fac1 = (
            1
            / (torch.sqrt(1 + 2 * sigma2 / gamma2))
            * torch.exp(-torch.pow(observation - mu, 2) / (gamma2 + 2 * sigma2))
        )
        fac2 = 1 / (2 * torch.sqrt(1 + 4 * sigma2 / gamma2))
        score = 0.5 - fac1 + fac2
        return score

    def multivariate_lora(
        self, observation: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        mu = prediction[..., 0]
        diag = prediction[..., 1]
        lowrank = prediction[..., 2:]
        gamma = torch.tensor(self.gamma, device=mu.device)
        gamma2 = torch.pow(gamma, 2)
        diff = observation - mu
        # Create diagonals and lowrank distributions
        diag1 = (2.0 / gamma2) * diag + torch.ones_like(diag).to(mu.device)
        diag2 = (4.0 / gamma2) * diag + torch.ones_like(diag).to(mu.device)
        lowrank1 = torch.sqrt(2.0 / gamma2) * lowrank
        lowrank2 = torch.sqrt(4.0 / gamma2) * lowrank
        lora1 = LowRankMultivariateNormal(loc=mu, cov_factor=lowrank1, cov_diag=diag1)
        lora2 = LowRankMultivariateNormal(loc=mu, cov_factor=lowrank2, cov_diag=diag2)

        # Calculate score
        M = _batch_lowrank_mahalanobis(
            lora1._unbroadcasted_cov_factor,
            lora1._unbroadcasted_cov_diag,
            diff,
            lora1._capacitance_tril,
        )
        det1 = (
            _batch_lowrank_logdet(
                lora1._unbroadcasted_cov_factor,
                lora1._unbroadcasted_cov_diag,
                lora1._capacitance_tril,
            )
            .clamp_max(80)
            .exp()
        )
        det2 = (
            _batch_lowrank_logdet(
                lora2._unbroadcasted_cov_factor,
                lora2._unbroadcasted_cov_diag,
                lora2._capacitance_tril,
            )
            .clamp_max(80)
            .exp()
        )

        fac1 = 1 / torch.sqrt(det1) * torch.exp(-1 / gamma2 * M)
        fac2 = 1 / (torch.sqrt(det2))
        score = 0.5 * (1 + fac2) - fac1
        return score

    def multivariate_cholesky(
        self, observation: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        mu = prediction[..., 0]
        L = prediction[..., 1:]
        cov = MultivariateNormal(loc=mu, scale_tril=L).covariance_matrix
        gamma = torch.tensor(self.gamma, device=mu.device)
        gamma2 = torch.pow(gamma, 2)
        diff = observation - mu
        id = (
            torch.eye(cov.shape[-1])
            .unsqueeze(0)
            .repeat(*cov.shape[:-2], 1, 1)
            .to(mu.device)
        )

        # Create both distributions distributions
        cov1 = (2.0 / gamma2) * cov + id
        cov2 = (4.0 / gamma2) * cov + id

        mvnorm1 = MultivariateNormal(loc=mu, covariance_matrix=cov1)
        mvnorm2 = MultivariateNormal(loc=mu, covariance_matrix=cov2)

        # Calculate score
        M = _batch_mahalanobis(mvnorm1._unbroadcasted_scale_tril, diff)
        det1 = 2 * mvnorm1._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).prod(-1)
        det2 = 2 * mvnorm2._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).prod(-1)

        fac1 = 1 / torch.sqrt(det1) * torch.exp(-1 / gamma2 * M)
        fac2 = 1 / (torch.sqrt(det2))
        score = 0.5 * (1 + fac2) - fac1
        return score

    def forward(
        self, observation: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            observation (torch.Tensor): Observed outcome. Shape = [batch_size, d0, .. dn].
            prediction (torch.Tensor): Predicted  of normal distribution.

        Returns:
            torch.Tensor: (Gaussian) kernel score
        """
        if self.dimension == "univariate":
            score = self.univariate(observation, prediction)
        elif self.dimension == "multivariate":
            if self.method == "lora":
                score = self.multivariate_lora(observation, prediction)
            elif self.method == "cholesky":
                score = self.multivariate_cholesky(observation, prediction)

        if self.reduction == "sum":
            return torch.sum(score)
        elif self.reduction == "mean":
            return torch.mean(score)
        else:
            return score


class QICE(nn.Module):
    """Implements QICE metric for quantile calibration, as pproposed in (https://arxiv.org/abs/2206.07275)."""

    def __init__(self, n_bins=10):
        super(QICE, self).__init__()
        self.n_bins = n_bins
        self.quantile_list = torch.linspace(0, 1, n_bins + 1)
        self.quantile_bin_count = torch.zeros(n_bins)
        self.n_samples = 0

    def aggregate(self, prediction, truth):
        """_summary_

        Args:
            prediction (_type_): Should be of shape (batch_size, ..., n_samples)
            truth (_type_): Should be of shape (batch_size, ...)
        """
        # Count number of samples
        sample_size = truth.nelement()
        self.n_samples += sample_size

        prediction = torch.flatten(prediction, start_dim=1, end_dim=-2)
        truth = torch.flatten(truth, start_dim=1).unsqueeze(0)
        quantiles = torch.quantile(prediction, self.quantile_list, dim=-1)
        quantile_membership = (truth - quantiles > 0).sum(axis=0)

        quantile_bin_count = torch.tensor(
            [(quantile_membership == v).sum() for v in range(self.n_bins + 2)],
            dtype=torch.int64,
        )
        # Combine outside (left/right) end intervals
        quantile_bin_count[1] += quantile_bin_count[0]
        quantile_bin_count[-2] += quantile_bin_count[-1]
        quantile_bin_count = quantile_bin_count[1:-1]

        # Add to quantile bin count
        self.quantile_bin_count += quantile_bin_count

    def compute(self):
        quantile_ratio = self.quantile_bin_count / self.n_samples
        # Compute QICE
        qice = torch.abs(torch.ones(self.n_bins) / self.n_bins - quantile_ratio).mean()
        return qice.item()


class NormalCRPS(nn.Module):
    """Computes the continuous ranked probability score (CRPS) for a predictive normal distribution and corresponding observations."""

    def __init__(
        self,
        reduction="mean",
    ) -> None:
        """Initialize loss.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'mean'`` | ``'sum'``.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self, observation: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            observation (torch.Tensor): Observed outcome. Shape = [batch_size, d0, .. dn].
            mu (torch.Tensor): Predicted mu of normal distribution. Shape = [batch_size, d0, .. dn].
            sigma (torch.Tensor): Predicted sigma of normal distribution. Shape = [batch_size, d0, .. dn].

        Raises:
            ValueError: If sizes of target mu and sigma don't match.

        Returns:
            crps: 1-D float `Tensor` with shape [batch_size] or Float if reduction in ["mean", "sum"]
        """

        if not (mu.size() == sigma.size() == observation.size()):
            raise ValueError("Mismatching target and prediction shapes")
        # Use absolute value of sigma
        sigma = torch.abs(sigma) + EPS
        loc = (observation - mu) / sigma
        Phi = 0.5 * (1 + torch.special.erf(loc / np.sqrt(2.0)))
        phi = 1 / (np.sqrt(2.0 * np.pi)) * torch.exp(-torch.pow(loc, 2) / 2.0)
        crps = sigma * (loc * (2.0 * Phi - 1) + 2.0 * phi - 1 / np.sqrt(np.pi))
        if self.reduction == "sum":
            return torch.sum(crps)
        elif self.reduction == "mean":
            return torch.mean(crps)
        else:
            return crps


class NormalMixtureCRPS(nn.Module):
    """Computes the continuous ranked probability score (CRPS) for a predictive mixture normal distribution and corresponding observations."""

    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        """Initialize loss.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'mean'`` | ``'sum'``.
        """
        super().__init__()
        self.reduction = reduction
        self.normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        self.norm_crps = NormalCRPS(reduction=None)

    def _A(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Helper function to calculate CRPS"""
        Phi = self.normal.cdf(mu / sigma)
        phi = self.normal.log_prob(mu / sigma).exp()
        out = mu * (2 * Phi - 1) + 2 * sigma * phi
        return out

    def forward(
        self,
        observation: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            observation (torch.Tensor): Observed outcome. Shape = [batch_size, d0, .. dn].
            prediction (torch.Tensor): Predicted  of normal distribution. Shape = [batch_size, d0, .. dn, n_components, 3].

        Returns:
            torch.Tensor: CRPS
        """
        # Split
        mu, sigma, weights = torch.split(prediction, 1, dim=-1)
        sigma = torch.abs(sigma) + EPS

        # First term
        diff = observation.unsqueeze(-1).unsqueeze(-1) - mu
        t1 = self._A(diff, sigma)
        t1 = (t1 * weights).sum(axis=(-2, -1)).squeeze()

        # Second term
        mu_diff = mu - mu.transpose(dim0=-2, dim1=-1)
        sigma_sum = torch.sqrt(
            torch.pow(sigma, 2) + torch.pow(sigma, 2).transpose(dim0=-2, dim1=-1)
        )
        t2 = self._A(mu_diff, sigma_sum)
        w = weights * weights.transpose(dim0=-2, dim1=-1)
        t2 = (w * t2).sum(axis=(-2, -1))

        crps = t1 - 0.5 * t2

        if self.reduction == "sum":
            return torch.sum(crps)
        elif self.reduction == "mean":
            return torch.mean(crps)
        else:
            return crps


class KernelScore(object):
    """
    A class for calculating the (Gaussian) kernel score between two tensors.
    """

    def __init__(
        self,
        reduction: str = "mean",
        gamma: int = 1,
    ):
        """Initialize class

        Args:
            reduction (str, optional): Reduction to apply. Defaults to "mean".
            gamma (int, optional): Kernel bandwidth. Defaults to 1.
        """
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma

    def forward(
        self, observation: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            observation (torch.Tensor): Observed outcome. Shape = [batch_size, d0, .. dn].
            prediction (torch.Tensor): Predicted  of normal distribution. Shape = [batch_size, d0, .. dn, n_samples].

        Returns:
            torch.Tensor: (Gaussian) kernel score
        """
        gamma = torch.tensor(self.gamma).to(observation.device)
        gamma2 = torch.pow(gamma, 2)
        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(
            torch.flatten(prediction, start_dim=1, end_dim=-2), 1, 2
        )
        y_flat = torch.swapaxes(
            torch.flatten(observation, start_dim=1, end_dim=-2), 1, 2
        )

        term_1 = -(-torch.cdist(x_flat, y_flat, p=2).mean(dim=(-2, -1)) / gamma2).exp()
        term_2 = -(-torch.cdist(x_flat, x_flat, p=2).mean(dim=(-2, -1)) / gamma2).exp()
        score = term_1 - 0.5 * term_2 + 0.5

        if self.reduction == "sum":
            return torch.sum(score)
        elif self.reduction == "mean":
            return torch.mean(score)
        else:
            return score


class GaussianNLL(object):
    """
    A class for calculating the Gaussian negative log likelihood between two tensors.
    The loss assumes that the input consists of several samples from which the mean and standard deviation can be calculated.
    """

    def __init__(
        self, reduction: str = "mean", reduce_dims: bool = True, **kwargs: dict
    ):
        """Initializes the Gaussian negative log likelihood class.

        Args:
            reduction (str, optional): Which reduction to apply. Defaults to "mean".
            reduce_dims (bool, optional): Which dimensions to reduce. Defaults to True.
        """
        super().__init__()
        self.reduction = reduction
        self.reduce_dims = reduce_dims

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the Gaussian negative log likelihood between two Tensors.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: Gaussian negative log likelihood
        """
        n_dims = len(x.shape) - 2

        # Calculate sample mean and standard deviation
        mu = torch.mean(x, dim=-1)
        sigma = torch.clamp(torch.std(x, dim=-1), min=1e-6, max=1e6)

        # Assert dimension
        assert mu.size() == y.size()

        # Calculate Gaussian NLL
        gaussian = torch.distributions.Normal(mu, sigma)
        score = -gaussian.log_prob(y)

        if self.reduce_dims:
            # Aggregate CRPS over spatial dimensions
            score = score.mean(dim=[d for d in range(1, n_dims + 1)])
        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


class Coverage(object):
    """
    A class for calculating the Coverage probability between two tensors. The corresponding quantiles are calculated from the model predictions.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        reduction: str = "mean",
        reduce_dims: bool = True,
        **kwargs: dict,
    ):
        """Initializes the Coverage probability class.

        Args:
            alpha (float, optional): Significance level. Defaults to 0.05.
            reduction (str, optional): Which reduction to apply. Defaults to "mean".
            reduce_dims (bool, optional): Which dimensions to reduce. Defaults to True.
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        # Kwargs for spherical loss
        self.weights = kwargs.get("weights", None)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def calculate_score(
        self, x: torch.Tensor, y: torch.Tensor, ensemble_dim: int = -1
    ) -> torch.Tensor:
        """Calculates the Coverage probability between two tensors with respect to the significance level alpha.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: Coverage probability.
        """
        n_dims = len(x.shape) - 2

        # Calculate quantiles
        q_lower = torch.quantile(x, self.alpha / 2, dim=ensemble_dim)
        q_upper = torch.quantile(x, 1 - self.alpha / 2, dim=ensemble_dim)

        # Assert dimension and alpha
        assert q_lower.size() == y.size()
        assert 0 < self.alpha < 1

        # Calculate coverage probability
        score = ((y > q_lower) & (y < q_upper)).float()
        # Weighting
        if self.weights is not None:
            weights = (self.weights / self.weights.sum()) * self.weights.size(0)
            score = score * weights

        if self.reduce_dims:
            # Aggregate over spatial and channel dimensions
            score = score.mean(dim=[d for d in range(1, n_dims + 1)])
        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


if __name__ == "__main__":
    # Example usage
    y = torch.randn(5, 1, 1)
    pred = torch.randn(5, 1, 10)

    loss = KernelScore(gamma=1.0, reduction=None)
    score = loss.forward(y, pred)
    print("Gaussian Kernel Score:", score.mean().item())
    print("Shape:", score.shape)
