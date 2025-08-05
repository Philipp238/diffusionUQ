# Implements the different losses used in the training of the models.

import math
import numpy as np
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.lowrank_multivariate_normal import (
    LowRankMultivariateNormal,
    _batch_lowrank_logdet,
    _batch_lowrank_mahalanobis,
)

EPS = 1e-9


class GaussianKernelScore(nn.Module):
    """Computes the Gaussian kernel score for a predictive normal distribution and corresponding observations."""

    def __init__(
        self, reduction="mean", dimension="univariate", gamma: float = 1.0
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.dimension = dimension

    def univariate(self, observation: Tensor, prediction: Tensor) -> Tensor:
        # Sizes are [B x c x d0 x ... x dn] for mu and sigma and observation
        mu, sigma = torch.split(prediction, 1, dim=-1)
        # Use power of sigma
        sigma2 = torch.pow(sigma, 2)
        # Flatten values
        mu = torch.flatten(mu, start_dim=1)
        sigma2 = torch.flatten(sigma2, start_dim=1)
        observation = torch.flatten(observation, start_dim=1)
        gamma = (
            torch.tensor(self.gamma, device=mu.device)
        )
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

    def multivariate(self, observation: Tensor, prediction: Tensor) -> Tensor:
        mu = prediction[..., 0]
        diag = prediction[..., 1]
        lowrank = prediction[..., 2:]
        gamma = (
            torch.tensor(self.gamma, device=mu.device)
        )
        gamma2 = torch.pow(gamma, 2)
        diff = observation - mu
        # Create diagonals and lowrank distributions
        diag1 = (2.0/gamma2)*diag +torch.ones_like(diag).to(mu.device)
        diag2 = (4.0/gamma2)*diag +torch.ones_like(diag).to(mu.device)
        lowrank1 = torch.sqrt(2.0/gamma2) * lowrank
        lowrank2 = torch.sqrt(4.0/gamma2) * lowrank
        lora1 = LowRankMultivariateNormal(loc=mu, cov_factor=lowrank1, cov_diag=diag1)
        lora2 = LowRankMultivariateNormal(loc=mu, cov_factor=lowrank2, cov_diag=diag2)

        # Calculate score
        M = _batch_lowrank_mahalanobis(
                lora1._unbroadcasted_cov_factor,
                lora1._unbroadcasted_cov_diag,
                diff,
                lora1._capacitance_tril,
            )
        det1 = _batch_lowrank_logdet(
            lora1._unbroadcasted_cov_factor,
            lora1._unbroadcasted_cov_diag,
            lora1._capacitance_tril,
        ).clamp_max(80).exp()
        det2 = _batch_lowrank_logdet(
            lora2._unbroadcasted_cov_factor,
            lora2._unbroadcasted_cov_diag,
            lora2._capacitance_tril,
        ).clamp_max(80).exp()

        fac1 = 1/torch.sqrt(det1) * torch.exp(-1/gamma2 * M)
        fac2 = 1 / (torch.sqrt(det2))
        score = 0.5*(1+fac2)-fac1
        return score 

    def forward(self, observation: Tensor, prediction: Tensor) -> Tensor:
        if self.dimension == "univariate":
            score = self.univariate(observation, prediction)
        elif self.dimension == "multivariate":
            score = self.multivariate(observation, prediction)

        if self.reduction == "sum":
            return torch.sum(score)
        elif self.reduction == "mean":
            return torch.mean(score)
        else:
            return score


class QICE(nn.Module):
    """Implements QICE metric for quantile calibration.

    Args:
        nn (_type_): _description_
    """

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
    """Computes the continuous ranked probability score (CRPS) for a predictive normal distribution and corresponding observations.

    Args:
        observation (Tensor): Observed outcome. Shape = [batch_size, d0, .. dn].
        mu (Tensor): Predicted mu of normal distribution. Shape = [batch_size, d0, .. dn].
        sigma (Tensor): Predicted sigma of normal distribution. Shape = [batch_size, d0, .. dn].
        reduce (bool, optional): Boolean value indicating whether reducing the loss to one value or to
            a Tensor with shape = `[batch_size]`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``.
    Raises:
        ValueError: If sizes of target mu and sigma don't match.

    Returns:
        quantile_score: 1-D float `Tensor` with shape [batch_size] or Float if reduction = True

    References:
      - Gneiting, T. et al., 2005: Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation. Mon. Wea. Rev., 133, 1098–1118
    """

    def __init__(
        self,
        reduction="mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, observation: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
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
    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        self.norm_crps = NormalCRPS(reduction=None)

    def _A(self, mu: torch.Tensor, sigma: torch.Tensor):
        Phi = self.normal.cdf(mu / sigma)
        phi = self.normal.log_prob(mu / sigma).exp()
        out = mu * (2 * Phi - 1) + 2 * sigma * phi
        return out

    def forward(
        self,
        observation: torch.torch.Tensor,
        prediction: torch.torch.Tensor,
    ) -> torch.Tensor:
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


class LpLoss(object):
    """
    A class for calculating the Lp loss between two tensors.
    """

    def __init__(
        self,
        d: int = 1,
        p: float = 2,
        L: float = 1.0,
        reduce_dims: List = [0],
        reduction: str = "mean",
        rel: bool = False,
    ):
        """Initializes the Lp loss class.

        Args:
            d (int, optional): Dimension of the domain. Defaults to 1.
            p (float, optional): Parameter for the Lp loss. Defaults to 2.
            L (float, optional): Grid spacing constant. Can be supplied as a list. Defaults to 1.0.
            reduce_dims (List, optional): Which dimensions to reduce loss across. Defaults to [0].
            reduction (str, optional): Which reduction should be applied. Defaults to "mean".
            rel (bool, optional): Whether to calculate relative or absolute loss. Defaults to False.
        """
        super().__init__()
        self.d = d
        self.p = p
        self.rel = rel

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reduction, str):
                assert reduction == "sum" or reduction == "mean"
                self.reductions = [reduction] * len(self.reduce_dims)
            else:
                for j in range(len(reduction)):
                    assert reduction[j] == "sum" or reduction[j] == "mean"
                self.reductions = reduction

        if isinstance(L, float):
            self.L = [L] * self.d
        else:
            self.L = L

    def uniform_h(self, x: torch.Tensor) -> float:
        """Calculates the integration constant for uniform mesh.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            float: Integration constant
        """
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j)
        return h

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def abs(self, x: torch.Tensor, y: torch.tensor, h=None) -> torch.Tensor:
        """Calculates the absolute Lp loss between two tensors.

        Args:
            x (torch.Tensor): Input tensor x
            y (torch.tensor): Input tensor y
            h (_type_, optional): Integration constant. Defaults to None.

        Returns:
            torch.Tensor: Lp loss between two tensors
        """
        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(y)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        const = torch.tensor(np.array(math.prod(h) ** (1.0 / self.p)), device=x.device)
        diff = const * torch.norm(
            torch.flatten(x, start_dim=1) - torch.flatten(y, start_dim=1),
            p=self.p,
            dim=-1,
            keepdim=False,
        )
        if self.reduce_dims is not None:
            diff = self.reduce(diff).squeeze()

        return diff

    def relative(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the relative Lp loss between two tensors.

        Args:
            x (torch.Tensor): Input tensor x
            y (torch.tensor): Input tensor y

        Returns:
            torch.Tensor: Relative Lp loss between two tensors
        """
        diff = torch.norm(
            torch.flatten(x, start_dim=1) - torch.flatten(y, start_dim=1),
            p=self.p,
            dim=-1,
            keepdim=False,
        )
        ynorm = torch.norm(
            torch.flatten(y, start_dim=1), p=self.p, dim=-1, keepdim=False
        )

        diff = diff / ynorm

        if self.reduce_dims is not None:
            diff = self.reduce(diff).squeeze()
        return diff

    def __call__(self, y_pred, y, **kwargs):
        if self.rel:
            return self.relative(y_pred, y)
        else:
            return self.abs(y_pred, y)


def lp_norm(
    x: torch.Tensor, y: torch.Tensor, const: float, p: float = 2
) -> torch.Tensor:
    """Calculates the Lp norm between two tensors with different sample sizes

    Args:
        x (torch.Tensor): First tensor of shape (B, n_samples_x, flatted_dims).
        y (torch.Tensor): Second tensor of shape (B, n_samples_y, flatted_dims).
        const (float): Integration constant.
        p (float, optional): Order of the norm. Defaults to 2.

    Returns:
        Tensor: Lp norm
    """
    norm = const * torch.cdist(x, y, p=p)
    return norm


class EnergyScore(object):
    """
    A class for calculating the energy score between two tensors.
    """

    def __init__(
        self,
        d: int = 1,
        type: str = "lp",
        reduction: str = "mean",
        reduce_dims: bool = True,
        **kwargs: dict,
    ):
        """Initializes the Energy score class.

        Args:
            d (int, optional): Dimension of the domain. Defaults to 1.
            type (str, optional): Type of metric to apply. Defaults to "lp".
            reduction (str, optional): Which reduction should be applied. Defaults to "mean".
            reduce_dims (List, optional): Which dimensions to reduce loss across. Defaults to [0].
            rel (bool, optional): Whether to calculate relative or absolute loss. Defaults to False.
        """
        super().__init__()
        self.d = d
        self.type = type
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        self.rel = kwargs.get("rel", False)
        self.p = kwargs.get("p", 2)
        self.L = kwargs.get("L", 1.0)
        # Arguments for spherical score
        if self.type == "spherical":
            self.nlon = kwargs.get("nlon", 256)
            self.weights = kwargs.get("weights", 1)
            self.dlon = 1 / self.nlon
            self.p = 2
            self.d = 2

        if isinstance(self.L, float):
            self.L = [self.L] * self.d

        self.norm = lp_norm

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

    def uniform_h(self, x: torch.Tensor) -> float:
        """Calculates the integration constant for uniform mesh.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            float: Integration constant
        """
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j - 1)
        return h

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor, h=None) -> torch.Tensor:
        """Calculates the energy score between two tensors for different metrics.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)
            h (_type_, optional): Integration constant. Defaults to None.

        Returns:
            torch.Tensor: Energy score
        """
        n_samples = x.size()[-1]

        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = y.unsqueeze(-1)

        if self.type == "spherical":
            weights = self.weights.unsqueeze(-1) / self.weights.sum()
            x = x * torch.sqrt(weights * self.dlon)
            y = y * torch.sqrt(weights * self.dlon)

        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        # Assume uniform mesh
        if self.type == "lp":
            const = torch.tensor(
                np.array(math.prod(h) ** (1.0 / self.p)), device=x.device
            )
        else:
            const = 1.0

        # Calculate energy score
        term_1 = torch.mean(
            self.norm(x_flat, y_flat, const=const, p=self.p), dim=(1, 2)
        )
        term_2 = torch.sum(self.norm(x_flat, x_flat, const=const, p=self.p), dim=(1, 2))

        if self.rel:
            ynorm = (
                const
                * torch.norm(
                    torch.flatten(y_flat, start_dim=-1), p=self.p, dim=-1, keepdim=False
                ).squeeze()
            )
            term_1 = term_1 / ynorm
            term_2 = term_2 / ynorm

        score = term_1 - term_2 / (2 * n_samples * (n_samples - 1))

        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


class CRPS(object):
    """
    A class for calculating the Continuous Ranked Probability Score (CRPS) between two tensors.
    """

    def __init__(
        self, reduction: str = "mean", reduce_dims: bool = True, **kwargs: dict
    ):
        """Initializes the CRPS class.

        Args:
            reduction (str, optional): Which reduction to apply. Defaults to "mean".
            reduce_dims (bool, optional): Which dimensions to reduce. Defaults to True.
        """
        super().__init__()
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        # Kwargs for spherical loss
        self.nlon = kwargs.get("nlon", 256)
        self.nlat = self.nlon / 2
        self.weights = kwargs.get("weights", None)
        self.dlon = 1 / self.nlon

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
        """Calculates the CRPS for two tensors.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: CRPS
        """
        n_samples = x.size()[-1]

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)

        # Dimensions of the input
        d = torch.prod(torch.tensor(x.shape[1:-1]))
        # Adjust weights for spherical grid
        if self.weights is not None:
            weights = self.weights / self.weights.sum()
            weights = weights.unsqueeze(-1) * self.dlon / (d / (self.nlon * self.nlat))
        else:
            weights = 1 / d
        x = x * weights
        y = y * weights

        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        # Calculate energy score
        term_1 = torch.mean(torch.cdist(x_flat, y_flat, p=1), dim=(1, 2))  # /d
        term_2 = torch.sum(torch.cdist(x_flat, x_flat, p=1), dim=(1, 2))  # /d

        score = term_1 - term_2 / (2 * n_samples * (n_samples - 1))
        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


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


class IntervalWidth(object):
    """
    A class for calculating the Intervalwidth between two tensors. The corresponding quantiles are calculated from the model predictions.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        reduction: str = "mean",
        reduce_dims: bool = True,
        **kwargs: dict,
    ):
        """Initializes the Intervalwidth class.

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

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the Intervalwidth of a specified quantile.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: Intervalwidth
        """
        n_dims = len(x.shape) - 2

        # Calculate quantiles
        q_lower = torch.quantile(x, self.alpha / 2, dim=-1)
        q_upper = torch.quantile(x, 1 - self.alpha / 2, dim=-1)

        # Assert dimension and alpha
        assert q_lower.size() == y.size()
        assert 0 < self.alpha < 1

        # Calculate interval width
        score = torch.abs(q_upper - q_lower)
        # Weighting
        if self.weights is not None:
            weights = self.weights / self.weights.sum() * self.weights.size(0)
            score = score * weights

        if self.reduce_dims:
            # Aggregate CRPS over spatial dimensions
            score = score.mean(dim=[d for d in range(1, n_dims + 1)])

        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


if __name__ == "__main__":
    # Example usage
    y = torch.randn(5, 1, 10)
    mu = torch.zeros(5, 1, 10,1)
    diag = torch.ones(5, 1, 10,1)
    lora = torch.rand(5,1,10,1)
    loss = GaussianKernelScore(gamma=1.0, reduction=None, dimension = "multivariate")
    score = loss(y, torch.cat([mu, diag, lora], dim=-1))
    print("Gaussian Kernel Score:", score.mean().item())
    print("Shape:", score.shape)
