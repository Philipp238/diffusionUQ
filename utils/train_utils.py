# Description: Utility functions for training the models.

from itertools import product
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from models.mlp_diffusion import MLP_diffusion_CARD
import utils.losses as losses
import scoringrules as sr
from models import (
    MLP,
    MLP_CARD,
    MLP_diffusion,
    MLP_diffusion_normal,
    MLP_diffusion_sample,
    MLP_diffusion_mixednormal,
    LA_Wrapper,
    UNetDiffusion,
    UNet_diffusion_normal,
    UNet_diffusion_mvnormal,
    UNet_diffusion_mixednormal,
    UNet_diffusion_sample,
)
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


def log_and_save_evaluation(value: float, key: str, results_dict: dict, logging):
    """Method to log and save evaluation results.

    Args:
        value (float): Value to save
        key (str): Results key
        results_dict (dict): Results dictionary
        logging (_type_): Logging object
    """
    value = float(np.round(value, decimals=7))
    logging.info(f"{key}: {value}")
    results_dict.setdefault(key, []).append(value)


def checkpoint(model, filename):
    """Save the model state dict to a file.

    Args:
        model (_type_): The torch model.
        filename (_type_): The filename including the path to save the model.
    """
    if isinstance(model, LA_Wrapper):
        model.save_state_dict(filename)
    else:
        torch.save(model.state_dict(), filename)


def resume(model, filename):
    """Load the model state dict from a file.

    Args:
        model (_type_): The torch model.
        filename (_type_): The filename including the path to load the model.
    """

    if isinstance(model, LA_Wrapper):
        model.load_state_dict(filename)
    else:
        model.load_state_dict(torch.load(filename))


def get_criterion(training_parameters, device):
    """Define criterion for the model.
    Criterion gets as arguments (truth, prediction) and returns a loss value.
    """
    method = training_parameters["mvnormal_method"]
    loss = training_parameters["loss"]
    if training_parameters["uncertainty_quantification"] == "diffusion":
        if training_parameters["distributional_method"] == "deterministic":
            criterion = nn.MSELoss()
        elif training_parameters["distributional_method"] == "normal":
            if loss == "crps":
                criterion = lambda truth, prediction: sr.crps_normal(
                    truth, prediction[..., 0], prediction[..., 1], backend="torch"
                ).mean()

            elif loss == "kernel":
                criterion = losses.GaussianKernelScore(dimension = "univariate", gamma = training_parameters["gamma"])
            elif loss == "log":
                criterion = lambda truth, prediction: (-1)* Normal(prediction[...,0], prediction[...,1]).log_prob(truth).mean()
            else:
                raise AssertionError("Loss function not implemented")

        elif training_parameters["distributional_method"] == "sample":
            criterion = lambda truth, prediction: sr.energy_score(
                truth.flatten(start_dim=1, end_dim=-1),
                prediction.flatten(start_dim=1, end_dim=-2),
                m_axis=-1,
                v_axis=-2,
                backend="torch",
            ).mean()
        elif training_parameters["distributional_method"] == "mixednormal":
            criterion = losses.NormalMixtureCRPS()
        elif training_parameters["distributional_method"] == "mvnormal":
            if method == "lora":               
                if loss == "log":
                    criterion = lambda truth, prediction: (-1)* LowRankMultivariateNormal(prediction[...,0], prediction[...,2:], prediction[...,1]).log_prob(truth).mean()
                else: # Kernel loss
                    criterion = losses.GaussianKernelScore(dimension = "multivariate", gamma = training_parameters["gamma"], method = "lora")
            elif method == "cholesky":
                if loss == "log":
                    criterion = lambda truth, prediction: (-1)* MultivariateNormal(loc = prediction[...,0], scale_tril=prediction[...,1:]).log_prob(truth).mean()
                else: # Kernel loss
                    criterion = losses.GaussianKernelScore(dimension = "multivariate", gamma = training_parameters["gamma"], method = "cholesky")
        else:
            raise ValueError(
                f'"distributional_method" must be any of the following: "deterministic", "normal", "sample" or'
                f'"mixednormal". You chose {training_parameters["distributional_method"]}.'
            )
    else:
        criterion = nn.MSELoss()
    return criterion


def initialize_weights(model, init):
    for name, param in model.named_parameters():
        if "weight" in name and param.data.dim() == 2:
            if init == "xavier":
                nn.init.xavier_uniform(param)
            elif init == "he":
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
            else:
                raise NotImplementedError(
                    f'Please choose init as "default", "xavier" or "he". You chose: {init}.'
                )


def setup_model(
    data_parameters: dict,
    training_parameters: dict,
    device,
    target_dim: int,
    input_dim: int,
):
    """Return the model specified by the training parameters.

    Args:
        training_parameters (dict): Dictionary of training parameters
        device (_type_): Device to run the model on.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        _type_: Specified model
    """
    if data_parameters["dataset_name"] in [
        "1D_Advection",
        "1D_ReacDiff",
        "1D_Burgers",
        "1D_KS",
        "WeatherBench",
    ]:
        if data_parameters["dataset_name"] == "WeatherBench":
            d = 2
        else:
            d = 1
        conditioning_dim = input_dim[0]
        backbone = UNetDiffusion(
            d=d,
            conditioning_dim=conditioning_dim,
            hidden_channels=training_parameters["hidden_dim"],
            init_features=training_parameters["hidden_dim"],
            domain_dim = target_dim
        )
        if training_parameters["distributional_method"] == "deterministic":
            hidden_model = backbone
        elif training_parameters["distributional_method"] == "normal" or training_parameters["distributional_method"] == "closed_form_normal":
            hidden_model = UNet_diffusion_normal(
                backbone=backbone,
                d=d,
                target_dim=1,
            )
        elif training_parameters["distributional_method"] == "mvnormal":
            hidden_model = UNet_diffusion_mvnormal(
                backbone=backbone,
                d=d,
                target_dim=1,
                domain_dim = target_dim[1:],
                rank = training_parameters["rank"],
                method = training_parameters["mvnormal_method"]
            )
        elif training_parameters["distributional_method"] == "sample":
            backbone = UNetDiffusion(
                d=d,
                conditioning_dim=conditioning_dim+1,
                hidden_channels=training_parameters["hidden_dim"],
                init_features=training_parameters["hidden_dim"],
                domain_dim = target_dim
            )
            hidden_model = UNet_diffusion_sample(
                backbone=backbone,
                d=d,
                target_dim=1,
                hidden_dim=training_parameters["hidden_dim"],
                n_samples=training_parameters["n_train_samples"],
            )
        elif training_parameters["distributional_method"] == "mixednormal":
            hidden_model = UNet_diffusion_mixednormal(
                backbone=backbone,
                d=d,
                target_dim=1,
                n_components=training_parameters["n_components"],
            )
    else:
        if training_parameters["uncertainty_quantification"] == "scoring-rule-reparam":
            raise NotImplementedError("Implement a model with parametrization trick.")
        elif training_parameters["uncertainty_quantification"] == "diffusion":
            use_regressor_pred = training_parameters["regressor"] is not None
            if training_parameters["backbone"] == "default":
                backbone = MLP_diffusion(
                    target_dim=target_dim,
                    conditioning_dim=input_dim,
                    concat=training_parameters["concat_condition_diffusion"],
                    use_regressor_pred=use_regressor_pred,
                    hidden_dim=training_parameters["hidden_dim"],
                    layers=training_parameters["n_layers"],
                    dropout=training_parameters["dropout"],
                )
            elif training_parameters["backbone"] == "CARD":
                hidden_dim = (
                    2 * training_parameters["hidden_dim"]
                    if training_parameters["concat_condition_diffusion"]
                    else training_parameters["hidden_dim"]
                )
                backbone = MLP_diffusion_CARD(
                    target_dim=target_dim,
                    conditioning_dim=input_dim,
                    hidden_dim=hidden_dim,
                    layers=training_parameters["n_layers"],
                    use_regressor_pred=use_regressor_pred,
                )
            else:
                raise KeyError(
                    f"No such backbone architecture: {training_parameters['backbone']}"
                )

            if training_parameters["distributional_method"] == "deterministic":
                hidden_model = backbone
            elif (
                training_parameters["distributional_method"] == "normal"
                or training_parameters["distributional_method"] == "closed_form_normal"
            ):
                hidden_model = MLP_diffusion_normal(
                    backbone=backbone,
                    target_dim=target_dim,
                    concat=training_parameters["concat_condition_diffusion"],
                    hidden_dim=training_parameters["hidden_dim"],
                )
            elif training_parameters["distributional_method"] == "sample":
                hidden_model = MLP_diffusion_sample(
                    backbone=backbone,
                    target_dim=target_dim,
                    hidden_dim=training_parameters["hidden_dim"],
                    n_samples=50,
                )
            elif (
                training_parameters["distributional_method"] == "mixednormal"
                or training_parameters["distributional_method"] == "closed_form_mixednormal"
            ):
                hidden_model = MLP_diffusion_mixednormal(
                    backbone=backbone,
                    target_dim=target_dim,
                    concat=training_parameters["concat_condition_diffusion"],
                    hidden_dim=training_parameters["hidden_dim"],
                    n_components=training_parameters["n_components"],
                )

        else:
            hidden_model = MLP(
                target_dim=target_dim,
                conditioning_dim=input_dim,
                dropout=training_parameters["dropout"],
                hidden_dim=training_parameters["hidden_dim"],
                layers=training_parameters["n_layers"],
            )

    return hidden_model.to(device)


class EarlyStopper:
    """
    Class for early stopping in training.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_hyperparameters_combination(hp_dict, except_keys=[]):
    """
    Gets all combinations of hyperparameters given in hp_dict and return a list,
    leaves the list of the keys of except_keys untouched though

    Args:
        hp_dict (_type_): Dictionary of hyperparameters
        except_keys (list, optional): Key to except. Defaults to [].

    Returns:
        _type_: Dictionary of hyperparameter combinations.
    """
    except_dict = {}
    for except_key in except_keys:
        if except_key in hp_dict.keys():
            except_dict[except_key] = hp_dict.pop(except_key)
    iterables = [
        value if isinstance(value, list) else [value] for value in hp_dict.values()
    ]
    all_combinations = list(product(*iterables))
    # Create a list of dictionaries for each combination
    combination_dicts = [
        {
            **{param: value for param, value in zip(hp_dict.keys(), combination)},
            **except_dict,
        }
        for combination in all_combinations
    ]
    return combination_dicts


def setup_CARD_model(
    image_dim: int,
    label_dim: int,
    hidden_layers=[100, 50],
    use_batchnorm=False,
    negative_slope=0.01,
    dropout_rate=0.1,
):
    return MLP_CARD(
        input_dim=label_dim,
        target_dim=image_dim,
        hid_layers=hidden_layers,
        use_batchnorm=use_batchnorm,
        negative_slope=negative_slope,
        dropout_rate=dropout_rate,
    )


def evaluate_CARD_model(model, loader, device, standardized=False):
    total_mse = 0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        images_pred = model(labels).detach().cpu()
        images = images.cpu()

        if standardized:
            images = loader.dataset.destandardize_output(images)
            images_pred = loader.dataset.destandardize_output(images_pred)

        mses = (images_pred - images) ** 2
        total_mse += mses.sum().item()
        count += images_pred.shape[0]

    return total_mse / count
