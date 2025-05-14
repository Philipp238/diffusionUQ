# Description: Utility functions for training the models.

from itertools import product
import torch
import torch.nn as nn
import numpy as np
import utils.losses as losses
import scoringrules as sr
from models import MLP, MLP_diffusion, MLP_diffusion_normal, LA_Wrapper


def log_and_save_evaluation(value: float, key: str, results_dict: dict, logging):
    """Method to log and save evaluation results.

    Args:
        value (float): Value to save
        key (str): Results key
        results_dict (dict): Results dictionary
        logging (_type_): Logging object
    """
    value = np.round(value, decimals=5)
    logging.info(f"{key}: {value}")
    if not key in results_dict.keys():
        results_dict[key] = []
    results_dict[key].append(value)


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
    if training_parameters["distributional_method"] == "deterministic":
        criterion = nn.MSELoss()
    elif training_parameters["distributional_method"] == "normal":
        criterion = lambda truth, prediction: sr.crps_normal(
            truth, prediction[..., 0], prediction[..., 1], backend="torch"
        ).mean()

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


def setup_model(training_parameters: dict, device, image_dim: int, label_dim: int):
    """Return the model specified by the training parameters.

    Args:
        training_parameters (dict): Dictionary of training parameters
        device (_type_): Device to run the model on.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        _type_: Specified model
    """
    if training_parameters["uncertainty_quantification"] == "scoring-rule-reparam":
        raise NotImplementedError("Implement a model with parametrization trick.")
        hidden_model = None
    elif training_parameters["uncertainty_quantification"] == "diffusion":
        if training_parameters["distributional_method"] == "deterministic":
            hidden_model = MLP_diffusion(
                target_dim=image_dim,
                conditioning_dim=label_dim,
                concat=training_parameters["concat_condition_diffusion"],
                hidden_dim=training_parameters["hidden_dim"],
                layers=training_parameters["n_layers"],
                dropout=training_parameters["dropout"],
            )
        elif training_parameters["distributional_method"] == "normal":
            hidden_model = MLP_diffusion_normal(
                target_dim=image_dim,
                conditioning_dim=label_dim,
                concat=training_parameters["concat_condition_diffusion"],
                hidden_dim=training_parameters["hidden_dim"],
                layers=training_parameters["n_layers"],
                dropout=training_parameters["dropout"],
            )
    else:
        hidden_model = MLP(
            target_dim=image_dim,
            conditioning_dim=label_dim,
            dropout=training_parameters["dropout"],
            hidden_dim=training_parameters["hidden_dim"],
            layers=training_parameters["n_layers"],
        )

    if training_parameters["uncertainty_quantification"] == "scoring-rule-dropout":
        # return PNO_Wrapper(hidden_model, n_samples=training_parameters["n_samples"]).to(device)
        return None
    else:
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
