import torch

from models import (
    generate_mcd_samples,
    generate_deterministic_samples,
    LA_Wrapper,
    generate_diffusion_samples_low_dimensional,
)
from utils import losses, train_utils
import numpy as np

from scoringrules import energy_score, crps_ensemble


def generate_samples(
    uncertainty_quantification: str,
    model,
    a: torch.Tensor,
    u: torch.Tensor,
    n_samples: int,
    x_T_sampling_method: str,
    distributional_method: str = "deterministic",
    regressor=None,
    
) -> torch.Tensor:
    """Mehtod to generate samples from the underlying model with the specified uncertainty quantification method.

    Args:
        uncertainty_quantification (str): Method for uncertainty quantification
        model (_type_): Neural network model
        a (torch.Tensor): Input data
        u (torch.Tensor): Target data
        n_samples (int): Number of samples to generate

    Returns:
        torch.Tensor: _description_
    """
    if uncertainty_quantification.endswith("dropout"):
        model.train()
    else:
        model.eval()
    if uncertainty_quantification == "dropout":
        out = generate_mcd_samples(model, a, u.shape, n_samples=n_samples).permute(
            0, 2, 1
        )
    elif uncertainty_quantification == "laplace":
        out = model.predictive_samples(a, n_samples=n_samples)
    elif uncertainty_quantification.startswith("scoring-rule"):
        out = model(a, n_samples=n_samples)
    elif uncertainty_quantification == "deterministic":
        out = generate_deterministic_samples(model, a, n_samples=n_samples).permute(
            0, 2, 1
        )
    elif uncertainty_quantification == "diffusion":
        out = generate_diffusion_samples_low_dimensional(
            model,
            labels=a,
            images_shape=u.shape,
            n_samples=n_samples,
            distributional_method=distributional_method,
            regressor=regressor,
            x_T_sampling_method=x_T_sampling_method
        ).permute(0, 2, 1)
    return out


def evaluate(model, training_parameters: dict, loader, device, regressor, standardized:bool = False):
    """Method to evaluate the given model.

    Args:
        model (_type_): Underlying model
        training_parameters (dict): Dictionary containing training parameters
        loader (_type_): Data loader
        device (_type_): Device to run the model on
        domain_range (_type_): Either list of domain range for LP based loss or nlon and weights for spherical loss

    Returns:
        _type_: Tuple of evaluation metrics
    """
    uncertainty_quantification = training_parameters["uncertainty_quantification"]
    mse = 0
    es = 0
    coverage = 0
    interval_width = 0
    crps = 0
    gaussian_nll = 0
    alpha = training_parameters["alpha"]

    d = len(next(iter(loader))[0].shape) - 2
    mse_loss = torch.nn.MSELoss()
    # energy_score = losses.EnergyScore(d=d, p=2, type="lp", L=domain_range)
    # crps_loss = losses.CRPS()
    gaussian_nll_loss = losses.GaussianNLL()
    coverage_loss = losses.Coverage(alpha)
    interval_width_loss = losses.IntervalWidth(alpha)

    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            images = images.to(device)
            batch_size = labels.shape[0]
            # predicted_images.shape = (batch_size, n_samples, n_variables)
            predicted_images = generate_samples(
                uncertainty_quantification,
                model,
                labels,
                images,
                training_parameters["n_samples_uq"],
                training_parameters['x_T_sampling_method'],
                training_parameters["distributional_method"],
                regressor
            )

            if standardized:
                images = loader.dataset.destandardize_image(images)
                predicted_images = loader.dataset.destandardize_image(predicted_images)

            mse += (
                mse_loss(predicted_images.mean(axis=1), images).item()
                * batch_size
                / len(loader.dataset)
            )
            es += (
                energy_score(images, predicted_images, backend="torch").mean().item()
                * batch_size
                / len(loader.dataset)
            )

            crps += (
                crps_ensemble(
                    images.cpu(),
                    predicted_images.permute(0, 2, 1).cpu(),
                    backend="torch",
                )
                .mean()
                .item()
                * batch_size
                / len(loader.dataset)
            )
            # gaussian_nll += (
            #     gaussian_nll_loss(predicted_images, images).item() * batch_size / len(loader.dataset)
            # )
            coverage += (
                coverage_loss(predicted_images, images, ensemble_dim=1).item()
                * batch_size
                / len(loader.dataset)
            )
            # interval_width += (
            #     interval_width_loss(predicted_images, images).item() * batch_size / len(loader.dataset)
            # )

    return mse, es, crps, coverage  # , gaussian_nll, , interval_width


def start_evaluation(
    model,
    training_parameters: dict,
    data_parameters: dict,
    train_loader,
    validation_loader,
    test_loader,
    results_dict: dict,
    device,
    logging,
    filename: str,
    regressor,
    **kwargs,
):
    """Performs evaluation of the model on the given data sets.

    Args:
        model (_type_): Underlying model
        training_parameters (dict): Dictionary of training parameters
        data_parameters (dict): Dictionary of data parameters
        train_loader (_type_): Train loader
        validation_loader (_type_): Validation loader
        test_loader (_type_): Test loader
        results_dict (dict): Dictionary to store results
        device (_type_): Device to run the model on
        domain_range (_type_): Either list of domain range for LP based loss or nlon and weights for spherical loss
        logging (_type_): Logging object
        filename (str): Filename to save the model
    """
    # Need to add additional train loader for autoregressive Laplace
    laplace_train_loader = kwargs.get("laplace_train_loader", None)
    logging.info(
        f'Starting evaluation: model {training_parameters["model"]} & uncertainty quantification {training_parameters["uncertainty_quantification"]}'
    )
    # Don't evaluate for train on era5 and SSWE for computational reasons
    if (
        data_parameters["dataset_name"] == "era5"
        or data_parameters["dataset_name"] == "SSWE"
    ):
        data_loaders = {"Validation": validation_loader, "Test": test_loader}
    else:
        data_loaders = {
            "Train": train_loader,
            "Validation": validation_loader,
            "Test": test_loader,
        }

    if training_parameters["uncertainty_quantification"] == "laplace" and (
        not isinstance(model, LA_Wrapper)
    ):
        model = LA_Wrapper(
            model,
            n_samples=training_parameters["n_samples_uq"],
            method="last_layer",
            hessian_structure="full",
            optimize=True,
        )
        if laplace_train_loader is not None:
            model.fit(laplace_train_loader)
        else:
            model.fit(train_loader)
        train_utils.checkpoint(model, filename)

    for name, loader in data_loaders.items():
        logging.info(f"Evaluating the model on {name} data.")

        mse, es, crps, coverage = evaluate(
            model, 
            training_parameters, 
            loader, 
            device, 
            regressor,
            standardized=data_parameters["standardize"]
        )
        # mse, es, crps, gaussian_nll, coverage, int_width = evaluate(model, training_parameters, loader, device, domain_range)

        train_utils.log_and_save_evaluation(mse, "MSE" + name, results_dict, logging)
        train_utils.log_and_save_evaluation(np.sqrt(mse), "RMSE" + name, results_dict, logging)
        train_utils.log_and_save_evaluation(
            es, "EnergyScore" + name, results_dict, logging
        )
        train_utils.log_and_save_evaluation(crps, "CRPS" + name, results_dict, logging)
        # train_utils.log_and_save_evaluation(
        #     gaussian_nll, "Gaussian NLL" + name, results_dict, logging
        # )
        train_utils.log_and_save_evaluation(
            coverage, "Coverage" + name, results_dict, logging
        )
        # train_utils.log_and_save_evaluation(
        #     int_width, "IntervalWidth" + name, results_dict, logging
        # )
