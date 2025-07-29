import matplotlib.pyplot as plt
import numpy as np
import torch
from scoringrules import crps_ensemble, energy_score

from models import (
    LA_Wrapper,
    generate_deterministic_samples,
    generate_diffusion_samples_low_dimensional,
    generate_mcd_samples,
)
from utils import losses, train_utils


def generate_samples(
    uncertainty_quantification: str,
    model,
    n_timesteps,
    a: torch.Tensor,
    u: torch.Tensor,
    n_samples: int,
    x_T_sampling_method: str,
    distributional_method: str = "deterministic",
    regressor=None,
    cfg_scale=3,
    ddim_sigma=1.0,
    noise_schedule=None,
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
        out = generate_mcd_samples(model, a, u.shape, n_samples=n_samples)
    elif uncertainty_quantification == "laplace":
        out = model.predictive_samples(a, n_samples=n_samples)
    elif uncertainty_quantification.startswith("scoring-rule"):
        out = model(a, n_samples=n_samples)
    elif uncertainty_quantification == "deterministic":
        out = generate_deterministic_samples(
            model, a, n_timesteps=n_timesteps, n_samples=n_samples
        )
    elif uncertainty_quantification == "diffusion":
        if u is not None and regressor is not None and distributional_method != "deterministic":
            out, crps_over_time, rmse_over_time, distr_over_time = (
                generate_diffusion_samples_low_dimensional(
                    model,
                    input=a,
                    n_timesteps=n_timesteps,
                    target_shape=u.shape,
                    n_samples=n_samples,
                    distributional_method=distributional_method,
                    regressor=regressor,
                    x_T_sampling_method=x_T_sampling_method,
                    cfg_scale=cfg_scale,
                    gt_images=u,
                    ddim_sigma=ddim_sigma,
                    noise_schedule=noise_schedule
                )
            )
            return out, crps_over_time, rmse_over_time, distr_over_time
        else:
            out = generate_diffusion_samples_low_dimensional(
                model,
                input=a,
                n_timesteps=n_timesteps,
                target_shape=u.shape,
                n_samples=n_samples,
                distributional_method=distributional_method,
                regressor=regressor,
                x_T_sampling_method=x_T_sampling_method,
                cfg_scale=cfg_scale,
                gt_images=u,
                ddim_sigma=ddim_sigma,
                noise_schedule=noise_schedule
            )
    return out


def evaluate(
    model,
    training_parameters: dict,
    loader,
    device,
    regressor,
    standardized: bool = False,
    filename=None,
):
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
    crps = 0
    gaussian_nll = 0
    alpha = training_parameters["alpha"]

    mse_loss = torch.nn.MSELoss()
    # energy_score = losses.EnergyScore(d=d, p=2, type="lp", L=domain_range)
    # crps_loss = losses.CRPS()
    gaussian_nll_loss = losses.GaussianNLL()
    coverage_loss = losses.Coverage(alpha)
    qice_loss = losses.QICE()

    if uncertainty_quantification == "diffusion":
        crps_over_time, rmse_over_time, distr_over_time = [], [], []

    cfg_scale = 3 if training_parameters["conditional_free_guidance_training"] else 0
    with torch.no_grad():
        for target, input in loader:
            input = input.to(device)
            target = target.to(device)
            batch_size = input.shape[0]
            # predicted_images.shape = (batch_size, n_samples, n_variables)
            res = generate_samples(
                uncertainty_quantification,
                model,
                training_parameters["n_timesteps"],
                input,
                target,
                training_parameters["n_samples_uq"],
                training_parameters["x_T_sampling_method"],
                training_parameters["distributional_method"],
                regressor,
                cfg_scale=cfg_scale,
                ddim_sigma=training_parameters["ddim_sigma"],
                noise_schedule=training_parameters["noise_schedule"]
            )

            if (
                uncertainty_quantification == "diffusion"
                and regressor is not None
                and training_parameters["distributional_method"] != "deterministic"
            ):
                (
                    predicted_images,
                    curr_crps_over_time,
                    curr_rmse_over_time,
                    curr_distr_over_time,
                ) = res
                if len(crps_over_time) == 0:
                    crps_over_time = curr_crps_over_time
                    rmse_over_time = curr_rmse_over_time
                    distr_over_time = curr_distr_over_time
                else:
                    crps_over_time = [
                        crps_over_time[i] + curr_crps_over_time[i]
                        for i in range(len(curr_crps_over_time))
                    ]
                    rmse_over_time = [
                        rmse_over_time[i] + curr_rmse_over_time[i]
                        for i in range(len(curr_rmse_over_time))
                    ]
                    distr_over_time = curr_distr_over_time
            else:
                predicted_images = res

            if standardized:
                target = loader.dataset.destandardize_output(target)
                predicted_images = loader.dataset.destandardize_output(predicted_images)

            mse += (
                mse_loss(predicted_images.mean(axis=-1), target).item()
                * batch_size
                / len(loader.dataset)
            )
            es += (
                energy_score(
                    target.flatten(start_dim=1, end_dim=-1),
                    predicted_images.flatten(start_dim=1, end_dim=-2),
                    m_axis=-1,
                    v_axis=-2,
                    backend="torch",
                )
                .mean()
                .item()
                * batch_size
                / len(loader.dataset)
            )

            crps += (
                crps_ensemble(
                    target.cpu(),
                    predicted_images.cpu(),
                    backend="torch",
                )
                .mean()
                .item()
                * batch_size
                / len(loader.dataset)
            )
            gaussian_nll += (
                gaussian_nll_loss(predicted_images.cpu(), target.cpu()).item()
                * batch_size
                / len(loader.dataset)
            )
            coverage += (
                coverage_loss(predicted_images, target, ensemble_dim=-1).item()
                * batch_size
                / len(loader.dataset)
            )
            qice_loss.aggregate(predicted_images.cpu(), target.cpu())

        crps_over_time = [x / len(loader.dataset) for x in crps_over_time]
        rmse_over_time = [np.sqrt(x / len(loader.dataset)) for x in rmse_over_time]

        qice = qice_loss.compute()

    return (
        mse,
        es,
        crps,
        coverage,
        gaussian_nll,
        qice,
        crps_over_time,
        rmse_over_time,
        distr_over_time,
    )


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
    directory = kwargs.get("directory", None)
    logging.info(
        f"Starting evaluation: model {training_parameters['model']} & uncertainty quantification {training_parameters['uncertainty_quantification']}"
    )
    # Don't evaluate for train on era5 and SSWE for computational reasons
    if data_parameters["dataset_name"].startswith("1D") or data_parameters[
        "dataset_name"
    ].startswith("2D"):
        data_loaders = {"Test": test_loader}
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
        if loader is None:
            continue
        logging.info(f"Evaluating the model on {name} data.")

        (
            mse,
            es,
            crps,
            coverage,
            gaussian_nll,
            qice,
            crps_over_time,
            rmse_over_time,
            distr_over_time,
        ) = evaluate(
            model,
            training_parameters,
            loader,
            device,
            regressor,
            standardized=data_parameters["standardize"],
            filename=filename,
        )
        # mse, es, crps, gaussian_nll, coverage, int_width = evaluate(model, training_parameters, loader, device, domain_range)

        train_utils.log_and_save_evaluation(mse, "MSE" + name, results_dict, logging)
        train_utils.log_and_save_evaluation(
            np.sqrt(mse), "RMSE" + name, results_dict, logging
        )
        train_utils.log_and_save_evaluation(
            es, "EnergyScore" + name, results_dict, logging
        )
        train_utils.log_and_save_evaluation(crps, "CRPS" + name, results_dict, logging)
        train_utils.log_and_save_evaluation(
            gaussian_nll, "Gaussian NLL" + name, results_dict, logging
        )
        train_utils.log_and_save_evaluation(
            coverage, "Coverage" + name, results_dict, logging
        )
        train_utils.log_and_save_evaluation(qice, "QICE" + name, results_dict, logging)

        # Plot CRPS and RMSE over the denoising timesteps
        reversed_timesteps = list(reversed(range(len(crps_over_time))))
        plt.plot(reversed_timesteps, crps_over_time, label="CRPS")
        plt.plot(reversed_timesteps, rmse_over_time, label="RMSE")
        plt.xlabel("timesteps")
        plt.legend()
        plt.title(f"Metrics {name}")
        plt.tight_layout()

        if directory is not None:
            plt.savefig(f"{directory}/{name}_metrics_over_timesteps.png")

        plt.close()

        NUM_SAMPLES = 5
        for idx_sample in range(NUM_SAMPLES):
            means_over_time = np.array(
                [distr_over_time[t][0][idx_sample] for t in range(len(crps_over_time))]
            ).squeeze()
            stds_over_time = np.array(
                [distr_over_time[t][1][idx_sample] for t in range(len(crps_over_time))]
            ).squeeze()

            plt.figure()
            plt.plot(reversed_timesteps, means_over_time, label="Prediction - Mean")
            plt.plot(
                reversed_timesteps,
                [loader.dataset[idx_sample][0].item() for _ in reversed_timesteps],
                label="Ground truth",
            )
            plt.fill_between(
                np.array(reversed_timesteps),
                means_over_time - stds_over_time,
                means_over_time + stds_over_time,
                color="blue",
                alpha=0.2,
                label="±1 Std Dev",
            )
            plt.xlabel("timesteps")
            plt.legend()
            plt.title(f"Predictive distribution {name}")
            plt.tight_layout()

            if directory is not None:
                plt.savefig(
                    f"{directory}/{name}_pred_distr_over_timesteps_sample{idx_sample}.png"
                )

            plt.close()
