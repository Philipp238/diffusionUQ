import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scoringrules import crps_ensemble, energy_score

from models import generate_diffusion_samples_low_dimensional
from utils import losses, train_utils


def generate_samples(
    uncertainty_quantification: str,
    model:nn.Module,
    n_timesteps:int,
    x: torch.Tensor,
    target: torch.Tensor,
    n_samples: int,
    x_T_sampling_method: str,
    distributional_method: str = "deterministic",
    closed_form: bool = False,
    regressor=None,
    cfg_scale:float=3,
    ddim_churn:float=1.0,
    noise_schedule:str=None,
    metrics_plots:bool=False,
    beta_endpoints:tuple=(1e-4, 0.02),
    tau:float=1,
) -> torch.Tensor:
    """Method to generate samples from the specified diffusion model.

    Args:
        uncertainty_quantification (str): Uncertainty quantification method.
        model (nn.Module): Underlying neural network.
        n_timesteps (int): Number of diffusion timesteps.
        x (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        n_samples (int): Number of samples to generate.
        x_T_sampling_method (str): Sampling method for diffusion process.
        distributional_method (str, optional): Type of distributional diffusion. Defaults to "deterministic".
        closed_form (bool, optional): Whether to use closed-form evaluation. Defaults to False.
        regressor (_type_, optional): Regressor. Defaults to None.
        cfg_scale (float, optional): CFG scale. Defaults to 3.
        ddim_churn (float, optional): Chrun parameter of the DDIM model. Defaults to 1.0.
        noise_schedule (str, optional): Noise schedule. Defaults to None.
        metrics_plots (bool, optional): Whether to plot the metrics. Defaults to False.
        beta_endpoints (tuple, optional): Beta endpoints of the noise schedule. Defaults to (1e-4, 0.02).
        tau (float, optional): Interpolation parameter for the covariance matrix. Defaults to 1.

    Returns:
        torch.Tensor: sampled predictions
    """
    model.eval()
    if uncertainty_quantification == "diffusion":
        if (
            metrics_plots
            and target is not None
            and regressor is not None
            and distributional_method != "deterministic"
        ):
            out, crps_over_time, rmse_over_time, distr_over_time = (
                generate_diffusion_samples_low_dimensional(
                    model,
                    input=x,
                    n_timesteps=n_timesteps,
                    target_shape=target.shape,
                    n_samples=n_samples,
                    distributional_method=distributional_method,
                    closed_form=closed_form,
                    regressor=regressor,
                    x_T_sampling_method=x_T_sampling_method,
                    cfg_scale=cfg_scale,
                    gt_target=target,
                    ddim_churn=ddim_churn,
                    noise_schedule=noise_schedule,
                    metrics_plots=metrics_plots,
                    beta_endpoints=beta_endpoints,
                    tau=tau,
                )
            )
            return out, crps_over_time, rmse_over_time, distr_over_time
        else:
            out = generate_diffusion_samples_low_dimensional(
                model,
                input=x,
                n_timesteps=n_timesteps,
                target_shape=target.shape,
                n_samples=n_samples,
                distributional_method=distributional_method,
                closed_form=closed_form,
                regressor=regressor,
                x_T_sampling_method=x_T_sampling_method,
                cfg_scale=cfg_scale,
                gt_target=target,
                ddim_churn=ddim_churn,
                noise_schedule=noise_schedule,
                metrics_plots=metrics_plots,
                beta_endpoints=beta_endpoints,
                tau=tau,
            )
    return out


def evaluate(
    model:nn.Module,
    training_parameters: dict,
    loader,
    device,
    regressor,
    standardized: bool = False,
    metrics_plots: bool = False,
)-> tuple:
    """Function to evaluate the given model.

    Args:
        model (nn.Module): Underlying model.
        training_parameters (dict): Dictionary containing training parameters.
        loader (_type_): Data loader.
        device (_type_): Device to run the model on.
        regressor (_type_): Regressor for the CARD model.
        standardized (bool): Whether data is standardized.
        metrics_plots (bool): Whether to plot metrics.

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
            # res.shape = (batch_size, n_samples, n_variables)
            res = generate_samples(
                uncertainty_quantification,
                model,
                training_parameters["n_timesteps"],
                input,
                target,
                training_parameters["n_samples_uq"],
                training_parameters["x_T_sampling_method"],
                training_parameters["distributional_method"],
                training_parameters["closed_form"],
                regressor,
                cfg_scale=cfg_scale,
                ddim_churn=training_parameters["ddim_churn"],
                noise_schedule=training_parameters["noise_schedule"],
                metrics_plots=metrics_plots,
                beta_endpoints=training_parameters["beta_endpoints"],
                tau=training_parameters["tau"],
            )

            if (
                uncertainty_quantification == "diffusion"
                and regressor is not None
                and training_parameters["distributional_method"] != "deterministic"
                and metrics_plots
            ):
                (
                    prediction,
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
                prediction = res

            if standardized:
                target = loader.dataset.destandardize_output(target)
                prediction = loader.dataset.destandardize_output(prediction)

            mse += (
                mse_loss(prediction.mean(axis=-1), target).item()
                * batch_size
                / len(loader.dataset)
            )
            es += (
                energy_score(
                    target.flatten(start_dim=1, end_dim=-1),
                    prediction.flatten(start_dim=1, end_dim=-2),
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
                    prediction.cpu(),
                    backend="torch",
                )
                .mean()
                .item()
                * batch_size
                / len(loader.dataset)
            )
            gaussian_nll += (
                gaussian_nll_loss(prediction.cpu(), target.cpu()).item()
                * batch_size
                / len(loader.dataset)
            )
            coverage += (
                coverage_loss(prediction, target, ensemble_dim=-1).item()
                * batch_size
                / len(loader.dataset)
            )
            qice_loss.aggregate(prediction.cpu(), target.cpu())

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
    model:nn.Module,
    training_parameters: dict,
    data_parameters: dict,
    train_loader,
    validation_loader,
    test_loader,
    results_dict: dict,
    device,
    logging,
    regressor,
    filename_ending: str,
    metrics_plots: bool,
    **kwargs,
):
    """Performs evaluation of the model on the given data sets.

    Args:
        model (nn.Module): Underlying model.
        training_parameters (dict): Dictionary of training parameters.
        data_parameters (dict): Dictionary of data parameters.
        train_loader (_type_): Train loader.
        validation_loader (_type_): Validation loader.
        test_loader (_type_): Test loader.
        results_dict (dict): Dictionary to store results.
        device (_type_): Device to run the model on.
        logging (_type_): Logging object.
        regressor (_type_): Regressor for the CARD model.
        filename_endig (str): Ending for the filename.
        metrics_plots (bool): Whether to plot metrics.
    """
    directory = kwargs.get("directory", None)
    logging.info(
        f"Starting evaluation: model {training_parameters['model']} & uncertainty quantification {training_parameters['uncertainty_quantification']}"
    )
    # Don't evaluate for train on era5 and SSWE for computational reasons
    if data_parameters["dataset_name"].startswith("1D") or data_parameters[
        "dataset_name"
    ].startswith("2D"):
        if training_parameters["val_only"]:
            data_loaders = {"Validation": validation_loader}
        else:
            data_loaders = {"Validation": validation_loader, "Test": test_loader}
    elif data_parameters["dataset_name"] == "WeatherBench":
        data_loaders = {
            "Test": test_loader,
        }
    else:
        data_loaders = {
            "Train": train_loader,
            "Validation": validation_loader,
            "Test": test_loader,
        }

    for name, loader in data_loaders.items():
        if loader is None:
            continue
        logging.info(f"Evaluating the model on {name} data.")
        t_eval_start = time.time()

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
            metrics_plots=metrics_plots,
        )
        # mse, es, crps, gaussian_nll, coverage, int_width = evaluate(model, training_parameters, loader, device, domain_range)
        t_elapsed = np.round(time.time() - t_eval_start, 3)
        logging.info(f"Evaluating the model on {name} data took {t_elapsed}s.")

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

        if metrics_plots:
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
                    [
                        distr_over_time[t][0][idx_sample]
                        for t in range(len(crps_over_time))
                    ]
                ).squeeze()
                stds_over_time = np.array(
                    [
                        distr_over_time[t][1][idx_sample]
                        for t in range(len(crps_over_time))
                    ]
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
                        f"{directory}/{name}_pred_distr_over_timesteps_sample{idx_sample}_{filename_ending}.png"
                    )

                plt.close()

        # Empty cache
        torch.cuda.empty_cache()
