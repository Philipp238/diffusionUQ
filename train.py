# Implements the training functions and scripts.

import os
import torch
from torch import optim
import matplotlib.pyplot as plt
from utils import train_utils
import resource
import psutil
import gc
from models import EMA, Diffusion, DistributionalDiffusion
import copy
import numpy as np
import configparser
from scoringrules import energy_score
import time 


def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # you can convert that object to a dictionary
    return f"{point}: mem (CPU python)={usage[2] / 1024.0}MB; mem (CPU total)={dict(psutil.virtual_memory()._asdict())['used'] / 1024**2}MB"


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}.")


def train(
    net,
    optimizer,
    target,
    input,
    criterion,
    gradient_clipping,
    batch_accumulation,
    length,
    idx,
    uncertainty_quantification,
    ema,
    ema_model,
    diffusion=None,
    conditional_free_guidance_training=True,
    regressor=None,
    **kwargs,
):
    """Function that perfroms a training step for a given model.

    Args:
        net (_type_): The model to be trained.
        optimizer (_type_): The optimizer to be used.
        input (_type_): The input data.
        target (_type_): The target data.
        criterion (_type_): The loss function.
        gradient_clipping (_type_): The gradient clipping value.

    Returns:
        _type_: Loss and gradient norm.
    """
    if uncertainty_quantification == "diffusion":
        assert not (diffusion is None)
        t = diffusion.sample_timesteps(target.shape[0]).to(device)
        if regressor is None:
            pred = None
        else:
            pred = regressor(input)

        x_t, noise = diffusion.noise_low_dimensional(target, t, pred=pred)
        if np.random.random() < 0.1 and conditional_free_guidance_training:
            input = None
        predicted_noise = net(x_t, t, input, pred = pred)
        loss = criterion(noise, predicted_noise)
    else:
        predicted_images = net(input)
        loss = criterion(target, predicted_images)

    loss = loss / batch_accumulation
    loss.backward()

    if ((idx + 1) % batch_accumulation == 0) or (idx == length - 1):
        # Update opimizer
        optimizer.step()
        ema.step_ema(ema_model, net)
        optimizer.zero_grad(set_to_none=True)

    loss = loss.item()

    return loss


def trainer(
    train_loader,
    val_loader,
    directory,
    training_parameters,
    data_parameters,
    logging,
    filename_ending,
    target_dim,
    input_dim,
    d_time,
    results_dict,
    regressor,
):
    """Trainer function that takes a parameter dictionaray and dataloaders, trains the models and logs the results.

    Args:
        train_loader (_type_): The training dataloader.
        val_loader (_type_): The validation dataloader.
        directory (_type_): The directory to save the results.
        training_parameters (_type_): The training parameter dictionary.
        data_parameters (_type_): The data parameter dictionary.
        logging (_type_): The logger.
        filename_ending (_type_): The filename.
        domain_range (_type_): The domain range of the dataset.
        d_time (_type_): The datetime.
        results_dict (_type_): Results dictionary.

    Returns:
        _type_: Trained model and corresponding filename.
    """

    if device == "cpu":
        assert not training_parameters["data_loader_pin_memory"]

    criterion = train_utils.get_criterion(
        training_parameters, device=device
    )  # Different loss functions for noise prediction
    eval_criterion = (
        energy_score  # torch.nn.MSELoss() # MSE loss for evaluating generated samples
    )

    model = train_utils.setup_model(data_parameters,training_parameters, device, target_dim, input_dim)

    if training_parameters["init"] != "default":
        train_utils.initialize_weights(model, training_parameters["init"])

    if training_parameters.get("finetuning", None):
        train_utils.resume(model, training_parameters.get("finetuning", None))

    n_parameters = 0
    for parameter in model.parameters():
        n_parameters += parameter.nelement()

    train_utils.log_and_save_evaluation(
        n_parameters, "NumberParameters", results_dict, logging
    )

    logging.info(f"GPU memory allocated: {torch.cuda.memory_reserved(device=device)}")
    logging.info(using("After setting up the model"))

    # create your optimizer
    lr = training_parameters["learning_rate"]
    if training_parameters["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=training_parameters["weight_decay"],
        )
    elif training_parameters["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=training_parameters["weight_decay"],
        )
    elif training_parameters["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr
        )

    # Gradiend accumulation
    batch_accumulation = training_parameters["batch_accumulation"]

    report_every = training_parameters["report_every"]
    early_stopper = train_utils.EarlyStopper(
        patience=int(training_parameters["early_stopping"] / report_every),
        min_delta=0.0001,
    )
    running_loss = 0

    training_loss_list = []
    validation_loss_list = []
    validation_loss_list_ema = []
    epochs = []

    best_loss = torch.inf

    lr_schedule = training_parameters["lr_schedule"]
    if lr_schedule == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    warmup_lr = training_parameters["warmup_lr"]
    if warmup_lr > 0:
        warmup_schedule = np.linspace(lr*0.1, lr, warmup_lr)
        

    # Additional parameters
    uncertainty_quantification = training_parameters["uncertainty_quantification"]
    distributional_method = training_parameters["distributional_method"]
    closed_form = training_parameters["closed_form"]
    noise_schedule = training_parameters['noise_schedule']
    if uncertainty_quantification == 'diffusion':
        if distributional_method == "deterministic":
            diffusion = Diffusion(
                noise_steps=training_parameters["n_timesteps"],
                img_size=target_dim,
                ddim_churn=training_parameters['ddim_churn'],
                device=device,
                x_T_sampling_method=training_parameters['x_T_sampling_method'],
                noise_schedule=noise_schedule
            )
        else:
            diffusion = DistributionalDiffusion(
                noise_steps=training_parameters["n_timesteps"],
                img_size=target_dim,
                device=device,
                ddim_churn=training_parameters['ddim_churn'],
                distributional_method=distributional_method,
                closed_form=closed_form,
                x_T_sampling_method=training_parameters['x_T_sampling_method'],
                noise_schedule=noise_schedule
            )
    else:
        diffusion = None

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    cfg_scale = 3 if training_parameters["conditional_free_guidance_training"] else 0

    # Iterate over autoregressive steps, if necessary
    logging.info(f"Training starts now.")

    filename = os.path.join(directory, f"Datetime_{d_time}_Loss_{filename_ending}.pt")

    # Gather training times
    t_training = []

    for epoch in range(training_parameters["n_epochs"]):
        # Set learning rate warm up
        if epoch < warmup_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_schedule[epoch]
        elif epoch == warmup_lr & warmup_lr != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logging.info(f"Warmup finished.")

        gc.collect()
        t_current_epoch = time.time()

        model.train()
        for idx, sample in enumerate(train_loader):
            target, input = sample
            target = target.to(device)
            input = input.to(device)

            batch_loss = train(
                model,
                optimizer,
                target,
                input,
                criterion,
                training_parameters["gradient_clipping"],
                batch_accumulation = batch_accumulation,
                length = len(train_loader),
                idx = idx,
                uncertainty_quantification=uncertainty_quantification,
                ema=ema,
                ema_model=ema_model,
                diffusion=diffusion,
                conditional_free_guidance_training=training_parameters[
                    "conditional_free_guidance_training"
                ],
                regressor=regressor,
            )
            running_loss += batch_loss

        # Get time
        t_elapsed = time.time() - t_current_epoch
        t_training.append(t_elapsed)

        if epoch % report_every == report_every - 1:
            logging.info(using(f"At the start of the epoch {epoch+1}"))
            epochs.append(epoch)
            training_loss_list.append(running_loss / report_every / (len(train_loader)))
            running_loss = 0.0

            logging_str = (
                f"[{epoch + 1:5d}] Training loss: {training_loss_list[-1]:.8f}"
            )

            if val_loader is not None:
                if not uncertainty_quantification.endswith("dropout"):
                    model.eval()

                validation_loss = 0
                validation_loss_ema = 0

                with torch.no_grad():
                    for target, input in val_loader:
                        target = target.to(device)
                        input = input.to(device)

                        if uncertainty_quantification == "diffusion":
                            n_samples = training_parameters["n_val_samples"]

                            if regressor is None:
                                repeated_pred = None
                            else:
                                pred = regressor(input)
                                repeated_pred = pred.repeat_interleave(n_samples, dim=0)

                            repeated_labels = input.repeat_interleave(n_samples, dim=0)
                            sampled_targets = diffusion.sample_low_dimensional(
                                model,
                                n=repeated_labels.shape[0],
                                conditioning=repeated_labels,
                                pred=repeated_pred,
                                cfg_scale=cfg_scale,
                            )
                            sampled_targets_ema = diffusion.sample_low_dimensional(
                                ema_model,
                                n=repeated_labels.shape[0],
                                conditioning=repeated_labels,
                                pred=repeated_pred,
                                cfg_scale=cfg_scale,
                            )
                            sampled_targets = sampled_targets.reshape(
                                input.shape[0], n_samples, *target.shape[1:]
                            ).moveaxis(1,-1)
                            sampled_targets_ema = sampled_targets_ema.reshape(
                                input.shape[0], n_samples, *target.shape[1:]
                            ).moveaxis(1,-1)
                        else:
                            sampled_targets = model(input)
                            sampled_targets_ema = ema_model(input)

                        validation_loss += eval_criterion(
                            target.flatten(start_dim=1, end_dim=-1),
                            sampled_targets.flatten(start_dim=1, end_dim=-2),
                            m_axis=-1,
                            v_axis=-2,
                            backend = "torch",
                        ).mean().item()
                        validation_loss_ema += eval_criterion(
                            target.flatten(start_dim=1, end_dim=-1),
                            sampled_targets_ema.flatten(start_dim=1, end_dim=-2),
                            m_axis=-1,
                            v_axis=-2,
                            backend = "torch",
                        ).mean().item()

                validation_loss_list.append(validation_loss / len(val_loader))
                validation_loss_list_ema.append(validation_loss_ema / len(val_loader))

                if validation_loss < best_loss:
                    best_loss = validation_loss
                    train_utils.checkpoint(model, filename)

                # Early stopping (If the model is only getting finetuned, run at least 5 epochs. Otherwise at least 50.)
                if training_parameters.get("finetuning", None):
                    min_n_epochs = 5
                else:
                    min_n_epochs = 50

                if training_parameters["early_stopping"] and (epoch > min_n_epochs):
                    if early_stopper.early_stop(validation_loss):
                        logging_str += (
                                ",Validation loss: "
                                f"{validation_loss_list[-1]:.8f}, Validation loss EMA: {validation_loss_list_ema[-1]:.8f}"
                            )
                        logging.info(logging_str)
                        logging.info(f"EP {epoch}: Early stopping")
                        break

                if lr_schedule == "step" and early_stopper.counter >= int(training_parameters["early_stopping"] // (report_every * 2)):
                    # stepwise scheduler only happens once per epoch and only if the validation has not been going down for at least 10 epochs
                    if scheduler.get_last_lr()[0] > 10e-9:
                        scheduler.step()
                        logging.info(f"Learning rate reduced to: {scheduler.get_last_lr()[0]}")

                logging_str += (
                    ",Validation loss: "
                    f"{validation_loss_list[-1]:.8f}, Validation loss EMA: {validation_loss_list_ema[-1]:.8f}"
                )
            logging.info(logging_str)

    # Save training time
    t_training = np.array(t_training)
    train_utils.log_and_save_evaluation(
        t_training.mean(), "t_training_avg", results_dict, logging
    )
    train_utils.log_and_save_evaluation(
        np.median(t_training), "t_training_med", results_dict, logging
    )
    train_utils.log_and_save_evaluation(
        t_training.std(), "t_training_std", results_dict, logging
    )

    logging.info(using("After finishing all epochs"))

    optimizer.zero_grad(set_to_none=True)
    try:
        train_utils.resume(model, filename)
    except:
        logging.info(
            f"Proceeding with diffusion model after {training_parameters['n_epochs']} epochs of training"
        )
        train_utils.checkpoint(model, filename)

    # Plot training and validation loss
    plt.plot(epochs, training_loss_list, label="training loss")
    if val_loader is not None:
        plt.plot(epochs, validation_loss_list, label="validation loss")
        plt.plot(epochs, validation_loss_list_ema, label="validation loss EMA")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(
        os.path.join(directory, f"Datetime_{d_time}_Loss_{filename_ending}.png")
    )
    # plt.plot(epochs, grad_norm_list, label="gradient norm")
    # plt.legend()
    # plt.yscale("log")
    # plt.tight_layout()
    # plt.savefig(
    #     os.path.join(directory, f"Datetime_{d_time}_analytics_{filename_ending}.png")
    # )
    plt.close()

    if data_parameters["dataset_name"] in ["x-squared", "uniform-regression"]:
        input = (
            (torch.rand(1024, dtype=torch.float32, device=device) * 3)
            .sort()
            .values.unsqueeze(-1)
        )
        with torch.no_grad():
            if uncertainty_quantification == "diffusion":
                sampled_targets = (
                    diffusion.sample_low_dimensional(
                        model, n=input.shape[0], conditioning=input
                    )
                    .squeeze(1)
                    .to("cpu")
                )
            else:
                sampled_targets = model(input).squeeze(1).to("cpu")

            plt.plot(input.cpu(), sampled_targets, "x")

            plt.savefig(os.path.join(directory, f"Datetime_{d_time}_visualisation.png"))
            plt.close()

    train_utils.checkpoint(model, filename)

    return model, filename
