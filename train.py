# Implements the training functions and scripts.

import configparser
import copy
import gc
import os
import resource
import time
import logging
import json

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.distributed as dist
from scoringrules import energy_score
from torch import optim
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from models import EMA, Diffusion, DistributionalDiffusion
from utils import train_utils
from data import get_datasets


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_dist()) or dist.get_rank() == 0

def ddp_avg(value: float, device: str):
    """Average a scalar across ranks (useful if you want to average train loss across GPUs)."""
    if not is_dist():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()

def sync_scalar_from_rank0(value: float | int, device: str, dtype=torch.float32):
    """Broadcast a python scalar from rank0 to all ranks; returns the scalar on every rank."""
    t = torch.tensor([value], device=device, dtype=dtype)
    if is_dist():
        dist.broadcast(t, src=0)
    return t.item()

def sync_bool_from_rank0(flag: bool, device: str):
    return bool(sync_scalar_from_rank0(1 if flag else 0, device, dtype=torch.int32))

def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # you can convert that object to a dictionary
    return f"{point}: mem (CPU python)={usage[2] / 1024.0}MB; mem (CPU total)={dict(psutil.virtual_memory()._asdict())['used'] / 1024**2}MB"


# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"
# print(f"Using {device}.")


def train(
    net,
    optimizer,
    device,
    distributed,
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
        
        if isinstance(diffusion, DistributionalDiffusion) and diffusion.distributional_method == "iDDPM":
            loss = criterion(noise, predicted_noise, t)
        else:
            loss = criterion(noise, predicted_noise)
    else:
        predicted_noise = net(input)
        loss = criterion(target, predicted_noise)


    loss = loss / batch_accumulation
    loss.backward()

    if ((idx + 1) % batch_accumulation == 0) or (idx == length - 1):
        # Update opimizer
        optimizer.step()
        ema.step_ema(ema_model, net.module if distributed else net)
        optimizer.zero_grad(set_to_none=True)

    loss = loss.item()

    return loss


def trainer(
    gpu_id,
    data_dir,
    directory,
    training_parameters,
    data_parameters,
    filename_ending,
    target_dim,
    input_dim,
    d_time,
    results_dict,
    regressor,
    world_size = None,
):
    """Trainer function that takes a parameter dictionaray and dataloaders, trains the models and logs the results.

    Args:
        train_loader (_type_): The training dataloader.
        val_loader (_type_): The validation dataloader.
        directory (_type_): The directory to save the results.
        training_parameters (_type_): The training parameter dictionary.
        data_parameters (_type_): The data parameter dictionary.
        logger (_type_): The logger.
        filename_ending (_type_): The filename.
        domain_range (_type_): The domain range of the dataset.
        d_time (_type_): The datetime.
        results_dict (_type_): Results dictionary.

    Returns:
        _type_: Trained model and corresponding filename.
    """

    # Get datasets and loaders
    seed = training_parameters["seed"]
    training_dataset, validation_dataset = get_datasets(
        data_dir,
        data_parameters,
        training_parameters,
        seed,
        test = False
    )    

    if training_parameters['distributed_training']:
        ddp_setup(rank=gpu_id, world_size=world_size)
        logging.basicConfig(filename=os.path.join(directory, f'experiment_{gpu_id}.log'), level=logging.INFO, force = True)
        logging.info('Starting the logger in the training process.')


    logger = logging.getLogger(__name__)

        # if gpu_id==0:
        # if gpu_id>-1:
        #     logger.basicConfig(filename=os.path.join(directory, f'experiment_{gpu_id}.log'), level=logger.INFO)
        #     logger.info('Starting the logger in the training process.')
        #     print('Starting the logger in the training process.')    
        # flag tensor for (early) stopping     
        #flag_tensor = torch.zeros(1).to(f'cuda:{gpu_id}')

    # Set device correctly
    if training_parameters['distributed_training']:
        device = f'cuda:{gpu_id}'
    else:
        device = "cpu" if not torch.cuda.is_available() else "cuda"

    if device == "cpu":
        assert not training_parameters["data_loader_pin_memory"]


    # Setup up parallel dataloaders
    if training_parameters["distributed_training"]:
        train_loader = DataLoader(
            training_dataset,
            batch_size=training_parameters["batch_size"],
            sampler=DistributedSampler(training_dataset,drop_last=True),
            drop_last = True,
            pin_memory = True
        )
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset,
                batch_size=training_parameters["eval_batch_size"],
                sampler=DistributedSampler(validation_dataset, drop_last=True),
                pin_memory = True
            )
        else:
            val_loader = None
    else:
        train_loader = DataLoader(
            training_dataset,
            batch_size=training_parameters["batch_size"],
            shuffle=True
        )
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset,
                batch_size=training_parameters["eval_batch_size"],
            )
        else:
            val_loader = None
            
    # UQ setup
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
                noise_schedule=noise_schedule,
                beta_endpoints=training_parameters["beta_endpoints"],
                tau = training_parameters["tau"]
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
                noise_schedule=noise_schedule,
                beta_endpoints=training_parameters["beta_endpoints"],
                tau = training_parameters["tau"]
            )
            
        beta = diffusion.beta # need it for iDDPM
    else:
        diffusion = None
        beta = None
        
    criterion = train_utils.get_criterion(
        training_parameters, device=device, beta=beta
    )  # Different loss functions for noise prediction
    eval_criterion = (
        energy_score  # torch.nn.MSELoss() # MSE loss for evaluating generated samples
    )

           
    model = train_utils.setup_model(data_parameters, training_parameters, device, target_dim, input_dim, beta)
    # Setup distributed model
    if training_parameters["distributed_training"]:
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True, broadcast_buffers=False)

    if training_parameters["init"] != "default":
        train_utils.initialize_weights(model, training_parameters["init"])

    if training_parameters.get("finetuning", None):
        train_utils.resume(model, training_parameters.get("finetuning", None))

    n_parameters = 0
    for parameter in model.parameters():
        n_parameters += parameter.nelement()

    train_utils.log_and_save_evaluation(
        n_parameters, "NumberParameters", results_dict, logger
    )
    if is_main_process():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_reserved(device=device)}")
        logger.info(using("After setting up the model"))

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
            betas=(0.9, 0.95),
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

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model.module if training_parameters["distributed_training"] else model)
    ema_model.eval().requires_grad_(False)
    ema_model.to(device)

    cfg_scale = 3 if training_parameters["conditional_free_guidance_training"] else 0

    # Iterate over autoregressive steps, if necessary
    if is_main_process():
        logger.info(f"Training starts now.")

    filename = os.path.join(directory, f"Datetime_{d_time}_Loss_{filename_ending}.pt")

    # Gather training times
    t_training = []

    for epoch in range(training_parameters["n_epochs"]):
        # Distributed training: set epoch for sampler
        if training_parameters["distributed_training"]:
            train_loader.sampler.set_epoch(epoch)
        # Set learning rate warm up
        if epoch < warmup_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_schedule[epoch]
        elif epoch == warmup_lr and warmup_lr != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if is_main_process():
                logger.info(f"Warmup finished.")
            early_stopper.counter = 0

        gc.collect()
        if training_parameters['distributed_training']:
            dist.barrier()

        t_current_epoch = time.time()

        model.train()
        for idx, sample in enumerate(train_loader):
            target, input = sample
            target = target.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)

            batch_loss = train(
                model,
                optimizer,
                device,
                training_parameters["distributed_training"],
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
        if training_parameters['distributed_training']:
            dist.barrier()

        t_elapsed = time.time() - t_current_epoch
        if training_parameters['distributed_training']:
            elapsed_tensor = torch.tensor(t_elapsed, device=device)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            t_elapsed = elapsed_tensor.item()

        if is_main_process():
            t_training.append(t_elapsed)

        do_report = (epoch % report_every == report_every - 1)

        if do_report:
            epoch_train_loss = running_loss / report_every / len(train_loader)
            running_loss = 0.0
            avg_epoch_train_loss = ddp_avg(epoch_train_loss, device) if training_parameters['distributed_training'] else epoch_train_loss
            if is_main_process():
                epochs.append(epoch+1)
                training_loss_list.append(avg_epoch_train_loss)            
                logging_str = f"[{epoch + 1:5d}] Training loss: {avg_epoch_train_loss:.8f}"
            else:
                logging_str = ""


        validation_loss = None
        validation_loss_ema = None

        if do_report and (val_loader is not None):
            model.eval()
            ema_model.eval()
            local_val_loss = 0.0
            local_val_loss_ema = 0.0
            local_batches = 0

            with torch.no_grad():
                for target, input in val_loader:
                    target = target.to(device, non_blocking=True)
                    input = input.to(device, non_blocking=True)

                    if uncertainty_quantification == "diffusion":
                        n_samples = training_parameters["n_val_samples"]
                        repeated_pred = regressor(input).repeat_interleave(n_samples, dim=0) if regressor else None
                        repeated_labels = input.repeat_interleave(n_samples, dim=0)
                        sampled_targets = diffusion.sample_low_dimensional(model, n=repeated_labels.shape[0],
                                                                        conditioning=repeated_labels, pred=repeated_pred, cfg_scale=cfg_scale)
                        sampled_targets_ema = diffusion.sample_low_dimensional(ema_model, n=repeated_labels.shape[0],
                                                                            conditioning=repeated_labels, pred=repeated_pred, cfg_scale=cfg_scale)
                        sampled_targets = sampled_targets.reshape(input.shape[0], n_samples, *target.shape[1:]).moveaxis(1, -1)
                        sampled_targets_ema = sampled_targets_ema.reshape(input.shape[0], n_samples, *target.shape[1:]).moveaxis(1, -1)
                    else:
                        sampled_targets = model(input)
                        sampled_targets_ema = ema_model(input)

                    # Evaluation
                    v  = eval_criterion(target.flatten(start_dim=1),
                                        sampled_targets.flatten(start_dim=1, end_dim=-2),
                                        m_axis=-1, v_axis=-2, backend="torch").mean()
                    ve = eval_criterion(target.flatten(start_dim=1),
                                        sampled_targets_ema.flatten(start_dim=1, end_dim=-2),
                                        m_axis=-1, v_axis=-2, backend="torch").mean()
                    local_val_loss += v
                    local_val_loss_ema += ve
                    local_batches += 1


            # Aggregate across ranks
            if training_parameters['distributed_training']:
                if local_batches == 0:  # rank has no validation data
                    val_loss_tensor = torch.tensor([0.0, 0.0, 0.0], device=device)
                else:
                    val_loss_tensor = torch.tensor([local_val_loss, local_val_loss_ema, local_batches], device=device)
                torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                validation_loss     = (val_loss_tensor[0] / val_loss_tensor[2]).item()
                validation_loss_ema = (val_loss_tensor[1] / val_loss_tensor[2]).item()
            else:
                validation_loss     = (local_val_loss / local_batches).item()
                validation_loss_ema = (local_val_loss_ema / local_batches).item()

            if is_main_process():
                    validation_loss_list.append(validation_loss)
                    validation_loss_list_ema.append(validation_loss_ema)
                    logging_str += f", Validation loss: {validation_loss:.8f}, Validation loss EMA: {validation_loss_ema:.8f}"


            # Early stopping and checkpointing
            stop_now = False
            if is_main_process():
                improved = validation_loss < best_loss
                if improved:
                    best_loss = validation_loss
                    if training_parameters['distributed_training']:
                        train_utils.checkpoint(model.module, filename)
                    else:
                        train_utils.checkpoint(model, filename)

                if lr_schedule == "step" and early_stopper.counter >= int(training_parameters["early_stopping"] // (report_every * 2)):
                    if scheduler.get_last_lr()[0] > 1e-8:  # avoid going too low
                        scheduler.step()
                        logger.info(f"Learning rate reduced to: {scheduler.get_last_lr()[0]:.8e}")


                if training_parameters["early_stopping"]:
                    stop_now = early_stopper.early_stop(validation_loss)

            if training_parameters['distributed_training']:
                stop_now  = sync_bool_from_rank0(stop_now, device)
                best_loss = sync_scalar_from_rank0(best_loss, device)

            model.train()
            ema_model.eval()

            if is_main_process():
                logger.info(logging_str)

            if stop_now:
                logger.info(f"Early stopping activated!")
                break

            # if validation_loss < best_loss:
            #     best_loss = validation_loss
            #     if training_parameters['distributed_training']:
            #         if gpu_id == 0:
            #             train_utils.checkpoint(model.module, filename)
            #     else:
            #         train_utils.checkpoint(model, filename)

            # Early stopping (If the model is only getting finetuned, run at least 5 epochs. Otherwise at least 50.)
            # if training_parameters.get("finetuning", None):
            #     min_n_epochs = 5
            # else:
            #     min_n_epochs = 50

            # if training_parameters["early_stopping"] and (epoch > min_n_epochs):
            #     if early_stopper.early_stop(validation_loss):
            #         logger_str += (
            #                 ",Validation loss: "
            #                 f"{validation_loss_list[-1]:.8f}, Validation loss EMA: {validation_loss_list_ema[-1]:.8f}"
            #             )
            #         if is_main_process():
            #             logger.info(logger_str)
            #             logger.info(f"EP {epoch}: Early stopping")
            #         if training_parameters['distributed_training']:
            #             pass #flag_tensor += 1
            #         else:
            #             break

            # if lr_schedule == "step" and early_stopper.counter >= int(training_parameters["early_stopping"] // (report_every * 2)):
            #     # stepwise scheduler only happens once per epoch and only if the validation has not been going down for at least 10 epochs
            #     if scheduler.get_last_lr()[0] > 10e-9:
            #         scheduler.step()
            #         if is_main_process():
            #             logger.info(f"Learning rate reduced to: {scheduler.get_last_lr()[0]}")

            # logger_str += (
            #     ",Validation loss: "
            #     f"{validation_loss_list[-1]:.8f}, Validation loss EMA: {validation_loss_list_ema[-1]:.8f}"
            # )


    if training_parameters['distributed_training'] and gpu_id != 0:
        return model

    if is_main_process():
        # Save training time
        t_training = np.array(t_training)
        train_utils.log_and_save_evaluation(
            t_training.mean(), "t_training_avg", results_dict, logger
        )
        train_utils.log_and_save_evaluation(
            np.median(t_training), "t_training_med", results_dict, logger
        )
        train_utils.log_and_save_evaluation(
            t_training.std(), "t_training_std", results_dict, logger
        )
    if is_main_process():
        logger.info(using("After finishing all epochs"))

    optimizer.zero_grad(set_to_none=True)
    try:
        if training_parameters['distributed_training']:
            train_utils.resume(model.module, filename)
        else:
            train_utils.resume(model, filename)
    except:
        if is_main_process():
            logger.info(
                f"Proceeding with diffusion model after {training_parameters['n_epochs']} epochs of training"
            )
            if training_parameters['distributed_training']:
                train_utils.checkpoint(model.module, filename)
            else:
                train_utils.checkpoint(model, filename)

    if is_main_process():
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

    # Dump intermediary files
    if is_main_process():
        with open(os.path.join(directory, "results.json"), "w") as f:
            json.dump(results_dict, f)

    if training_parameters['distributed_training']:
        net = model.module
    else:
        net = model
        train_utils.checkpoint(net, filename)

    if training_parameters['distributed_training']:
        dist.destroy_process_group()

    return net
