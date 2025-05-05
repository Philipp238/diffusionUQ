# Implements the training functions and scripts.

import os
import torch
from torch import optim
import matplotlib.pyplot as plt
from utils import train_utils, losses
import resource
import psutil
import gc
from models import EMA, Diffusion
import copy
import numpy as np

def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # you can convert that object to a dictionary
    return f'{point}: mem (CPU python)={usage[2]/1024.0}MB; mem (CPU total)={dict(psutil.virtual_memory()._asdict())["used"] / 1024**2}MB'


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}.")


def train(net, optimizer, images, labels, criterion, gradient_clipping, uncertainty_quantification, ema, ema_model, diffusion=None, **kwargs):
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
    optimizer.zero_grad(set_to_none=True)        
    
    if uncertainty_quantification.startswith('diffusion'):
        assert not (diffusion is None)
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_low_dimensional(images, t)
        if np.random.random() < 0.1:
            labels = None
        predicted_noise = net(x_t, t, labels)
        loss = criterion(noise, predicted_noise)
    else:
        predicted_images = net(labels)
        loss = criterion(images, predicted_images)

    loss.backward()

    optimizer.step()
    ema.step_ema(ema_model, net)

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
    image_dim,
    label_dim,
    d_time,
    results_dict,
):
    """ Trainer function that takes a parameter dictionaray and dataloaders, trains the models and logs the results.

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

    # criterion = train_utils.get_criterion(training_parameters, domain_range=None, d=image_dim, device=device)
    uncertainty_quantification = training_parameters["uncertainty_quantification"]
    criterion = torch.nn.MSELoss()
    if uncertainty_quantification == "diffusion":
        train_criterion = torch.nn.MSELoss()
    elif uncertainty_quantification == "diffusion_crps":
        train_criterion = losses.NormalCRPS()
    
    model = train_utils.setup_model(
        training_parameters, device, image_dim, label_dim
    )

    if training_parameters["init"] != "default":
        train_utils.initialize_weights(model, training_parameters["init"])

    if training_parameters.get('finetuning', None):
        train_utils.resume(model, training_parameters.get('finetuning', None))
    
    n_parameters = 0
    for parameter in model.parameters():
        n_parameters += parameter.nelement()

    train_utils.log_and_save_evaluation(
        n_parameters, "NumberParameters", results_dict, logging
    )

    logging.info(f"GPU memory allocated: {torch.cuda.memory_reserved(device=device)}")
    logging.info(using("After setting up the model"))

    # create your optimizer
    if training_parameters["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_parameters["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=training_parameters["weight_decay"],
        )
    elif training_parameters["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_parameters["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=training_parameters["weight_decay"],
        )
    elif training_parameters["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=training_parameters["learning_rate"]
        )

    report_every = training_parameters['report_every']
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

    # Additional parameters
    uncertainty_quantification = training_parameters["uncertainty_quantification"]
    if uncertainty_quantification.startswith("diffusion"):
        diffusion = Diffusion(img_size=image_dim, device=device)
    else:
        diffusion = None
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # Iterate over autoregressive steps, if necessary
    logging.info(f"Training starts now.")

    for epoch in range(training_parameters["n_epochs"]):
        gc.collect()
        # logging.info(using("At the start of the epoch"))

        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            batch_loss = train(
                model,
                optimizer,
                images,
                labels,
                train_criterion,
                training_parameters["gradient_clipping"],
                uncertainty_quantification=uncertainty_quantification,
                ema=ema, 
                ema_model=ema_model, 
                diffusion=diffusion
            )
            running_loss += batch_loss

        if lr_schedule == "step" and early_stopper.counter > 5:
            # stepwise scheduler only happens once per epoch and only if the validation has not been going down for at least 10 epochs
            if scheduler.get_last_lr()[0] > 0.0001:
                scheduler.step()
                logging.info(
                    f"Learning rate reduced to: {scheduler.get_last_lr()[0]}"
                )

        if epoch % report_every == report_every - 1:
            epochs.append(epoch)
            if not uncertainty_quantification.endswith("dropout"):
                model.eval()

            validation_loss = 0
            validation_loss_ema = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    if uncertainty_quantification == 'diffusion':
                        n_samples = 10
                        repeated_labels = labels.repeat_interleave(n_samples, dim=0)
                        sampled_images = diffusion.sample_low_dimensional(model, n=repeated_labels.shape[0], conditioning=repeated_labels)
                        sampled_images_ema = diffusion.sample_low_dimensional(ema_model, n=repeated_labels.shape[0], conditioning=repeated_labels)
                        sampled_images = sampled_images.reshape(labels.shape[0], n_samples, labels.shape[1]).mean(dim=1)
                        sampled_images_ema = sampled_images_ema.reshape(labels.shape[0], n_samples, images.shape[1]).mean(dim=1)
                    elif uncertainty_quantification == 'diffusion_crps':
                        n_samples = 10
                        repeated_labels = labels.repeat_interleave(n_samples, dim=0)
                        sampled_images = diffusion.sample_crps_low_dimensional(model, n=repeated_labels.shape[0], conditioning=repeated_labels)
                        sampled_images_ema = diffusion.sample_crps_low_dimensional(ema_model, n=repeated_labels.shape[0], conditioning=repeated_labels)
                        sampled_images = sampled_images.reshape(labels.shape[0], n_samples, labels.shape[1]).mean(dim=1)
                        sampled_images_ema = sampled_images_ema.reshape(labels.shape[0], n_samples, images.shape[1]).mean(dim=1)
                    else:
                        sampled_images = model(labels)
                        sampled_images_ema = ema_model(labels)
                    
                    validation_loss += criterion(sampled_images, images).item()
                    validation_loss_ema += criterion(sampled_images_ema, images).item()
            
            validation_loss_list.append(
                validation_loss / len(val_loader)
            )
            validation_loss_list_ema.append(
                validation_loss_ema / len(val_loader)
            )
            training_loss_list.append(
                running_loss / report_every / (len(train_loader))
            )
            running_loss = 0.0

            if validation_loss < best_loss:
                best_loss = validation_loss
                filename = os.path.join(
                    directory, f"Datetime_{d_time}_Loss_{filename_ending}.pt"
                )
                train_utils.checkpoint(model, filename)

            # Early stopping (If the model is only getting finetuned, run at least 5 epochs. Otherwise at least 50.)
            if training_parameters.get('finetuning', None):
                min_n_epochs = 5
            else:
                min_n_epochs = 50
                
            if training_parameters['early_stopping'] and (epoch > min_n_epochs):

                if early_stopper.early_stop(validation_loss):
                    logging.info(f"EP {epoch}: Early stopping")
                    break
            
            logging.info(
                    f"[{epoch + 1:5d}] Training loss: {training_loss_list[-1]:.8f}, Validation loss: "
                    f"{validation_loss_list[-1]:.8f}, Valdiation loss EMA: {validation_loss_list_ema[-1]:.8f}"
                )
        
    logging.info(using("After finishing all epochs"))

    optimizer.zero_grad(set_to_none=True)
    train_utils.resume(model, filename)

    # Plot training and validation loss
    plt.plot(epochs, training_loss_list, label="training loss")
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
    
    labels = (torch.randn(1024, dtype=torch.float32, device=device) * 3 ).sort().values.unsqueeze(-1)
    
    with torch.no_grad():
        if uncertainty_quantification == 'diffusion':
            sampled_images = diffusion.sample_low_dimensional(model, n=labels.shape[0], conditioning=labels).squeeze(1).to('cpu')
        elif uncertainty_quantification == 'diffusion_crps':
            sampled_images = diffusion.sample_crps_low_dimensional(model, n=labels.shape[0], conditioning=labels).squeeze(1).to('cpu')
        else:
            sampled_images = model(labels).squeeze(1).to('cpu')

        plt.plot(labels.cpu(), sampled_images, 'x')

        plt.savefig(os.path.join(directory, f"visualisation.png"))
        plt.close()

    train_utils.checkpoint(model, filename)

    return model, filename