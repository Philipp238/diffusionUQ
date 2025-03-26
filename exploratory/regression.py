import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA, MLP_diffusion
import logging
from diffusion import Diffusion

# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    setup_logging(args.run_name)
    device = args.device
    # TODO:
    train_loader, val_loader, test_loader, target_dim, conditioning_dim = get_data(args)
    
    # model = UNet_conditional(num_classes=args.num_classes).to(device)
    model = MLP_diffusion(target_dim=target_dim, conditioning_dim=conditioning_dim, layers=args.layer, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=target_dim, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    training_loss_list = []
    validation_loss_list = []
    ema_validation_loss_list = []
    
    l = len(train_loader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    running_loss = 0
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)
        model.train()
        for i, (conditioning, target) in enumerate(pbar):
            conditioning = conditioning.to(device)
            target = target.to(device).unsqueeze(-1)
            t = diffusion.sample_timesteps(conditioning.shape[0]).to(device)
            x_t, noise = diffusion.noise_low_dimensional(target, t)
            if np.random.random() < 0.1 and False:
                target = None
            predicted_noise = model(x_t, t, conditioning)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            running_loss += loss.item()
            
            # logging.info(f'Training batch loss: {loss.item():.5f}')
            
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        logging.info(f'Training loss: {running_loss / len(train_loader)}')
        training_loss_list.append(running_loss / len(train_loader))
        
        if epoch % 10 == 0:
            validation_loss = 0
            ema_validation_loss = 0
            
            with torch.no_grad():
                for conditioning, target in val_loader:
                    conditioning = conditioning.to(device)
                    target = target.to(device)
                    
                    prediction = diffusion.sample_low_dimensional(model, n=len(target), conditioning=conditioning)
                    ema_prediction = diffusion.sample_low_dimensional(ema_model, n=len(target), conditioning=conditioning)
                    
                    validation_loss += mse(prediction, target).item()
                    ema_validation_loss += mse(ema_prediction, target).item()

                validation_loss /= len(val_loader)
                ema_validation_loss /= len(val_loader)
            
            validation_loss_list.append(validation_loss)
            ema_validation_loss_list.append(ema_validation_loss)
            
            logging.info(
                f"[{epoch + 1:5d}] Training loss: {training_loss_list[-1]:.8f}, Validation loss: "
                f"{validation_loss:.8f}, Ema validation loss: {ema_validation_loss:.8f}"
            )

            
            # sampled_images = diffusion.sample(model, n=len(target), labels=target)
            # ema_sampled_images = diffusion.sample(ema_model, n=len(target), labels=target)
            # plot_images(sampled_images)
            # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            # save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_regression"
    args.epochs = 100
    args.batch_size = 32
    args.batch_size_pred = 1024
    args.image_size = 64
    args.num_classes = 10
    args.dataset_name = 'x_squared'  # x_squared, housing_prices
    args.dataset_path = '/home/groups/ai/scholl/diffusion/cifar10'
    args.device = "cuda"
    args.layer = 0 
    args.hidden_dim = 128
    
    args.lr = 3e-5
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
