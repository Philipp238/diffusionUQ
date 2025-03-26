import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet, MLP_diffusion, MLP, EMA
import logging
# from torch.utils.tensorboard import SummaryWriter
from diffusion import Diffusion
from scipy import stats
import numpy as np
import copy


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    setup_logging(args.run_name)
    device = args.device
    if args.dataset_name in ['uniform-regression', 'housing_prices']:
        dataloader, validation_dataloader, test_dataloader, target_dim, conditioning_dim = get_data(args)
        # dataloader = get_data(args)
    else:
        dataloader = get_data(args)
    
    if args.diffusion:
        if args.conditioning:
            model = MLP_diffusion(target_dim=target_dim, conditioning_dim=conditioning_dim, concat=False, layers=args.layers).to(device)
        else:
            model = MLP_diffusion(target_dim=1, layers=args.layers).to(device)
    else:
        model = MLP(target_dim=target_dim, conditioning_dim=conditioning_dim, layers=args.layers).to(device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    training_loss_list = []
    validation_loss_list = []
    ema_validation_loss_list = []
    
    l = len(dataloader)
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        
        if "SLURM_JOB_ID" in os.environ:
            print('Using Slurm!')
            pbar = dataloader
        else:
            print('Not using Slurm!')
            pbar = tqdm(dataloader)
            
        running_loss = 0
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            if not args.conditioning:
                labels = None
            if args.diffusion:
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_low_dimensional(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = model(x_t, t, labels)
                loss = mse(noise, predicted_noise)
            else:
                predicted_images = model(labels)
                loss = mse(images, predicted_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            running_loss += loss.item()
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        logging.info(f'Training loss: {running_loss}')
        training_loss_list.append(loss)
            
        if epoch % 10 == 0:
            if args.dataset_name in ['uniform-regression', 'housing_prices']:
                validation_loss = 0
                ema_validation_loss = 0
                
                with torch.no_grad():
                    for images, labels in validation_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        if args.diffusion:
                            prediction = diffusion.sample_low_dimensional(model, n=len(labels), conditioning=labels)
                            ema_prediction = diffusion.sample_low_dimensional(ema_model, n=len(labels), conditioning=labels)
                        else:
                            prediction = model(labels)                
                        validation_loss += mse(prediction, images).item()
                        ema_validation_loss += mse(ema_prediction, images).item()

                    validation_loss /= len(validation_dataloader)
                    ema_validation_loss /= len(validation_dataloader)
                
                validation_loss_list.append(validation_loss)
                ema_validation_loss_list.append(ema_validation_loss)
                
                logging.info(
                    f"[{epoch + 1:5d}] Training loss: {training_loss_list[-1]:.8f}, Validation loss: "
                    f"{validation_loss:.8f}, Ema validation loss: {ema_validation_loss:.8f}"
                )
                
            if not args.dataset_name == 'housing_prices':
                if args.dataset_name == 'uniform-regression':
                    labels = (torch.randn(args.batch_size_pred, dtype=torch.float32, device=device) * 3 ).sort().values.unsqueeze(-1)
                elif args.dataset_name == 'uniform-conditioning':
                    labels = torch.randint(args.n_classes, (args.batch_size_pred, 1), dtype=torch.float32, device=device) 
                if not args.conditioning:
                    labels = None
                sampled_images = diffusion.sample_low_dimensional(model, n=args.batch_size_pred, conditioning=labels).squeeze(1).to('cpu')

                if args.dataset_name == 'uniform-conditioning' and args.conditioning:
                    labels = labels.to('cpu')[:,0]
                    for label in range(args.n_classes):
                        plt.hist(sampled_images[labels==label], label=label)
                        plt.legend()
                elif args.dataset_name == 'uniform-regression' and args.conditioning:
                    labels = labels.to('cpu')[:,0]
                    plt.plot(labels, sampled_images, 'x')
                else:
                    if args.conditioning:
                        ks_stat, p_value = stats.kstest((sampled_images - labels.to('cpu').squeeze(1)), stats.uniform.cdf)
                        print(f"KS Statistic: {ks_stat}, p-value: {p_value}")
                    
                    plt.hist(sampled_images, bins=50)
                    if args.conditioning:
                        plt.title(f'Conditioning: {labels[0,0].item():.2f}')
            plt.savefig(os.path.join("results", args.run_name, f"{epoch}.jpg"))
            plt.close()
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            # plt.plot(training_loss_list, label='Training loss')
            plt.plot(validation_loss_list, label='Validation loss')
            plt.legend()
            plt.yscale('log')
            plt.savefig(os.path.join("results", args.run_name, f"loss.jpg"))
            plt.close()
            
            
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 50
    args.batch_size = 4096 * 8
    args.batch_size_pred = 4096 * 8
    args.image_size = 1
    args.dataset_name = 'uniform-regression'
    args.conditioning = True
    args.run_name = args.dataset_name
    args.dataset_path = r"/home/groups/ai/scholl/diffusion/image-landscapes"
    args.image_num = 10000
    args.dataset_size = 4096 * 128
    args.device = "cuda"
    args.lr = 1e-3
    args.layers = 3
    args.diffusion = True
    args.n_classes = 5
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()