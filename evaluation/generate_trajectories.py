import os
import sys
from models import (
    UNetDiffusion,
    UNet_diffusion_normal,
    UNet_diffusion_sample,
    UNet_diffusion_mixednormal,
    UNet_diffusion_mvnormal,
)
import torch
from models import (
    Diffusion,
    generate_diffusion_samples_low_dimensional,
    DistributionalDiffusion,
)
import matplotlib.pyplot as plt
import numpy as np
from data import PDE1D

device = "cuda"


def get_trajectory(
    model, input, target, distributional_method, t_steps, n_samples, device
):
    grid_input = input[:, 2:3, :]
    t0 = input[:, 1:2, :]
    full_array = torch.zeros(
        input.shape[0], 1, input.shape[2], t_steps, n_samples, device=device
    )

    for i in range(n_samples):
        autoregressive_input = input.clone()
        pred_array = torch.zeros(
            input.shape[0], 1, input.shape[2], t_steps, device=device
        )
        for t in range(t_steps):
            pred = generate_diffusion_samples_low_dimensional(
                model=model,
                input=autoregressive_input,
                n_timesteps=50,
                target_shape=target.shape,
                n_samples=1,
                distributional_method=distributional_method,
                x_T_sampling_method="standard",
                cfg_scale=0,
                noise_schedule="linear",
                beta_endpoints=beta_endpoints,
            )
            if t == 0:
                pred_array[..., t] = pred.squeeze(-1) + t0
            else:
                pred_array[..., t] = pred.squeeze(-1) + pred_array[..., t - 1]

            if t == 1:
                autoregressive_input = torch.cat(
                    [t0, pred_array[..., 0], grid_input], dim=1
                )
            elif t > 1:
                autoregressive_input = torch.cat(
                    [pred_array[..., t - 1], pred_array[..., t], grid_input], dim=1
                )
        # Save
        full_array[..., i] = pred_array
    return full_array.cpu()


if __name__ == "__main__":
    torch.manual_seed(3)
    np.random.seed(3)
    save_path = "evaluation/trajectories/"

    test_dataset = PDE1D(
        data_dir="data/",
        pde="KS",
        var="test",
        downscaling_factor=1,
        normalize=True,
        last_t_steps=2,
        temporal_downscaling_factor=2,
        select_timesteps="zero",
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=3,
        shuffle=True,
        num_workers=0,
    )

    target_dim, input_dim = (
        (1, *test_dataset.get_dimensions()),
        (3, *test_dataset.get_dimensions()),
    )
    grid = test_dataset.get_coordinates()[0]

    # Parameters
    n_samples = 50
    t_steps = 50

    # Generate test data
    n_test = len(test_dataset)
    n = 5
    indices = np.random.choice(n_test, n, replace=False)
    input = []
    target = []
    trajectory = []
    for idx in indices:
        target_tensor, input_tensor = test_dataset.get_trajectory(idx, length=t_steps)
        target_tensor = target_tensor.unsqueeze(0).to(device)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        input.append(input_tensor)
        target.append(target_tensor[:, :, 2] - input_tensor[:, -2])
        trajectory.append(target_tensor)
    input = torch.cat(input, dim=0)
    target = torch.cat(target, dim=0)
    trajectory = torch.cat(trajectory, dim=0).cpu()
    np.save(save_path + "KS_truth.npy",trajectory)

    # Deterministic
    ckpt_path = "results/KS/deterministic/Datetime_20250902_163740_Loss_1D_KS_UNet_diffusion_deterministic_T50_DDIM1.pt"
    distributional_method = "deterministic"
    beta_endpoints = (0.001, 0.35)

    model = UNetDiffusion(
        d=1,
        conditioning_dim=3,
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
        init_features=64,
        domain_dim=target_dim,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    trajectory = get_trajectory(
        model, input, target, distributional_method, t_steps, n_samples, device
    )
    np.save(save_path + f"KS_{distributional_method}.npy",trajectory)

    # Normal
    ckpt_path = "results/KS/normal/Datetime_20250831_213836_Loss_1D_KS_UNet_diffusion_normal_T50_DDIM1.pt"
    distributional_method = "normal"
    beta_endpoints = (0.001, 0.2)

    backbone = UNetDiffusion(
        d=1,
        conditioning_dim=3,
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
        init_features=64,
        domain_dim=target_dim,
    )
    model = UNet_diffusion_normal(
        backbone=backbone,
        d=1,
        target_dim=1,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    trajectory = get_trajectory(
        model, input, target, distributional_method, t_steps, n_samples, device
    )
    np.save(save_path + f"KS_{distributional_method}.npy",trajectory)

    # Mixed normal
    ckpt_path = "results/KS/mixednormal/Datetime_20250901_093001_Loss_1D_KS_UNet_diffusion_mixednormal_T50_DDIM1.pt"
    distributional_method = "mixednormal"
    beta_endpoints = (0.001, 0.2)
    backbone = UNetDiffusion(
        d=1,
        conditioning_dim=3,
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
        init_features=64,
        domain_dim=target_dim,
    )
    model = UNet_diffusion_mixednormal(
        backbone=backbone,
        d=1,
        target_dim=1,
        n_components=50,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    trajectory = get_trajectory(
        model, input, target, distributional_method, t_steps, n_samples, device
    )
    np.save(save_path + f"KS_{distributional_method}.npy",trajectory)

    # Multivariate normal
    ckpt_path = "results/KS/mvnormal/Datetime_20250901_160327_Loss_1D_KS_UNet_diffusion_mvnormal_T50_DDIM1.pt"
    distributional_method = "mvnormal"
    beta_endpoints = (0.001, 0.35)
    backbone = UNetDiffusion(
        d=1,
        conditioning_dim=3,
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
        init_features=64,
        domain_dim=target_dim,
    )
    model = UNet_diffusion_mvnormal(
        backbone=backbone,
        d=1,
        target_dim=1,
        domain_dim=target_dim[1:],
        rank=1,
        method="lora",
    )

    dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(dict)
    model = model.to(device)
    trajectory = get_trajectory(
        model, input, target, distributional_method, t_steps, n_samples, device
    )
    np.save(save_path + f"KS_{distributional_method}.npy",trajectory)

    # Sample
    ckpt_path = "results/KS/sample/Datetime_20250901_212819_Loss_1D_KS_UNet_diffusion_sample_T50_DDIM1.pt"
    distributional_method = "sample"
    beta_endpoints = (0.001, 0.35)

    backbone = UNetDiffusion(
        d=1,
        conditioning_dim=4,
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
        init_features=64,
        domain_dim=target_dim[1:],
    )
    model = UNet_diffusion_sample(backbone=backbone, d=1, target_dim=1, hidden_dim=64)

    dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(dict)
    model = model.to(device)
    trajectory = get_trajectory(
        model, input, target, distributional_method, t_steps, n_samples, device
    )
    np.save(save_path + f"KS_{distributional_method}.npy",trajectory)
