# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
import sys
from pathlib import Path
import argparse 
import random 
import matplotlib.pyplot as plt

import numpy as np
import logging
# Keep the import below for registering all model definitions
from DiffusionMBIR.models import ddpm, ncsnv2, ncsnpp, unet
from DiffusionMBIR import losses, sampling, likelihood, sde_lib
from DiffusionMBIR.models import utils as mutils
from DiffusionMBIR.models.ema import ExponentialMovingAverage
import DiffusionMBIR.datasets as datasets
#import evaluation
from absl import flags
import torch
from torch import nn
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from DiffusionMBIR.utils import save_checkpoint, restore_checkpoint, get_mask, kspace_to_nchw, root_sum_of_squares
#from configs.ve.knee_320_ncsnpp_continuous import get_config
from configs.ve.brain_256_ncsnpp_continuous import get_config
from data.datasets import FastMRIDataset, FastMRIBrainDataset
from torch.utils.data import DataLoader
from DiffusionMBIR import run_lib
from data.utils import log, ToTensor
import torch.nn.functional as F 

FLAGS = flags.FLAGS


def test_diff(config, workdir, target_dir, logging, test_dl, seed, epochs=[60,40,30,20,10,50]):
    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    normalizer = lambda x: (x - x.min()) / (x.max() - x.min())

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting

    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting)
    print("Device:", config.device)
    for epoch_no in epochs:
        logging.info(f"Epoch: {epoch_no}")
        for seed_i in [seed+400, seed+31, seed+61]: # seed+21, seed+200
            sample_dir = os.path.join(target_dir, f"epoch{epoch_no}/seed{seed_i}")
            Path(sample_dir).mkdir(parents=True, exist_ok=True)

            ckpt_path = os.path.join(list((workdir / "checkpoints").glob(f"*{epoch_no}.pth"))[0])
            # load the checkpoint model 
            state = restore_checkpoint(ckpt_path, state, device=config.device)
            ema.copy_to(score_model.parameters())
            (mag_batch, phase_batch) = test_dl.random_scan(seed_i) #  + 2
            # Building the sampling function
            sampling_shape = (mag_batch.shape[0], config.data.num_channels - 1,
                                config.data.image_size, config.data.image_size)
            sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
            # Start sampling 
            logging.info(f'Sampling for epoch: {epoch_no}, seed: {seed_i}')
            mag_batch = scaler(mag_batch.to(config.device))
            phase_batch = scaler(phase_batch.to(config.device))
            # Generate synthetic phase images and save them 
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            sample, n = sampling_fn(score_model, mag_batch)
            ema.restore(score_model.parameters())
            mse_score = F.mse_loss(sample, phase_batch)
            # sample save directory 
            nrow = int(np.sqrt(sample.shape[0]))
            mag_batch = inverse_scaler(mag_batch)
            phase_batch = inverse_scaler(phase_batch)
            diff_grid = make_grid(normalizer(sample), nrow, padding=2) # for synthetic phase images 
            gt_grid = make_grid(normalizer(phase_batch), nrow, padding=2) # for real phase images 
            mag_grid = make_grid(normalizer(mag_batch), nrow, padding=2) # for real magnitude images 
            combined_grid = torch.cat((mag_grid, gt_grid, diff_grid), dim=2)
            # Save image files as torch tensor
            torch.save(mag_batch.cpu(), os.path.join(sample_dir, "magnitude_phase.pt"))
            torch.save(phase_batch.cpu(), os.path.join(sample_dir, "real_phase.pt"))
            torch.save(sample.cpu(), os.path.join(sample_dir, "synthetic_phase.pt"))
            # Save images to the local disk 
            save_image(diff_grid, os.path.join(sample_dir, "synthetic_phase.png"))
            save_image(gt_grid, os.path.join(sample_dir, "real_phase.png"))
            save_image(mag_grid, os.path.join(sample_dir, "magnitude.png"))
            # save the combined grid 
            combined_grid = combined_grid.permute(1,2,0).cpu().numpy()
            num_grids = 3
            dpi = 500
            grid_width = combined_grid.shape[1]
            grid_height = combined_grid.shape[0]
            fig_width = grid_width / dpi
            fig_height = (grid_height / dpi) * num_grids
            # Create the figure with improved resolution
            plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

            # Display the combined grid using Matplotlib
            plt.imshow(combined_grid, cmap='gray')
            plt.axis('off')  # Turn off the axis
            plt.title(f"Diffusion {epoch_no}th epoch. cumulative MSE, GT and synthesized phase = {mse_score:.4f}", fontsize=15)
            plt.savefig(os.path.join(sample_dir, f"epoch{epoch_no}_seed{seed_i}.png"), transparent=True, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default="info", type=Path, help="name of the folder that contains information for the experiment")
    parser.add_argument("--log_name", default="no_name", type=Path, help="name of the log file")
    parser.add_argument("--target_dir", default="diff_results", type=Path, help="save_dir for representative cases of diffusion model")
    parser.add_argument("--dataset_type", default="brain", type=str, help="type of the dataset to be tested")
    args = parser.parse_args()

    # for reproducible results (important for train and test split)
    seed = 101
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    torch.autograd.set_detect_anomaly(True)

    config = get_config()
    print("CUDA device index:", config.device.index)

    if args.dataset_type == "knee":
        val_data = FastMRIDataset(data_path="./KneeMRI/singlecoil_kspace_train", select_size=-1, patient_dict="./KneeMRI/gen_train_val_test_dicts/val_dict.pkl")
    elif args.dataset_type == "brain":
        val_data = FastMRIBrainDataset(data_path="./BrainMRI/singlecoil_train", patient_dict="./BrainMRI/gen_train_val_dicts/val_dict.pkl")
    val_data.eval = True
    # get logger 
    logger = log("logs", args.log_name)
    logger.info(config)
    test_diff(config, args.work_dir, args.target_dir, logger, val_data, seed)
    logger.info("test diffusion ends successfully!")