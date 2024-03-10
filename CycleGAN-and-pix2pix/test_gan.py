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
import pickle

import numpy as np
# Keep the import below for registering all model definitions
#import evaluation
from absl import flags
import torch
from torch import nn
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from options.test_options import TestOptions
from models import create_model
from data import FastMRIDataset, FastMRIBrainDataset
from models.pix2pix_model import Pix2PixModel

FLAGS = flags.FLAGS


def test_gan(args, test_dl, seed, epochs=[10,20,30,40,50,60,70,80,90,100]):
    normalizer = lambda x: (x - x.min()) / (x.max() - x.min())
    target_dir = args.target_dir
    # set the model options
    opt = TestOptions().parse()
    #opt = Pix2PixModel().parse()
    opt.name = args.exp_name
    opt.netG = args.netG
    opt.dataset_type = args.dataset_type 
    model = create_model(opt)
    #print(opt)
    #model = Pix2PixModel(opt)

    val_dl = DataLoader(test_dl, batch_size=32, num_workers=16, shuffle=False)
    val_losses = []

    for epoch_no in epochs:
        print(f"Epoch: {epoch_no}")
        for seed_i in [101, seed+200]: # seed+21, seed+200, seed+400
            sample_dir = target_dir / f"epoch{epoch_no}/{seed_i}"
            Path(sample_dir).mkdir(parents=True, exist_ok=True)

            opt.epoch = f"{epoch_no}"
            # load the checkpoint model
            model.setup(opt)
            model.eval()
            # get a random scan from the dataset
            (mag_batch, phase_batch) = test_dl.random_scan(seed_i) #  + 2
            import sys; sys.exit()
            # Start sampling 
            print(f'Generating for epoch: {epoch_no}, seed: {seed_i}')
            # Generate synthetic phase images and save them 
            model.set_input((mag_batch, phase_batch))
            model.forward()
            fake_batch = model.fake_B.to(phase_batch.device)
            fake_batch = (fake_batch + 1) / 2
            mse_score = F.mse_loss(fake_batch, phase_batch)
            # sample save directory 
            nrow = int(np.sqrt(fake_batch.shape[0]))
            diff_grid = make_grid(normalizer(fake_batch), nrow, padding=2) # for synthetic phase images 
            gt_grid = make_grid(normalizer(phase_batch), nrow, padding=2) # for real phase images 
            mag_grid = make_grid(normalizer(mag_batch), nrow, padding=2) # for real magnitude images 
            combined_grid = torch.cat((mag_grid, gt_grid, diff_grid), dim=2)
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
            plt.title(f"GAN {epoch_no}th epoch. cumulative MSE, GT and synthesized phase = {mse_score:.4f}", fontsize=10)
            plt.savefig(os.path.join(sample_dir, f"epoch{epoch_no}_seed{seed_i}.png"), transparent=True, bbox_inches="tight")
            plt.close()
        
        # calculate validation loss 
        with torch.no_grad():
            total_mse = 0
            for i, data in enumerate(val_dl):
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.forward()
                total_mse += F.mse_loss(model.fake_B, model.real_B)
            print(f"Epoch {epoch_no} validation loss: {total_mse / len(val_dl)}")
            val_losses.append(total_mse / len(val_dl))

    with open("val_losses.pkl", 'wb') as file:
        pickle.dump(val_losses, file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_name", default="no_name", type=Path, help="name of the log file")
    parser.add_argument("--target_dir", default="knee_gan_e4_256results", type=Path, help="save_dir for representative cases of GAN model")
    parser.add_argument("--netG", default="unet_256", type=str, help="type of the generator in GAN")
    parser.add_argument("--exp_name", default="knee_exp_e4_256", type=Path, help="name of the training whose models will be tested")
    parser.add_argument("--dataset_type", default="knee", type=str, help="type of the dataset to be used in test")
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

    if args.dataset_type == "knee":
        val_data = FastMRIDataset(data_path="../KneeMRI/singlecoil_kspace_train", select_size=-1, patient_dict="../KneeMRI/gen_train_val_test_dicts/val_dict.pkl")
    elif args.dataset_type == "brain":
        val_data = FastMRIBrainDataset(data_path="../BrainMRI/singlecoil_train", patient_dict="../BrainMRI/gen_train_val_dicts/val_dict.pkl")
    val_data.eval = True
    # get logger 
    test_gan(args, val_data, seed)
    print("test GAN ends successfully!")