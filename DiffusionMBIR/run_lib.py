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

import numpy as np
import logging
# Keep the import below for registering all model definitions
from .models import ddpm, ncsnv2, ncsnpp, unet
from . import losses, sampling, likelihood, sde_lib
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
import DiffusionMBIR.datasets as datasets
#import evaluation
from absl import flags
import torch
from torch import nn
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from .utils import save_checkpoint, restore_checkpoint, get_mask, kspace_to_nchw, root_sum_of_squares

FLAGS = flags.FLAGS


def train(config, workdir, logging, train_dl, test_dl, seed):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  Path(sample_dir).mkdir(parents=True, exist_ok=True)

  tb_dir = os.path.join(workdir, "tensorboard")
  Path(tb_dir).mkdir(parents=True, exist_ok=True)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
  Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
  Path(checkpoint_meta_dir).mkdir(parents=True, exist_ok=True)

  # Resume training when intermediate checkpoints are detected
  initial_step = 0

  # Build pytorch dataloader for training
  num_data = len(train_dl.dataset)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

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
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  ckpt_path = os.path.join(checkpoint_dir, config.checkpts.name)
  if config.checkpts.load:
    # load the checkpoint model 
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    logging.info(f"Model is loaded from {config.checkpts.name}")
    initial_step = int(state["step"])
  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
  for epoch in range(config.training.init_epoch, config.training.epochs):
    logging.info('=================================================')
    logging.info(f'Epoch: {epoch}')
    logging.info('=================================================')

    for step, (mag_img, phase_img) in enumerate(train_dl, start=1):
      mag_img = scaler(mag_img.to(config.device))
      phase_img = scaler(phase_img.to(config.device))
      # Execute one training step
      loss = train_step_fn(state, (mag_img, phase_img))
      if step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e" % (step, loss))
        global_step = num_data * epoch + step
        writer.add_scalar("training_loss", scalar_value=loss, global_step=global_step)
    if (step != 0 and epoch % 10 == 0):
      save_checkpoint(checkpoint_meta_dir, state)
      # Report the loss on an evaluation dataset periodically
      eval_mloss = 0 
      for (mag_img_val, phase_img_val) in test_dl:
        mag_img_val = scaler(mag_img_val).to(config.device)
        phase_img_val = scaler(phase_img_val).to(config.device)
        eval_loss = eval_step_fn(state, (mag_img_val, phase_img_val))
        eval_mloss += eval_loss
      eval_mloss /= len(test_dl)
      logging.info("epoch: %d, eval_loss: %.5e" % (epoch, eval_mloss))
      global_step = num_data * epoch + step
      writer.add_scalar("eval_loss", scalar_value=eval_loss, global_step=global_step) 
    if epoch % 5 == 0 or epoch == config.training.epochs-1 or epoch == 0:
      # Save a checkpoint for every five epoch 
      save_checkpoint(checkpoint_dir, state, name=f'checkpoint_{epoch}.pth')
      print("Model is saved.")

    if config.training.snapshot_sampling and epoch % 10 == 0: 

      # Get a random scan/volume
      #if val_idx_on: 
      #  (mag_batch, phase_batch) = train_dl.dataset.get_val_scan(test_dl, seed)
      #else:
      (mag_batch, phase_batch) = test_dl.dataset.random_scan(seed) #  + 2

      # Building the sampling function
      sampling_shape = (mag_batch.shape[0], config.data.num_channels - 1,
                        config.data.image_size, config.data.image_size)
      sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
      # Start sampling 
      logging.info('sampling')
      mag_batch = scaler(mag_batch.to(config.device))
      phase_batch = scaler(phase_batch.to(config.device))
      # Generate synthetic phase images and save them 
      ema.store(score_model.parameters())
      ema.copy_to(score_model.parameters())
      sample, n = sampling_fn(score_model, mag_batch)
      if config.data.is_complex:
        sample = root_sum_of_squares(sample, dim=1).unsqueeze(dim=0)
      ema.restore(score_model.parameters())
      this_sample_dir = os.path.join(sample_dir, "iter_{}".format(epoch))
      Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
      nrow = int(np.sqrt(sample.shape[0]))
      mag_batch = inverse_scaler(mag_batch)
      phase_batch = inverse_scaler(phase_batch)
      image_grid = make_grid(sample, nrow, padding=2) # for synthetic phase images 
      image_grid2 = make_grid(phase_batch, nrow, padding=2) # for real phase images 
      image_grid3 = make_grid(mag_batch, nrow, padding=2) # for real magnitude images 
      #sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
      #np.save(os.path.join(this_sample_dir, "sample"), sample)
      # Save images to tensorboard 
      writer.add_image("synthetic_phase", image_grid, epoch)
      writer.add_image("real_phase", image_grid2, epoch)
      writer.add_image("magnitudes", image_grid3, epoch)
      # Save images to the local disk 
      save_image(image_grid, os.path.join(this_sample_dir, "synthetic_phase.png"))
      save_image(image_grid2, os.path.join(this_sample_dir, "real_phase.png"))
      save_image(image_grid3, os.path.join(this_sample_dir, "magnitude.png"))

  
