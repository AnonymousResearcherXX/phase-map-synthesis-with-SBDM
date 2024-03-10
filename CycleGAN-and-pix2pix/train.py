"""
Training code is modified from 

General-purpose trang script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch.nn.functional as F 
import torch 
import random
import numpy as np 
import sys 
from datasets import CardiacDataset, FastMRIKneeDataset, OCMRDataset, FastMRIBrainDataset
from torch.utils.data import DataLoader
import pickle 
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path 

# remove later 
from options.test_options import TestOptions

if __name__ == '__main__':
    seed = 101 
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    torch.autograd.set_detect_anomaly(True)

    opt = TrainOptions().parse()   # get training options
    if opt.dataset_type == 'cardiac':
        train_dict = pickle.load(open("../data/split_dict_cardiac/train_dict.pkl", 'rb'))
        train_data = CardiacDataset(data_dict=train_dict)  # create a dataset given opt.dataset_mode and other options
        train_data.augmentation=True
    elif opt.dataset_type == 'knee':
        # load the dictionary files for train-val-test split 
        train_data = FastMRIKneeDataset(data_path="../KneeMRI/singlecoil_kspace_train", select_size=-1, patient_dict="../KneeMRI/gen_train_val_test_dicts/train_dict.pkl")
        val_data = FastMRIKneeDataset(data_path="../KneeMRI/singlecoil_kspace_train", select_size=-1, patient_dict="../KneeMRI/gen_train_val_test_dicts/val_dict.pkl")
        val_data.eval = True
    elif opt.dataset_type == "brain":
        train_data = FastMRIBrainDataset(data_path="../BrainMRI/singlecoil_train", patient_dict="../BrainMRI/gen_train_val_dicts/train_dict.pkl")
        val_data = FastMRIBrainDataset(data_path="../BrainMRI/singlecoil_train", patient_dict="../BrainMRI/gen_train_val_dicts/val_dict.pkl")
        val_data.eval = True 
        print("Train dict len:", len(train_data.data_dict), "patients")
        print("Val dict len:", len(val_data.data_dict), "patients")
        print("Train data size:", len(train_data), "slices")
        print("Val data size:", len(val_data), "slices")
    elif opt.dataset_type == "ocmr":
        train_dict = pickle.load(open("../data/split_dict_ocmr/train_dict.pkl", 'rb'))
        data_path = Path("./OCMR/singlecoil_ocmr")
        train_data = OCMRDataset(train_dict, data_path, test=False)
    else:
        raise ValueError("Invalid dataset type.")
    # create dataloaders 
    train_dl = DataLoader(train_data, batch_size=opt.batch_size, num_workers=16, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=opt.batch_size, num_workers=16, shuffle=False)

    dataset_size = len(train_data)    # get the number of images in the dataset.
    print("Train patient size:", len(train_data.data_dict))
    print("Validation patient size:", len(val_data.data_dict))
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    writer = SummaryWriter(log_dir=model.save_dir+"/logs")
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_dl):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    pass
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            """
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'epoch_%d' % epoch #if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            """

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs (opt.save_epoch_freq)
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(str(epoch))
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        if epoch % 10 == 0: 
            model.isTrain = False
            with torch.no_grad():
                total_mse = 0
                for i, data in enumerate(val_dl):
                    model.set_input(data)         # unpack data from dataset and apply preprocessing
                    model.forward()
                    total_mse += F.mse_loss(model.fake_B, model.real_B)
                print(f"Epoch {epoch} validation loss: {total_mse / len(val_dl)}")
                writer.add_scalar("Val MSE Loss", total_mse / len(val_dl), epoch)
                model.isTrain = True

