
import argparse
import pickle
import torchvision.transforms as tvt
import torchvision.transforms.functional as F 

import random
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Sequence, Dict
import os
import sys 
#from data.utils import log
import logging
import re 
import h5py 
import fastmri 
from fastmri.data.transforms import to_tensor
import time 

import DiffusionMBIR.models.utils as mutils
from DiffusionMBIR.models.ema import ExponentialMovingAverage
from DiffusionMBIR.utils import restore_checkpoint
from DiffusionMBIR import sde_lib, datasets, losses, sampling
from DiffusionMBIR.models import ncsnpp


# add parent directory 
directory = os.path.dirname(os.path.abspath("__file__"))
# setting path
sys.path.append(directory)
#from configs.ve.knee_320_ncsnpp_continuous import get_config 
from configs.ve.brain_256_ncsnpp_continuous import get_config 

def generate_synthetic_phase_k(patient_dict, config, ckpt_path, logger, save_dir):

    def get_test_scans(scan_list: Path, test_dict: dict):
        new_list = []
        for list_of_scan in test_dict.values():
            for scan_name in list_of_scan:
                for scan_path in scan_list:
                    if scan_path.name == scan_name:
                        new_list.append(scan_path)
        return new_list
    
    def save_batch(mag_batch : torch.Tensor, gt_phase : torch.Tensor, syn_phase : torch.Tensor, new_scan_path : Path) -> None:
        # generate folders 
        new_scan_path.mkdir()
        gt_path = new_scan_path / "gt_phase"
        syn_path = new_scan_path / "diff_phase"
        mag_path = new_scan_path / "gt_mag"
        gt_path.mkdir()
        syn_path.mkdir()
        mag_path.mkdir()
        # save generated scans and gt scans to folders 
        num_slices = gt_phase.shape[0]
        gt_phase = gt_phase.detach().cpu().numpy()
        syn_phase = syn_phase.detach().cpu().numpy()
        mag_batch = mag_batch.detach().cpu().numpy()
        for k in range(num_slices):
            slice1 = gt_phase[k]
            slice2 = syn_phase[k]
            slice3 = mag_batch[k]
            np.save(gt_path / f"slice{k+1}.npy", slice1)
            np.save(syn_path / f"slice{k+1}.npy", slice2)
            np.save(mag_path / f"slice{k+1}.npy", slice3)

    def normalize(tensor):
        min_tensor = torch.min(torch.min(tensor, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        max_tensor = torch.max(torch.max(tensor, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        tensor = (tensor - min_tensor) / (max_tensor - min_tensor)
        return tensor

    scan_list = list(Path("KneeMRI/singlecoil_val").glob("*.h5")) + list(Path("KneeMRI/singlecoil_train").glob("*.h5"))
    scan_list = [scan_name.with_suffix("") for scan_name in scan_list]
    logger.info(f"Total scan number: {len(scan_list)}")
    scan_list = get_test_scans(scan_list, patient_dict)
    logger.info(f"Selected scan number: {len(scan_list)}")
    
    # Initialize a diffusion model 
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)    
    
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
    
    # Load the checkpoint 
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    logger.info(f"Model is loaded from {ckpt_path}")

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    # new directory for the dataset to be generated 
    save_dir.mkdir(exist_ok=True)
    for scan_no, path in enumerate(scan_list, 1):
        logger.info(f"Scan No: {scan_no}/{len(scan_list)}")
        # create the folder for a scan file (ground truth and synthetic)
        new_scan_path = save_dir / path.name
        if new_scan_path.exists():
            continue
        scan_file = h5py.File(str(path) + ".h5")
        scan_kspace = to_tensor(np.array(scan_file['kspace'])).unsqueeze(dim=1)
        ifft_tensor = fastmri.fftc.ifft2c_new(scan_kspace)
        # get magnitude and phase images 
        mag_batch = torch.norm(ifft_tensor, dim=4)
        phase_batch = torch.arctan(ifft_tensor[...,1] / ifft_tensor[...,0])
        # crop and upsample images 
        crop = tvt.CenterCrop((320, 320))
        resize = tvt.Resize(size=(384, 384), antialias=False)
        mag_batch = resize(crop(mag_batch))
        phase_batch = resize(crop(phase_batch))
        # normalize images 
        mag_batch = normalize(mag_batch)
        phase_batch = normalize(phase_batch)
        # Building the sampling function
        sampling_shape = (mag_batch.shape[0], config.data.num_channels - 1,
                            config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
        logger.info(f"Number of slices: {mag_batch.shape[0]}")
        ## Sampling
        # Scale images
        mag_batch = scaler(mag_batch.to(config.device))
        phase_batch = scaler(phase_batch.to(config.device))
        # Generate synthetic phase images and save them 
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        # add frame indices 
        start_time = time.time()
        syn_phase, n = sampling_fn(score_model, mag_batch)
        end_time = time.time()
        elapsed_time = end_time - start_time 
        
        logger.info(f"Elapsed time for sample generation is {int(elapsed_time // 60)} minutes, {int(elapsed_time % 60)} seconds.")
        ema.restore(score_model.parameters())      
        # save the ground truth batch and synthetic batch
        save_batch(mag_batch.squeeze(dim=1), phase_batch.squeeze(dim=1), syn_phase.squeeze(dim=1), new_scan_path)
        logger.info("#"*20)
    logger.info("Synthetic phase images were generated succesfully!")

def generate_synthetic_phase_b(patient_dict, config, ckpt_path, logger, save_dir):
    
    def normalize(tensor):
        min_tensor = torch.min(torch.min(tensor, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        max_tensor = torch.max(torch.max(tensor, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        tensor = (tensor - min_tensor) / (max_tensor - min_tensor)
        return tensor

    def get_scan(scan_path: Path): 
        file_list = sorted(list(scan_path.glob("*.npy")), key=lambda x: int(x.name.split("slice")[1].split(".npy")[0]))
        sample = torch.from_numpy(np.load(file_list[0]))
        num_slice = len(file_list)
        ifft_tensor = torch.zeros((num_slice, 1, sample.shape[0], sample.shape[1]), dtype=torch.complex64)
        for idx, file_path in enumerate(file_list):
            ifft_tensor[idx,0] = torch.from_numpy(np.load(file_path))
        ifft_tensor = torch.view_as_real(ifft_tensor).to(torch.float32)
        # magnitude & phase 
        mag_tensor = torch.norm(ifft_tensor, dim=-1)
        phase_tensor = torch.arctan(ifft_tensor[...,1] / ifft_tensor[...,0])
        # transforms 
        crop = tvt.CenterCrop((320, 320))
        resize = tvt.Resize(size=256, antialias=False)
        mag_tensor = normalize(resize(crop(mag_tensor)))
        phase_tensor = normalize(resize(crop(phase_tensor)))
        return mag_tensor, phase_tensor
    
    def save_batch(mag_batch : torch.Tensor, gt_phase : torch.Tensor, syn_phase : torch.Tensor, new_scan_path : Path) -> None:
        # generate folders 
        new_scan_path.mkdir(exist_ok=True)
        gt_path = new_scan_path / "gt_phase"
        syn_path = new_scan_path / "diff_phase"
        mag_path = new_scan_path / "gt_mag"
        gt_path.mkdir(exist_ok=True)
        syn_path.mkdir(exist_ok=True)
        mag_path.mkdir(exist_ok=True)
        # save generated scans and gt scans to folders 
        num_slices = gt_phase.shape[0]
        gt_phase = gt_phase.detach().cpu().numpy()
        syn_phase = syn_phase.detach().cpu().numpy()
        mag_batch = mag_batch.detach().cpu().numpy()
        for k in range(num_slices):
            slice1 = gt_phase[k]
            slice2 = syn_phase[k]
            slice3 = mag_batch[k]
            np.save(gt_path / f"slice{k+1}.npy", slice1)
            np.save(syn_path / f"slice{k+1}.npy", slice2)
            np.save(mag_path / f"slice{k+1}.npy", slice3)

    all_scan_list = list(Path("./BrainMRI/singlecoil_train").glob("*/*"))
    scan_list = []
    for scan_path in all_scan_list:
        for key, value in patient_dict.items():
            for scan_name in value:
                if scan_name in str(scan_path):
                    scan_list.append(scan_path)
    logger.info(f"Total scan number: {len(scan_list)}")
    
    # Initialize a diffusion model 
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)    
    
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
    
    # Load the checkpoint 
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    logger.info(f"Model is loaded from {ckpt_path}")

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    # new directory for the dataset to be generated 
    save_dir.mkdir(exist_ok=True)
    for scan_no, path in enumerate(scan_list, 1):
        logger.info(f"Scan No: {scan_no}/{len(scan_list)}")
        # create the folder for a scan file (ground truth and synthetic)
        new_scan_path = save_dir / path.name
        if (new_scan_path / "diff_phase").exists():
            continue
        mag_batch, phase_batch = get_scan(scan_path=path)
        # Building the sampling function
        sampling_shape = (mag_batch.shape[0], config.data.num_channels - 1,
                            config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
        logger.info(f"Number of slices: {mag_batch.shape[0]}")
        ## Sampling
        # Scale images
        mag_batch = scaler(mag_batch.to(config.device))
        phase_batch = scaler(phase_batch.to(config.device))
        # Generate synthetic phase images and save them 
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        # add frame indices 
        start_time = time.time()
        syn_phase, n = sampling_fn(score_model, mag_batch)
        end_time = time.time()
        elapsed_time = end_time - start_time 
        
        logger.info(f"Elapsed time for sample generation is {int(elapsed_time // 60)} minutes, {int(elapsed_time % 60)} seconds.")
        ema.restore(score_model.parameters())      
        # save the ground truth batch and synthetic batch
        save_batch(mag_batch.squeeze(dim=1), phase_batch.squeeze(dim=1), syn_phase.squeeze(dim=1), new_scan_path)
        logger.info("#"*20)
    logger.info("Synthetic phase images were generated succesfully!")

def load_dicts(train_pkl: Path, val_pkl: Path, test_pkl: Path) -> Sequence[Dict[str, Sequence[str]]]:
    train_file = open(train_pkl, 'rb')
    val_file = open(val_pkl, 'rb')
    test_file = open(test_pkl, 'rb')

    train_dict = pickle.load(train_file)
    val_dict = pickle.load(val_file)
    test_dict = pickle.load(test_file)

    train_file.close()
    val_file.close()
    test_file.close()
    return train_dict, val_dict, test_dict

def update_dict(
    main_dict: Dict[str, Dict[str, Sequence[str]]],
    data_dict: Dict[str, Dict[str, Sequence[str]]]
) -> Dict[str, Dict[str, Sequence[str]]]:
    """add the paths in the data dictionary to main dictionary"""
    main_dict["magnitude"]["path"] += data_dict["magnitude"]["path"]
    main_dict["magnitude"]["scan"] += data_dict["magnitude"]["scan"]
    main_dict["phase"]["path"] += data_dict["phase"]["path"]
    main_dict["phase"]["scan"] += data_dict["phase"]["scan"]
    return main_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_name", default="logs", type=str, help="directory for the log file")
    parser.add_argument("--data_type", default="knee", type=str, choices=("knee", "brain"), help="type of the dataset (knee or brain)")
    args = parser.parse_args()

    # for reproducible results (important for train and test split)
    seed = 101
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmarks =False
    torch.autograd.set_detect_anomaly(True)

    config = get_config()
    #get logger 
    logging.basicConfig(filename="diff_gen.log", level=logging.INFO)
    logging.info(config)

    # generate synthetic phase images using pre-trained diff models 
    if args.data_type == "knee":
        # start generation 
        with open("KneeMRI/rec_train_val_test_dicts/train_dict.pkl", "rb") as file:
            patient_dict = pickle.load(file)
        ckpt_path = Path("knee_diff_e4/checkpoints/checkpoint_30.pth")
        save_dir = Path("KneeMRI/gen_test_data")
        generate_synthetic_phase_k(patient_dict, config, ckpt_path, logging, save_dir)
    elif args.data_type == "brain":
        with open("BrainMRI/rec_train_val_test_dicts/train_dict.pkl", 'rb') as file:
            patient_dict = pickle.load(file)
        print("Number of patients that will be generated:", len(patient_dict))
        ckpt_path = Path("brain_diff_e4/checkpoints/checkpoint_40.pth")
        save_dir = Path("BrainMRI/gen_test_data")
        generate_synthetic_phase_b(patient_dict, config, ckpt_path, logging, save_dir)
    else:
        raise ValueError(f"Unknown data type: {args.data_type}".)
