


import argparse
from DiffusionMBIR import run_lib 
#from DiffusionMBIR.configs.ve.fastmri_knee_320_ncsnpp_continuous_complex_magpha import get_config
from configs.ve.brain_256_ncsnpp_continuous import get_config
import torchvision.transforms as tvt
from data.datasets import FastMRIDataset, BrainDataset, FastMRIBrainDataset
from torch.utils.data import DataLoader
from data.utils import log, ToTensor
import random
import torch 
import numpy as np
import pickle 
from pathlib import Path
from collections import OrderedDict 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", default="info", type=str, help="name of the folder that contains information for the experiment")
    parser.add_argument("--log_name", default="no_name", type=str, help="name of the log file")
    parser.add_argument("--dataset", default="fastmri_brain", choices=("fastmri_knee", "fastmri_brain"), type=str, help="type of the dataset that diffusion model will be trained")
    parser.add_argument("--ratio", default=.5, type=float, help="ratio for train-test split")
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
    
    # initalize datasets
    if args.dataset == "fastmri_knee":
        #train_data = FastMRIDataset(data_path="./KneeMRI/singlecoil_kspace_train", select_size=-1, patient_dict="./KneeMRI/dict_paths/train_dict.pkl")
        #val_data = FastMRIDataset(data_path="./KneeMRI/singlecoil_kspace_train", select_size=-1, patient_dict="./KneeMRI/dict_paths/val_dict.pkl")
        #val_data.eval = True
    
        data = FastMRIDataset(data_path="./KneeMRI/singlecoil_kspace_train", select_size=360, patient_dict="./KneeMRI/train_dict.pkl")
        train_data, val_data = data.split(patient1=330, patient2=30)
        val_data.eval = True
        # save the data dictionaries 
        save_dir = Path("./KneeMRI/dict_paths")
        save_dir.mkdir(exist_ok=True)
        
        with open(save_dir / "train_dict.pkl", 'wb') as file:
            pickle.dump(train_data.data_dict, file)
        with open(save_dir / "val_dict.pkl", 'wb') as file:
            pickle.dump(val_data.data_dict, file)
        print("Train dict len:", len(train_data.data_dict), "patients")
        print("Val dict len:", len(val_data.data_dict), "patients")

        train_dl = DataLoader(train_data, batch_size=config.training.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        val_dl = DataLoader(val_data, batch_size=config.training.batch_size, num_workers=8, shuffle=False, pin_memory=True)
        #print("Scan list: ", train_data.scan_list)
        # get logger 
        logger = log("logs", args.log_name)
        logger.info(config)

        run_lib.train(config, args.folder_name, logger, train_dl, val_dl, seed)
        logger.info("training ends successfully!")
    elif args.dataset == "fastmri_brain":
        train_data = FastMRIBrainDataset(data_path="./BrainMRI/singlecoil_train", patient_dict="./BrainMRI/gen_train_val_dicts/train_dict.pkl")
        val_data = FastMRIBrainDataset(data_path="./BrainMRI/singlecoil_train", patient_dict="./BrainMRI/gen_train_val_dicts/val_dict.pkl")
        val_data.eval = True 
        print("Train dict len:", len(train_data.data_dict), "patients")
        print("Val dict len:", len(val_data.data_dict), "patients")
        print("Train data size:", len(train_data), "slices")
        print("Val data size:", len(val_data), "slices")
        train_dl = DataLoader(train_data, batch_size=config.training.batch_size, num_workers=16, shuffle=True, pin_memory=True)
        val_dl = DataLoader(val_data, batch_size=config.training.batch_size, num_workers=16, shuffle=False, pin_memory=True)
        logger = log("logs", args.log_name)
        logger.info(config)
        run_lib.train(config, args.folder_name, logger, train_dl, val_dl, seed)
        logger.info("training ends successfully!")
    else:
        raise ValueError("There is no dataset called", args.dataset)
