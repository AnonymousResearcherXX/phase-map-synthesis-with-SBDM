


import argparse
from DiffusionMBIR import run_lib 
#from DiffusionMBIR.configs.ve.fastmri_knee_320_ncsnpp_continuous_complex_magpha import get_config
#from configs.ve.cardiac_128_ncsnpp_continuous import get_config
#from configs.ve.knee_320_ncsnpp_continuous import get_config
from configs.ve.brain_256_ncsnpp_continuous import get_config
#from configs.ve.ocmr_ncsnpp_continuous import get_config
import torchvision.transforms as tvt
from data.datasets import FastMRIDataset, CardiacDataset, OCMRDataset, BrainDataset, FastMRIBrainDataset
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
    parser.add_argument("--dataset", default="cardiac", choices=("fastmri_knee", "ocmr", "cardiac", "fastmri_brain"), type=str, help="type of the dataset that diffusion model will be trained")
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
    elif args.dataset == "cardiac":
        dataset = CardiacDataset(data_path="cardiac_data")

        # prepare train-val-test datasets 
        train_data, others = dataset.split(70, 24, seed)
        val_data, test_data = others.split(4, 20, seed)
        train_data.augmentation=True

        train_dl = DataLoader(train_data, batch_size=config.training.batch_size, num_workers=8, shuffle=True, pin_memory=True) 
        val_dl = DataLoader(val_data, batch_size=config.training.batch_size, num_workers=8, shuffle=False, pin_memory=True)
        #test_dl = DataLoader(test_data, batch_size=config.training.batch_size, num_workers=2, shuffle=False)

        #get logger 
        logger = log("logs", args.log_name)
        logger.info(config)

        logger.info(f"Validation dataset: {val_data.get_patients()}")

        run_lib.train(config, args.folder_name, logger, train_dl, val_dl, seed)
        logger.info("training ends successfully!")
    elif args.dataset == "ocmr":
        # load the index dict for data splitting 
        data_path = Path("./OCMR/singlecoil_ocmr")
        # create data dictionary 
        data_dict = OrderedDict()
        patient_list = sorted(list(data_path.glob("*")))
        for patient in patient_list:
            patient_id = patient.name
            keys = sorted(list(data_dict.keys()))
            slices = [slice.name for slice in sorted(list(patient.glob("*.npy")))]
            if patient_id in keys:
                data_dict[patient_id] += slices
            else:
                data_dict[patient_id] = slices
        # train-test split 
        num_patients = len(patient_list)
        idxs = np.arange(num_patients)
        train_size = round(num_patients * args.ratio)
        test_size = num_patients - train_size
        train_idxs = np.random.choice(idxs, size=train_size, replace=False)
        test_idxs = np.setdiff1d(idxs, train_idxs)
        import sys; sys.exit()
        # save indices for later use  
        np.save(Path(data_path.parent) / "train_idxs.npy", train_idxs)
        np.save(Path(data_path.parent) / "test_idxs.npy", test_idxs)
        # create train_and test datasets 
        dataset = OCMRDataset(data_dict, data_path, test=False)
        train_data, test_data = dataset.split(train_idxs, test_idxs)
        with open("test_dict.pkl", 'wb') as file:
            pickle.dump(test_data.patient_dict, file)
        with open("train_dict.pkl", "wb") as file:
            pickle.dump(train_data.patient_dict, file)
        import sys; sys.exit()
        test_data.test = False
        print("Train set number of slices:", len(train_data))
        print("Test set number of slices:", len(test_data))

        train_dl = DataLoader(train_data, batch_size=1, num_workers=20, shuffle=True)
        test_dl = DataLoader(test_data, batch_size=1, shuffle=False)
        #test_dl = DataLoader(test_data, batch_size=1, shuffle=False)

        #get logger 
        logger = log("logs", args.log_name)
        logger.info(config)

        logger.info(f"train-test split (slice): {len(train_data)}-{len(test_data)}")

        run_lib.train(config, args.folder_name, logger, train_dl, test_dl, seed)
        logger.info("training ends successfully!")
    elif args.dataset == "fastmri_brain":
        train_path = Path("./BrainMRI/train_data")
        val_path = Path("./BrainMRI/val_data")
        data_paths = (train_path, val_path)
        data = BrainDataset(paths=data_paths)
        # train-test split 
        train_data, val_data, test_idxs = data.split(ratio=args.ratio, seed=seed)
        # save the test idxs to be used later 
        #print(train_data.path_list)
        np.save("./BrainMRI/test_idxs.npy", test_idxs)
        train_dl = DataLoader(train_data, batch_size=config.training.batch_size, num_workers=20, shuffle=True)
        val_dl = DataLoader(val_data, batch_size=1, shuffle=False)

        #get logger 
        logger = log("logs", args.log_name)
        logger.info(config)

        logger.info(f"train-test split: {len(train_data)}-{len(val_data)}")

        run_lib.train(config, args.folder_name, logger, train_dl, val_dl, seed)
        logger.info("training ends successfully!")
        
    else:
        raise ValueError("There is no dataset called", args.dataset)