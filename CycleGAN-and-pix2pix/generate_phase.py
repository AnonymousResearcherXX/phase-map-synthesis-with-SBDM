
import torch 
from pathlib import Path 
from typing import (
    List, 
    Tuple,
    Dict,
    Sequence
)

import re 
import numpy as np 
import os 
import logging 
# GAN imports 
from models import create_model
from options.test_options import TestOptions 
import sys 
import random
import pickle 
from data import CardiacDataset
from models.pix2pix_model import Pix2PixModel
import argparse 
import torchvision.transforms as tvt 
from fastmri.data.transforms import to_tensor
import h5py
import fastmri 


def generate_synthetic_phase_knee(patient_dict, args, logger, save_dir):
        
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
            #gt_path = new_scan_path / "gt_phase"
            syn_path = new_scan_path / "gan2_phase"
            #mag_path = new_scan_path / "gt_mag"
            #gt_path.mkdir()
            syn_path.mkdir()
            #mag_path.mkdir()
            # save generated scans and gt scans to folders 
            num_slices = gt_phase.shape[0]
            #gt_phase = gt_phase.detach().cpu().numpy()
            syn_phase = syn_phase.detach().cpu().numpy()
            #mag_batch = mag_batch.detach().cpu().numpy()
            for k in range(num_slices):
                #slice1 = gt_phase[k]
                slice2 = syn_phase[k]
                #slice3 = mag_batch[k]
                #np.save(gt_path / f"slice{k+1}.npy", slice1)
                np.save(syn_path / f"slice{k+1}.npy", slice2)
                #np.save(mag_path / f"slice{k+1}.npy", slice3)
        
        def normalize(tensor):
            min_tensor = torch.min(torch.min(tensor, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
            max_tensor = torch.max(torch.max(tensor, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
            tensor = (tensor - min_tensor) / (max_tensor - min_tensor)
            return tensor

        scan_list = list(Path("../KneeMRI/singlecoil_val").glob("*.h5")) + list(Path("../KneeMRI/singlecoil_train").glob("*.h5"))
        scan_list = [scan_name.with_suffix("") for scan_name in scan_list]
        logger.info(f"Total scan number: {len(scan_list)}")
        scan_list = get_test_scans(scan_list, patient_dict)
        logger.info(f"Selected scan number: {len(scan_list)}")

        # No augmentation on new dataset 
        opt = TestOptions().parse()  # get test options
        opt.name = args.name 
        opt.netG = args.netG 
        opt.epoch = args.epoch
        opt.dataset_type = args.dataset_type
        #model = create_model(opt)
        model = Pix2PixModel(opt)
        model.setup(opt)
        
        if opt.eval:
            model.eval()

        for scan_no, path in enumerate(scan_list, 1):
            logger.info(f"Scan No: {scan_no}/{len(scan_list)}")
            # create the folder for a scan file (ground truth and synthetic)
            new_scan_path = save_dir / path.name / 'gan2_phase'
            if new_scan_path.exists():
                continue 
            if (save_dir / path.name / 'diff_phase').exists() is False:
                raise ValueError("WRONG LOCATION")
            scan_file = h5py.File(str(path) + ".h5")
            scan_kspace = to_tensor(np.array(scan_file['kspace'])).unsqueeze(dim=1)
            ifft_tensor = fastmri.fftc.ifft2c_new(scan_kspace)
            # get magnitude and phase images 
            mag_batch = torch.norm(ifft_tensor, dim=4)
            phase_batch = torch.arctan(ifft_tensor[...,1] / ifft_tensor[...,0])
            # crop and upsample images 
            crop = tvt.CenterCrop((320, 320))
            upsample = tvt.Resize(size=(256, 256), antialias=False)
            mag_batch = upsample(crop(mag_batch))
            phase_batch = upsample(crop(phase_batch))
            # normalize images 
            mag_batch = normalize(mag_batch)
            phase_batch = normalize(phase_batch)
            model.set_input((mag_batch, phase_batch))
            model.forward()
            syn_batch = (model.fake_B + 1) / 2
            save_batch(mag_batch.squeeze(dim=1), phase_batch.squeeze(dim=1), syn_batch.squeeze(dim=1), new_scan_path)
            logger.info("#"*20)
        logger.info("Synthetic phase images were generated succesfully!")   

def generate_synthetic_phase_brain(patient_dict, args, logger, save_dir):

    def save_batch(mag_batch : torch.Tensor, gt_phase : torch.Tensor, syn_phase: torch.Tensor,  new_scan_path : Path) -> None:
        # generate folders 
        new_scan_path.mkdir(exist_ok=True)
        gt_path = new_scan_path / "gt_phase"
        syn_path = new_scan_path / "gan_phase"
        mag_path = new_scan_path / "gt_mag"
        gt_path.mkdir(exist_ok=True)
        syn_path.mkdir(parents=True)
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
    
    def normalize(tensor):
        min_tensor = torch.min(torch.min(tensor, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        max_tensor = torch.max(torch.max(tensor, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        tensor = (tensor - min_tensor) / (max_tensor - min_tensor)
        return tensor


    all_scan_list = list(Path("../BrainMRI/singlecoil_train").glob("*/*"))
    scan_list = []
    for scan_path in all_scan_list:
        for key, scan_name in patient_dict.items():
            if scan_name in str(scan_path):
                scan_list.append(scan_path)
    print("Length of scan_list:", len(scan_list))

    logger.info(f"Total scan number: {len(scan_list)}")
    # No augmentation on new dataset 
    opt = TestOptions().parse()
    opt.name = "brain_gan_e4" #"knee_exp_e4"
    opt.netG = "unet_256"
    opt.epoch = 100
    opt.dataset_type = args.dataset_type 
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # Create data normalizer and its inverse

    save_dir.mkdir(exist_ok=True)
    for scan_no, path in enumerate(scan_list, 1):
        logger.info(f"Scan No: {scan_no}/{len(scan_list)}")
        # create the folder for a scan file (ground truth and synthetic)
        new_scan_path = save_dir / path.name
        if (new_scan_path / "gan_phase").exists():
            continue 
        mag_batch, phase_batch = get_scan(path)
        model.set_input((mag_batch, phase_batch))
        model.forward()
        syn_batch = (model.fake_B + 1) / 2 
        save_batch(mag_batch.squeeze(dim=1), phase_batch.squeeze(dim=1), syn_batch.squeeze(dim=1), new_scan_path)
        #save_batch(mag_batch.squeeze(dim=1), phase_batch.squeeze(dim=1), new_scan_path)
        logger.info("#"*20)
    logger.info("Synthetic phase images were generated succesfully!")   



def generate_synthetic_phase(dataset, logger):

    def get_batch(mag_path: Path, phase_path: Path) -> torch.tensor:
        mag_img_paths = sorted(list(mag_path.glob("*.npy")), key=lambda x: int(re.findall(r'\d+', str(x.name))[0]))
        phase_img_paths = sorted(list(phase_path.glob("*.npy")), key=lambda x: int(re.findall(r'\d+', str(x.name))[0]))
        assert len(mag_img_paths) == len(phase_img_paths)
        num_slices = len(mag_img_paths)
        # Initialize volumes/batches for phase sampling
        mag_batch = torch.zeros((num_slices, 1, 128, 128), dtype=torch.float32)
        phase_batch = torch.zeros_like(mag_batch)
        # Start loading data 
        for i in range(num_slices):
            mag_img = torch.from_numpy(np.load(mag_img_paths[i]))
            phase_img = torch.from_numpy(np.load(phase_img_paths[i]))
            mag_img, phase_img = dataset.transform(mag_img, phase_img)
            mag_batch[i,...], phase_batch[i,...] = mag_img, phase_img

        return mag_batch, phase_batch

    def save_batch(path: Path, sample: torch.tensor) -> None:
        num_slices = sample.shape[0]
        numpy_sample = sample.detach().cpu().numpy()
        for k in range(num_slices):
            img = numpy_sample[k,0]
            np.save(path / f"slice{k+1}.npy", img)


    # No augmentation on new dataset 
    dataset.augmentation=False 
    opt = TestOptions().parse()  # get test options
    model = create_model(opt)
    model.setup(opt)
    
    if opt.eval:
        model.eval()

    get_new_path = lambda old_path : Path(str(old_path).replace("phase", "synthetic_gan3D_phase"))

    # Create data normalizer and its inverse
    scaler = lambda x: x * 2 - 1 
    inverse_scaler = lambda x: (x + 1) / 2

    for patient_no, path in enumerate(dataset.patient_list, 1):
        patient_path = Path(path)
        logger.info(f"Patient No: {patient_no}/{len(dataset.patient_list)}")
        logger.info(f"Patient Name: {patient_path.name}")
        logger.info("#"*20)

        scan_path_list = list(patient_path.glob("*"))
        for scan_no, scan_path in enumerate(scan_path_list, 1):
            logger.info(f"Scan No: {scan_no}/{len(scan_path_list)}")
            logger.info(f"Scan Name: {scan_path.name}")
            logger.info(f"-"*20)
            # Save original magnitude images 
            mag_paths = sorted(list(scan_path.glob("magnitude?")), key=lambda x: int(re.findall(r'\d+', str(x.name))[0]))
            phase_paths = sorted(list(scan_path.glob("phase?")), key=lambda x: int(re.findall(r'\d+', str(x.name))[0]))

            # Loop for slices of the same patient 
            for slice_no in range(len(mag_paths)):
                logger.info(f"Slice No: {slice_no+1}/{len(mag_paths)}")
                # Get PyTorch batches 
                mag_path, phase_path = mag_paths[slice_no], phase_paths[slice_no]
                mag_batch, phase_batch = get_batch(mag_path, phase_path)
                frame_no = torch.arange(len(mag_batch))
                # Scale images
                mag_batch = scaler(mag_batch.to("cuda"))
                phase_batch = scaler(phase_batch.to("cuda"))
                # Generate synthetic phase images and save them 
                model.set_input((mag_batch, phase_batch, frame_no))
                model.forward()
                fake_batch = model.fake_B

                # Save the sample 
                syn_phase_path = get_new_path(phase_path)
                if not syn_phase_path.exists(): syn_phase_path.mkdir(parents=True)
                save_batch(syn_phase_path, fake_batch)
            logger.info("-"*20)
        logger.info("#"*20)
    logger.info("Synthetic phase images were generated succesfully!")    

def generate_synthetic_phase_ocmr(args):

    def get_batch(mag_path: Path, phase_path: Path) -> torch.tensor:
        mag_img_paths = sorted(list(mag_path.glob("*.npy")), key=lambda x: int(re.findall(r'\d+', str(x.name))[0]))
        phase_img_paths = sorted(list(phase_path.glob("*.npy")), key=lambda x: int(re.findall(r'\d+', str(x.name))[0]))
        assert len(mag_img_paths) == len(phase_img_paths)
        num_slices = len(mag_img_paths)
        # Initialize volumes/batches for phase sampling
        shape = np.load(mag_img_paths[0]).shape[-1]
        mag_batch = torch.zeros((num_slices, 1, shape, shape), dtype=torch.float32)
        phase_batch = torch.zeros_like(mag_batch)
        # Start loading data 
        for i in range(num_slices):
            mag_img = torch.from_numpy(np.load(mag_img_paths[i]))
            phase_img = torch.from_numpy(np.load(phase_img_paths[i]))
            mag_batch[i,...], phase_batch[i,...] = mag_img, phase_img
        print("Current shape: ", shape)
        if shape == 96:
            pad = tvt.Pad(padding=16)
            mag_batch = pad(mag_batch)
            phase_batch = pad(phase_batch)
        elif shape == 160:
            pad = tvt.Pad(padding=48)
            mag_batch = pad(mag_batch)
            phase_batch = pad(phase_batch)
        elif shape == 120:
            pad = tvt.Pad(padding=4)
            mag_batch = pad(mag_batch)
            phase_batch = pad(phase_batch)
        return mag_batch, phase_batch, shape 

    def save_batch(path: Path, sample: torch.tensor, org_shape) -> None:
        num_slices = sample.shape[0]
        shape = sample.shape[-1]
        crop = tvt.CenterCrop(size=org_shape)
        sample = crop(sample)
        numpy_sample = sample.detach().cpu().numpy()
        for k in range(num_slices):
            img = numpy_sample[k,0]
            np.save(path / f"frame{k+1}.npy", img)

    # No augmentation on new dataset 
    opt = TestOptions().parse()  # get test options
    opt.name = args.name 
    opt.netG = args.netG 
    opt.epoch = args.epoch
    opt.dataset_type = args.dataset_type
    target_path = Path(args.data_path) 
    model = create_model(opt)
    #model = Pix2PixModel(opt)
    #model = create_model(opt)
    model.setup(opt)
        
    if opt.eval:
        model.eval()
    # Create data normalizer and its inverse
    scaler = lambda x: x * 2 - 1 
    inv_scaler = lambda x: (x-x.min())/ (x-x.min()).max()
    # use the data that haven't been used in training 
    patient_dict = pickle.load(open('../data/split_dict_ocmr/test_dict.pkl', 'rb'))
    for i, (patient_id, slices) in enumerate(patient_dict.items()):
        logger.info(f"Patient No: {i+1}/{len(patient_dict)}")
        # create gan path 
        for j, slice_name in enumerate(slices):
            logger.info(f"Slice No: {j+1}/{len(slices)}")
            slice_name = slice_name.split(".npy")[0]
            slice_path = target_path / patient_id / slice_name
            gan_path = slice_path / "gan_phase"
            mag_path = slice_path / "gt_mag"
            phase_path = slice_path / "gt_phase"
            gan_path.mkdir(parents=False, exist_ok=True)
            mag_batch, phase_batch, smp_shape = get_batch(mag_path, phase_path)
            mag_batch = scaler(mag_batch.to("cuda:0"))
            phase_batch = scaler(phase_batch.to("cuda:0"))
            # Generate synthetic phase images and save them 
            model.set_input((mag_batch, phase_batch))
            model.forward()
            fake_batch = model.fake_B

            # Save the sample 
            save_batch(gan_path, fake_batch, smp_shape)
            logger.info("-"*20)
        logger.info("#"*20)
    logger.info("Synthetic phase images were generated succesfully!")  


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../OCMR/ocmr_recon_data", type=str, help="")
    parser.add_argument("--dataset_type", default="knee", choices=("knee", "cardiac", "ocmr", "brain"), help="type of the dataset to gen. phase map")
    parser.add_argument('--name', type=str, default='knee_exp_e4_256', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--netG', type=str, default='unet_256', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--epoch', type=str, default='90', help='which epoch to load? set to latest to use latest cached model')
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

    logging.basicConfig(level=logging.INFO)
    if args.dataset_type == "cardiac":
        test_dict = pickle.load(open('../data/split_dict/test_dict.pkl', 'rb'))
        dataset = CardiacDataset(data_dict=test_dict)
        dataset.augmentation = False 
        generate_synthetic_phase(dataset, logging)
    elif args.dataset_type == "knee":
        with open('../KneeMRI/rec_train_val_test_dicts/train_dict.pkl', 'rb') as file:
            patient_dict = pickle.load(file)
            file.close()
        save_dir = Path("../KneeMRI/gen_test_data")
        generate_synthetic_phase_knee(patient_dict, args, logging, save_dir)
    elif args.dataset_type == "brain":
        # start generation for recTrain
        with open("../BrainMRI/rec_train_val_test_dicts/test_dict.pkl", "rb") as file:
            patient_dict = pickle.load(file)
            file.close()
        save_dir = Path("../BrainMRI/gen_test_data")
        generate_synthetic_phase_brain(patient_dict, args, logging, save_dir)
    elif args.dataset_type == "ocmr":
        generate_synthetic_phase_ocmr(args)
        print("good job")