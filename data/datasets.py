
import os 
import h5py 
import time 
import numpy as np 
import random
import sys 
import pickle
import copy 
import re
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)


# torch imports
import torch 
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torchvision.transforms as tvt
import torchvision.transforms.functional as F
from torchvision.utils import save_image, make_grid
from torchvision.utils import save_image
# my imports 
from .utils import generate_dataset
from .utils import ToTensor
# fastMRI imports 
import fastmri
import fastmri.fftc as fftc
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
from fastmri.data.subsample import EquispacedMaskFractionFunc, EquiSpacedMaskFunc
from collections import OrderedDict

class BrainDataset(Dataset):
    def __init__(self, paths=None, path_list=None, test=False):
        self.test = test
        if path_list == None:
            path_list1 = sorted(list(paths[0].glob("*/*")))
            path_list2 = sorted(list(paths[1].glob("*/*"))) 
            path_list = path_list1 + path_list2
            # remove small batches & shapes 
            self.scan_list = []
            self.tresh_list = []
            for path in path_list:
                # check number of scans 
                files = list(path.rglob("*.npy"))
                img = np.load(files[0])
                height, width = img.shape[0], img.shape[1]
                num_files = len(files)
                if height == width and height == 320 and num_files >= 4: 
                    self.scan_list.append(path)
                else:
                    self.tresh_list.append(path)
        else:
            self.scan_list = path_list

        # create .npy list for training
        self.file_list = [] 
        for filename in self.scan_list:
            self.file_list += list(filename.glob("*.npy"))
        self.file_tresh = []
        for filename in self.tresh_list:
            self.file_tresh += list(filename.glob("*.npy"))

    def filter(self, idx_list):
        new_data_list = [self.scan_list[idx] for idx in idx_list]
        return BrainDataset(path_list=new_data_list, test=self.test)
            
    def split(self, ratio=.5, seed=2):
        np.random.seed(seed)
        num_scans = len(self.scan_list)
        train_size = int(num_scans * ratio)
        train_idxs = np.random.choice(num_scans, train_size)
        test_idxs = np.setdiff1d(np.arange(num_scans), train_idxs)
        # create path lists 
        train_list = [self.scan_list[idx] for idx in train_idxs]
        test_list = [self.scan_list[idx] for idx in test_idxs]
        return BrainDataset(path_list=train_list), BrainDataset(path_list=test_list, test=True), test_idxs

    def transforms(self, mag_img, phase_img, p=.3): # p=0.3
        # Standard scale
        scaler = ToTensor()
        mag_img = scaler(mag_img)
        phase_img = scaler(phase_img)

        p = 0 if self.test else p
        if random.random() < p:
            ## Random Transforms
            # Random Rotation 
            angle = random.random() * 120.0 - 60.0
            mag_img = F.rotate(mag_img, angle)
            phase_img = F.rotate(phase_img, angle)        
        return mag_img, phase_img
    
    def _read_scan(self, path):
        file_list = list(path.glob("*.npy"))
        num_files = len(file_list)
        img = np.load(file_list[0])
        scan = np.empty((num_files, 1, img.shape[0], img.shape[1], 2), dtype=np.float32)
        scan[0] = img
        for i in range(1, num_files):
            img = np.load(file_list[i])
            scan[i] = img
        # convert to batch form 
        scan = torch.from_numpy(scan)
        mag_img = torch.sum(torch.pow(scan, 2), dim=-1)
        phase_img = torch.atan(torch.div(scan[...,1], scan[...,0]))
        mag_img, phase_img  = self.transforms(mag_img, phase_img)
        return mag_img, phase_img

    def random_scan(self, seed=1):
        np.random.seed(seed)
        idx = np.random.randint(len(self.scan_list))
        scan_path = self.scan_list[idx]
        return self._read_scan(scan_path)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        filepath = self.file_list[index]
        scan = np.load(filepath)
        scan = torch.from_numpy(scan)
        # get mag and phase components
        mag_img = torch.sum(torch.pow(scan, 2), dim=-1)
        phase_img = torch.atan(torch.div(scan[...,1], scan[...,0]))
        mag_img, phase_img = self.transforms(mag_img[None,...], phase_img[None,...])
        return mag_img, phase_img

class FastMRIDataset(Dataset):
    def __init__(self, data_path=None, path_scan_list=None, select_size=330, patient_dict=None):
        if isinstance(patient_dict, str) or isinstance(patient_dict, Path):
            with open(patient_dict, 'rb') as file:
                patient_dict = pickle.load(file)
                # create the subsampled dictionary 
                if select_size == -1:
                    self.patient_ids = list(patient_dict.keys())
                else:
                    self.patient_ids = random.sample(list(patient_dict.keys()), k=select_size)
                self.data_dict = {key: value for key, value in patient_dict.items() if key in self.patient_ids}
                scan_names = [name for scan_list in self.data_dict.values() for name in scan_list]
        elif isinstance(patient_dict, dict):
            self.data_dict = patient_dict
            self.patient_ids = list(self.data_dict.keys())
        else:
            raise ValueError("Invalid patient_dict type:", type(patient_dict))

        if path_scan_list is not None:
            self.path_list, self.scan_list = path_scan_list 
        else:
            self.path_list = []
            self.scan_list = []
            folder_list = os.listdir(data_path)
            for folder_name in folder_list:
                if folder_name not in scan_names:
                    continue
                folder_path = os.path.join(data_path, folder_name)
                self.scan_list.append(folder_path)
                file_list = os.listdir(folder_path)
                for f in file_list:
                    full_path = os.path.join(folder_path, f)
                    self.path_list.append(full_path)
            print("Dataset is constructed!")
        self.data_path = data_path
        self.eval = False 
        self.color_jitter = tvt.ColorJitter(brightness=0.2, contrast=0.2)
        #self.padder = tvt.Pad(padding=32)
        self.resize = tvt.Resize(size=(256, 256), antialias=False)

    def split(self, patient1=330, patient2=30):
        select_patient2 = random.sample(self.patient_ids, patient2)
        data_dict1, data_dict2 = {}, {}
        scan_names1, scan_names2 = [], []
        # get the scan names and patient splits
        for id, scan_list in self.data_dict.items():
            if id in select_patient2:
                data_dict2[id] = self.data_dict[id]
                scan_names2 += scan_list
            else:
                data_dict1[id] = self.data_dict[id]
                scan_names1 += scan_list
                self.path_list = []

        scan_list1, scan_list2 = [], []
        path_list1, path_list2 = [], []
        folder_list = os.listdir(self.data_path)
        for folder_name in folder_list:
            if folder_name in scan_names1:
                scan_list = scan_list1
                path_list = path_list1
            elif folder_name in scan_names2:
                scan_list = scan_list2
                path_list = path_list2
            else:
                continue
            folder_path = os.path.join(self.data_path, folder_name)
            scan_list.append(folder_path)
            file_list = os.listdir(folder_path)
            for f in file_list:
                full_path = os.path.join(folder_path, f)
                path_list.append(full_path)
        path_scan_list1 = (path_list1, scan_list1)
        path_scan_list2 = (path_list2, scan_list2)
        return (FastMRIDataset(data_path=self.data_path, path_scan_list=path_scan_list1, select_size=-1, patient_dict=data_dict1),
                FastMRIDataset(data_path=self.data_path, path_scan_list=path_scan_list2, select_size=-1, patient_dict=data_dict2))
         
    def transforms(self, mag_img, phase_img): # p=0.3

        # Standard scale
        scaler = ToTensor()

        if self.eval is False:
            if random.random() < 0.1:
                mag_img = F.vflip(mag_img)
                phase_img = F.vflip(phase_img)
            if random.random() < 0.1:
                mag_img = F.hflip(mag_img)
                phase_img = F.hflip(phase_img)

            if random.random() < 0.2:
                ## Random Transforms
                # Random Rotation 
                angle = random.random() * 60.0 - 30.0
                if angle < 0:
                    angle += 360
                mag_img = F.rotate(mag_img, angle)
                phase_img = F.rotate(phase_img, angle)
            # crop the images
            crop = tvt.CenterCrop((320, 320))
            mag_img = crop(mag_img)
            phase_img = crop(phase_img)
            # resize the image to 256x256
            mag_img = self.resize(mag_img)
            phase_img = self.resize(phase_img)
            # contrast & brightness augmentation
            if random.random() < 0.1:
                mag_img = self.color_jitter(mag_img)
        else:
            # crop the images
            crop = tvt.CenterCrop((320, 320))
            mag_img = crop(mag_img)
            phase_img = crop(phase_img)
            mag_img = self.resize(mag_img)
            phase_img = self.resize(phase_img)
        # normalize the images
        mag_img = scaler(mag_img)
        phase_img = scaler(phase_img)
        return mag_img, phase_img
    
    def get_val_scan(self, val_idxs, random_seed=None):
        if random_seed != None:
            random.seed(random_seed)
        random_idx = random.randint(0, len(val_idxs))
        # get val data list 
        val_path = Path("KneeMRI/singlecoil_val")
        val_list = sorted(list(val_path.glob("*")))
        file_path = val_list[val_idxs[random_idx]]
        # get kspace
        file = h5py.File(file_path)
        kspace = np.array(file['kspace'])
        ## Generate magnitude and phase images from k-space image 
        kspace = to_tensor(kspace).unsqueeze(dim=1)
        ifft_tensor = fftc.ifft2c_new(kspace)
        # Magnitude 
        mag_tensor = torch.norm(ifft_tensor, dim=4)
        # Phase 
        phase_tensor = torch.arctan(ifft_tensor[...,1] / ifft_tensor[...,0])
        mag_tensor, phase_tensor = self.transforms(mag_tensor, phase_tensor, p=0)
        return mag_tensor, phase_tensor

    def random_scan(self, random_seed=None):
        if random_seed != None:
            random.seed(random_seed)
        random_idx = random.randint(0, len(self.path_list))
        file_path = self.path_list[random_idx]
        dirname = os.path.dirname(file_path)
        print("dirname:", dirname)
        # get the indices 
        indices = []
        for idx, path in enumerate(self.path_list):
            if dirname in path:
                indices.append(idx)
        volume = DataLoader(Subset(self, indices), batch_size=len(indices), shuffle=False)
        return next(iter(volume))


    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        file_path = self.path_list[idx]
        kspace = np.load(file_path)

        ## Generate magnitude and phase images from k-space image 
        kspace = to_tensor(kspace).unsqueeze(dim=0)
        ifft_tensor = fftc.ifft2c_new(kspace)
        # Magnitude 
        mag_tensor = torch.norm(ifft_tensor, dim=3)
        # Phase 
        phase_tensor = torch.arctan(ifft_tensor[...,1] / ifft_tensor[...,0])
        mag_tensor, phase_tensor = self.transforms(mag_tensor, phase_tensor)
        return mag_tensor, phase_tensor

class FastMRIBrainDataset(Dataset):
    def __init__(self, data_path=None, image_size=256, patient_dict=None):
        if isinstance(patient_dict, str) or isinstance(patient_dict, Path):
            with open(patient_dict, 'rb') as file:
                patient_dict = pickle.load(file)
            self.patient_ids = list(patient_dict.keys())
            self.data_dict = {key: value for key, value in patient_dict.items() if key in self.patient_ids}
            scan_names = [name for scan_list in self.data_dict.values() for name in scan_list]
        else:
            raise ValueError("Invalid patient_dict type:", type(patient_dict))

        self.path_list = []
        self.scan_list = []
        data_path = Path(data_path) if isinstance(data_path, str) else data_path
        folder_list = list(data_path.glob("*/file_brain*"))
        for folder_path in folder_list:
            if folder_path.name not in scan_names:
                continue
            self.scan_list.append(folder_path)
            file_list = list(folder_path.glob("*.npy"))
            for f in file_list:
                self.path_list.append(f)
        print("Dataset is constructed!")
        self.data_path = data_path
        self.eval = False 
        self.color_jitter = tvt.ColorJitter(brightness=0.2, contrast=0.2)
        #self.padder = tvt.Pad(padding=32)
        self.resize = tvt.Resize(size=image_size, antialias=False)

    def transforms(self, mag_img, phase_img): # p=0.3

        # Standard scale
        scaler = ToTensor()
        if self.eval is False:
            if random.random() < 0.1:
                mag_img = F.vflip(mag_img)
                phase_img = F.vflip(phase_img)
            if random.random() < 0.1:
                mag_img = F.hflip(mag_img)
                phase_img = F.hflip(phase_img)

            if random.random() < 0.2:
                ## Random Transforms
                # Random Rotation 
                angle = random.random() * 60.0 - 30.0
                if angle < 0:
                    angle += 360
                mag_img = F.rotate(mag_img, angle)
                phase_img = F.rotate(phase_img, angle)
            # crop the images
            crop = tvt.CenterCrop((320, 320))
            mag_img = crop(mag_img)
            phase_img = crop(phase_img)
            # resize image to 256x256
            mag_img = self.resize(mag_img)
            phase_img = self.resize(phase_img)
            # contrast & brightness augmentation
            if random.random() < 0.1:
                mag_img = self.color_jitter(mag_img)
        else:
            # crop the images
            crop = tvt.CenterCrop((320, 320))
            mag_img = crop(mag_img)
            phase_img = crop(phase_img)
            # resize de the image to 256x256
            mag_img = self.resize(mag_img)
            phase_img = self.resize(phase_img)
        # normalize the images
        mag_img = scaler(mag_img)
        phase_img = scaler(phase_img)
        return mag_img, phase_img
    
    def random_scan(self, random_seed=None):
        if random_seed != None:
            random.seed(random_seed)
        random_idx = random.randint(0, len(self.path_list))
        file_path = self.path_list[random_idx]
        dirname = os.path.dirname(file_path)
        print("dirname:", dirname)
        # get the indices 
        indices = []
        for idx, path in enumerate(self.path_list):
            if dirname in str(path):
                indices.append(idx)
        volume = DataLoader(Subset(self, indices), batch_size=len(indices), shuffle=False)
        return next(iter(volume))


    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        file_path = self.path_list[idx]
        ifft_tensor = torch.from_numpy(np.load(file_path))
        ifft_tensor = torch.view_as_real(ifft_tensor).unsqueeze(dim=0).to(torch.float32)
        # magnitude 
        mag_tensor = torch.norm(ifft_tensor, dim=3)
        # phase 
        phase_tensor = torch.arctan(ifft_tensor[...,1] / ifft_tensor[...,0])
        mag_tensor, phase_tensor = self.transforms(mag_tensor, phase_tensor)
        return mag_tensor, phase_tensor
