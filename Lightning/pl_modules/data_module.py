
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union, Sequence
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as tvt 

import pickle 
import lightning as L
import torch
import os 
import sys


from data import CardiacSliceDataset, FastMRIDataset, KneeDataVarNet, OCMRDataset, BrainDataVarNet

import fastmri
from fastmri.pl_modules.data_module import worker_init_fn
from fastmri.data.mri_data import SliceDataset, CombinedSliceDataset
from sklearn.model_selection import KFold
import numpy as np

## FUNCTIONS


def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        SliceDataset, CombinedSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedSliceDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))


def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True

    return False

## CLASSES

# Data module for phase synthesis
class OCMRDataModule(L.LightningDataModule):
    """
    Data module class for Cardiac data sets.
    """

    def __init__(
        self,
        dict_path: Path,
        msk_fnc: Callable,
        batch_size: int = 16,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        phase_type: str = "gt_phase",
        seed: Optional[int] = 101,
        train_transform: tvt.Compose = None,
        val_transform: tvt.Compose = None, 
        test_transform: tvt.Compose = None, 
        fold: int = 1,
        cross_val: bool = False,
        num_folds: int = 5, 
        train_path: Optional[Path] = None, 
        normalize_input: Optional[bool] = False,
        split: Union[list, float] = 0.2, 
        pred_path: Union[Path, str] = None
    ):
        super().__init__()
        self.prepare_data_per_node=False
        self.dict_path = dict_path 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.msk_fnc = msk_fnc
        self.seed = seed
        self.data_dict = None 
        self.phase_type = phase_type
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.fold = fold
        self.cross_val = cross_val
        self.num_folds = num_folds
        self.normalize_input = normalize_input        
        self.split = split 
        self.pred_path = pred_path

        # I DONT LIKE THIS 
        self.gt_recon = None
    def setup(self, stage):

        dataset = OCMRDataset(
            patient_dict=self.dict_path,
            msk_fnc=self.msk_fnc,
            phase_type=self.phase_type,
            seed=self.seed
        )

        num_patients = len(dataset.patients) 
        print(f"Total number of patients: {num_patients}")

        if self.cross_val:
            # Prepare data for k-fold cross-validation
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            indices = torch.arange(num_patients)
            all_splits = [k for k in kf.split(indices)]
            train_idxs, test_idxs = all_splits[self.fold]
            self.train_data, self.val_data = dataset.split(train_idxs, test_idxs)
            self.test_data = self.val_data 
        elif isinstance(self.split, list):
            generator = torch.Generator().manual_seed(self.seed)
            if len(self.split) == 2:
                # No cross-validation, just train-test split
                train_idxs, test_idxs = random_split(torch.arange(num_patients), self.split, generator)
                self.train_data, self.val_data = dataset.split(train_idxs, test_idxs)
                self.test_data = self.val_data
            elif len(self.split) == 3:
                # No cross-validation, just train-val-test split 
                # generate train data 
                train_idxs, others = random_split(torch.arange(num_patients), [self.split[0], sum(self.split[1:])], generator)
                self.train_data, others = dataset.split(train_idxs, others)
                # generate val and test data 
                generator2 = torch.Generator().manual_seed(self.seed)
                val_idxs, test_idxs = random_split(torch.arange(len(others.patient_list)), self.split[1:], generator2)
                self.val_data, self.test_data = dataset.split(val_idxs, test_idxs)
                print(f"Val data size: {len(self.val_data)}")
            else:
                raise ValueError(f"Split should have 2 or 3 elements not {len(self.split)}")
        elif isinstance(self.split, float):
            train_size = round(num_patients * self.split) 
            test_size = num_patients - train_size
            generator = torch.Generator().manual_seed(self.seed)
            train_idxs, test_idxs = random_split(torch.arange(num_patients), [train_size, test_size], generator)
            self.train_data, self.val_data = dataset.split(train_idxs, test_idxs)
            self.test_data = self.val_data
            print(f"Train data usage ratio: {self.split}.")
            print(f"Number of patients in train dataset: {len(train_idxs)}")
            print(f"Number of patients in test dataset: {len(test_idxs)}")
        else:
            raise ValueError(f"Unkown split entry: {self.split}")
        
        # set transformations
        self.train_data.transforms = self.train_transform
        self.val_data.transforms = self.val_transform
        self.test_data.transforms = self.test_transform
    
    
    def train_dataloader(self):
        #sampler = torch.utils.data.DistributedSampler(self.train_data)
        knee_train = DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #sampler=sampler,
            shuffle=True
        )
        return knee_train

    def val_dataloader(self):
        knee_val = DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        return knee_val

    def test_dataloader(self):
        knee_test = DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        return knee_test

    def predict_dataloader(self):

        """
        This function is for inference for our phase synthesis diffusion model. 
        It picks a random slice of a random patient from the dataset. 
        """
        gt_recon = self.gt_recon
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, batch):
                self.masked_kspace, self.mask, self.num_low_frequencies, self.target = batch  
                self.target = (self.target[0], gt_recon[0]) if gt_recon is not None else self.target[0]
            def __len__(self):
                return self.masked_kspace.shape[0]
            def __getitem__(self, idx):
                return self.masked_kspace[idx], self.mask[idx], self.num_low_frequencies[idx], self.target
        batch = self.test_data.random_scan(slice_path=self.pred_path, random_seed=51) # 101 11 51
        dataset = CustomDataset(batch)
        data = DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            num_workers=self.num_workers,
            shuffle=False
        )
        return data


    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--dict_path",
            default=None,
            type=Path,
            help="Path to cardiac data root",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=eval,
            help="Whether to cache dataset metadata in a pkl file",
        )
        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4, # this was 4
            type=int,
            help="Number of workers to use in data loader",
        )

        parser.add_argument(
            "--phase_type",
            default="gt_phase",
            choices=("gt_phase", "diff_phase", "rand_phase", "zero_phase", "gan_phase", "gaus_phase", "gan2_phase"),
            type=str,
            help="type of the phase data to be used in VarNet training"
        )

        parser.add_argument(
            "--val_idx_path",
            default=Path("../val_idxs.npy"),
            type=Path,
            help="Path object pointing to dictionary consisting of frame paths"
        )

        return parser

# Data module for phase synthesis
class KneeDataModule(L.LightningDataModule):
    """
    Data module class for Cardiac data sets.
    """

    def __init__(
        self,
        train_dict: Union[dict, Path],
        val_dict: Union[dict, Path],
        msk_fnc: Callable,
        batch_size: int = 16,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        phase_type: str = "gt_phase",
        seed: Optional[int] = 101,
        train_transform: tvt.Compose = None,
        val_transform: tvt.Compose = None, 
        test_transform: tvt.Compose = None, 
        fold: int = 1,
        cross_val: bool = False,
        num_folds: int = 5, 
        normalize_input: Optional[bool] = False
    ):
        super().__init__()
        self.prepare_data_per_node=False
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.msk_fnc = msk_fnc
        self.seed = seed
        self.phase_type = phase_type
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.fold = fold
        self.cross_val = cross_val
        self.num_folds = num_folds
        self.normalize_input = normalize_input
        self.train_dict = pickle.load(open(train_dict, "rb")) if isinstance(train_dict, Path) else train_dict   
        self.val_dict = pickle.load(open(val_dict, "rb")) if isinstance(val_dict, Path) else val_dict         
    
    def setup(self, stage):

        if self.cross_val:
            dataset = KneeDataVarNet(
                msk_fnc=self.msk_fnc,
                patient_dict=self.dict_path, 
                phase_type=self.phase_type, 
                normalize_input=self.normalize_input
            )
            num_patients = len(dataset.patients) 
            print(f"Total number of patients: {num_patients}")
            # Prepare data for k-fold cross-validation
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            indices = torch.arange(num_patients)
            all_splits = [k for k in kf.split(indices)]
            train_idxs, test_idxs = all_splits[self.fold]
            print(f"Train patients: {len(train_idxs)}")
            print(f"Test patients: {len(test_idxs)}")
            self.train_data, self.val_data = dataset.split(train_idxs, test_idxs)
            self.test_data = self.val_data 
        else:
            self.train_data = KneeDataVarNet(
                msk_fnc=self.msk_fnc,
                patient_dict=self.train_dict,
                phase_type=self.phase_type,
                normalize_input=self.normalize_input,
            )
            self.val_data = KneeDataVarNet(
                msk_fnc=self.msk_fnc,
                patient_dict=self.val_dict,
                phase_type="gt_phase" if self.phase_type != "zero_phase" else "zero_phase",
                normalize_input=self.normalize_input
            )
            print("Train data size:", len(self.train_data))
            print("Test data size:", len(self.val_data))
            self.test_data = self.val_data 

        # set transformations
        self.train_data.transforms = self.train_transform
        self.val_data.transforms = self.val_transform
        self.test_data.transforms = self.test_transform
    
    def train_dataloader(self):
        #sampler = torch.utils.data.DistributedSampler(self.train_data)
        knee_train = DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #sampler=sampler,
            shuffle=True
        )
        return knee_train

    def val_dataloader(self):
        knee_val = DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        return knee_val

    def test_dataloader(self):
        knee_test = DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        return knee_test

    def predict_dataloader(self):

        """
        This function is for inference for our phase synthesis diffusion model. 
        It picks a random slice of a random patient from the dataset. 
        """

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, batch):
                self.masked_kspace, self.mask, self.num_low_frequencies, self.target = batch  
            def __len__(self):
                return self.masked_kspace.shape[0]
            def __getitem__(self, idx):
                return self.masked_kspace[idx], self.mask[idx], self.num_low_frequencies[idx], self.target[idx]
            
        batch = self.test_data.random_scan(random_seed=51) # 101 11 51
        dataset = CustomDataset(batch)
        
        data = DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            num_workers=self.num_workers,
            shuffle=False
        )
        return data

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--train_dict",
            default=Path("../KneeMRI/rec_train_val_test_dicts/train_dict.pkl"),
            type=Path,
            help="Path to knee train dict",
        )
        parser.add_argument(
            "--val_dict",
            default=Path("../KneeMRI/rec_train_val_test_dicts/test_dict.pkl"),
            type=Path,
            help="Path to knee train dict",
        )

        parser.add_argument(
            "--normalize_input",
            default=True,
            type=eval, 
            help="flag for normalizing input"
        )

        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=eval,
            help="Whether to cache dataset metadata in a pkl file",
        )
        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4, # this was 4
            type=int,
            help="Number of workers to use in data loader",
        )

        parser.add_argument(
            "--phase_type",
            default="gt_phase",
            choices=("gt_phase", "gan_phase", "diff_phase", "rand_phase", "zero_phase", "gaus_phase", "gan2_phase"),
            type=str,
            help="type of the phase data to be used in VarNet training"
        )
        return parser
        

class BrainDataModule(L.LightningDataModule):

    def __init__(
        self,
        train_dict: Union[dict, Path],
        val_dict: Union[dict, Path],
        msk_fnc: Callable,
        batch_size: int = 32,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        phase_type: str = "gt_phase",
        seed: Optional[int] = 101,
        train_transform: tvt.Compose = None,
        val_transform: tvt.Compose = None, 
        test_transform: tvt.Compose = None, 
        fold: int = 1,
        cross_val: bool = False,
        num_folds: int = 5, 
        normalize_input: Optional[bool] = False
    ):
        super().__init__()
        self.prepare_data_per_node=False
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.msk_fnc = msk_fnc
        self.seed = seed
        self.phase_type = phase_type
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.fold = fold
        self.cross_val = cross_val
        self.num_folds = num_folds
        self.normalize_input = normalize_input
        self.train_dict = pickle.load(open(train_dict, "rb")) if isinstance(train_dict, Path) else train_dict   
        self.val_dict = pickle.load(open(val_dict, "rb")) if isinstance(val_dict, Path) else val_dict         
    
    def setup(self, stage):
        self.train_data = BrainDataVarNet(
            msk_fnc=self.msk_fnc,
            patient_dict=self.train_dict,
            phase_type=self.phase_type,
            normalize_input=self.normalize_input,
        )
        self.val_data = BrainDataVarNet(
            msk_fnc=self.msk_fnc,
            patient_dict=self.val_dict,
            phase_type="gt_phase", #"gt_phase" if self.phase_type not in ("zero_phase", "gaus_phase") else self.phase_type,
            normalize_input=self.normalize_input
        )
        print("Train data size:", len(self.train_data))
        print("Test data size:", len(self.val_data))
        self.test_data = self.val_data 

        # set transformations
        self.train_data.transforms = self.train_transform
        self.val_data.transforms = self.val_transform
        self.test_data.transforms = self.test_transform
    
    def train_dataloader(self):
        #sampler = torch.utils.data.DistributedSampler(self.train_data)
        knee_train = DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #sampler=sampler,
            shuffle=True
        )
        return knee_train

    def val_dataloader(self):
        knee_val = DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        return knee_val

    def test_dataloader(self):
        knee_test = DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        return knee_test

    def predict_dataloader(self):

        """
        This function is for inference for our phase synthesis diffusion model. 
        It picks a random slice of a random patient from the dataset. 
        """

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, batch):
                self.masked_kspace, self.mask, self.num_low_frequencies, self.target = batch  
            def __len__(self):
                return self.masked_kspace.shape[0]
            def __getitem__(self, idx):
                return self.masked_kspace[idx], self.mask[idx], self.num_low_frequencies[idx], self.target[idx]
            
        batch = self.test_data.random_scan(random_seed=51) # 101 11 51
        dataset = CustomDataset(batch)
        
        data = DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            num_workers=self.num_workers,
            shuffle=False
        )
        return data

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--train_dict",
            default=Path("../BrainMRI/rec_train_val_test_dicts/train_dict.pkl"),
            type=Path,
            help="Path to brain train dict",
        )
        parser.add_argument(
            "--val_dict",
            default=Path("../BrainMRI/rec_train_val_test_dicts/test_dict.pkl"),
            type=Path,
            help="Path to brain test dict",
        )

        parser.add_argument(
            "--normalize_input",
            default=True,
            type=eval, 
            help="flag for normalizing input"
        )

        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=eval,
            help="Whether to cache dataset metadata in a pkl file",
        )
        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4, # this was 4
            type=int,
            help="Number of workers to use in data loader",
        )

        parser.add_argument(
            "--phase_type",
            default="gt_phase",
            choices=("gt_phase", "gan_phase", "diff_phase", "rand_phase", "gaus_phase", "zero_phase"),
            type=str,
            help="type of the phase data to be used in VarNet training"
        )
        return parser

# Data module for phase synthesis
class CardiacDataModule(L.LightningDataModule):
    """
    Data module class for Cardiac data sets.
    """

    def __init__(
        self,
        data_path: Path,
        msk_fnc: Callable,
        split: Optional[Sequence[int]] = [13, 2, 5],
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        data_dict_path: Optional[Path] = None,
        phase_type: str = "ground_truth",
        seed: Optional[int] = 101,
        train_transform: tvt.Compose = None,
        val_transform: tvt.Compose = None, 
        test_transform: tvt.Compose = None, 
        fold: int = 1,
        cross_val: bool = False,
        num_folds: int = 5 
    ):
        super().__init__()
        self.prepare_data_per_node=False
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.msk_fnc = msk_fnc
        self.split = split
        self.seed = seed
        self.data_dict = None 
        self.phase_type = phase_type
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        # train-val-test datasets 
        if not 2 <= len(split) <= 3:
            raise ValueError(f"Invalid split. {split} is undefined!")
        #assert (data_dict_path == None) ^ (split == None) # either one of them should be None
        assert 0 <= fold <= 4, "Fold can be {0,2,3,4}"
        self.data_dict_path = data_dict_path
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.fold = fold
        self.cross_val = cross_val
        self.num_folds = num_folds
    
    def setup(self, stage):

        if self.data_dict_path != None:
            # load data 
            f = open(self.data_dict_path, 'rb')
            self.data_dict = pickle.load(f)
            f.close() 

            dataset = CardiacSliceDataset(
                msk_fnc=self.msk_fnc,
                data_dict=self.data_dict,
                phase_type=self.phase_type,
            )
        else:
            dataset = CardiacSliceDataset(
                root=self.data_path, 
                msk_fnc=self.msk_fnc,
                phase_type=self.phase_type   
            )

        num_patients = len(dataset.patient_list)

        if self.cross_val:
            # Prepare data for k-fold cross-validation
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            indices = torch.arange(num_patients)
            all_splits = [k for k in kf.split(indices)]
            train_idxs, test_idxs = all_splits[self.fold] ## CHANGE THIS TO 00!!!! (it was 1 )
            print("TRAIN_IDXS:", train_idxs)
            print("TEST_IDXS:", test_idxs)
            self.train_data, self.val_data = dataset.split(train_idxs, test_idxs)
            self.test_data = self.val_data 
        else:
            generator = torch.Generator().manual_seed(self.seed)
            if len(self.split) == 2:
                # No cross-validation, just train-test split
                train_idxs, test_idxs = random_split(torch.arange(num_patients), self.split, generator)
                self.train_data, self.val_data = dataset.split(train_idxs, test_idxs)
                self.test_data = self.val_data
            elif len(self.split) == 3:
                # No cross-validation, just train-val-test split 
                # generate train data 
                train_idxs, others = random_split(torch.arange(num_patients), [self.split[0], sum(self.split[1:])], generator)
                self.train_data, others = dataset.split(train_idxs, others)
                # generate val and test data 
                generator2 = torch.Generator().manual_seed(self.seed)
                val_idxs, test_idxs = random_split(torch.arange(len(others.patient_list)), self.split[1:], generator2)
                self.val_data, self.test_data = dataset.split(val_idxs, test_idxs)
                print(f"Val data size: {len(self.val_data)}")
            else:
                raise ValueError(f"Split should have 2 or 3 elements not {len(self.split)}")
        
        # set transformations
        self.train_data.transforms = self.train_transform
        self.val_data.transforms = self.val_transform
        self.test_data.transforms = self.test_transform
    
    
    def train_dataloader(self):
        #sampler = torch.utils.data.DistributedSampler(self.train_data)
        cardiac_train = DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #sampler=sampler,
            shuffle=True
        )
        return cardiac_train

    def val_dataloader(self):
        cardiac_val = DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        return cardiac_val

    def test_dataloader(self):
        cardiac_test = DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        return cardiac_test

    def predict_dataloader(self):

        """
        This function is for inference for our phase synthesis diffusion model. 
        It picks a random slice of a random patient from the dataset. 
        """

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, batch):
                self.masked_kspace, self.mask, self.num_low_frequencies, self.target = batch  
            def __len__(self):
                return self.masked_kspace.shape[0]
            def __getitem__(self, idx):
                return self.masked_kspace[idx], self.mask[idx], self.num_low_frequencies[idx], self.target[idx]
            
        batch = self.test_data.random_scan(random_seed=101) # 101 11 51
        dataset = CustomDataset(batch)

        cardiac_pred = DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            num_workers=self.num_workers,
            shuffle=False
        )

        return cardiac_pred


    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to cardiac data root",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=eval,
            help="Whether to cache dataset metadata in a pkl file",
        )
        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4, # this was 4
            type=int,
            help="Number of workers to use in data loader",
        )

        parser.add_argument(
            "--phase_type",
            default="ground_truth",
            choices=("ground_truth", "diffusion", "gan", "random", "zero", "diffusion_3D", "gan_3D"),
            type=str,
            help="type of the phase data to be used in VarNet training"
        )

        parser.add_argument(
            "--data_dict_path",
            default=Path("data/datadict.pkl"),
            type=Path,
            help="Path object pointing to dictionary consisting of frame paths"
        )

        # augmentations 
        parser.add_argument(
            "--center_crop",
            default=128,
            type=int,
            help="size of the image after cropping the center"
        )
        
        parser.add_argument(
            "--angle",
            type=float,
            nargs="+",
            default=[-45.0, 45.0],
            help="angle interval for random affine transform"
        )

        parser.add_argument(
            "--scale",
            type=float,
            nargs="+",
            default=[.8, 1.2],
            help="scale interval for random affine transform"
        )

        parser.add_argument(
            "--shear",
            type=float,
            nargs="+",
            default=[-20, 20],
            help="shear interval for random affine transform"
        )

        return parser


class KneeTrainDataModule(L.LightningDataModule):
    """
    Data module class for Cardiac data sets.
    """

    def __init__(
        self,
        data_path: Path,
        msk_fnc: Callable,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        data_dict_path: Optional[Path] = None,
        seed: Optional[int] = 101,
        train_transform: tvt.Compose = None,
        val_transform: tvt.Compose = None,  
    ):
        super().__init__()
        self.prepare_data_per_node=False
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.msk_fnc = msk_fnc
        self.seed = seed
        self.data_dict = None 
        self.train_transform = train_transform
        self.val_transform = val_transform
        #assert (data_dict_path == None) ^ (split == None) # either one of them should be None
        self.data_dict_path = data_dict_path
        self.train_data = None
        self.val_data = None
    
    def setup(self, stage):

        self.train_data = FastMRIDataset(data_path='KneeMRI/singlecoil_kspace_train')
        self.val_data = FastMRIDataset(data_path='KneeMRI/singlecoil_kspace_train')
        
        # set transformations
        self.train_data.transforms = self.train_transform
        self.val_data.transforms = self.val_transform
    
    
    def train_dataloader(self):
        #sampler = torch.utils.data.DistributedSampler(self.train_data)
        knee_train = DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #sampler=sampler,
            shuffle=True
        )
        return knee_train

    def val_dataloader(self):
        knee_val = DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        return knee_val


    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to cardiac data root",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=eval,
            help="Whether to cache dataset metadata in a pkl file",
        )
        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4, # this was 4
            type=int,
            help="Number of workers to use in data loader",
        )
        return parser


"""
class BrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        num_coils: int = 16,
        phase_type: str = "ground_truth"
    ):
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError(
                "Can set val_sample_rate or val_volume_sample_rate, but not both."
            )
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError(
                "Can set test_sample_rate or test_volume_sample_rate, but not both."
            )

        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.num_coils = num_coils
        self.phase_type = phase_type

    def train_dataloader(self):
        dataset = BrainDataset(
            num_coils=self.num_coils,
            root=self.data_path / "multicoil_train",
            challenge=self.challenge,
            transform=self.train_transform,
            use_dataset_cache=False,
            sample_rate=self.sample_rate,
            phase_type=self.phase_type 
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #worker_init_fn=worker_init_fn,
            #sampler=torch.utils.data.DistributedSampler(dataset),
            shuffle=True
        )
    
    def val_dataloader(self):
        dataset = BrainDataset(
            num_coils=self.num_coils,
            root=self.data_path / "multicoil_val",
            challenge=self.challenge,
            transform=self.val_transform,
            use_dataset_cache=False,
            sample_rate=self.sample_rate,
            phase_type="ground_truth"
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #worker_init_fn=worker_init_fn,
            #sampler=torch.utils.data.DistributedSampler(dataset),
            shuffle=False
        )

    
    def test_dataloader(self):
        return None 
        dataset = BrainDataset(
            partition='test',
            num_coils=self.num_coils,
            root=self.data_path,
            challenge=self.challenge,
            transform=self.test_transform,
            use_dataset_cache=False,
            sample_rate=self.sample_rate,
            phase_type=self.phase_type
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=torch.utils.data.DistributedSampler(dataset),
            shuffle=False
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = FastMriDataModule.add_data_specific_args(parent_parser)
        ## New arguments 

        # dataset arguments
        parser.add_argument(
            "--num_coils",
            default=16,
            type=int,
            help="number of coils used in parallel imaging"
        )

        # different phase types 
        parser.add_argument(
            "--phase_type",
            default="ground_truth",
            choices=("ground_truth", "zero", "random"),
            type=str,
            help="type of the phase to be used in varnet training"
        )
        return parser 
"""