

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import torchvision.transforms as tvt

import lightning as L
import pytorch_lightning as pl
import torch
import yaml 

from sklearn.model_selection import KFold

from fastmri.data.subsample import create_mask_for_mask_type
from pl_modules import VarNetModule, CardiacDataModule, KneeDataModule, CascadeNetModule, OCMRDataModule
from data import VarNetDataTransform, CenterCrop, AffineTransform, HorizontalFlip, VerticalFlip, GaussianBlur, ElasticTransform
from data import CardiacSliceDataset, FastMRIDataset, OCMRDataset
import torch.distributed as dist
from pathlib import Path 
import pickle
import numpy as np 
import random
from PIL import Image 
import matplotlib.pyplot as plt 


def cli_main(args):
    # for high performance 
    torch.set_float32_matmul_precision("medium")

    L.seed_everything(args.seed)

    # ------------
    # data
    # -----------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        "equispaced", [.1], [4]
    )
    train_transform = val_transform = test_transform = None
    # ptl data module - this handles data loaders 
    if args.dataset_type == 'cardiac':
        # Set augmentations and transformations 
        train_transform = tvt.Compose([
        # Augmentations
        HorizontalFlip(p=args.p_hflip),
        #VerticalFlip(p=args.p_vflip),
        #AffineTransform(args.angle, args.scale, args.shear),
        GaussianBlur(args.p_blur, args.kernel_size, args.sigma),
        CenterCrop(args.center_crop)
        ])
        val_transform = CenterCrop(args.center_crop)
        test_transform = CenterCrop(args.center_crop)

        data_module = CardiacDataModule(
            data_path=args.data_path,
            msk_fnc=mask,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler= "ddp", #(args.accelerator in ("ddp", "ddp_cpu"))
            data_dict_path=args.data_dict_path,
            phase_type=args.phase_type,
            seed=args.seed,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            split=[17, 3],
            cross_val=args.cv,
            fold=args.fold_idx
        )
    elif args.dataset_type == "knee":
        data_module = KneeDataModule(
            dict_path=args.dict_path,
            split=args.train_ratio,
            msk_fnc=mask, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler= "ddp", #(args.accelerator in ("ddp", "ddp_cpu"))
            seed=args.seed, 
            phase_type=args.phase_type,
            train_transform=train_transform, 
            val_transform=val_transform, 
            test_transform=test_transform, 
            fold=args.fold_idx, 
            cross_val=args.cv
        )
    elif args.dataset_type == "ocmr":
        data_module = OCMRDataModule(
            dict_path=args.dict_path,
            msk_fnc=mask, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed_sampler= "ddp", #(args.accelerator in ("ddp", "ddp_cpu"))
            seed=args.seed, 
            phase_type=args.phase_type,
            train_transform=train_transform, 
            val_transform=val_transform, 
            test_transform=test_transform, 
            fold=args.fold_idx, 
            cross_val=args.cv, 
            pred_path=args.pred_path
        )
    else:
        raise ValueError("Invalid dataset type!")
    
    # ------------
    # model
    # ------------
    if args.model == "varnet":
        model = VarNetModule(
            num_cascades=args.num_cascades,
            pools=args.pools,
            chans=args.chans,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            single_coil=args.single_coil,
            args=args
        )
    elif args.model == "dccnn":
        model = CascadeNetModule(
            num_cascades=args.num_cascades, 
            n_convs=args.n_convs, 
            batchnorm=args.batchnorm, 
            no_dc=args.no_dc, 
            args=args
        )
    else:
        raise ValueError(f"Unkown model type: {args.model}")
    
    trainer = L.Trainer(
        accelerator="gpu", 
        strategy=args.strategy,
        devices=[args.device_idx],
        max_epochs=15,
        deterministic=args.deterministic,
        default_root_dir=args.default_root_dir,
        log_every_n_steps=40, # because there are 45 steps in one epoch
    )
    
    trainer.predict(model, datamodule=data_module, ckpt_path=args.ckpt_path)[0]



def build_args():
    parser = ArgumentParser()

    # basic args
    #path_config = pathlib.Path("../../fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = 1 if backend == "ddp" else 1
    model = "varnet"

    # set defaults based on optional directory config
    #data_path = pathlib.Path("../cardiac_data")
    dict_path = pathlib.Path("../OCMR/test_dict.pkl") #pathlib.Path("../KneeMRI/recon_dict.pkl")
    #data_path = pathlib.Path("../KneeMRI")
    default_root_dir = pathlib.Path("dcnn_trains_auge") # for synthetic varnet train

    parser.add_argument(
        "--device_idx",
        default=0,
        type=int, 
        help="index of the gpu to be used"
    ) 

    parser.add_argument(
        "--model", 
        default="varnet", 
        choices=("varnet", "dccnn"), 
        type=str, 
        help="reconstruction model to be trained"
    )

    parser.add_argument(
        "--cv",
        default=True,
        type=eval,
        help="5-fold cross-validation"
    )
    parser.add_argument(
        "--fold_idx",
        default=0,
        type=int,
        help="index of the current fold"
    )

    parser.add_argument(
        "--loss_type",
        default="ssim",
        choices=("l1", "mse", "ssim"),
        type=str,
        help="type of the training loss"
    )

    # dataset type 
    parser.add_argument(
        "--dataset_type",
        default="cardiac",
        choices=("cardiac", "knee", "ocmr"),
        type=str,
        help="type of the dataset to be used for VarNet"
    )

    # single/multi-coil for VarNet
    parser.add_argument(
        "--single_coil",
        default=True,
        type=eval,
        help="Use single coil or multi coil"
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced", "equispaced_fraction"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.1], # 0.1
        type=list,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument(
        "--num_gpus",
        default=num_gpus,
        type=int,
        help="number of gpus in the training"
    )
    parser.add_argument(
        "--replace_sampler_ddp",
        default=False,
        type=eval,
        help="this is necessary for volume dispatch during val"
    )
    parser.add_argument(
        "--strategy",
        default="ddp",  # ddp_find_unused_parameters_true
        type=str,
        help="strategy for training"
    )
    parser.add_argument(
        "--seed",
        default=101,
        type=int,
        help="seed for the randomness"
    )
    parser.add_argument(
        "--deterministic",
        default=True,
        type=eval,
        help="for reproducible results"
    )
    parser.add_argument(
        "--default_root_dir",
        default=default_root_dir,
        type=pathlib.Path,
        help="root directory"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        type=eval,
        help="continue training from checkpoint"
    )
    parser.add_argument(
        "--ckpt_path", 
        default=pathlib.Path("Jan15/acc4/diff/lightning_logs/version_0/epoch=10-step=869.ckpt"),
        type=Path, 
        help="ckpt load directory"
    )

    # variables for prediction & display steps
    parser.add_argument(
        "--pred_path", 
        default=None,
        type=Path,
        help="path of the predict slice"
    )
    parser.add_argument(
        "--display_path",
        default=Path("predictions"), 
        type=Path, 
        help="path of the directory where images will be saved"
    )
    parser.add_argument(
        "--frame_idx",
        default=0,
        type=int, 
        help="index of the frame in a slice"
    )
    parser.add_argument(
        "--load", 
        default=False, 
        type=eval,
        help="switch for using loaded images or generate those images via pre-trained models"
    )
    # data config 
    parser = OCMRDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        dict_path=dict_path,  # path to Knee
        mask_type="equispaced",  # VarNet uses equispaced mask
        challenge="singlecoil",  # only multicoil implemented for VarNet
        batch_size=1,  # number of samples per batch
        test_path=None # path for test split, overwrites data_path
    )
    
    
    # module config
    if model == "varnet":
        parser = VarNetModule.add_model_specific_args(parser)
        parser.set_defaults(
            num_cascades=12,  # number of unrolled iterations (previously num_cascades=8)
            pools=4,  # number of pooling layers for U-Net
            chans=18,  # number of top-level channels for U-Net
            lr=0.0003,  # Adam learning rate (previously lr=0.001)
            lr_step_size=10,  # epoch at which to decrease learning rate (default = 40)
            lr_gamma=0.1,  # extent to which to decrease learning rate
            weight_decay=0.0,  # weight regularization strength
        )
    elif model == "dccnn":
        parser = CascadeNetModule.add_model_specific_args(parser)
        parser.set_defaults(
            num_cascades=12, 
            hidden_chans=64, 
            lr=0.0003,
            lr_step_size=40, 
            lr_gamma=0.1, 
            weight_decay=0.0
        )
    else:
        raise ValueError(f"Unknown model type: {model}")

    args = parser.parse_args()
    return args


def run_cli():
    # ---------------------
    # GENERATE IMAGES 
    # ---------------------
    model_dir = Path("./ocmr_latest/acc4")
    mask = create_mask_for_mask_type(
        "equispaced", [0.1], [4]
    )

    args = build_args()
    args.mode = "display"
    if args.load == False:
        phase_list = ["gt_phase", "diff_phase", "gan_phase", "rand_phase"]
        slice_list = [
                #Path("../OCMR/ocmr_recon_data/fs_0028_3T/slice1"),
                #Path("../OCMR/ocmr_recon_data/fs_0082_1_5T/slice1"),
                #Path("../OCMR/ocmr_recon_data/fs_0031_3T/slice1"), 
                #Path("../OCMR/ocmr_recon_data/fs_0078_1_5T/slice1")
                #Path("../OCMR/ocmr_recon_data/fs_0079_1_5T/slice1")
                Path("../OCMR/ocmr_recon_data/fs_0016_3T/slice1")
            ]
        slice_dict = {}
        
        dataset = OCMRDataset(
            patient_dict=args.dict_path,
            msk_fnc=mask,
            phase_type=args.phase_type,
            seed=args.seed
        )
        num_patients = len(dataset.patients)
        print(f"Total number of patients: {num_patients}")
        # Get the folds for each representative case
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        indices = torch.arange(num_patients)
        all_splits = [k for k in kf.split(indices)]
        for i in range(5):
            train_idxs, test_idxs = all_splits[i]
            train_data, val_data = dataset.split(train_idxs, test_idxs)
            for slice_path in slice_list:
                if slice_path in val_data.slice_list:
                    slice_dict[slice_path] = i 

        args.cv = True 
        for (path, fold_idx) in slice_dict.items():
            for phase_type in phase_list:
                args.phase_type = phase_type
                args.fold_idx = fold_idx 
                args.pred_path = path
                phase_type = phase_type.split("_")[0]
                args.ckpt_path = str(list((model_dir / phase_type / f"lightning_logs/version_{fold_idx}").glob("*epoch*"))[0])
                for i in range(3):
                    args.frame_idx = i
                    cli_main(args)   
        
    display_list = ["ground_truth", "gt_phase", "gt_phase_recon", 
                    "gt_phase_err", "us_mag", "undsamp_err", 
                    "rand_phase_recon", "rand_phase_err", 
                    "gan_phase_recon", "gan_phase", "gan_phase_err",
                    "diff_phase_recon", "diff_phase", "diff_phase_err"]
    cols = len(display_list)
    scan_list = list(args.display_path.glob("*"))
    for scan in scan_list:
        frame_list = list(scan.glob("*"))
        rows = len(frame_list)
        fig, axs = plt.subplots(rows, cols, figsize=(30, 15), dpi=300)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for row_idx, frame_path in enumerate(frame_list): 
            for col_idx, img_name in enumerate(display_list):
                img = Image.open(frame_path / (img_name + ".png"))
                axs[row_idx, col_idx].imshow(img, aspect='equal')
                axs[row_idx, col_idx].axis("off")
                axs[row_idx, col_idx].set_xticks([])
                axs[row_idx, col_idx].set_yticks([])
        #plt.tight_layout()
        plt.savefig(scan.name + ".png")

if __name__ == "__main__":
    # for reproducible results (important for train and test split)
    seed = 101
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    torch.autograd.set_detect_anomaly(True)

    run_cli()

