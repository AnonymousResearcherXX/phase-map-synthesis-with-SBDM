"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
from pathlib import Path 
from argparse import ArgumentParser
import torchvision.transforms as tvt
import lightning as L
import torch
import yaml

from fastmri.data.subsample import create_mask_for_mask_type
from pl_modules import VarNetModule, CardiacDataModule, KneeDataModule, CascadeNetModule, OCMRDataModule, BrainDataModule
from data import VarNetDataTransform, CenterCrop, AffineTransform, HorizontalFlip, VerticalFlip, GaussianBlur, ElasticTransform
#import torch.distributed as dist
import pickle
import numpy as np 
import random


def cli_main(args):
    # for high performance 
    torch.set_float32_matmul_precision("medium")

    L.seed_everything(args.seed, workers=True)

    # ------------
    # data
    # -----------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, [args.center_fractions], [args.accelerations]
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
            train_dict=args.train_dict,
            val_dict=args.val_dict,
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
            normalize_input=args.normalize_input
        )
    elif args.dataset_type  == "brain":
        data_module = BrainDataModule(
            train_dict=args.train_dict,
            val_dict=args.val_dict,
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
            normalize_input=args.normalize_input
        )
    elif args.dataset_type == "ocmr":
        data_module = OCMRDataModule(
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
            args=args, 
            data_module=data_module
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
        max_epochs=args.max_epochs,
        deterministic=args.deterministic,
        default_root_dir=args.default_root_dir,
        log_every_n_steps=80, # because there are 45 steps in one epoch 
        callbacks=args.callbacks
    )

    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)    
    elif args.mode == "ood_test":
        acs_dict = {0.0625: "acs16l", 0.1: "acs26l", 0.12: "acs31l"}
        #acs = acs_dict[args.center_fractions]
        #test_path = args.ckpt_dir / "acs26l" / "R4"
        #root = args.default_root_dir
        #if test_path.exists():
        #    ckpt_paths = sorted(list(test_path.rglob("*_phase/lightning_logs/version_0/*epoch=14*.ckpt")), key=lambda x: x.parent.parent.parent.name)
        #    # create model dict 
        #    phase_list = [path.name for path in sorted(list(test_path.glob("*_phase")), key=lambda x : x.name)]
        #    model_dict = dict(zip(phase_list, ckpt_paths))

        #else:
        #    return None
        """    
        for phase in phase_list:
            save_dir = root / acs / f"R{args.accelerations}" / phase
            save_dir.mkdir(exist_ok=True, parents=True)
            trainer = L.Trainer(
                accelerator="gpu", 
                strategy=args.strategy,
                devices=[args.device_idx],
                max_epochs=args.max_epochs,
                deterministic=args.deterministic,
                default_root_dir=save_dir,
                log_every_n_steps=80, # because there are 45 steps in one epoch 
                callbacks=args.callbacks
            )
            """
        #args.ckpt_path = model_dict[phase]
        trainer.test(model, datamodule=data_module)#, ckpt_path=args.ckpt_path)
    elif args.mode == "test_all":
        folders = args.default_root_dir.glob("*phase*")
        #args.ckpt_path = Path("brain_final_results/acs16l/R3/zero_phase/lightning_logs/version_0/epoch=14-step=43230.ckpt")
        #trainer.test(model, datamodule=data_module, ckpt_path=args.ckpt_path)
        #import sys; sys.exit()
        for folder in folders: 
            if "rand_phase" not in str(folder):
                args.ckpt_path = list((folder / "lightning_logs/version_0").glob("*epoch=14*.ckpt"))[0]
                trainer.test(model, datamodule=data_module, ckpt_path=args.ckpt_path)
    elif args.mode == "display":
        gt_recon = trainer.predict(model, datamodule=data_module, ckpt_path=args.ckpt_path)[0]
    else:
        raise ValueError(f"unrecognized mode {args.mode}")



def build_args():
    parser = ArgumentParser()

    # basic args
    #path_config = pathlib.Path("../../fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = 1 if backend == "ddp" else 1
    model = "varnet"

    # set defaults based on optional directory config
    default_root_dir = Path("varnet_brain")
    #default_root_dir = pathlib.Path("dcnn_trains_auge") # for synthetic varnet train
    log_path = Path("lightning_logs")

    parser.add_argument(
        "--save_every_n_epoch", 
        default=5, 
        type=int, 
        help="get the results every n epoch (results=model checkpoint and rep. cases)"
    )

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
        "--num_folds",
        default=5, 
        type=int, 
        help="number of folds for cross-validation"
    )

    parser.add_argument(
        "--cv",
        default=False,
        type=eval,
        help="5-fold cross-validation"
    )
    parser.add_argument(
        "--fold_idx",
        default=0,
        type=int,
        help="index of the current fold"
    )

    # gaussian blur parameters 
    parser.add_argument(
        "--p_blur",
        default=0,
        type=float, 
        help="probability of guassian blur"
    )

    parser.add_argument(
        "--kernel_size",
        default=0,
        type=int, 
        help="kernel size of the gaussian blur"
    )

    parser.add_argument(
        "--sigma",
        default=0.0,
        type=float,
        help="standard deviation of gaussian noise"
    )

    # horizontal/vertical flip probs 
    parser.add_argument(
        "--p_hflip",
        default=0.0,
        type=float,
        help='probability of horizontal flip'
    )

    parser.add_argument(
        "--p_vflip",
        default=0.0,
        type=float,
        help="probability of vertical flip"
    )

    parser.add_argument(
        "--loss_type",
        default="ssim",
        choices=("l1", "mse", "ssim"),
        type=str,
        help="type of the training loss"
    )

    parser.add_argument(
        "--train_ratio",
        default=1, 
        type=float, 
        help="ratio of the training part for reconstruction model"
    )

    # dataset type 
    parser.add_argument(
        "--dataset_type",
        default="brain",
        choices=("cardiac", "knee", "ocmr", "brain"),
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

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test", "display", "test_all", "display_all", "test_ckpts", "ood_test"),
        type=str,
        help="Operation mode",
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
        #nargs="+",
        default=0.1, 
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        #nargs="+",
        default=4,
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
        default=303,
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
        type=Path,
        help="root directory"
    )
    parser.add_argument(
        "--ckpt_dir",
        default=Path("."),
        type=Path, 
        help="ckpt load directory"
    )
    parser.add_argument(
        "--max_epochs",
        default=30,
        type=int,
        help="maximum number of epochs for training"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        type=eval,
        help="continue training from checkpoint"
    )

    parser.add_argument(
        "--ckpt_path", 
        default=Path("brain_final_results/acs16l/R3/gt_phase/lightning_logs/version_0/"),
        type=Path, 
        help="ckpt load directory"
    )
    """
    parser = KneeDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        normalize_input=True,
        mask_type="equispaced",  # VarNet uses equispaced mask
        challenge="singlecoil",  # only multicoil implemented for VarNet
        batch_size=16,  # number of samples per batch
        test_path=None # path for test split, overwrites data_path
    )
    """
    parser = BrainDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        normalize_input=True,
        mask_type="equispaced",  # VarNet uses equispaced mask
        challenge="singlecoil",  # only multicoil implemented for VarNet
        batch_size=16,  # number of samples per batch
        test_path=None # path for test split, overwrites data_path
    )
    
    # module config
    if model == "varnet":
        parser = VarNetModule.add_model_specific_args(parser)
        parser.set_defaults(
            num_cascades=12,  # number of unrolled iterations (this was 12)
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

    versions = list((args.default_root_dir / log_path).glob('*_*'))
    max_version = max([int(fname.name.split("_")[1]) for fname in versions]) if len(versions) != 0 else -1

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if args.mode != "test_all":
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)

        args.callbacks = [
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=args.default_root_dir / log_path / f"version_{max_version+1}", 
                save_top_k=-1,
                #save_last=True,
                verbose=True,
                monitor="val_ssim",
                every_n_epochs=args.save_every_n_epoch
            )
        ]

        # set default checkpoint if one exists in our checkpoint directory
        if args.resume_from_checkpoint is None:
            ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
            if ckpt_list:
                args.resume_from_checkpoint = str(ckpt_list[-1])
    else: 
        args.callbacks=None

    return args 


def run_cli():
    # ---------------------
    # RUN TRAINING
    # ---------------------
    args = build_args()
    if args.cv == False or args.mode == "display":
        args.num_folds = 1
        args.fold_idx = 0
        cli_main(args)
    elif args.mode == "test":
        result_path = Path("./varnet_knee/acs10/batch16/checkpoints")
        ckpt_paths = list(result_path.glob("*.ckpt"))
        for path in ckpt_paths:
            args.ckpt_path = path
            epoch_no = int(str(path).split("=")[1].split("-")[0])
            args.epoch_no = epoch_no
            args.mode = "test"
            cli_main(args)
    elif args.mode == "ood_test":
        acs_list = [0.12, 0.1, 0.0625] # 31l, 26l, 16l
        acs_dict = {0.12: "acs31", 0.1: "acs26l", 0.0625: "acs16l"}
        acc_list = [2,3,4,6]
        for acs in acs_list:
            for acc in acc_list:
                args.center_fractions = acs
                args.accelerations = acc
                args.default_root_dir = args.default_root_dir / acs_dict[acs] / f"R{acc}"
                print("ACS:", acs)
                print("ACC:", acc)
                cli_main(args)
                args.default_root_dir = args.default_root_dir.parent.parent

    elif args.mode == "test_all": 
        args.num_folds = 1 
        args.fold_idx = 0 
        cli_main(args)
    else:
        # cross-validation training
        for fold_idx in range(args.num_folds):
            args.fold_idx = fold_idx 
            cli_main(args)
            args = build_args()


if __name__ == "__main__":
    # for reproducible results (important for train and test split)
    seed = 303 # 101 202 303 
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    torch.autograd.set_detect_anomaly(True)
    print("GPU name:", torch.cuda.get_device_name())
    run_cli()





# data config
    """
    parser = CardiacDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to Cardiac
        mask_type="equispaced",  # VarNet uses equispaced mask
        challenge="singlecoil",  # only multicoil implemented for VarNet
        batch_size=64,  # number of samples per batch
        test_path=None # path for test split, overwrites data_path
        )
    
    parser = OCMRDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        dict_path=dict_path,  # path to Knee
        mask_type="equispaced",  # VarNet uses equispaced mask
        challenge="singlecoil",  # only multicoil implemented for VarNet
        batch_size=1,  # number of samples per batch
        test_path=None # path for test split, overwrites data_path
    )
    """