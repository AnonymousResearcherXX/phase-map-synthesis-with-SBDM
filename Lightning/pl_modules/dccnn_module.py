

from argparse import ArgumentParser

import torch
import numpy as np
from torch import nn
from models.dccnn import CascadeNet
#from ..models.varnet import VarNet
import fastmri
import fastmri.fftc as fftc
from fastmri.data import transforms
from torchvision.utils import save_image 

from fastmri.pl_modules.mri_module import MriModule
from torchmetrics.metric import Metric
import lightning as L
from pathlib import Path 

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError 

from typing import Dict, Union

import yaml 
from .losses import NMSE

class CascadeNetModule(L.LightningModule):

    def __init__(
        self,
        hidden_channels: int = 64, 
        n_convs: int = 5, 
        batchnorm: bool = False, 
        no_dc: bool = True, 
        num_cascades: int = 10,
        args: any = None
    ):
        super().__init__()
        # save hyperparameters 
        if args != None:
            params = vars(args)
            self.save_hyperparameters(params)

        self.batch_size = params["batch_size"]
        
        self.cascade_net = CascadeNet(
            hidden_channels=hidden_channels, 
            n_convs=n_convs, 
            batchnorm=batchnorm, 
            no_dc=no_dc,
            num_cascades=num_cascades
        )
        # other losses for validation 
        self.ssim = fastmri.SSIMLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        if args.loss_type == "ssim":
            self.train_loss = self.ssim
        elif args.loss_type == "mse":
            self.train_loss = self.mse
        elif args.loss_type == "l1":
            self.train_loss = self.l1
        else:
            raise ValueError(f"Unkown loss type: {args.loss_type}.")

        self.psnr = PeakSignalNoiseRatio()
        self.loss_type = args.loss_type
        # optimizer parameters 
        self.lr = args.lr
        self.lr_step_size = args.lr_step_size
        self.lr_gamma = args.lr_gamma
        self.weight_decay = args.weight_decay
    
    def forward(self, masked_kspace, mask):
        return self.cascade_net(masked_kspace, mask)

    def process_loss(self, target, pred, _loss_fn, mask=None):
        # calculate the maximum values
        max_target = torch.max(torch.max(fastmri.complex_abs(target), dim=-1)[0], dim=-1)[0].squeeze(dim=-1)
        max_pred = torch.max(torch.max(fastmri.complex_abs(pred), dim=-1)[0], dim=-1)[0].squeeze(dim=-1)
        # do normalization 
        #target = fastmri.complex_abs(target / max_target[:,None,None,None]).unsqueeze(dim=1)
        #pred = fastmri.complex_abs(pred / max_pred[:,None,None,None]).unsqueeze(dim=1)
        target = target / max_target[:,None,None,None]
        pred = pred / max_pred[:,None,None,None]
        
        if "ssim" in str(_loss_fn).lower():
            max_value = torch.max(torch.max(fastmri.complex_abs(target), dim=-1)[0], dim=-1)[0].squeeze(dim=-1)
            def loss_fn(x, y):
                x = fastmri.complex_abs(x).unsqueeze(dim=1)
                y = fastmri.complex_abs(y).unsqueeze(dim=1)
                return _loss_fn(x, y, data_range=max_value)
        elif "psnr" in str(_loss_fn).lower():
            def loss_fn(x, y):
                x = fastmri.complex_abs(x).unsqueeze(dim=1)
                y = fastmri.complex_abs(y).unsqueeze(dim=1)
                return _loss_fn(x, y)
        else:
            def loss_fn(x, y):
                """Calculate other loss"""
                return _loss_fn(x, y)
        return loss_fn(target, pred)

    def training_step(self, batch):
        masked_kspace, mask, _, target = batch
        mask = mask.unsqueeze(dim=1)
        pred = self.forward(masked_kspace, mask)
        #target = torch.view_as_complex(target).unsqueeze(dim=1)
        loss = self.process_loss(target, pred, self.train_loss)
        
        # Calculate regularization term
        params = list(self.parameters())
        # Calculate the L2 norm of the concatenated parameters
        l2_norm = torch.sqrt(sum(p.pow(2).sum() for p in params))
        loss += l2_norm * 1e-7
        self.log(str(self.train_loss), loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, _, target = batch 
        mask = mask.unsqueeze(dim=1)
        pred = self.forward(masked_kspace, mask)
        # calculate magnitude image 
        ssim = self.process_loss(target, pred, self.ssim)
        psnr = self.process_loss(target, pred, self.psnr)
        mse = self.process_loss(target, pred, self.mse)
        l1 = self.process_loss(target, pred, self.l1)
        self.log("val_ssim", 1 - ssim, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_psnr", psnr, batch_size=self.batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mse", mse, batch_size=self.batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_l1", l1, batch_size=self.batch_size, prog_bar=True, logger=True, sync_dist=True)
        
        return {
            "batch_idx": batch_idx,
            "pred": pred, 
            "target": target, 
            "val_loss": ssim
        }

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, _, target = batch 
        mask = mask.unsqueeze(dim=1)
        pred = self.forward(masked_kspace, mask)
        max_value, _ = torch.max(target.view(target.shape[0], target.shape[1], -1), dim=2)
        loss = self.ssim(
            pred, target, data_range=max_value
        )
        psnr = self.psnr(pred, target)
        mse = self.mse(pred, target)
        l1 = self.l1(pred, target)
        self.log("test_ssim", 1 - loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_psnr", psnr, batch_size=self.batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_mse", mse, batch_size=self.batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_l1", l1, batch_size=self.batch_size, prog_bar=True, logger=True, sync_dist=True)
        
        return {
            "batch_idx": batch_idx,
            "pred": pred, 
            "target": target, 
            "val_loss": loss
        }
    
    def predict_step(self, batch, dataloader_idx=5):
        masked_kspace, mask, _, target = batch
        mask = mask.unsqueeze(dim=1)
        pred = self.forward(masked_kspace, mask)
        self.display_preds(masked_kspace, pred, target, 5)
        # calculate magnitude image 
        ssim = self.process_loss(target, pred, self.ssim)
        psnr = self.process_loss(target, pred, self.psnr)
        mse = self.process_loss(target, pred, self.mse)
        l1 = self.process_loss(target, pred, self.l1)
        print(f"val_ssim: {1-ssim:.4f}")
        print(f"val_psnr: {psnr:.4f}")
        print(f"val_mse: {mse:.4f}")
        print(f"val_l1: {l1:.4f}")
        return pred 

    def display_preds(
        self,
        masked_kspace: torch.Tensor,
        output: torch.Tensor,
        ground_truth: torch.Tensor,
        idx: int
    ):
        main_dir = Path("predictions") / f"BatchNo{idx}" 
        if not main_dir.exists():
            main_dir.mkdir(parents=True)

        def scaler(tensor: torch.Tensor()):
            tensor -= tensor.min()
            tensor /= tensor.max()
            return tensor
        undersampled_mag = scaler(fastmri.complex_abs(fftc.ifft2c_new(masked_kspace)))
        undersampled_mag = undersampled_mag.unsqueeze(dim=1)
        output = scaler(fastmri.complex_abs(output)).unsqueeze(dim=1)
        ground_truth = scaler(fastmri.complex_abs(ground_truth)).unsqueeze(dim=1)
        save_image(undersampled_mag[7:-7], fp=main_dir / "undersampled.png", nrow=round(output.shape[0] ** .5))
        save_image(output[7:-7], fp=main_dir / "predictions.png", nrow=round(output.shape[0] ** .5))
        save_image(ground_truth[7:-7], fp=main_dir / "ground_truth.png", nrow=round(output.shape[0] ** .5))

    
    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)
        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of CNN block cascades",
        )
        parser.add_argument(
            "--n_convs", 
            default=5, 
            type=int, 
            help="Number of convolutions in a CascadeBlock"
        )
        parser.add_argument(
            "--hidden_chans",
            default=64,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )

        parser.add_argument(
            "--batchnorm", 
            default=False, 
            type=bool, 
            help="Switch for batch normalization"
        )

        parser.add_argument(
            "--no_dc", 
            default=True, 
            type=bool, 
            help="Switch for soft data-consistency in DCCNN"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=3e-4, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=1e-7,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser

