"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch

from models.varnet import VarNet
#from ..models.varnet import VarNet
import fastmri
import fastmri.fftc as fftc
from fastmri.data import transforms

from fastmri.pl_modules.mri_module import MriModule
from torchmetrics.metric import Metric
import lightning as L
from pathlib import Path 
from torchvision.utils import save_image 
import matplotlib.pyplot as plt 
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError 
import torchvision.transforms.functional as F 
from matplotlib import cm
import numpy as np 
from torchvision.utils import make_grid, save_image
import pickle

from typing import Dict, Union
import yaml 
from .losses import NMSE, nmse, mse, rmse, nrmse 

class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity

class VarNetModule(L.LightningModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        single_coil: bool = False,
        args = None, 
        data_module = None,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
 
        # save hyperparameters 
        if args is not None:
            params = vars(args)
            #self.save_hyperparameters(params)

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.single_coil = single_coil

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            single_coil=single_coil
        )  
        self.loss = fastmri.SSIMLoss()
        self.psnr = PeakSignalNoiseRatio()
        self.result = {"mse": [], "nmse": [], "rmse": [], "psnr": [], "ssim": []} if args.mode == "test_all" else None
        self.dataset_type = args.dataset_type
        self.args = args
        self.data_module = data_module 
        self.batch_size = params["batch_size"] 


    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.varnet(masked_kspace, mask, num_low_frequencies)

    def training_step(self, batch, batch_idx):
        if self.single_coil:
            if self.dataset_type in ("knee", "brain"):
                masked_kspace, mask, num_low_frequencies, target = batch
            elif self.dataset_type == "ocmr": # varnet model for cardiac data 
                masked_kspace, mask, num_low_frequencies, target = batch   
                mask = mask.squeeze(dim=0)
                masked_kspace = masked_kspace.squeeze(dim=0)
                target = target.squeeze(dim=0)
            elif self.dataset_type == "cardiac":
                masked_kspace, mask, num_low_frequencies, target = batch  
                mask = mask.unsqueeze(dim=1)
            output = self.forward(masked_kspace, mask, num_low_frequencies)
            # calculate SSIM loss
            max_vals = torch.max(torch.max(target, dim=-1)[0], dim=-1)[0]
            loss = self.loss(
                    output, target, data_range=max_vals
                )
        self.log("train_ssim", 1 - loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch[-1].shape[0]
        if self.dataset_type in ("knee", "brain"): 
            masked_kspace, mask, num_low_frequencies, target = batch
        elif self.dataset_type == "ocmr": # varnet model for cardiac data 
            masked_kspace, mask, num_low_frequencies, target = batch
            masked_kspace = masked_kspace.squeeze(dim=0)
            target = target.squeeze(dim=0)
            mask = mask.squeeze(dim=0)
            #target = fastmri.complex_abs(target).unsqueeze(dim=1)
        elif self.dataset_type == "cardiac":
            masked_kspace, mask, num_low_frequencies, target = batch   
            mask = mask.unsqueeze(dim=1)
        # forward pass
        output = self.forward(masked_kspace, mask, num_low_frequencies)
        # calculate SSIM loss
        max_vals = torch.max(torch.max(target, dim=-1)[0], dim=-1)[0]
        loss = self.loss(
            output, target, data_range=max_vals
        )
        #self.display_preds(masked_kspace, output, target, batch_idx)
        psnr = self.psnr(output, target)
        # get representative images 
        # calculate other losses
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        nmse_loss = nmse(target, output)
        mse_loss = mse(target, output)
        rmse_loss = rmse(target, output)
        nrmse_loss = nrmse(target, output)

        self.log("val_ssim", 1 - loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)   
        self.log("val_psnr", psnr, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_nmse", nmse_loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mse", mse_loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_rmse", rmse_loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_nrmse", nrmse_loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)

        return {
            "batch_idx": batch_idx,
            "output": output,
            "target": target,
            "val_loss": loss
        }

    """
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.args.save_every_n_epoch == 0: 
            for i in range(10):
                batch, scan_name = self.data_module.val_data.random_scan(random_seed=101 * i)
                self.predict_step(batch, scan_name=scan_name)
    """
    
    def on_test_epoch_start(self):
        if self.args.mode == "test_all":
            #for i in range(20):
            #    batch, scan_name = self.data_module.test_data.random_scan(random_seed=101 * i) # 101 and 30
            #    print(scan_name, i)
            #    continue
            #    self.predict_step(batch, scan_name=scan_name, test=True)
            #import sys; sys.exit()
            #return 
            i = 11 # 9 10 (knee) 11 (brain)
            batch, scan_name = self.data_module.test_data.random_scan(random_seed=101 * i) # 101 and 30
            self.predict_step(batch, scan_name=scan_name, test=True)
        super().on_test_epoch_start()
    


    def test_step(self, batch, batch_idx):
        #return None # YOU NEED TO PUT THIS FOR MODE='test_all'
        batch_size = batch[-1].shape[0]
        if self.dataset_type in ("knee", "brain"): 
            masked_kspace, mask, num_low_frequencies, target = batch
        elif self.dataset_type == "ocmr": # varnet model for cardiac data 
            masked_kspace, mask, num_low_frequencies, target = batch
            masked_kspace = masked_kspace.squeeze(dim=0)
            target = target.squeeze(dim=0)
            mask = mask.squeeze(dim=0)
            #target = fastmri.complex_abs(target).unsqueeze(dim=1)
        elif self.dataset_type == "cardiac":
            masked_kspace, mask, num_low_frequencies, target = batch   
            mask = mask.unsqueeze(dim=1)
        # forward pass
        us_mag = fastmri.complex_abs(fastmri.ifft2c(masked_kspace))
        output = us_mag
        #output = self.forward(masked_kspace, mask, num_low_frequencies)
        # calculate SSIM loss
        max_vals = torch.max(torch.max(target, dim=-1)[0], dim=-1)[0]
        loss = self.loss(
            output, target, data_range=max_vals
        )
        #self.display_preds(masked_kspace, output, target, batch_idx)
        psnr = self.psnr(output, target)
        # get representative images 
        # calculate other losses
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        nmse_loss = nmse(target, output)
        mse_loss = mse(target, output)
        rmse_loss = rmse(target, output)
        nrmse_loss = nrmse(target, output)

        self.log("val_ssim", 1 - loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)   
        self.log("val_psnr", psnr, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_nmse", nmse_loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mse", mse_loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_rmse", rmse_loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_nrmse", nrmse_loss, batch_size=batch_size, prog_bar=True, logger=True, sync_dist=True)

        return {
            "batch_idx": batch_idx,
            "output": output,
            "target": target,
            "val_loss": loss
        }


    def predict_step(self, batch, dataloader_idx=5, scan_name=None, test=False):
        if self.dataset_type in ("knee", "brain"):
            masked_kspace, mask, num_low_frequencies, target = batch 
            idx2 = target.shape[0] // 2
            idxs = [idx2 - 5, idx2, idx2 + 3]
            target = target[idxs].to(self.device)
            mask = mask[idxs].to(self.device)
            masked_kspace = masked_kspace[idxs].to(self.device)
        elif self.dataset_type == "ocmr":
            masked_kspace = masked_kspace.squeeze(dim=0)
            mask = mask.squeeze(dim=0)
            target = target.squeeze(dim=0)
        elif self.dataset_type == "cardiac":
            mask = mask.unsqueeze(dim=1)
        # forward pass
        output = self.forward(masked_kspace, mask, num_low_frequencies)
        us_mag = fastmri.complex_abs(fastmri.ifft2c(masked_kspace))
        # calculate SSIM loss
        max_vals = torch.max(torch.max(target, dim=-1)[0], dim=-1)[0]
        pred_ssim = 1 - self.loss(output, target, data_range=max_vals)
        us_ssim = 1 - self.loss(us_mag, target, data_range=max_vals)

        psnr = self.psnr(output, target)
        # calculate undersampled image's losses
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        us_mag = us_mag.detach().cpu().numpy()
        pred_nrmse = nrmse(target, output)
        us_nrmse = nrmse(target, us_mag)
        #us_nrmse = [nrmse(target[i], us_mag[i]) for i in range(target.shape[0])]
        #pred_nrmse = [nrmse(target[i], output[i]) for i in range(target.shape[0])]

        if test == False:
            self.display_gtvspred(
                us_mag,
                output, 
                target, 
                [us_ssim, pred_ssim], 
                [us_nrmse, pred_nrmse], 
                scan_name
                )
        else:      
            # create directory for tensors 
            save_path = Path(self._trainer.default_root_dir) / "tensors" / scan_name
            save_path.mkdir(exist_ok=True, parents=True)
            # save the output and target tensors 
            phase_type = self.args.ckpt_path.parent.parent.parent.name
            np.save(save_path / (phase_type + ".npy"), output)
            np.save(save_path / "gt_mag.npy", target)
            # save the naive recon 
            np.save(save_path / "us_recon.npy", us_mag)
            losses = {"ssim": pred_ssim, "nrmse": pred_nrmse}
            with open(save_path / (phase_type + "_loss.pkl"), "wb") as file:
                pickle.dump(losses, file)
                file.close()
            losses2 = {"ssim": us_ssim, "nrmse": us_nrmse}
            with open(save_path / ("us_recon_loss.pkl"), "wb") as file:
                pickle.dump(losses2, file)
                file.close()
            
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def display_gtvspred(
        self,
        undersampled: torch.Tensor,
        output: np.array,
        target: np.array,
        ssim_list: float,
        nrmse_list: float, 
        scan_name: str
    ):
        def normalizer(x: torch.Tensor):
            max_vals = torch.max(torch.max(x, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
            min_vals = torch.min(torch.min(x, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
            x = (x - min_vals) / (max_vals - min_vals)
            return x 
        output = normalizer(torch.from_numpy(output))
        target = normalizer(torch.from_numpy(target))
        undersampled = normalizer(torch.from_numpy(undersampled))
        # construct grids
        nrow = 1
        output_grid = make_grid(output, nrow, padding=8)
        target_grid = make_grid(target, nrow, padding=8)
        undersamp_grid = make_grid(undersampled, nrow, padding=8)
        combined_grid = torch.cat((target_grid, undersamp_grid, output_grid), dim=2)
        combined_grid = combined_grid.permute(1,2,0).cpu().numpy()
        # save the figure of combination of grids
        num_grids = 3
        dpi = 400
        grid_width = combined_grid.shape[1]
        grid_height = combined_grid.shape[0]
        fig_width = grid_width / dpi
        fig_height = (grid_height / dpi) * num_grids
        # Create the figure with improved resolution
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        column_width = grid_width // 3

        phase_dict = {"rand_phase": "random", "gan2_phase": "GAN", "gan_phase": "GAN", "diff_phase": "Diffusion", "gt_phase": "GT", "zero_phase": "zero", "gaus_phase": "Gaussian"}
        phase_name = phase_dict[self.args.phase_type]
        title_list = ["GT mag.", "US mag.", f"{phase_name} recon."]
        plt.text(0.5 * column_width, -12, title_list[0], fontsize=4, ha='center')
        for i in range(1, 3):
            column_text = f"SSIM:{ssim_list[i-1]:.4f}\nNRMSE:{nrmse_list[i-1]:.4f}"
            column_center = (i - .5 + 1) * column_width
            plt.text(column_center, -12, title_list[i], fontsize=4, ha='center')
            plt.text(column_center, grid_height + 80, column_text, fontsize=4, ha='center')
        # Display the combined grid using Matplotlib
        plt.imshow(combined_grid, cmap='gray')
        plt.axis('off')  # Turn off the axis
        plt.title(f"VarNet {self.current_epoch}th epoch, R: {self.args.accelerations}, ACS: {self.args.center_fractions}, Model: {phase_name} phase.", fontsize=5)
        save_path = Path(self._trainer.default_root_dir) / "images" / scan_name
        save_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path / f"recon_epoch{self.current_epoch}", transparent=True, bbox_inches="tight") 
        plt.close()

    def display_preds(
        self,
        masked_kspace: torch.Tensor,
        output: torch.Tensor,
        ground_truth: torch.Tensor,
        idx: int, 
        nrmse_recon: float, 
        nrmse_us: float,
        ssim_recon: float,
        ssim_us: float
    ):
        scan_name = self.args.pred_path.parts[-2]
        scan_dir = self.args.display_path / scan_name / f"frame{idx}" 
        scan_dir.mkdir(parents=True, exist_ok=True)
        undersampled_mag = fastmri.complex_abs(fftc.ifft2c_new(masked_kspace))
        # calculate error maps
        normalizer = lambda x : ( x - x.min() ) / (x.max() - x.min())
        output = normalizer(output)
        ground_truth = normalizer(ground_truth)
        undersampled_mag = normalizer(undersampled_mag)
        err_map = lambda pred, target: torch.abs(pred - target)
        output = F.hflip(output)
        ground_truth = F.hflip(ground_truth)
        undersampled_mag = F.hflip(undersampled_mag)
        pred_err = err_map(output, ground_truth)
        undersamp_err = err_map(undersampled_mag, ground_truth)
        # change cmap 
        save_image(ground_truth[idx], fp=scan_dir / "ground_truth.png", nrow=1)
        max, min = 1, 0
        # NEW IMAGE 
        # Plot reconstruction error map 
        fig, ax = plt.subplots()
        im = ax.imshow(pred_err[idx,0].cpu().numpy() * 1.5, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        pos = (.97, .03)
        #ax.text(pos[0], pos[1], f"{nrmse_recon * 100:.1f}%", transform=ax.transAxes, fontsize=36, weight='black', ha='right', va='bottom', color='yellow')
        plt.savefig(str(scan_dir / f"{self.args.phase_type}_err.png"), transparent=True)
        plt.show()
        plt.close()
        # NEW IMAGE 
        fig, ax = plt.subplots()
        # Plot undersampled error map with 
        im = ax.imshow(undersamp_err[idx,0].cpu().numpy() * 1.5, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        #fig.colorbar(im, ax=ax)
        pos = (.97, .03)
        #ax.text(pos[0], pos[1], f"{nrmse_us * 100:.1f}%", transform=ax.transAxes, fontsize=36, weight='black', ha='right', va='bottom', color='yellow')
        plt.savefig(str(scan_dir / "undsamp_err.png"), transparent=True)
        plt.show()
        plt.close()
        # NEW IMAGE 
        fig, ax = plt.subplots()
        # Plot reconstruction image with NRMSE %
        recon = output[idx, 0].cpu().numpy()
        im = ax.imshow(recon, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        #fig.colorbar(im, ax=ax)
        pos = (.03, .03)
        ax.text(pos[0], pos[1], f"NRMSE: {nrmse_recon:.3f}\nSSIM: {ssim_recon:.3f}", transform=ax.transAxes, fontsize=20, weight='semibold', ha='left', va='bottom', color='white')
        plt.savefig(str(scan_dir / f"{self.args.phase_type}_recon.png"), transparent=True)
        plt.show()
        plt.close()
        fig, ax = plt.subplots()
        # Plot undersampled image with NMSE %
        us_mag = undersampled_mag[idx, 0].cpu().numpy()
        im = ax.imshow(us_mag, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        #fig.colorbar(im, ax=ax)
        pos = (.03, .03) #.97 .03
        ax.text(pos[0], pos[1], f"NRMSE: {nrmse_us:.3f}\nSSIM: {ssim_us:.3f}", transform=ax.transAxes, fontsize=20, weight='semibold', ha='left', va='bottom', color='white')
        plt.savefig(str(scan_dir / "us_mag.png"), transparent=True)
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        # save the phase to the target folder
        if self.args.phase_type in ("gt_phase", "diff_phase", "gan_phase"):
            phase_path = Path("../OCMR/ocmr_recon_data") / scan_name / self.args.pred_path.name / self.args.phase_type / f"frame{idx}.npy"
            img = np.load(phase_path)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            im = ax.imshow(np.flip(img, axis=1), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(str(scan_dir / f"{self.args.phase_type}"), transparent=True)
            plt.show()
            plt.close()


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
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
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
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser

