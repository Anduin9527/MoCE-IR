from typing import List

import os
import pathlib
import numpy as np

from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

try:
    import swanlab
except ImportError:
    swanlab = None
    print("Warning: swanlab not installed.")

from net.moce_ir import MoCEIR

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11
from data.custom_dataset import CustomAIODataset, CustomTestDataset
from utils.loss_utils import FFTLoss
from utils.val_utils import compute_psnr_ssim


class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.balance_loss_weight = opt.balance_loss_weight

        self.net = MoCEIR(
            dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_dec_blocks=opt.num_dec_blocks,
            levels=len(opt.num_blocks),
            heads=opt.heads,
            num_refinement_blocks=opt.num_refinement_blocks,
            topk=opt.topk,
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity,
            depth_type=opt.depth_type,
            stage_depth=opt.stage_depth,
            rank_type=opt.rank_type,
            complexity_scale=opt.complexity_scale,
        )

        if opt.loss_type == "fft":
            self.loss_fn = nn.L1Loss()
            self.aux_fn = FFTLoss(loss_weight=self.opt.fft_loss_weight)
        else:
            self.loss_fn = nn.L1Loss()

    def on_fit_start(self):
        if self.opt.use_swanlab and self.trainer.is_global_zero:
            if swanlab is None:
                print("Swanlab not installed, skipping init.")
                return

            swanlab.init(
                project=self.opt.swanlab_project,
                name=self.opt.swanlab_experiment_name,
                config=vars(self.opt),
            )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch, de_id)
        balance_loss = self.net.total_loss

        if self.opt.loss_type == "fft":
            l1_loss = self.loss_fn(restored, clean_patch)
            fft_loss = self.aux_fn(restored, clean_patch)
            loss = l1_loss + fft_loss
            self.log("loss/l1", l1_loss, sync_dist=True)
            self.log("loss/fft", fft_loss, sync_dist=True)
        else:
            l1_loss = self.loss_fn(restored, clean_patch)
            loss = l1_loss.clone()
            self.log("loss/l1", l1_loss, sync_dist=True)

        loss += self.balance_loss_weight * balance_loss
        self.log("loss/total", loss, sync_dist=True)
        self.log("loss/balance", balance_loss, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, sync_dist=True)

        if self.opt.use_swanlab and self.trainer.is_global_zero:
            log_dict = {
                "train/loss_total": loss.item(),
                "train/loss_balance": balance_loss.item(),
                "train/lr": lr,
            }
            if self.opt.loss_type == "fft":
                log_dict["train/loss_l1"] = l1_loss.item()
                log_dict["train/loss_fft"] = fft_loss.item()
            else:
                log_dict["train/loss_l1"] = l1_loss.item()

            swanlab.log(log_dict)

        return loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        # Forward pass
        restored = self.net(degrad_patch, de_id)
        if isinstance(restored, List) and len(restored) == 2:
            restored, _ = restored

        # Clamp output
        restored = torch.clamp(
            restored, 0, 1
        )  # Assuming input is [0,1] or handled by dataset

        # Compute metrics
        psnr, ssim, _ = compute_psnr_ssim(restored, clean_patch)

        self.log("val/psnr", psnr, sync_dist=True, on_epoch=True)
        self.log("val/ssim", ssim, sync_dist=True, on_epoch=True)
        return {"val_psnr": psnr, "val_ssim": ssim}

    def on_validation_epoch_end(self):
        if self.opt.use_swanlab and self.trainer.is_global_zero:
            # Get metrics from trainer.callback_metrics which handles sync/aggregation
            psnr = self.trainer.callback_metrics.get("val/psnr")
            ssim = self.trainer.callback_metrics.get("val/ssim")

            log_dict = {}
            if psnr is not None:
                log_dict["val/psnr"] = psnr.item()
            if ssim is not None:
                log_dict["val/ssim"] = ssim.item()

            if log_dict:
                swanlab.log(log_dict, step=self.global_step)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=150
        )

        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer, warmup_epochs=1, max_epochs=self.opt.epochs
            )
        return [optimizer], [scheduler]


def main(opt):
    print("Options")
    print(opt)
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    if opt.use_swanlab:
        opt.swanlab_project = opt.swanlab_project if opt.swanlab_project else "MoCE-IR"
        if not opt.swanlab_experiment_name:
            opt.swanlab_experiment_name = opt.model + "_" + time_stamp

        if swanlab is None:
            raise ImportError(
                "Please install swanlab to use it as a logger: pip install swanlab"
            )
        # SwanLab initialization is now handled in PLTrainModel.on_fit_start
        logger = False  # Disable default logger if using SwanLab manually (or use None/TensorBoardLogger as backup)
        # To avoid double logging if we still want TensorBoard, we could use TensorBoardLogger here.
        # But user request implies replacing logic. Let's just disable Lightning logger for now or fallback to False (no logger).
        # Actually, let's keep TensorBoardLogger as a local fallback if desired?
        # The prompt said "abandon lighting related logic" for logger.
        # So setting logger to specific Lightning logger is probably not what's wanted for the main logging.
        # But Lightning needs a logger or False.

    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    # Create model
    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt
        )
    else:
        model = PLTrainModel(opt)

    print(model)
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, every_n_epochs=5, save_top_k=-1, save_last=True
    )

    # Create datasets and dataloaders
    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
    elif opt.trainset == "custom":
        trainset = CustomAIODataset(opt)
    else:
        trainset = AIOTrainDataset(opt)

    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=opt.accum_grad,
        deterministic=True,
    )

    # Optionally resume from a checkpoint
    if opt.resume_from:
        checkpoint_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
    else:
        checkpoint_path = None

    # Create Validation Dataset if custom
    val_loader = None
    if opt.trainset == "custom" and opt.val:
        val_set = CustomTestDataset(opt)
        if len(val_set) > 0:
            val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
            print(f"Validation enabled. Loaded {len(val_set)} validation samples.")
        else:
            print("Warning: Custom dataset root validation set empty or missing.")
    elif not opt.val:
        print("Validation disabled.")

    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path,  # Specify the checkpoint path to resume from
    )


if __name__ == "__main__":
    train_opt = train_options()
    main(train_opt)
