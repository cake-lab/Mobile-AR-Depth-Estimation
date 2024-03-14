# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import csv
import os
import uuid
import warnings
from datetime import datetime as dt
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from zoedepth.utils.config import flatten
from zoedepth.utils.misc import RunningAverageDict, colorize, colors


def is_rank_zero(args):
    return args.rank == 0


class BaseTrainer:
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        """ Base Trainer class for training a model."""

        self.config = config
        self.metric_criterion = "abs_rel"
        if device is None:
            device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

    def resize_to_target(self, prediction, target):
        if prediction.shape[2:] != target.shape[-2:]:
            prediction = nn.functional.interpolate(
                prediction, size=target.shape[-2:], mode="bilinear", align_corners=True
            )
        return prediction

    def load_ckpt(self, checkpoint_dir="./checkpoints", ckpt_type="best"):
        import glob
        import os

        from zoedepth.models.model_io import load_wts

        if hasattr(self.config, "checkpoint"):
            checkpoint = self.config.checkpoint
        elif hasattr(self.config, "ckpt_pattern"):
            pattern = self.config.ckpt_pattern
            matches = glob.glob(os.path.join(
                checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
            if not (len(matches) > 0):
                raise ValueError(f"No matches found for the pattern {pattern}")
            checkpoint = matches[0]
        else:
            return
        model = load_wts(self.model, checkpoint)
        # TODO : Resuming training is not properly supported in this repo. Implement loading / saving of optimizer and scheduler to support it.
        print("Loaded weights from {0}".format(checkpoint))
        warnings.warn(
            "Resuming training is not properly supported in this repo. Implement loading / saving of optimizer and scheduler to support it.")
        self.model = model

    def init_optimizer(self):
        m = self.model.module if self.config.multigpu else self.model

        if self.config.same_lr:
            print("Using same LR")
            if hasattr(m, 'core'):
                m.core.unfreeze()
            params = self.model.parameters()
        else:
            print("Using diff LR")
            if not hasattr(m, 'get_lr_params'):
                raise NotImplementedError(
                    f"Model {m.__class__.__name__} does not implement get_lr_params. Please implement it or use the same LR for all parameters.")

            params = m.get_lr_params(self.config.lr)

        return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.wd)

    def init_scheduler(self):
        lrs = [l['lr'] for l in self.optimizer.param_groups]
        return optim.lr_scheduler.OneCycleLR(self.optimizer, lrs, epochs=self.config.epochs, steps_per_epoch=len(self.train_loader),
                                             cycle_momentum=self.config.cycle_momentum,
                                             base_momentum=0.85, max_momentum=0.95, div_factor=self.config.div_factor, final_div_factor=self.config.final_div_factor, pct_start=self.config.pct_start, three_phase=self.config.three_phase)

    def train_on_batch(self, batch, train_step):
        raise NotImplementedError

    def validate_on_batch(self, batch, val_step):
        raise NotImplementedError

    def raise_if_nan(self, losses):
        for key, value in losses.items():
            if torch.isnan(value):
                raise ValueError(f"{key} is NaN, Stopping training")

    @property
    def iters_per_epoch(self):
        return len(self.train_loader)

    @property
    def total_iters(self):
        return self.config.epochs * self.iters_per_epoch

    def should_early_stop(self):
        if self.config.get('early_stop', False) and self.step > self.config.early_stop:
            return True

    def train(self):
        print(f"Training {self.config.name}")
        if self.config.uid is None:
            self.config.uid = str(uuid.uuid4()).split('-')[-1]
        run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{self.config.uid}"
        self.config.run_id = run_id
        self.config.experiment_id = f"{self.config.name}{self.config.version_name}_{run_id}"
        self.should_write = ((not self.config.distributed)
                             or self.config.rank == 0)
        self.should_log = self.should_write  # and logging
        if self.should_log:
            tags = self.config.tags.split(',') if self.config.tags != '' else None
            wandb.init(project=self.config.project, name=self.config.experiment_id, config=flatten(self.config), dir=self.config.root,
                       tags=tags, notes=self.config.notes, settings=wandb.Settings(start_method="fork"))

            with open('./logs/training_log_trian.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'Step','loss_function', 'loss', 'rmse','rmse_log','silog', 'log_10' ,'a1', 'a2', 'a3', 'abs_rel', 'sq_rel'])
            with open('./logs/validation_log_trian.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'Step','loss_function', 'loss', 'rmse','rmse_log','silog', 'log_10' ,'a1', 'a2', 'a3', 'abs_rel', 'sq_rel'])
            with open('./logs/validation_log_int_trian.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'Step','loss_function', 'loss', 'rmse','rmse_log','silog', 'log_10' ,'a1', 'a2', 'a3', 'abs_rel', 'sq_rel'])
        self.model.train()
        self.step = 0
        best_loss = np.inf
        validate_every = int(self.config.validate_every * self.iters_per_epoch)

        if self.config.prefetch:

            for i, batch in tqdm(enumerate(self.train_loader), desc=f"Prefetching...", total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader):
                pass

        losses = {}
        def stringify_losses(L): return "; ".join(map(
            lambda kv: f"{colors.fg.purple}{kv[0]}{colors.reset}: {round(kv[1].item(),3):.4e}", L.items()))
        
        for epoch in range(self.config.epochs):
            if self.should_early_stop():
                break
            print(epoch, self.step)
            self.epoch = epoch
            if epoch == 0:
                self.model.eval()
                metrics, test_losses = self.validate()
                wandb.log({f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)
                wandb.log({f"Metrics/{k}": v for k,
                                      v in metrics.items()}, step=self.step)
            ################################# Train loop ##########################################################
            if self.should_log:
                print(f"Epoch {epoch}")
                wandb.log({"Epoch": epoch}, step=self.step)
            pbar = tqdm(enumerate(self.train_loader), desc=f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train",
                        total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader)
            for i, batch in pbar:
                if self.should_early_stop():
                    print("Early stopping")
                    break
                t_metrics, losses = self.train_on_batch(batch, i) # forward pass
                self.raise_if_nan(losses)
                if is_rank_zero(self.config) and self.config.print_losses:
                    pbar.set_description(
                        f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train. Losses: {stringify_losses(losses)}")
                    
                self.scheduler.step() # update learning rate

                if self.should_log: 
                    wandb.log({f"Train/{name}": loss.item() for name, loss in losses.items()}, step=self.step)
                    

                    with open('./logs/training_log_trian.csv', mode='a') as file:
                        writer = csv.writer(file) 
                        for name, loss in losses.items():
                            writer.writerow(
                                [epoch, self.step, f"Train/{name}", loss.item(), t_metrics['rmse'], t_metrics['rmse_log'], t_metrics['silog'], t_metrics['log_10'], t_metrics['a1'], t_metrics['a2'], t_metrics['a3'], t_metrics['abs_rel'], t_metrics['sq_rel']])

                self.step += 1

                ########################################################################################################
                if self.test_loader:
                    if (self.step % validate_every) == 0:
                        self.model.eval()
                        self.save_checkpoint(
                                f"{self.config.experiment_id}_latest_trian.pt")
                        ################################# Validation loop ##################################################
                        # validate on the entire validation set in every process but save only from rank 0, I know, inefficient, but avoids divergence of processes
                        metrics, test_losses = self.validate()
                        
                        if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                            wandb.log(
                                {f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)

                            wandb.log({f"Metrics/{k}": v for k,
                                      v in metrics.items()}, step=self.step)
                            
                            self.save_checkpoint(
                                f"{self.config.experiment_id}_best.pt")
                            best_loss = metrics[self.metric_criterion]
                        self.model.train()

                        if self.config.distributed:
                            dist.barrier()
                        # print(f"Validated: {metrics} on device {self.config.rank}")
                    #################################################################################################
        # Save / validate at the end
        self.step += 1  # log as final point
        self.model.eval()
        self.save_checkpoint(f"{self.config.experiment_id}_latest_trian.pt")
        if self.test_loader:
            ################################# Validation loop ##################################################
            metrics, test_losses = self.validate()
            if self.should_log:
                wandb.log({f"Test/{name}": tloss for name,
                          tloss in test_losses.items()}, step=self.step)
                wandb.log({f"Metrics/{k}": v for k,
                          v in metrics.items()}, step=self.step)
                if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                    self.save_checkpoint(
                        f"{self.config.experiment_id}_best_trian.pt")
                    best_loss = metrics[self.metric_criterion]

        self.model.train()

    def validate(self):
        with torch.no_grad():
            losses_avg = RunningAverageDict()
            metrics_avg = RunningAverageDict()
            per_batch = []
            for i, batch in tqdm(enumerate(self.test_loader), desc=f"Epoch: {self.epoch + 1}/{self.config.epochs}. Loop: Validation", total=len(self.test_loader), disable=not is_rank_zero(self.config)):
                metrics, losses = self.validate_on_batch(batch, val_step=i)
                if len(per_batch) == self.config.batch_size:
                    average_list = [sum(x) / len(x) for x in zip(*per_batch)]
                    
                    with open('./logs/validation_log_int_trian.csv', mode='a') as file:
                        writer = csv.writer(file) 
                        writer.writerow(
                                    [self.epoch, self.step, f"validation/SILog", average_list[0], average_list[1], average_list[2], average_list[3], average_list[4], average_list[5], average_list[6], average_list[7], average_list[8], average_list[9]]) 
                    average_list = []
                    per_batch = []
                else: 
                    per_batch.append([losses['SILog'],  metrics['rmse'], metrics['rmse_log'], metrics['silog'], metrics['log_10'], metrics['a1'], metrics['a2'], metrics['a3'], metrics['abs_rel'], metrics['sq_rel']])
                    
                    
                if losses:
                    losses_avg.update(losses)
                if metrics:
                    metrics_avg.update(metrics)
            
            if self.should_log:
                with open('./logs/validation_log_trian.csv', mode='a') as file:
                    writer = csv.writer(file) 
                    writer.writerow(
                            [self.epoch, self.step, f"validation/SILog", losses_avg.get_value()['SILog'], metrics_avg.get_value()['rmse'], metrics_avg.get_value()['rmse_log'], metrics_avg.get_value()['silog'], metrics_avg.get_value()['log_10'], metrics_avg.get_value()['a1'], metrics_avg.get_value()['a2'], metrics_avg.get_value()['a3'], metrics_avg.get_value()['abs_rel'], metrics_avg.get_value()['sq_rel']])
            return metrics_avg.get_value(), losses_avg.get_value()

    def save_checkpoint(self, filename):
        if not self.should_write:
            return
        root = self.config.save_dir
        if not os.path.isdir(root):
            os.makedirs(root)

        fpath = os.path.join(root, filename)
        m = self.model.module if self.config.multigpu else self.model
        torch.save(
            {
                "model": m.state_dict(),
                "optimizer": None,  # TODO : Change to self.optimizer.state_dict() if resume support is needed, currently None to reduce file size
                "epoch": self.epoch
            }, fpath)

    def log_images(self, rgb: Dict[str, list] = {}, depth: Dict[str, list] = {}, scalar_field: Dict[str, list] = {}, prefix="", scalar_cmap="jet", min_depth=None, max_depth=None):
        if not self.should_log:
            return

        if min_depth is None:
            try:
                min_depth = self.config.min_depth
                max_depth = self.config.max_depth
            except AttributeError:
                min_depth = None
                max_depth = None

        depth = {k: colorize(v, vmin=min_depth, vmax=max_depth)
                 for k, v in depth.items()}
        scalar_field = {k: colorize(
            v, vmin=None, vmax=None, cmap=scalar_cmap) for k, v in scalar_field.items()}
        images = {**rgb, **depth, **scalar_field}
        wimages = {
            prefix+"Predictions": [wandb.Image(v, caption=k) for k, v in images.items()]}
        wandb.log(wimages, step=self.step)

    def log_line_plot(self, data):
        if not self.should_log:
            return

        plt.plot(data)
        plt.ylabel("Scale factors")
        wandb.log({"Scale factors": wandb.Image(plt)}, step=self.step)
        plt.close()

    def log_bar_plot(self, title, labels, values):
        if not self.should_log:
            return

        data = [[label, val] for (label, val) in zip(labels, values)]
        table = wandb.Table(data=data, columns=["label", "value"])
        wandb.log({title: wandb.plot.bar(table, "label",
                  "value", title=title)}, step=self.step)
