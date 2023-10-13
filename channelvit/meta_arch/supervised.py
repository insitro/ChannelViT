# Evaluate the DINO embedding through linear probing
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

import channelvit.backbone as backbone
import channelvit.data as data
import channelvit.utils as utils


class Supervised(pl.LightningModule):
    def __init__(self, main_supervised_cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = self.set_cfg(main_supervised_cfg)

        # load the data cfg from the root cfg
        self.val_data_cfg = main_supervised_cfg.val_data_dict

        # load the transformation cfg from the root cfg
        # note that in the transformation, we can play with different channel masking
        # baselines
        self.val_transform_cfg = main_supervised_cfg.val_transformations

        self.save_hyperparameters()

        # get the backbone
        self.backbone = getattr(backbone, self.cfg.backbone.name)(
            **self.cfg.backbone.args
        )

        self.classifier = nn.Linear(self.backbone.embed_dim, self.cfg.num_classes)

        self.compute_loss = torch.nn.CrossEntropyLoss(
            label_smoothing=self.cfg.label_smoothing
        )

        # We treat the prediction target as one of the covariates. Here we specify the
        # prediction target key.
        self.target = self.cfg.target

        self.validation_step_outputs = []
        self.best_validation_results = {}

        self.configure_scheduler()

    def set_cfg(self, main_supervised_cfg: DictConfig) -> DictConfig:
        cfg = main_supervised_cfg.meta_arch

        # set the optimization configurations
        with open_dict(cfg):
            try:
                cfg.total_batch_size = (
                    main_supervised_cfg.trainer.devices
                    * main_supervised_cfg.train_data.loader.batch_size
                    * main_supervised_cfg.trainer.accumulate_grad_batches
                )
                cfg.num_batches = (
                    main_supervised_cfg.train_data.loader.num_batches
                    // main_supervised_cfg.trainer.accumulate_grad_batches
                )
                cfg.num_batches_original = (
                    main_supervised_cfg.train_data.loader.num_batches
                )
            except Exception as e:
                print(e)
                cfg.total_batch_size = (
                    main_supervised_cfg.trainer.devices
                    * main_supervised_cfg.train_data.loader.batch_size
                )
                cfg.num_batches = main_supervised_cfg.train_data.loader.num_batches
                cfg.num_batches_original = (
                    main_supervised_cfg.train_data.loader.num_batches
                )

            cfg.max_epochs = main_supervised_cfg.trainer.max_epochs
            cfg.backbone.patch_size = cfg.patch_size

        print(cfg)
        return cfg

    def val_dataloader(self):
        """Create the validation data loaders for the linear probing.
        """
        print("Loading the validation data loaders.")
        val_loaders = []
        val_data_name_list = []
        for val_data_name, val_data_cfg in self.val_data_cfg.items():
            val_data_name_list.append(val_data_name)
            print(f"Loading {val_data_name}")
            val_data = getattr(data, val_data_cfg.name)(
                is_train=False,
                transform_cfg=self.val_transform_cfg,
                **val_data_cfg.args,
            )
            val_loaders.append(
                DataLoader(
                    val_data, **val_data_cfg.loader, collate_fn=val_data.collate_fn
                )
            )

        self.val_data_name_list = val_data_name_list

        return val_loaders

    def train_single_batch(self, batch):
        # unpack the batch
        imgs, covariates = batch

        output = self.backbone(imgs, covariates)

        logit = self.classifier(output)

        labels = covariates[self.target]

        loss = self.compute_loss(logit, labels)

        return loss

    def training_step(self, batch, batch_idx):
        # We use manual optimization here
        # get the optimizer and set the lr / wd rate
        optimizer = self.optimizers(use_pl_optimizer=True)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[self.global_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[self.global_step]

        self.log("learning_rate", self.lr_schedule[self.global_step])
        self.log("weight_decay", self.wd_schedule[self.global_step])

        if type(batch) is dict:
            # We average the loss across all batches for all data loaders
            current_ratio = float(batch_idx) / self.cfg.num_batches_original
            for key, single_batch in batch.items():
                if current_ratio < self.cfg.data_ratio:
                    if key == "jumpcp" or key == "so2sat_hard_sentinel_1":
                        loss = self.train_single_batch(single_batch)
                        break
                    else:
                        continue
                else:
                    if key != "jumpcp" and key != "so2sat_hard_sentinel_1":
                        loss = self.train_single_batch(single_batch)
                        break
                    else:
                        continue

        else:
            loss = self.train_single_batch(batch)

        self.log(f"train_loss", loss, on_step=True, prog_bar=True, rank_zero_only=True)

        return loss

    def inference(self, batch, batch_idx):
        """Make predictions from the trained linear classifier"""
        # unpack the batch
        imgs, covariates = batch

        output = self.backbone(imgs, covariates)
        labels = covariates[self.target]

        pred = self.classifier(output)

        return {"logit": pred, "true": labels}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.inference(batch, batch_idx)
        outputs["dataloader_idx"] = dataloader_idx

        self.validation_step_outputs.append(outputs)

        return outputs

    def on_validation_epoch_end(self):
        val_outputs_per_loader = {}

        for outputs in self.validation_step_outputs:
            idx = outputs["dataloader_idx"]
            if idx not in val_outputs_per_loader:
                val_outputs_per_loader[idx] = {"logit": [], "true": []}

            val_outputs_per_loader[idx]["logit"].append(outputs["logit"])
            val_outputs_per_loader[idx]["true"].append(outputs["true"])

        for loader_idx, outputs_dict in val_outputs_per_loader.items():
            all_preds = torch.cat(outputs_dict["logit"])
            all_trues = torch.cat(outputs_dict["true"])

            loss = F.cross_entropy(all_preds, all_trues).mean()
            acc = (torch.argmax(all_preds, dim=-1) == all_trues).float().mean()

            dist.all_reduce(loss)
            dist.all_reduce(acc)

            loss = loss / dist.get_world_size()
            acc = acc / dist.get_world_size()

            self.log(
                f"val_{loader_idx}_acc",
                acc,
                on_epoch=True,
                prog_bar=True,
                rank_zero_only=True,
            )
            self.log(
                f"val_{loader_idx}_loss",
                loss,
                on_epoch=True,
                prog_bar=False,
                rank_zero_only=True,
            )

        # do something with all preds
        self.validation_step_outputs.clear()  # free memory

    def predict_step(self, batch, batch_idx):
        return self.inference(batch, batch_idx)

    def configure_optimizers(self):
        """Loading optimizer and learning rate / weight decay schedulers"""

        params_groups = utils.get_params_groups(self.backbone)
        params_groups[0]["params"] += self.classifier.parameters()

        print("Creating optimizer.")
        if self.cfg.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        else:
            raise ValueError("DINO only supports optimizer adaw, sgd and lars.")

        return optimizer

    def configure_scheduler(self) -> None:
        print("Creating learning rate, weight decay and momentum scheduler.")
        # Note that these schedulers are not the typical pytorch schedulers. they are
        # just simple np arrays. The length of each array equals to the total number of
        # steps.
        print(self.cfg.lr)
        print(self.cfg.total_batch_size)
        self.lr_schedule = utils.cosine_scheduler(
            self.cfg.lr * self.cfg.total_batch_size / 256.0,  # linear scaling rule
            self.cfg.min_lr,
            self.cfg.max_epochs,
            self.cfg.num_batches,
            warmup_epochs=self.cfg.warmup_epochs,
        )

        self.wd_schedule = utils.cosine_scheduler(
            self.cfg.weight_decay,
            self.cfg.weight_decay_end,
            self.cfg.max_epochs,
            self.cfg.num_batches,
        )
