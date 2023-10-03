# Evaluate the supervised embedding through linear probing
import boto3
import pytorch_lightning as pl
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import amlssl.data as data
import amlssl.utils as utils
from amlssl.meta_arch import Supervised


def get_checkpoint_path(ckpt):
    '''Get the exact checkpoint path based on the prefix and the epoch number.
    Note that the original checkpoint is named after both the epochs and the number of
    updates. This method enables us to automatically fetch the latest ckpt based on the
    epoch number.
    '''
    assert ckpt.startswith("s3://insitro-user/")
    prefix = ckpt[len("s3://insitro-user/"):]

    try:
        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket="insitro-user",
                                      Prefix=prefix,
                                      MaxKeys=100)
        assert (len(response['Contents']) == 1,
            "Received more than one files with the given ckpt prefix")

        ckpt = "s3://insitro-user/" + response['Contents'][0]['Key']
        print(f"Getting ckpt from {ckpt}")

        return ckpt

    except Exception as e:
        print(prefix)
        raise e


class SupervisedProb(pl.LightningModule):
    def __init__(self, main_supervised_prob_cfg: DictConfig) -> None:
        super().__init__()
        # read the cfg for the linear prob meta arch
        # e.g.: amlssl/config/meta_arch/linear_prob.yaml
        self.cfg = main_supervised_prob_cfg.meta_arch

        # load the data cfg from the root cfg
        self.train_data_cfg = main_supervised_prob_cfg.train_data
        self.val_data_cfg = main_supervised_prob_cfg.val_data_dict

        # load the transformation cfg from the root cfg
        # note that in the transformation, we can play with different channel masking
        # baselines
        self.transform_cfg = main_supervised_prob_cfg.transformations

        self.save_hyperparameters()

        if self.cfg.checkpoint != None:
            print("Loading pre-trained SSL model.")
            checkpoint = get_checkpoint_path(self.cfg.checkpoint)
            self.model = Supervised.load_from_checkpoint(checkpoint)
            print("Print the checkpoint configurations.")
            print(self.model.cfg, flush=True)

        else:
            raise ValueError(
                "No checkpoint available.i\n"
                "You can set the checkpoint by running "
                "python contextvit/main/main_linear_prob.py "
                "meta.arch.checkpoint={PATH_TO_CKPT}"
            )

        self.compute_loss = torch.nn.CrossEntropyLoss(
            label_smoothing=self.cfg.label_smoothing
        )
        self.n_last_blocks = self.cfg.n_last_blocks


        # linear layer for prediction
        input_dim = self.model.backbone.embed_dim * self.n_last_blocks
        if self.cfg.covariate.name != None:
            input_dim += self.cfg.covariate.embedding_dim
            self.covariate_embed = nn.Embedding(self.cfg.covariate.num_embeddings,
                                                self.cfg.covariate.embedding_dim)
        else:
            self.covariate_embed = None

        if self.cfg.use_mlp:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, self.cfg.num_classes)
            )
        else:
            self.classifier = nn.Linear(input_dim, self.cfg.num_classes)

        # We treat the prediction target as one of the covariates. Here we specify the
        # prediction target key.
        self.target = self.cfg.target

        self.validation_step_outputs = []
        self.best_validation_results = {}

        with open_dict(self.cfg):
            # override the T_max based on the maximum number of epochs of the trainer.
            self.cfg.scheduler.args.T_max = main_supervised_prob_cfg.trainer.max_epochs

    def train_dataloader(self):
        """Create the training data loader for the linear probing.
        """
        print("Loading the training data loader.")
        assert len(self.train_data_cfg) == 1, "Only one training data is allowed."
        train_data_cfg = next(iter(self.train_data_cfg.values()))
        train_data = getattr(data, train_data_cfg.name)(
            is_train=True, transform_cfg=self.transform_cfg, **train_data_cfg.args
        )
        train_loader = DataLoader(train_data, **train_data_cfg.loader)

        return train_loader

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
                is_train=False, transform_cfg=self.transform_cfg, **val_data_cfg.args
            )
            val_loaders.append(DataLoader(val_data, **val_data_cfg.loader))

        self.val_data_name_list = val_data_name_list

        return val_loaders

    def training_step(self, batch, batch_idx):
        """Train the linear probing model for one step"""
        with torch.no_grad():
            # Use the pre-trained model to generate the embeddings
            self.model.eval()
            imgs, covariates = batch
            features = self.model.backbone.get_intermediate_layers(imgs, covariates,
                                                                   self.n_last_blocks)
            features = torch.cat([x[:, 0] for x in features], dim=-1)

        labels = covariates[self.target]

        if self.cfg.covariate.name != None:
            covariate_embed = self.covariate_embed(covariates[self.cfg.covariate.name])
            features = torch.cat([features, covariate_embed], dim=-1)

        pred = self.classifier(features)

        loss = self.compute_loss(pred, labels)
        self.log(f"train_loss", loss, on_step=True, prog_bar=True, rank_zero_only=True)

        return loss

    def inference(self, batch, batch_idx):
        """Make predictions from the trained linear classifier"""
        imgs, covariates = batch
        features = self.model.backbone.get_intermediate_layers(imgs, covariates,
                                                               self.n_last_blocks)
        features = torch.cat([x[:, 0] for x in features], dim=-1)
        labels = covariates[self.target]

        if self.cfg.covariate.name != None:
            covariate_embed = self.covariate_embed(covariates[self.cfg.covariate.name])
            features = torch.cat([features, covariate_embed], dim=-1)

        pred = self.classifier(features)

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

            # compute f1
            pred_true = torch.stack(
                [torch.argmax(all_preds, dim=-1), all_trues], dim=1
            )
            pred_true_list = [torch.zeros_like(pred_true) for _ in
                              range(dist.get_world_size())]
            dist.all_gather(pred_true_list, pred_true)
            pred_true_list = torch.cat(pred_true_list, dim=0)
            f1 = f1_score(
                y_true=pred_true_list[:,1].cpu().numpy(),
                y_pred=pred_true_list[:,0].cpu().numpy(),
            )
            self.log(f"val_{loader_idx}_f1", f1, on_epoch=True, prog_bar=True,
                     rank_zero_only=True)

            # random guess baseline
            random_acc = np.mean(pred_true_list[:,1].cpu().numpy())
            random_acc = max(1 - random_acc, random_acc)
            self.log(f"val_{loader_idx}_random", random_acc, on_epoch=True,
                     prog_bar=False, rank_zero_only=True)
            self.log(f"val_{loader_idx}_data_size", len(pred_true_list[:,1]),
                     on_epoch=True, prog_bar=False, rank_zero_only=True)


        # do something with all preds
        self.validation_step_outputs.clear()  # free memory

    def predict_step(self, batch, batch_idx):
        return self.inference(batch, batch_idx)

    def configure_optimizers(self):
        if self.cfg.covariate.name != None:
            params_groups = utils.get_params_groups([self.classifier,
                                                     self.covariate_embed])
        else:
            params_groups = utils.get_params_groups(self.classifier)

        optimizer = getattr(torch.optim, self.cfg.optimizer.name)(
            params_groups, **self.cfg.optimizer.args
        )

        scheduler = getattr(torch.optim.lr_scheduler, self.cfg.scheduler.name)(
            optimizer, **self.cfg.scheduler.args
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
