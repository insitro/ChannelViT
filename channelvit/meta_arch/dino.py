# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Note:
# We adapted the DINO code from https://github.com/facebookresearch/dino

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict

import channelvit.backbone as backbone
import channelvit.utils as utils


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head, channel_drop: float = 0.0):
        super(MultiCropWrapper, self).__init__()

        # disable layers dedicated to ImageNet labels classification
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()

        if hasattr(backbone, "head"):
            backbone.head = nn.Identity()

        self.backbone = backbone
        self.head = head
        if channel_drop > 0:
            self.channel_drop = nn.Dropout2d(channel_drop)
        else:
            self.channel_drop = nn.Identity()

    def _repeat_extra_tokens(self, extra_tokens, start_idx, end_idx):
        """Repeat the covariates for the same view"""
        cur_covariates = {}
        for k, v in extra_tokens.items():
            if v is None:
                cur_covariates[k] = None
            elif v.dim() == 1:
                cur_covariates[k] = v.repeat(end_idx - start_idx)
            else:
                repeats = torch.ones(v.dim()).long()
                repeats[0] = end_idx - start_idx
                cur_covariates[k] = v.repeat(*repeats)

        return cur_covariates

    def forward(self, x, extra_tokens):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]), return_counts=True
            )[1],
            0,
        )

        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            # We group inputs with the same sptial dimensions together from multiple
            # crops
            # For the covariate values, we need to repeat them the the right dimension
            # as well.
            cur_extra_tokens = self._repeat_extra_tokens(
                extra_tokens, start_idx, end_idx
            )
            _out = self.backbone(
                self.channel_drop(torch.cat(x[start_idx:end_idx])), cur_extra_tokens
            )
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class DINO(pl.LightningModule):
    def __init__(self, main_dino_cfg: DictConfig) -> None:
        super().__init__()
        print("Initializing meta arch DINO")
        self.save_hyperparameters()
        self.cfg = self.set_cfg(main_dino_cfg)

        print("Building student and teacher network.")
        # initialize the student and teacher backbone
        student = getattr(backbone, self.cfg.backbone.name)(**self.cfg.backbone.args)
        # we will set drop path rate to 0 for the teacher network
        teacher = getattr(backbone, self.cfg.backbone.name)(
            drop_path_rate=0.0,
            **{
                k: v
                for k, v in self.cfg.backbone.args.items()
                if k not in ["drop_path_rate"]
            },
        )

        # combine the backbone with the dino head
        self.embed_dim = student.out_dim
        self.student = MultiCropWrapper(
            student,
            DINOHead(
                self.embed_dim,
                self.cfg.out_dim,
                use_bn=self.cfg.use_bn_in_head,
                norm_last_layer=self.cfg.norm_last_layer,
            ),
            channel_drop=self.cfg.student_channel_drop,
        )
        self.teacher = MultiCropWrapper(
            teacher,
            DINOHead(self.embed_dim, self.cfg.out_dim, use_bn=self.cfg.use_bn_in_head),
        )

        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ============ preparing loss ... ============
        self.dino_loss = DINOLoss(
            self.cfg.out_dim,
            # total number of crops = 2 global crops + local_crops_number
            self.cfg.local_crops_number + 2,
            self.cfg.warmup_teacher_temp,
            self.cfg.teacher_temp,
            self.cfg.warmup_teacher_temp_epochs,
            self.cfg.max_epochs,
        )

        self.configure_scheduler()

    def set_cfg(self, main_dino_cfg: DictConfig) -> DictConfig:
        cfg = main_dino_cfg.meta_arch

        # set the optimization configurations
        with open_dict(cfg):
            cfg.total_batch_size = (
                main_dino_cfg.trainer.devices * main_dino_cfg.data.loader.batch_size
            )

            cfg.max_epochs = main_dino_cfg.trainer.max_epochs
            cfg.num_batches = main_dino_cfg.data.loader.num_batches
            cfg.local_crops_number = (
                main_dino_cfg.transformations.args.local_crops_number
            )
            cfg.backbone.patch_size = cfg.patch_size

        print(cfg)
        return cfg

    def training_step(self, batch, batch_idx):
        #######################################
        # preparing inputs and learning rates #
        #######################################
        # unpack the batch
        imgs, covariates = batch

        # We use manual optimization here
        # get the optimizer and set the lr / wd rate
        optimizer = self.optimizers(use_pl_optimizer=True)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[self.global_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[self.global_step]

        ####################################
        # teacher and student forward pass #
        ####################################
        teacher_output = self.teacher(
            imgs[:2], covariates
        )  # only the 2 global views pass through the teacher
        student_output = self.student(imgs, covariates)
        loss = self.dino_loss(student_output, teacher_output, self.current_epoch)

        self.log("learning_rate", self.lr_schedule[self.global_step])
        self.log("weight_decay", self.wd_schedule[self.global_step])
        self.log("train_loss", loss, on_step=True)

        return loss

    def on_after_backward(self) -> None:
        """Apply gradient clipping and freeze last layer"""
        if self.cfg.clip_grad:
            utils.clip_gradients(self.student, self.cfg.clip_grad)

        cancel_gradients_last_layer(
            self.current_epoch, self.student, self.cfg.freeze_last_layer
        )

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """EMA update for the teacher"""
        with torch.no_grad():
            m = self.momentum_schedule[self.global_step]
            for param_q, param_k in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        n_last_blocks: int = 1,
    ) -> Any:
        img, covariates = batch
        if isinstance(img, list):
            raise TypeError("We do not use multi-view crops for inference.")

        if self.use_teacher_for_pred:
            model = self.teacher.backbone
        else:
            model = self.student.backbone

        if n_last_blocks == 1:
            covariates["features"] = model(img, covariates)
        else:
            intermediate_output = model.get_intermediate_layers(
                img, covariates, n_last_blocks
            )
            covariates["features"] = torch.cat(
                [x[:, 0] for x in intermediate_output], dim=-1
            )

        return covariates

    def configure_optimizers(self):
        """Loading optimizer and learning rate / weight decay schedulers"""

        params_groups = utils.get_params_groups(self.student)

        print("Creating optimizer.")
        if self.cfg.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif self.cfg.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                params_groups, lr=0, momentum=0.9
            )  # lr is set by scheduler
        elif self.cfg_optimizer == "lars":
            optimizer = utils.optim.LARS(params_groups)
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

        self.momentum_schedule = utils.cosine_scheduler(
            self.cfg.momentum_teacher, 1, self.cfg.max_epochs, self.cfg.num_batches
        )

    def configure_prediction(self, use_teacher_for_pred=True) -> None:
        """Set what features to use for prediction. This method is called only in
        predict.py after we have loaded the saved checkpoint. It doesn't impact
        training. Note that main_dino_cfg is the root configuration file.
        """
        self.use_teacher_for_pred = use_teacher_for_pred
        print(f"Use teacher for prediction: {self.use_teacher_for_pred}.")
