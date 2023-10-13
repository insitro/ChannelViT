# Evaluate the DINO embedding through linear probing
import pytorch_lightning as pl
import torch
import torch.distributed as dist


class CorrelationComputer(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.validation_step_outputs = []
        self.best_validation_results = {}

    def inference(self, batch, batch_idx):
        """Make predictions from the trained linear classifier"""
        # unpack the batch
        imgs, covariates = batch

        # imgs has dim: BatchSize * Channel * H * W
        imgs = imgs.permute(1, 0, 2, 3).flatten(1)  # channel, rest
        channel_sum = imgs.sum(dim=0)
        nonzero_idx = channel_sum.nonzero().squeeze()
        imgs = imgs[:, nonzero_idx]

        return {"correlation": torch.corrcoef(imgs)}

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
                val_outputs_per_loader[idx] = {"correlation": []}

            val_outputs_per_loader[idx]["correlation"].append(outputs["correlation"])

        for loader_idx, outputs_dict in val_outputs_per_loader.items():
            correlations = torch.stack(outputs_dict["correlation"], dim=0).mean(dim=0)
            correlations_list = [
                torch.zeros_like(correlations) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(correlations_list, correlations)
            correlations_list = torch.stack(correlations_list, dim=0)
            correlations = torch.mean(correlations_list, dim=0)

            print(correlations)

        # do something with all preds
        self.validation_step_outputs.clear()  # free memory

    def predict_step(self, batch, batch_idx):
        return self.inference(batch, batch_idx)
