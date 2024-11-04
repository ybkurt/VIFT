import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
import hydra

class WeightedVIOLitModule(LightningModule):
    def __init__(
            self,
            net,
            optimizer,
            scheduler,
            criterion,
            compile,
            tester,
            metrics_calculator,
        ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.criterion = criterion
        self.tester = tester
        self.metrics_calculator = metrics_calculator


    def forward(self, x, target):
        return self.net(x, target)

    def training_step(self, batch, batch_idx):
        x, target = batch
        out = self.forward(x, target)
        weight = x[-1]
        loss = self.criterion(out, target, weight)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # Log the current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        out = self.forward(x, target)
        weight = x[-1]
        loss = self.criterion(out, target, weight, use_weighted_loss=False)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # This method is not used for our custom testing
        pass

    def on_test_epoch_end(self):
        with torch.inference_mode():
            results = self.tester.test(self.net)
        metrics = self.metrics_calculator.calculate_metrics(results)
        metric_sum = 0
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
            metric_sum += value
        self.log("hp_metric", metric_sum)
        
        save_dir = self.trainer.logger.log_dir
        self.tester.save_results(results, save_dir)

    def setup(self, stage):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
