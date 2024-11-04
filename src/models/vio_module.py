import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
import hydra

class VIOLitModule(LightningModule):
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
        """ Initialize a `VIOLitModule`.

            :param net: the model to train.
            :param optimizer: the optimizer to use for training.
            :param scheduler: the learning rate scheduler to use for training.
            :param citerion: the loss function to use for training
        """
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
        loss = self.criterion(out, target)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        out = self.forward(x, target)
        loss = self.criterion(out, target)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss



    def test_step(self, batch, batch_idx):
        # This method is not used for our custom testing
        pass

    def on_test_epoch_end(self):
        results = self.tester.test(self.net)
        metrics = self.metrics_calculator.calculate_metrics(results)
        
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        
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

        
if __name__ == "__main__":
    _ = VIOLitModule(None, None, None, None, None)
