from textwrap import dedent
import torch.nn as nn
import torch
import torchmetrics
import pytorch_lightning as pl
from dataModule import DataModule
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class Convolutional_AE(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, 3, padding=1)
        # self.conv5 = nn.Conv2d(1024, 2048, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)

        # Decoder

        # self.t_conv1 = nn.ConvTranspose2d(2048, 1024, 3, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(1024, 512, 3, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(512, 128, 3, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.t_conv4 = nn.ConvTranspose2d(64, 1, 3, padding=1)

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.BCELoss()
        # self.loss_fn= nn.KLDivLoss
        self.accuracy = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

    def forward(self, x):
        # Encoder forward pass
        x = F.relu(self.conv1(x))
        # x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))

        # Decoder forward pass
        # x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        # x = F.sigmoid(self.t_conv3(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        # x = F.relu(self.t_conv5(x))

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        # print(loss)

        self.log("train/loss", loss, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        # self.accuracy(logits.argmax(dim=-1), labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        self.accuracy(outputs.argmax(dim=-1), labels)
        self.log("test/accuracy", self.accuracy, on_epoch=True, prog_bar=True)

        return loss


if __name__ == "__main__":
    # Load model and datamodule for pytorch lightning
    model = Convolutional_AE()
    datamodule = DataModule()

    # Run model using pytorch lightning trainer
    trainer = Trainer(gpus=[0, 1, 2, 3], max_epochs=50, log_every_n_steps=85,
                      default_root_dir='/homes/erv01/DL4AM/CNN_AE_model_medium')  # gpus=[0,1,2,3],strategy='dp'
    trainer.fit(model, datamodule)

    # Save model
    torch.save(model.state_dict(), 'CNN_AE_medium2.pt')

# accelerator='gpu', devices=1,