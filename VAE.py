from cmath import log
from grpc import xds_server_credentials
import torch.nn as nn
import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from dataModule import DataModule


class VAE(pl.LightningModule):
    def __init__(self, image_channels=1, h_dim=47488, z_dim=256, learning_rate=1e-4):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=(2, 4), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.batch2encode = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # Bottleneck layers #87360 works for no padding
        self.linear1 = nn.Linear(h_dim, z_dim)
        self.linear2 = nn.Linear(h_dim, z_dim)
        self.linear3 = nn.Linear(z_dim, h_dim)

        # Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 14, 106))

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, image_channels, kernel_size=(5, 4), stride=1, padding=1)

        # Define loss function (pytorch functions)
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()

        self.accuracy = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

    # Helper function for KLD loss
    def loss_fn(self, outputs, labels, mu, logvar):
        # BCE = F.binary_cross_entropy(outputs, labels, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        return KLD  # can also return + BCE but leads to CUDA error

    # Helper function for l1 loss
    # def loss_fn(self, outputs, labels, mu, logvar):
    #     return torch.nn.functional.l1_loss(outputs, labels)

    def forward(self, x):
        ### Encoder ###
        x = F.relu(self.conv1(x))
        # print('1st convolution:', x.shape)
        x = F.relu(self.conv2(x))
        # print('2nd convolution:', x.shape)
        x = F.relu(self.conv3(x))
        # print('3rd convolution:', x.shape)

        x = torch.flatten(x, start_dim=1)  ## shape here: batch_size x 47488
        # print('Flatten:', x.shape)

        ### Bottleneck ###
        z, mu, logvar = self.bottleneck(x)
        z = self.linear3(z)
        # print('Z-dimension shape:', z.shape)

        ### Decoder ###
        x = self.unflatten(z)
        # print('Unflatten:', x.shape)

        x = F.relu(self.t_conv1(x))
        # print('1st transposed convolution:', x.shape)
        x = F.relu(self.t_conv2(x))
        # print('2nd transposed convolution:', x.shape)
        x = F.relu(self.t_conv3(x))
        # print('3rd transposed convolution:', x.shape)

        return x, mu, logvar  ## return just x for other loss functions

    def bottleneck(self, x):
        mu = self.linear1(x)
        logvar = self.linear2(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # outputs = self(inputs)
        outputs, mu, logvar = self(inputs)
        loss = self.loss_fn(outputs, labels, mu, logvar)
        # loss = self.loss_fn(outputs, labels)
        # print(loss)

        self.log("train/loss", loss, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        # outputs = self(inputs)
        outputs, mu, logvar = self(inputs)
        loss = self.loss_fn(outputs, labels, mu, logvar)
        # loss = self.loss_fn(outputs, labels)

        # self.accuracy(logits.argmax(dim=-1), labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, mu, logvar = self(inputs)
        # outputs = self(inputs)
        loss = self.loss_fn(outputs, labels, mu, logvar)
        # loss = self.loss_fn(outputs, labels)

        self.accuracy(outputs.argmax(dim=-1), labels)
        self.log("test/accuracy", self.accuracy, on_epoch=True, prog_bar=True)

        return loss

# Runs through the model using pytorch lightning
if __name__ == "__main__":
    model = VAE()
    datamodule = DataModule()
    trainer = pl.Trainer(gpus=[0, 1, 2, 3], strategy='dp', max_epochs=100)
    trainer.fit(model, datamodule)
    torch.save(model.state_dict(), 'CNN_VAE2.pt')
