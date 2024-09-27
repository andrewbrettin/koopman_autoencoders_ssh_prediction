"""
cnn.py

"""

__all__ = [
    "ConvBlock",
    "ConvEncoder",
    "ConvDecoder",
    "CNNAutoencoder",
    "CNNKoopmanAutoencoder"
]

import os
import itertools
import yaml

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.base import BaseModule
from src.train import losses

# Modified ConvBlock class
# Modified ConvBlock class
class ConvBlock(nn.Module):
    def __init__(
        self, 
        C_in: int,
        C_out: int,
        block_size: int = 1,
        kernel_size: int = 3,
        decoder_block: bool=False,
        **conv_kwargs
    ):
        """
        Creates a convolutional block for CNNs. A convolutional block consists
        of convolutional blocks plus ReLU activations, repeated block_size 
        times.

        Parameters:
            C_in: int
                Number of input channels.
            C_out: int
                Number of channels for each hidden layer.
            block_size: int
                Number of convolutional layers with ReLU activations.
            kernel_size: int
                Filter size.
            decoder_block: bool
                If True, then all conv layers except for the first have channels (C_out, C_out). 
                If False, then all conv layers except for the last have channels (C_in, C_in).
            conv_kwargs: dict
                Various kwargs to be passed to nn.Conv2d.
        """
        super().__init__()
        if not decoder_block:
            first_layer = [nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs), nn.ReLU()]
            subsequent_layers = (block_size-1) * [
                nn.Conv2d(C_out, C_out, kernel_size, **conv_kwargs),
                nn.ReLU()
            ]
            self.stack = nn.ModuleList([*first_layer, *subsequent_layers])
        else:
            initial_layers = (block_size-1) * [
                nn.Conv2d(C_in, C_in, kernel_size, **conv_kwargs),
                nn.ReLU()
            ]
            output_layer = [nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs), nn.ReLU()]
            self.stack = nn.ModuleList([*initial_layers, *output_layer])

    def forward(self, x):
        for module in self.stack:
            x = module(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        D: int,
        hiddens: list,
        block_size: int = 1,
        kernel_size: int = 3,
        **conv_kwargs
    ):
        """
        Convolutional encoder.

        Parameters:
            C: int
                Number of channels in input tensor.
            H: int
                Input field height.
            W: int
                Input field width.
            D: int
                Dimensionality of latent space.
            hiddens: list of int
                List giving the number of output channels in each convolutional
                block in the encoder.
            block_size: int
                Number of convolutional layers in each convolutional block.
            kernel_size: int
                Filter size.
            **conv_kwargs: 
                Kwargs passed to nn.Conv2d, for instance, padding='same'
                
        Convolutional blocks are used to designate convolutional layers with 
        ReLU activations which are not followed by a max pooling layer. For instance,
        if hiddens=[8, 16] and block_size=2, then we will have two convolutional
        layers with 8 output channels prior to any pooling, followed by two 
        convolutional layers with 16 output channels on the pooled fields.
        """
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.D = D
        self.hiddens = hiddens
        
        # Compute number of poolings from list of hiddens
        self.n_pools = len(self.hiddens)
        if self.H % (2**self.n_pools) != 0 or self.W % (2**self.n_pools) != 0:
            raise ValueError(
                "Input height and width is not divisible by 2^p, where p is "
                f"the number of pooling layers (H={self.H}, W={self.W}, "
                f"p={self.n_pools}). This is required so that"
            )
        self.H_out = self.H // (2**self.n_pools)
        self.W_out = self.W // (2**self.n_pools)
        
        # Encoder
        encoder_layers = nn.ModuleList()
        encoder_layers.append(ConvBlock(C, self.hiddens[0], block_size, kernel_size, **conv_kwargs))
        encoder_layers.append(nn.MaxPool2d(kernel_size=2))
        for C_n, C_np1 in itertools.pairwise(self.hiddens):
            encoder_layers.append(ConvBlock(C_n, C_np1, block_size, kernel_size, **conv_kwargs))
            encoder_layers.append(nn.MaxPool2d(kernel_size=2))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Linear layer to latent space
        self.linear = nn.Linear(self.hiddens[-1] * self.H_out * self.W_out, self.D)

    def forward(self, x):
        out = self.encoder(x)
        # Flatten using view
        out = out.view(out.size(0), self.hiddens[-1] * self.H_out * self.W_out)
        # Apply linear layer
        out = self.linear(out)
        return out


class ConvDecoder(nn.Module):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        D: int,
        hiddens,
        block_size: int = 1,
        kernel_size: int = 3,
        **conv_kwargs
    ):
        """
        Convolutional decoder.

        See ConvEncoder class for information about init parameters.
        """
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.D = D
        self.hiddens = hiddens
        # Compute number of poolings from list of hiddens
        self.n_pools = len(self.hiddens)
        if self.H % (2**self.n_pools) != 0 or self.W % (2**self.n_pools) != 0:
            raise ValueError(
                "Input height and width is not divisible by 2^p, where p is "
                f"the number of pooling layers (H={self.H}, W={self.W}, "
                f"p={self.n_pools}). This is required so that"
            )
        self.H_out = self.H // (2**self.n_pools)
        self.W_out = self.W // (2**self.n_pools)

        # Linear layer from latents to first convolutional layer
        self.linear = nn.Linear(D, self.hiddens[-1] * self.H_out * self.W_out)

        # Decoder
        decoder_layers = nn.ModuleList()
        for C_np1, C_n in itertools.pairwise(self.hiddens[::-1]):
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            decoder_layers.append(
                ConvBlock(C_np1, C_n, block_size=block_size, kernel_size=kernel_size, decoder_block=True, **conv_kwargs)
            )
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        decoder_layers.append(ConvBlock(self.hiddens[0], C, block_size, kernel_size, decoder_block=True, **conv_kwargs))
        
        self.decoder = nn.Sequential(*decoder_layers)

        # Output layer to undo ReLU
        self.output_layer = nn.Conv2d(C, C, kernel_size=(1,1), padding='same')

    def forward(self, z):
        out = self.linear(z)
        # Unflatten
        out = out.view(out.size(0), self.hiddens[-1], self.H_out, self.W_out)
        # Decoder
        out = self.decoder(out)
        # Flatten and apply linear layer
        out = self.output_layer(out)
        return out


class CNNAutoencoder(pl.LightningModule, BaseModule):
    def __init__(self, configs):
        super(CNNAutoencoder, self).__init__()

        self.C = configs['C']
        self.H = configs['H']
        self.W = configs['W']
        self.D = configs['D']

        self.conv_encoder = ConvEncoder(
            self.C,
            self.H,
            self.W,
            self.D,
            configs['hiddens'],
            block_size=configs['block_size'],
            kernel_size=configs['kernel_size'],
            **configs['conv_kwargs']
        )

        self.conv_decoder = ConvDecoder(
            self.C,
            self.H,
            self.W,
            self.D,
            configs['hiddens'],
            block_size=configs['block_size'],
            kernel_size=configs['kernel_size'],
            **configs['conv_kwargs']
        )

        # Weights
        # https://lightning.ai/docs/pytorch/stable/integrations/ipu/prepare.html \
        # #init-tensors-using-tensor-to-and-register-buffer
        self.register_buffer('weights', configs['weights'])
        self.weighted_mse = losses.WeightedMSE2dLoss()
        
        self.configs = configs

    def forward(self, x):
        out = self.conv_encoder(x)
        out = self.conv_decoder(out)
        return out

    def training_step(self, batch, batch_idx):
        # Reconstruction_loss
        X = batch[:, 0:self.C, :, :]
        X_reconstruction = self.conv_decoder(self.conv_encoder(X))
        loss = self.weighted_mse(X, X_reconstruction, self.weights)

        # Alias for MSE "loss" included for consistency
        self.log('train/mse', loss, on_step=False, on_epoch=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Reconstruction_loss
        X = batch[:, 0:self.C, :, :]
        X_reconstruction = self.conv_decoder(self.conv_encoder(X))
        loss = self.weighted_mse(X, X_reconstruction, self.weights)

        # Alias for MSE "loss" included for consistency
        self.log('val/mse', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # Set weight decay
        weight_decay = 0
        if 'l2' in self.configs:
            weight_decay = self.configs['l2']

        # Set other optimizer kwargs
        optimizer_kwargs = {}
        if 'optimizer_kwargs' in self.configs:
            optimizer_kwargs = self.configs['optimizer_kwargs']
        
        optimizer_func = getattr(torch.optim, self.configs['optimizer'])
        optimizer = optimizer_func(
            self.parameters(),
            lr=self.configs['lr'],
            weight_decay=weight_decay,
            **optimizer_kwargs
        )

        # If not using learning rate scheduler, return optimizer
        if 'scheduler' not in self.configs or self.configs['scheduler'] is None:
            return optimizer
        else:
            if 'scheduler_kwargs' not in self.configs:
                scheduler_kwargs = {}
            else:
                scheduler_kwargs = self.configs['scheduler_kwargs']

            scheduler_class = getattr(torch.optim.lr_scheduler, self.configs['scheduler'])
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class CNNKoopmanAutoencoder(pl.LightningModule, BaseModule):
    def __init__(self, configs):
        super(CNNKoopmanAutoencoder, self).__init__()

        # CNN dimensions
        self.C = configs['C']
        self.H = configs['H']
        self.W = configs['W']

        # Latent space dimensions
        self.D = configs['D']

        # Parameters and loss functions
        self.conv_encoder = ConvEncoder(
            self.C,
            self.H,
            self.W,
            self.D,
            configs['hiddens'],
            block_size=configs['block_size'],
            kernel_size=configs['kernel_size'],
            **configs['conv_kwargs']
        )

        self.linear_embedding = nn.Linear(
            configs['D'], configs['D'], bias=False
        )

        self.conv_decoder = ConvDecoder(
            self.C,
            self.H,
            self.W,
            self.D,
            configs['hiddens'],
            block_size=configs['block_size'],
            kernel_size=configs['kernel_size'],
            **configs['conv_kwargs']
        )

        # Weights
        # https://lightning.ai/docs/pytorch/stable/integrations/ipu/prepare.html \
        # #init-tensors-using-tensor-to-and-register-buffer
        self.register_buffer('weights', configs['weights'])
        self.weighted_mse = losses.WeightedMSE2dLoss()

        # Koopman stuff
        self.k = configs['k']
        self.alphas = configs['alphas']
        
        self.configs = configs

    def forward(self, x):
        out = self.conv_encoder(x)
        out = self.linear_embedding(out)
        out = self.conv_decoder(out)
        return out

    def autoencoder(self, x):
        out = self.conv_encoder(x)
        out = self.conv_decoder(out)
        return out

    def multistep_prediction(self, x, k):
        """
        Predicts state using k recurrent passes.
        """
        out = self.conv_encoder(x)
        # Apply recurrent passes through linear layer
        for i in range(k):
            out = self.linear_embedding(out)
        out = self.conv_decoder(out)
        return out
    
    def training_step(self, batch, batch_idx):
        X = batch

        # Reconstruction loss
        X_n = X[:, 0:self.C, :, :]
        X_reconstructed = self.conv_decoder(self.conv_encoder(X_n))
        reconstruction_loss = self.weighted_mse(X_n, X_reconstructed, self.weights)

        # Multipass prediction loss
        prediction_loss = 0.0
        for i in range(1, self.k+1):
            X_npk = X[:, (i*self.C):(i+1)*self.C, :, :]
            X_pred = self.multistep_prediction(X_n, i)
            prediction_loss += self.weighted_mse(X_npk, X_pred, self.weights)
        # Average over each prediction
        prediction_loss /= self.k

        # Linear loss
        X_np1 = X[:, self.C:2*self.C, :, :]
        linear_loss = F.mse_loss(
            self.conv_encoder(X_np1),
            self.linear_embedding(self.conv_encoder(X_n))
        )

        # Net loss
        net_loss = (
            self.alphas[0] * reconstruction_loss
            + self.alphas[1] * prediction_loss
            + self.alphas[2] * linear_loss
        )

        # Logs
        log_kw = dict(on_step=False, on_epoch=True)
        self.log('train/reconstruction_mse', reconstruction_loss, **log_kw, sync_dist=True)
        self.log('train/prediction_mse', prediction_loss, **log_kw, sync_dist=True)
        self.log('train/latent_mse', linear_loss, **log_kw, sync_dist=True)
        self.log('train/loss', net_loss, **log_kw, sync_dist=True)

        return net_loss

    def validation_step(self, batch, batch_idx):
        X = batch

        # Reconstruction loss
        X_n = X[:, 0:self.C, :, :]
        X_reconstructed = self.conv_decoder(self.conv_encoder(X_n))
        reconstruction_loss = self.weighted_mse(X_n, X_reconstructed, self.weights)

        # Multipass prediction loss
        prediction_loss = 0.0
        for i in range(1, self.k+1):
            X_npk = X[:, (i*self.C):(i+1)*self.C, :, :]
            X_pred = self.multistep_prediction(X_n, i)
            prediction_loss += self.weighted_mse(X_npk, X_pred, self.weights)
        # Average over each prediction
        prediction_loss /= self.k

        # Linear loss
        X_np1 = X[:, self.C:2*self.C, :, :]
        linear_loss = F.mse_loss(
            self.conv_encoder(X_np1),
            self.linear_embedding(self.conv_encoder(X_n))
        )

        # Net loss
        net_loss = (
            self.alphas[0] * reconstruction_loss
            + self.alphas[1] * prediction_loss
            + self.alphas[2] * linear_loss
        )

        # Logs
        log_kw = dict(on_step=False, on_epoch=True)
        self.log('val/reconstruction_mse', reconstruction_loss, **log_kw, sync_dist=True)
        self.log('val/prediction_mse', prediction_loss, **log_kw, sync_dist=True)
        self.log('val/latent_mse', linear_loss, **log_kw, sync_dist=True)
        self.log('val/loss', net_loss, **log_kw, sync_dist=True)

        return net_loss

    def configure_optimizers(self):
        # Set weight decay
        weight_decay = 0
        if 'l2' in self.configs:
            weight_decay = self.configs['l2']

        # Set other optimizer kwargs
        optimizer_kwargs = {}
        if 'optimizer_kwargs' in self.configs:
            optimizer_kwargs = self.configs['optimizer_kwargs']

        # Decine optimizer
        optimizer_func = getattr(torch.optim, self.configs['optimizer'])
        optimizer = optimizer_func(
            self.parameters(),
            lr=self.configs['lr'],
            weight_decay=weight_decay,
            **optimizer_kwargs
        )

        # Set up learning rate schedule, if applicable
        if 'scheduler' not in self.configs or self.configs['scheduler'] is None:
            return optimizer
        else:
            if 'scheduler_kwargs' not in self.configs:
                scheduler_kwargs = {}
            else:
                scheduler_kwargs = self.configs['scheduler_kwargs']

            scheduler_class = getattr(torch.optim.lr_scheduler, self.configs['scheduler'])
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
