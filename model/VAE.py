import numpy as np

import torch
import torch.nn as nn

class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)

# 560792
class Encoder(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Encoder, self).__init__()

        self.mean_model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim),
        )

        self.variance_model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim),
            ExpActivation()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        mean = self.mean_model(img_flat)
        variance = self.variance_model(img_flat)
        return mean, variance

class Decoder(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Decoder, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.img_shape[0], self.img_shape[1], self.img_shape[2])
        return img
    
# 1412640
class Encoder_CNN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Encoder_CNN, self).__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        ds_size = img_shape[1] // 2 ** 4
        self.mean_model = nn.Sequential(
            *encoder_block(img_shape[0], 16, bn=False),
            *encoder_block(16, 32),
            *encoder_block(32, 64),
            *encoder_block(64, 128),
        )
        self.mean_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, latent_dim))

        self.variance_model = nn.Sequential(
            *encoder_block(img_shape[0], 16, bn=False),
            *encoder_block(16, 32),
            *encoder_block(32, 64),
            *encoder_block(64, 128),
        )
        self.variance_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, latent_dim), ExpActivation())

    def forward(self, img):
        bz = img.shape[0]
        mean = self.mean_layer(self.mean_model(img).view(bz, -1))
        variance = self.variance_layer(self.variance_model(img).view(bz, -1))
        return mean, variance

class Decoder_CNN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Decoder_CNN, self).__init__()

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))  # 100 ——> 128 * 8 * 8 = 8192

        self.decoder_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.decoder_blocks(out)
        return img
    