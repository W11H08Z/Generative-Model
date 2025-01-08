import numpy as np

import torch
from torchvision import transforms

from model.GAN import Generator, Discriminator, Generator_CNN, Discriminator_CNN
from model.VAE import Encoder, Decoder, Encoder_CNN, Decoder_CNN
from model.NNHVM import NNHVM, NNHVM_CNN
from model.Diffusion import Unet, GaussianDiffusion
from trainer import GAN_trainer, VAE_trainer, NNHVM_trainer, Diffusion_trainer
from data import TrophyDataset
from config import args_parse

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(args):
    img_shape = (args.channels, args.img_size, args.img_size)
    fixed_z = torch.tensor(np.random.normal(0, 1, (25, args.latent_dim))).to(torch.float)

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    data = TrophyDataset(args.dataset_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True)

    if args.model == 'GAN':
        generator = Generator(args.latent_dim, img_shape)
        discriminator = Discriminator(img_shape)
        GAN_trainer(args, generator, discriminator, train_loader, fixed_z, device)
    elif args.model == 'VAE':
        encoder = Encoder(args.latent_dim, img_shape)
        decoder = Decoder(args.latent_dim, img_shape)
        VAE_trainer(args, encoder, decoder, train_loader, fixed_z, device)
    elif args.model == 'NNHVM':
        generator = NNHVM(args.latent_dim, img_shape)
        NNHVM_trainer(args, generator, train_loader, fixed_z, device)
    elif args.model == 'GAN_CNN':
        generator = Generator_CNN(args.latent_dim, img_shape)
        discriminator = Discriminator_CNN(img_shape)
        GAN_trainer(args, generator, discriminator, train_loader, fixed_z, device)
    elif args.model == 'VAE_CNN':
        encoder = Encoder_CNN(args.latent_dim, img_shape)
        decoder = Decoder_CNN(args.latent_dim, img_shape)
        VAE_trainer(args, encoder, decoder, train_loader, fixed_z, device)
    elif args.model == 'NNHVM_CNN':
        generator = NNHVM_CNN(args.latent_dim, img_shape)
        NNHVM_trainer(args, generator, train_loader, fixed_z, device)
    elif args.model == 'Diffusion_Model':
        model = Unet(args.img_size)
        diffusion = GaussianDiffusion()
        Diffusion_trainer(args, model, diffusion, train_loader, device)



if __name__ == '__main__':
    args = args_parse()
    train(args)
