import os
import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F


def GAN_trainer(args, generator, discriminator, train_loader, fixed_z, device):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    fixed_z = fixed_z.to(device)
    adversarial_loss = torch.nn.BCELoss().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=(args.lr * 8 / 9), betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    total_params = sum(p.numel() for p in generator.parameters())+sum(p.numel() for p in discriminator.parameters())
    print(f"Total number of parameters: {total_params}")

    for epoch in range(args.epochs):
        for i, (imgs) in enumerate(train_loader):
            # adversarial ground truths
            valid = torch.ones(imgs.shape[0], 1).to(device)
            fake = torch.zeros(imgs.shape[0], 1).to(device)

            real_imgs = imgs.to(device)

            #############    Train Generator    ################
            optimizer_G.zero_grad()

            # sample noise as generator input
            z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))).to(torch.float).to(device)

            # Generate a batch of images
            gen_imgs = generator(z)

            # G-Loss
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            #############  Train Discriminator ################
            optimizer_D.zero_grad()

            # D-Loss
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G Loss: %f]"
                % (epoch, args.epochs, i, len(train_loader), d_loss.item(), g_loss.item())
            )


            batches_done = epoch * len(train_loader) + i
            os.makedirs(f"{args.model}_images", exist_ok=True)
            if batches_done % args.sample_interval == 0:
                save_image(generator(fixed_z), f"{args.model}_images/{batches_done}.png", nrow=fixed_z.shape[0] // 5, normalize=True)

    torch.save(generator.state_dict(), args.save_model_path)


def NNHVM_trainer(args, generator, train_loader, fixed_z, device):
    generator = generator.to(device)
    fixed_z = fixed_z.to(device)
    mse_loss = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Total number of parameters: {total_params}")

    for epoch in range(args.epochs):
        for i, (imgs) in enumerate(train_loader):
            imgs = imgs.to(device)
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0]*args.k, args.latent_dim))).to(torch.float).to(device)
            gen_imgs = generator(z).view(imgs.shape[0], args.k, imgs.shape[1], imgs.shape[2], imgs.shape[3])
            loss = mse_loss(gen_imgs, imgs.unsqueeze(1).expand_as(gen_imgs))
            loss.backward()
            optimizer.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                % (epoch, args.epochs, i, len(train_loader), loss.item())
            )

            batches_done = epoch * len(train_loader) + i
            os.makedirs(f"{args.model}_images_bz{args.batch_size}_ldim{args.latent_dim}_k{args.k}", exist_ok=True)
            if batches_done % args.sample_interval == 0:
                save_image(generator(fixed_z), f"{args.model}_images_bz{args.batch_size}_ldim{args.latent_dim}_k{args.k}/{batches_done}.png", nrow=fixed_z.shape[0] // 5,
                           normalize=True)

    torch.save(generator.state_dict(), args.save_model_path)

class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()

    def forward(self, mean, variance, gen_imgs, imgs):
        mse_loss = F.mse_loss(gen_imgs, imgs.expand_as(gen_imgs))
        kl_loss = torch.sum(mean.pow(2)+variance-torch.log(variance)-1)
        return mse_loss, kl_loss


def VAE_trainer(args, encoder, decoder, train_loader, fixed_z, device):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    fixed_z = fixed_z.to(device)
    vae_loss = VAE_loss().to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    total_params = sum(p.numel() for p in decoder.parameters())+sum(p.numel() for p in encoder.parameters())
    print(f"Total number of parameters: {total_params}")

    for epoch in range(args.epochs):
        for i, (imgs) in enumerate(train_loader):
            imgs = imgs.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            mean, variance = encoder(imgs)
            z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], args.k, args.latent_dim))).to(torch.float).to(device)
            z = torch.sqrt(variance).unsqueeze(1).expand_as(z)*z+mean.unsqueeze(1).expand_as(z)
            gen_imgs = decoder(z.view(-1, args.latent_dim)).view(-1, args.k, imgs.shape[1], imgs.shape[2], imgs.shape[3])
            mse_loss, kl_loss = vae_loss(mean, variance, gen_imgs, imgs.unsqueeze(1))
            loss = 0.5*(mse_loss+kl_loss)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [mse loss: %f] [kl loss: %f]"
                % (epoch, args.epochs, i, len(train_loader), mse_loss.item(), kl_loss.item())
            )

            batches_done = epoch * len(train_loader) + i
            os.makedirs(f"{args.model}_images_bz{args.batch_size}_ldim{args.latent_dim}_k{args.k}", exist_ok=True)
            if batches_done % args.sample_interval == 0:
                save_image(decoder(fixed_z), f"{args.model}_images_bz{args.batch_size}_ldim{args.latent_dim}_k{args.k}/{batches_done}_sample.png", nrow=fixed_z.shape[0] // 5,
                           normalize=True)
                save_image(gen_imgs[:25], f"{args.model}_images_bz{args.batch_size}_ldim{args.latent_dim}_k{args.k}/{batches_done}_recons.png", nrow=5,
                           normalize=True)
                save_image(imgs[:25], f"{args.model}_images_bz{args.batch_size}_ldim{args.latent_dim}_k{args.k}/{batches_done}_true.png", nrow=5,
                           normalize=True)
            

    torch.save(decoder.state_dict(), args.save_model_path)

def Diffusion_trainer(args, model, diffusion, train_loader, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(args.epochs):
        for i, (imgs) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            t = torch.randint(low=0, high=diffusion.timesteps, size=(imgs.shape[0], )).to(device)
            loss = diffusion.train_losses(model, imgs, t)
            loss.backward()
            optimizer.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                % (epoch, args.epochs, i, len(train_loader), loss.item())
            )

            batches_done = epoch * len(train_loader) + i
            os.makedirs(f"{args.model}_images", exist_ok=True)
            if batches_done % args.sample_interval == 0:
                save_image(torch.tensor(diffusion.sample(model, args.img_size, batch_size=25)[-1]), f"{args.model}_images/{batches_done}.png", nrow=5,
                           normalize=True)

    torch.save(model.state_dict(), args.save_model_path)