import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from core.models.maskGIT.discriminator import Discriminator, weights_init
from core.models.maskGIT.vqgan import VQGAN
from core.models.maskGIT.lpips import LPIPS
from core.data import train_lightings_loader

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()

        self.train(args)

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs(args.ckpt_path, exist_ok=True)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(list(self.vqgan.encoder.parameters()) +
                                  list(self.vqgan.decoder.parameters()) +
                                  list(self.vqgan.codebook.parameters()) +
                                  list(self.vqgan.quant_conv.parameters()) +
                                  list(self.vqgan.post_quant_conv.parameters()),
                                  lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        return opt_vq, opt_disc

    def train(self, args):
        train_dataset = train_lightings_loader(args)
        steps_one_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (lightings, label) in zip(pbar, train_dataset):
                    imgs = lightings.reshape(-1, 3, args.image_size, args.image_size) 
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch * steps_one_epoch + i,
                                                          threshold=args.disc_start)
                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    nll_loss = args.perceptual_loss_factor * perceptual_loss + args.l2_loss_factor * rec_loss
                    nll_losss = nll_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(nll_losss, g_loss)
                    loss_vq = nll_losss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    loss_gan = disc_factor * .5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    loss_vq.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    loss_gan.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            both = torch.cat((imgs.add(1).mul(0.5)[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(both, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(VQ_Loss=np.round(loss_vq.cpu().detach().numpy().item(), 5),
                                     GAN_Loss=np.round(loss_gan.cpu().detach().numpy().item(), 3))
                    pbar.update(0)
                torch.save(self.vqgan.state_dict(), os.path.join(args.ckpt_path, f"vqgan_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image_size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num_codebook_vectors', type=int, default=256, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image_channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--batch_size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.0, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=0, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--data_path', default="/mnt/home_6T/public/jayliu0313/datasets/Eyecandies/", type=str)
    parser.add_argument('--ckpt_path', default="./checkpoints/vqgan")

    parser.add_argument('--CUDA', type=int, default=0, help="choose the device of CUDA")

    args = parser.parse_args()

    cuda_idx = str(args.CUDA)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("current device:", args.device)
   
    train_vqgan = TrainVQGAN(args)

