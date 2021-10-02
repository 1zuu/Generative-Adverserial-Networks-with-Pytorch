import os, torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from variables import *
torch.manual_seed(seed)

if torch.cuda.is_available():
    print("Using CUDA")
else:
    print("CUDA not available")

def show_tensor_images(image_tensor, cur_step, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(f"results/{cur_step}.png")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.image_chan = image_chan
        self.hidden_dim = hidden_dim
        self.generator = nn.Sequential(
                        self.make_gen_block(z_dim, hidden_dim * 4),
                        self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
                        self.make_gen_block(hidden_dim * 2, hidden_dim),
                        self.make_gen_block(hidden_dim, self.image_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.image_chan = image_chan
        self.hidden_dim = 16
        self.discriminator = nn.Sequential(
            self.make_disc_block(self.image_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    def forward(self, image):
        disc_pred = self.discriminator(image)
        return disc_pred.view(len(disc_pred), -1)

class DCGAN(object):
    def __init__(self):
        self.z_dim = z_dim
        self.im_dim = image_dim ** 2
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()

        transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                        ])

        self.dataloader = DataLoader(
                                MNIST(
                                    '.', 
                                    download=True if not os.path.exists('MNIST') else False,
                                    transform=transform
                                    ),
                                batch_size=batch_size,
                                shuffle=True
                                )

        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.gen_opt  = torch.optim.Adam(
                                self.generator.parameters(), 
                                lr=learning_rate, 
                                betas=(beta_1, beta_2)
                                        )
        self.disc_opt = torch.optim.Adam(
                                self.discriminator.parameters(), 
                                lr=learning_rate, 
                                betas=(beta_1, beta_2)
                                        )


    def get_noise(self, n_samples):
        return torch.randn(n_samples,self.z_dim,device=self.device)

    def get_disc_loss(self, real, num_images):
        fake_noise = self.get_noise(num_images)
        fake = self.generator(fake_noise)  # (batch_size, channels ,width, height)
        disc_fake_pred = self.discriminator(fake.detach()) # detach from the generator to avoid backpropagation
        disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

        disc_real_pred = self.discriminator(real)
        disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))

        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss

    def get_gen_loss(self, num_images):
        fake_noise = self.get_noise(num_images)
        fake = self.generator(fake_noise) 
        disc_fake_pred = self.discriminator(fake)
        gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss

    def train(self):
        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        gen_loss = False
        for _ in range(n_epochs):
            for real, _ in tqdm(self.dataloader):
                cur_batch_size = len(real)

                # Flatten the batch of real images from the dataset
                real = real.to(self.device)

                ### Update discriminator ###
                self.disc_opt.zero_grad()
                disc_loss = self.get_disc_loss(real, cur_batch_size)
                disc_loss.backward(retain_graph=True)
                self.disc_opt.step()

                ### Update generator ###
                self.gen_opt.zero_grad()
                gen_loss = self.get_gen_loss(cur_batch_size)
                gen_loss.backward()
                self.gen_opt.step()

                # Keep track of the average generator & discriminator loss
                mean_discriminator_loss += disc_loss.item() / display_step
                mean_generator_loss += gen_loss.item() / display_step

                ### Visualization ###
                if (cur_step % display_step == 0) and (cur_step > 0):
                    print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                    fake_noise = self.get_noise(cur_batch_size)
                    fake = self.generator(fake_noise)
                    show_tensor_images(fake, cur_step)
                    # show_tensor_images(real, cur_step)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1


gan = DCGAN()
gan.train()