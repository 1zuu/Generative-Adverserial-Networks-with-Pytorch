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
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()
    plt.savefig(f"results/{cur_step}.png")

def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
            )

def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
         nn.Linear(input_dim, output_dim),
         nn.LeakyReLU(0.2, inplace=True)
                        )       

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
                        get_generator_block(z_dim, hidden_dim),
                        get_generator_block(hidden_dim, hidden_dim * 2),
                        get_generator_block(hidden_dim * 2, hidden_dim * 4),
                        get_generator_block(hidden_dim * 4, hidden_dim * 8),
                        nn.Linear(hidden_dim * 8, im_dim),
                        nn.Sigmoid()
                                 )
    def forward(self, noise):
        return self.gen(noise)

class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
                    get_discriminator_block(im_dim, hidden_dim * 4),
                    get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
                    get_discriminator_block(hidden_dim * 2, hidden_dim),
                    nn.Linear(hidden_dim, 1)
                            )

    def forward(self, image):
        return self.disc(image)

class BasicGAN(object):
    def __init__(self):
        self.z_dim = z_dim
        self.im_dim = image_dim ** 2
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()

        self.dataloader = DataLoader(
                                MNIST(
                                    '.', 
                                    download=True if not os.path.exists('MNIST') else False,
                                    transform=transforms.ToTensor()
                                    ),
                                batch_size=batch_size,
                                shuffle=True
                                )

        self.generator = Generator(self.z_dim, self.im_dim, hidden_dim).to(self.device)
        self.discriminator = Discriminator(self.im_dim, hidden_dim).to(self.device)
        self.gen_opt  = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)


    def get_noise(self, n_samples):
        return torch.randn(n_samples,self.z_dim,device=self.device)

    def get_disc_loss(self, real, num_images):
        fake_noise = self.get_noise(num_images)
        fake = self.generator(fake_noise) 

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
                real = real.view(cur_batch_size, -1).to(self.device)

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


gan = BasicGAN()
gan.train()