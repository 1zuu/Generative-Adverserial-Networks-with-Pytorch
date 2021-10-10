import os, torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from variables import *
from utils import *

set_seed()

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

def make_grad_hook():
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook

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

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.z_dim = z_dim
        self.image_chan = image_chan
        self.hidden_dim = 16
        self.critic = nn.Sequential(
            self.make_critic_block(self.image_chan, hidden_dim),
            self.make_critic_block(hidden_dim, hidden_dim * 2),
            self.make_critic_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_critic_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
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
        critic_pred = self.critic(image)
        return critic_pred.view(len(critic_pred), -1)

class WGAN(object):
    def __init__(self):
        self.z_dim = z_dim
        self.im_dim = image_dim ** 2
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.critic = Critic().to(self.device)

        self.generator = self.generator.apply(weights_initialization)
        self.critic = self.critic.apply(weights_initialization)

        self.gen_opt  = torch.optim.Adam(
                                self.generator.parameters(), 
                                lr=learning_rate, 
                                betas=(beta_1, beta_2)
                                        )
        self.critic_opt = torch.optim.Adam(
                                self.critic.parameters(), 
                                lr=learning_rate, 
                                betas=(beta_1, beta_2)
                                        )


    def get_noise(self, n_samples):
        return torch.randn(n_samples,self.z_dim,device=self.device)

    def get_gradient(self, real, fake, epsilon):
        mixed_images = real * epsilon + fake * (1 - epsilon)

        mixed_scores = self.critic(mixed_images)
        
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
                            inputs=mixed_images,
                            outputs=mixed_scores,
                            grad_outputs=torch.ones_like(mixed_scores), 
                            create_graph=True,
                            retain_graph=True,
                                    )[0]
        return gradient

    def get_gen_loss(self, crit_fake_pred):
        gen_loss = -torch.mean(crit_fake_pred)

        '''
            If Critic output for Generator fake images is too high, then Generator is learning properly. In order to
            minimize that loss we add negative sign to the loss. This will make the Generator to learn to generate 
            fake images which gives the Critic a higher positive prediction.

        '''
        return gen_loss

    def get_critic_loss(self, crit_fake_pred, crit_real_pred, gp, c_lambda):
        critic_fake_loss = -torch.mean(crit_fake_pred)
        critic_real_loss = torch.mean(crit_real_pred)
        panelty_loss = critic_fake_loss + critic_real_loss + c_lambda * gp

        critic_loss =  critic_fake_loss + critic_real_loss + panelty_loss
        return critic_loss

    def train(self):
        cur_step = 0
        generator_losses = []
        critic_losses = []
        for _ in range(n_epochs):
            for real, _ in tqdm(self.dataloader):
                cur_batch_size = len(real)

                # Flatten the batch of real images from the dataset
                real = real.to(self.device)

                mean_iteration_critic_loss = 0
                for _ in range(crit_repeats):
                    
                    ### Update critic ###
                    self.critic_opt.zero_grad()

                    fake_noise = self.get_noise(cur_batch_size)
                    fake = self.generator(fake_noise)
                    crit_fake_pred = self.critic(fake.detach())
                    crit_real_pred = self.critic(real)

                    epsilon = torch.rand(len(real), 1, 1, 1, device=self.device, requires_grad=True)
                    gradient = self.get_gradient(real, fake.detach(), epsilon)
                    gp = gradient_penalty(gradient)

                    critic_loss = self.get_critic_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                    mean_iteration_critic_loss += critic_loss.item() / crit_repeats
                    critic_loss.backward(retain_graph=True)
                    self.critic_opt.step()

                critic_losses += [mean_iteration_critic_loss]

                ### Update generator ###
                self.gen_opt.zero_grad()
                fake_noise_ = self.get_noise(cur_batch_size)
                fake_ = self.generator(fake_noise_)
                crit_fake_pred = self.critic(fake_)
                gen_loss = self.get_gen_loss(crit_fake_pred)
                gen_loss.backward()
                self.gen_opt.step()

                generator_losses += [gen_loss.item()]

                ### Visualization code ###
                if cur_step % display_step == 0 and cur_step > 0:
                    gen_mean = sum(generator_losses[-display_step:]) / display_step
                    crit_mean = sum(critic_losses[-display_step:]) / display_step
                    print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                    show_tensor_images(fake, cur_step)
                    # show_tensor_images(real, cur_step)
                    # step_bins = 20
                    # num_examples = (len(generator_losses) // step_bins) * step_bins
                    # plt.plot(
                    #     range(num_examples // step_bins), 
                    #     torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    #     label="Generator Loss"
                    # )
                    # plt.plot(
                    #     range(num_examples // step_bins), 
                    #     torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    #     label="Critic Loss"
                    # )
                    # plt.legend()
                    # plt.show()

                cur_step += 1


gan = WGAN()
gan.train()