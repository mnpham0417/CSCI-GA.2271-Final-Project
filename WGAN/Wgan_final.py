#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# In[49]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
lr = 0.00005
batch_size = 128
image_size = 64
channels_img = 3
z_dim = 100
num_epochs = 5
features_disc = 64
features_gen = 64
LAMBDA_GP = 10
num_epochs = 10
real_label = 1.
fake_label = 0.


workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 5e-5

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# In[4]:


dataroot = "/celeba"


# In[36]:


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# In[37]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[50]:


dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# In[53]:


netD = Discriminator(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)
# Print the model
print(netD)


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)


# In[63]:


step = 0
netG.train()
netD.train()


# In[68]:


def get_noise(n_samples, noise_dim, device='cpu'):
    '''
    Generate noise vectors from the random normal distribution with dimensions (n_samples, noise_dim),
    where
        n_samples: the number of samples to generate based on  batch_size
        noise_dim: the dimension of the noise vector
        device: device type can be cuda or cpu
    '''
    
    return  torch.randn(n_samples,noise_dim, 1,1,device=device)


# In[66]:


def gradient_penalty( critic, real_image, fake_image, device):
    #print(real_image.shape)
    batch_size, channel, height, width= real_image.shape
    #alpha is selected randomly between 0 and 1
    #print(batch_size)
    alpha= torch.rand(batch_size,1,1,1).repeat(1, channel, height, width).to(device)
    # interpolated image=randomly weighted average between a real and fake image
    #interpolated image ← alpha *real image  + (1 − alpha) fake image
    real_image = real_image.to(device)
    fake_image = fake_image.to(device)
    #print(fake_image.shape)
    interpolatted_image=(alpha*real_image) + (1-alpha) * fake_image
    
    interpolatted_image = interpolatted_image.to(device)
    # calculate the critic score on the interpolated image
    interpolated_score= critic(interpolatted_image)
    interpolated_score = interpolated_score.to(device)
    # take the gradient of the score wrt to the interpolated image
    gradient= torch.autograd.grad(inputs=interpolatted_image,
                                  outputs=interpolated_score,
                                  retain_graph=True,
                                  create_graph=True,
                                  grad_outputs=torch.ones_like(interpolated_score)                          
                                 )[0]
    gradient= gradient.view(gradient.shape[0],-1)
    gradient_norm= gradient.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    return gradient_penalty


# In[72]:

for epoch in range(num_epochs):
  for batch_idx, (real, _) in enumerate(dataloader):
    real = real.to(device)
    cur_batch_size = real.shape[0]
    for _ in range(5):
       netD.zero_grad()
       fake_noise = get_noise(cur_batch_size, z_dim, device=device)   
       fake = netG(fake_noise)
       critic_real = netD(real).reshape(-1)
       critic_fake = netD(fake).reshape(-1)
       #loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
       gp =gradient_penalty(netD, real, fake, device)
       gp = gp.to(device)
       loss_critic = -(torch.mean(critic_real) -torch.mean(critic_fake)) + LAMBDA_GP *gp
       netD.zero_grad()
       loss_critic.backward(retain_graph = True)
       optimizerD.step()

       
    
    netG.zero_grad()
    output = netD(fake).reshape(-1)
    loss_gen = -torch.mean(output)
    
    loss_gen.backward()
    optimizerG.step()
    
    if batch_idx%100 == 0:
      print (
          f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \ Loss D: {loss_critic :.4f}, loss G: {loss_gen:.4f}"
      ) 


model_file = 'WGAN_discriminator_model_new4.pth'
torch.save(netD, model_file)

model_file = 'WGAN_generator_model_new4.pth'
torch.save(netG, model_file)





img_list = []
fake = netG(fixed_noise).detach().cpu()
img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())






