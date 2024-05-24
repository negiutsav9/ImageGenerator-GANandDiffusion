'''################################################################################################
Filename: hw8.py
Author: Utsav Negi
Purpose: To implement image modelling using Generative Adversarial Nets and
    Denoising Diffusion using a subset of CelebA dataset.
################################################################################################'''
import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as tvt

from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import (
    calculate_activation_statistics,
    calculate_frechet_distance,
)

# setting seeds for consistency (reference: Lecture 2 and HW 2)
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ["PYTHONHASHSEED"] = str(seed)

##################### FUNCTIONS AND CLASSES FOR THE MODEL AND ITS TRAINING ########################

# Discriminator model to get the probability of image belonging to a label
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 4-2-1
        self.conv1 = nn.Conv2d(3, 16, 4, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv_final = nn.Conv2d(256, 1, 2, 1, 0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, True)
        x = self.bn1(self.conv2(x))
        x = F.leaky_relu(x, 0.1, True)
        x = self.bn2(self.conv3(x))
        x = F.leaky_relu(x, 0.1, True)
        x = self.bn3(self.conv4(x))
        x = F.leaky_relu(x, 0.1, True)
        x = self.bn4(self.conv5(x))
        x = F.leaky_relu(x, 0.1, True)
        x = self.out(self.conv_final(x))
        return x

# Generator model to generate an image from a latent Gaussian Noise
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convtrans1 = nn.ConvTranspose2d(100, 256, 2, 1, 0, bias=False)
        self.convtrans2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.convtrans3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.convtrans4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.convtrans5 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False)
        self.convtrans_final = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)
        self.out = nn.Tanh()

    def forward(self, x):
        x = self.convtrans1(x)
        x = F.leaky_relu(self.bn1(x), negative_slope=0.1)
        x = self.convtrans2(x)
        x = F.leaky_relu(self.bn2(x), negative_slope=0.1)
        x = self.convtrans3(x)
        x = F.leaky_relu(self.bn3(x), negative_slope=0.1)
        x = self.convtrans4(x)
        x = F.leaky_relu(self.bn4(x), negative_slope=0.1)
        x = self.convtrans5(x)
        x = F.leaky_relu(self.bn5(x), negative_slope=0.1)
        x = self.out(self.convtrans_final(x))
        return x

# training function to train the Discriminator and the Generator
def training(discriminator, generator, dataloader, results_dir, device):

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    nz = 100
    netD = discriminator.to(device).apply(weights_init)
    netG = generator.to(device).apply(weights_init)
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label, fake_label = 1, 0
    optimD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.75, 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.75, 0.999))
    criterion = nn.BCELoss()
    G_losses = []
    D_losses = []
    iterations = 0
    print("\n\nStarting Training Loop....\n\n")
    for epoch in range(500):
        g_losses_per_print = []
        d_losses_per_print = []
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_images = data[0].to(device)
            
            # getting loss based on 1s
            label = torch.full(
                (batch_size,), real_label, device=device, dtype=torch.float
            )
            output = netD(real_images).view(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()

            # getting loss based on 0s
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach()).view(-1)
            lossD_fakes = criterion(output, label)
            lossD_fakes.backward()

            # adding the loss and optimizing the discriminator model
            overall_lossD = lossD_real + lossD_fakes
            d_losses_per_print.append(overall_lossD)
            optimD.step()

            # getting loss for optimizing the generator model
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_images).view(-1)
            lossG = criterion(output, label)
            g_losses_per_print.append(lossG)
            lossG.backward()
            optimG.step()

            # print status
            if i % 500 == 499:
                average_dloss = torch.mean(torch.FloatTensor(d_losses_per_print)).cpu()
                average_gloss = torch.mean(torch.FloatTensor(g_losses_per_print)).cpu()
                print(
                    "Epoch:",
                    epoch + 1,
                    "=> ",
                    "Discriminator Loss:",
                    torch.round(average_dloss, decimals=6),
                    "Generator Loss:",
                    torch.round(average_gloss, decimals=6),
                )

            G_losses.append(lossG.item())
            D_losses.append(overall_lossD.item())

            # validating the performance of the generator
            if (i % 1000 == 0) or ((epoch == 499) and (i == len(dataloader) - 1)):
                if not os.path.exists(results_dir):
                    os.mkdir(results_dir)
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                image = tvt.ToPILImage()(
                    make_grid(fake, padding=1, pad_value=1, normalize=True)
                )
                image.save(results_dir + str(epoch) + "_" + str(i) + ".jpg", "JPEG")

            iterations += 1

    return G_losses, D_losses

####################################### MAIN PIPELINE #############################################

# data directory for celebA dataset
data_dir = "./dataset"

# setting device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
torch.cuda.set_device("cuda:0")

# creating torch Dataset
dataset = datasets.ImageFolder(
    root=data_dir,
    transform=tvt.Compose(
        [tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
)

# creating dataloader by wrapping the Dataset Object
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size)

discriminator = Discriminator()
generator = Generator()

num_learnable_params_disc = sum(
    p.numel() for p in discriminator.parameters() if p.requires_grad
)
print(
    "\n\nThe number of learnable parameters in the Discriminator: %d\n"
    % num_learnable_params_disc
)
num_learnable_params_gen = sum(
    p.numel() for p in generator.parameters() if p.requires_grad
)
print(
    "\nThe number of learnable parameters in the Generator: %d\n"
    % num_learnable_params_gen
)
num_layers_disc = len(list(discriminator.parameters()))
print("\nThe number of layers in the discriminator: %d\n" % num_layers_disc)
num_layers_gen = len(list(generator.parameters()))
print("\nThe number of layers in the generator: %d\n\n" % num_layers_gen)

G_losses, D_losses = training(
    discriminator, generator, dataloader, "./results/", device
)

torch.save(generator.state_dict(), "./gen.pth")
torch.save(discriminator.state_dict(), "./dis.pth")

################################### MODEL EVALUATION ##############################################

# Creating a plot to show incurred during Generator and Discriminator Training 
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./gen_and_disc_loss_training.png")
plt.show()

# Generating 2048 samples using the Generator model for evaluation
gen_testing_net = Generator()
gen_testing_net.load_state_dict(torch.load("./gen.pth"))
gen_testing_net = gen_testing_net.to(device)
for i in range(2048):
    nz = 100
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)
    transform = tvt.Compose(
        [
            tvt.Normalize((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
            tvt.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
            tvt.ToPILImage(),
        ]
    )
    fake_img = transform(
        torch.squeeze((gen_testing_net(fixed_noise).detach().cpu()), dim=0)
    )
    fake_img.save("./fake_images/" + str(i) + ".jpeg", "JPEG")
    
# Determining the FID value for GAN-based images and real images    
real_paths = []
fake_paths = []
paths = os.listdir("./real_images")
for path in paths:
    path = "./real_images/" + path
    real_paths.append(path)
paths = os.listdir("./fake_images")
for path in paths:
    path = "./fake_images/" + path
    fake_paths.append(path)

dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device=device)
m1, s1 = calculate_activation_statistics(real_paths, model, device=device)
m2, s2 = calculate_activation_statistics(fake_paths, model, device=device)
fid_value = calculate_frechet_distance(m1, s1, m2, s2)
print(f"FID: {fid_value:.2f}")


# Generating 2048 samples using the Denoising Diffusion method for evaluation
data = np.load("./samples_2048x64x64x3.npz")
print("\n\n[visualize_sample.py]  the data object: ", data)
print("\n\n[visualize_sample.py]  type of the data object: ", type(data))
print("\n\n[visualize_sample.py]  shape of the object data['arr_0']: ", data['arr_0'].shape)
for i, img in enumerate(data['arr_0']):
    image = Image.fromarray(img)
    #plt.axis("off")
    image.save(f"./visualize_samples/test_{i}.jpeg", "JPEG")

# Determining the FID value for images generated by Denoising Diffusion method and real images    
real_paths = []
fake_paths = []
paths = os.listdir("./real_images")
for path in paths:
    path = "./real_images/" + path
    real_paths.append(path)
paths = os.listdir("./visualize_samples")
for path in paths:
    path = "./visualize_samples/" + path
    fake_paths.append(path)
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device=device)
m1, s1 = calculate_activation_statistics(real_paths, model, device=device)
m2, s2 = calculate_activation_statistics(fake_paths, model, device=device)
fid_value = calculate_frechet_distance(m1, s1, m2, s2)
print(f"FID: {fid_value:.2f}")

######################################## END OF FILE ##############################################