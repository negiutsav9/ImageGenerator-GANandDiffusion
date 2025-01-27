# Image Generation using Generative Adversarial Networks (GANs) and Denoising Diffusion
by Utsav Negi

## Aim
The aim of this report is to demonstrate the approach used to implement image modelling using
Generative Adversarial Nets and Denoising Diffusion using a subset of CelebA dataset.

## Source Code Detail
For this assignment, the details about the source code are as follows:
<ul><li>main.py: Source code consisting of all the essential classes, functions & main ML pipeline for
training and evaluating GANs along with code to evaluate the samples generated using Denoising
Di usion.</li></ul>

## Image Generation using GANs

This implementation consists of two nets: Discriminator and Generator. The role of the discriminator is to
return the probability that the input belongs to the distribution representing the training data. The role of
the Generator is to create images from noisy data by transforming the noisy data distribution to a
distribution representing the training data. For this assignment, the Discriminator and Generator
architecture is like the DLStudio implementation, while the number of input-output channels and the
number of layers is tuned to convert the CelebA image data into its latent form for the Discriminator using
Conv2D and vice versa for the Generator using ConvTranspose2D.

For training the nets, weights for convolutional and batch normalization layers are initialized to stabilize
the nets. Two separate ADAM optimizers with the same learning rate and hyperparameters are used to
determine the new parameters for the Discriminator and the Generator. To calculate the loss, the
criterion is set to BinaryCrossEntropy Loss Function as the values are returned in the form of a
probability. In each iteration, the outputs of the Discriminator net are compared with the labels filled with
1 to determine the loss based on the training images. At the same time, the output image of the
Generator net, generated from a Gaussian noise, is used as an input to the Discriminator net. The output
values yielded by the Discriminator are compared with the labels filled with 0 to determine the loss based
on the fake images. These losses are backpropagated and added to determine the new parameters for
the Discriminator net. Using the fake image generated by the Generator net, the newly trained
Discriminator net generates an output which is compared with the labels filled with 1 to determine the
loss used to calculate the new parameters for the Generator net. In this implementation, the number of
epochs is set to 500 while the learning rate and the β for both optimizers are set to 1e-4 and (0.75, 0.999)
respectively.

## Image Generation using Denoising Diffusion

The method of Denoising Diffusion involves two Markov chain processes taking place simultaneously.
One process called Di usion takes place by injecting Gaussian noise into a training image at each
timestep until the image data gets substituted with an isotropic Gaussian noise. Another process called
Denoising takes place by removing Gaussian noise from a sample of isotropic Gaussian noise at each
timestep until the sample gets converted into a recognizable image. During these processes, a denoising
neural network is trained to remove the same amount of noise which was added during the diffusion for
the same timestep transition.

For this assignment, pre-trained weights are added to the Unet model to generate images due to
hardware constraints. The number of timesteps are set to 100 and around 2048 sample images are
generated for evaluating the performance.
