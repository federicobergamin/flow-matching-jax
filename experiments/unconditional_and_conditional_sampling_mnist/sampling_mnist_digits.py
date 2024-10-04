"""
We sample both unconditionally and using reconstruction guidance on the MNIST dataset.
"""


import numpy as np
import jax
import jax.numpy as jnp
import flax
import torch
import math
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from PIL import Image, ImageDraw, ImageFont
from jax import random
from matplotlib import pyplot as plt
import argparse
from cfm_jax.models.unet.unet import Unet
from cfm_jax.models.models import ConvNet
from cfm_jax.datautils import sample_gaussian, sample_8gaussians
from cfm_jax.utils import (
    split_key,
    plot_data,
    euclidian_distance,
    image_to_numpy,
    numpy_collate,
    image_to_numpy_zero_one_interval,
)
from cfm_jax.losses import cfm_loss
from cfm_jax.conditional_flow_matching import ConditionalFlowMatchingModel
import optax
from tqdm import tqdm
from jax import flatten_util
import pickle as pkl
from cfm_jax.losses import cross_entropy_loss


def main(args):
    seed = args.seed
    prng_key = jax.random.PRNGKey(seed)

    saving_dir = "trained_models/"
    img_saving_dir = "plots/"

    # define the model
    model = Unet(embedding_dim=32, output_channels=1, dim_mults=(1, 2, 4))

    data_rng, time_rng, init_rng, prng_key = split_key(prng_key, num=4)

    # function to apply the model
    model_apply = lambda p, x, t: model.apply({"params": p}, x, t)

    ## now I can load the parameters
    params_dict = pkl.load(open(saving_dir + "cfm_mnist_weights_cfm.pickle", "rb"))
    print(params_dict.keys())

    params_vec, unflatten = flatten_util.ravel_pytree(params_dict)
    n_params = len(params_vec)
    print(f"Number of parameters: {n_params}")

    print("Sampling from the model")
    n_samples = 64
    key_samples, prng_key = split_key(prng_key, num=2)
    x0_samples = sample_gaussian(n_samples, dimension=28 * 28, key=key_samples)
    x0_samples = np.reshape(x0_samples, (n_samples, 28, 28, 1))
    t = jnp.zeros((n_samples,))

    N = 500
    xt = x0_samples
    for i in tqdm(range(N), desc="Sampling"):
        pred = model_apply(params_dict["params"], xt, t)
        xt = xt + pred * (1 / N)
        t = t + 1 / N

    ## save results
    xt_numpy = np.array(xt)
    torch.save(xt_numpy, saving_dir + "cfm_mnist_samples.pt")

    print(xt_numpy.shape)

    # I can plot the samples in a grid
    fig, axs = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            axs[i, j].imshow(xt_numpy[i * 8 + j, :, :, 0], cmap="gray")
            axs[i, j].axis("off")
    plt.title("Unconditional samples from the model")
    # plt.savefig(img_saving_dir + "cfm_mnist_samples.png")
    plt.show()


    ##################################################
    ###
    ###        Reconstruction guidance
    ###
    ##################################################
    print("Reconstruction guidance")
    guidance_strength = 1
    # now I can sample using reconstruction guidance
    # I need to define and load the best classifier model
    num_classes = 10
    channels = [8, 16]
    kernel_size = 3
    classifier_model = ConvNet(channels=channels, num_classes=num_classes, act_fn=jax.nn.relu, kernel_size=kernel_size)
    with open(saving_dir+"classification_best_model_noised.pkl", "rb") as f:
        classifier_model_weights = pkl.load(f)

    classifier_model_apply = lambda p, x: classifier_model.apply({"params": p}, x)
    class_conditioning = 1

    # I need have to get the gradient of the loss wrt the input
    # loss function definition
    ce_loss = cross_entropy_loss(classifier_model_apply)
    # we have to get the gradient wrt the input
    classifier_loss_grad_fn = jax.jit(jax.value_and_grad(ce_loss, argnums=1, has_aux=True))
    
    # now we can sample using reconstruction guidance
    print("Sampling from the model")
    n_samples = args.n_samples
    key_samples, prng_key = split_key(prng_key, num=2)
    x0_samples = sample_gaussian(n_samples, dimension=28 * 28, key=key_samples)
    x0_samples = np.reshape(x0_samples, (n_samples, 28, 28, 1))
    conditioning_label = jnp.ones((n_samples,), dtype=jnp.int32) * class_conditioning
    ## the initial time is 0
    ## but it can be problematic for certain computations
    ## so we can add a small epsilon
    t = jnp.zeros((n_samples,)) + 1e-3

    N = 500
    xt = x0_samples
    for i in tqdm(range(N), desc="Sampling"):
        vecot_field_pred = model_apply(params_dict["params"], xt, t)
        # here we have to compute the guidance
        # I have to compute the gradient of the loss wrt the input
        (loss, logits), grads = classifier_loss_grad_fn(classifier_model_weights["params"], xt, conditioning_label)
        # printing just to check what the classifier is doing
        print(jnp.argmax(logits, axis=1))

        # I think this is the correct way of applying reconstruction guidance
        # Euler step
        xt = xt + vecot_field_pred * (1 / N)
        xt = xt - guidance_strength*grads
        t = t + 1 / N

    xt_numpy = np.array(xt)
    # torch.save(xt_numpy, saving_dir + "cfm_mnist_samples_reconstruction_guidance.pt")

    print(xt_numpy.shape)

    # I can plot the samples in a grid
    fig, axs = plt.subplots(int(jnp.sqrt(n_samples)), int(jnp.sqrt(n_samples)), figsize=(8, 8))
    for i in range(int(jnp.sqrt(n_samples))):
        for j in range(int(jnp.sqrt(n_samples))):
            axs[i, j].imshow(xt_numpy[i * int(jnp.sqrt(n_samples)) + j, :, :, 0], cmap="gray")
            axs[i, j].axis("off")
    plt.savefig(img_saving_dir + "cfm_mnist_samples_reconstruction_guidance.png")
    plt.show()


    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic interpolants Gaussian-MNIST")
    parser.add_argument("--seed", "-s", type=int, default=1, help="seed")
    parser.add_argument("--n_samples", "-ns", type=int, default=64, help="Number of samples to generate")

    args = parser.parse_args()
    main(args)
