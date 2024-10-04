"""
Here I want to create a small experiment to do image inpainting.
Like missing value inputation of a single MNIST image by using reconstruction guidance.
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
from cfm_jax.losses import cross_entropy_loss, guided_loss


def main(args):
    seed = args.seed
    prng_key = jax.random.PRNGKey(seed)

    saving_dir = "trained_models/"
    img_saving_dir = "plots/"
    dataset_path = args.dataset_path
    batch_size = args.batch_size
        
    ## type of inverse problem
    # infilling
    infilling_exp = False # if False we will have central missing values block

    # define the model
    model = Unet(embedding_dim=32, output_channels=1, dim_mults=(1, 2, 4))

    data_rng, time_rng, init_rng, prng_key = split_key(prng_key, num=4)

    # function to apply the model
    model_apply = lambda p, x, t: model.apply({"params": p}, x, t)

    ## now I can load the parameters
    params_dict = pkl.load(open(saving_dir + "cfm_mnist_weights_1000epochs.pickle", "rb"))
    print(params_dict.keys())

    params_vec, unflatten = flatten_util.ravel_pytree(params_dict)
    n_params = len(params_vec)
    print(f"Number of parameters: {n_params}")

    # now I should get the test data
    test_dataset = MNIST(root=dataset_path, transform=image_to_numpy_zero_one_interval, train=False, download=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)

    # now i get a single image
    x_test, y_test = test_dataset[100]

    plt.imshow(x_test, cmap="gray")
    plt.show()

    # now i have to split the keys and sample a mask
    mask_rng, prng_key = split_key(prng_key, num=2)

    if infilling_exp:
        mask = jax.random.binomial(mask_rng, n=1, p=0.25, shape=x_test.shape)
    else:
        mask = jnp.ones_like(x_test)
        mask = mask.at[7:21, 7:21].set(0.)

    print(mask.shape)
    print(mask)

    plt.imshow(mask, cmap="gray")
    plt.show()

    # now i have to apply the mask to the image
    x_test_masked = (x_test * mask) + (1-mask) * 0.5

    plt.imshow(x_test_masked, cmap="gray")
    plt.show()

    ###################################
    #
    #  Inverse problem version 1
    #
    ###################################

    ## now I can use guidance to do inpainting
    ## I have to define the guidance loss
    # n_samples = args.n_samples
    # key_samples, prng_key = split_key(prng_key, num=2)
    # x0_samples = sample_gaussian(n_samples, dimension=28 * 28, key=key_samples)
    # x0_samples = np.reshape(x0_samples, (n_samples, 28, 28, 1))

    # # now I have to multiply the number of masks as n_samples
    # mask = jnp.reshape(mask, (1, 28, 28, 1))
    # mask = jnp.tile(mask, (n_samples, 1, 1, 1))

    # # test
    # # print(mask.shape)
    # # plt.imshow(mask[10].reshape(28, 28), cmap="gray")
    # # plt.show()

    # # now I can define the loss
    # guidance = guided_loss(x0_samples, mask)
    # # we have to get the gradient wrt the input
    # guidance_grad_fn = jax.jit(jax.value_and_grad(guidance, argnums=0, has_aux=True))
    
    # guidance_strength = 1
    
    # t = jnp.zeros((n_samples,)) + 1e-3
    # N = 500
    # t1 = t.reshape(-1, *([1] * (len(x0_samples.shape) - 1))) 

    # # xt = (1-t) *x0_samples
    # xt = x0_samples
    # for i in tqdm(range(N), desc="Sampling"):
    #     vector_field_pred = model_apply(params_dict["params"], xt, t)
    #     # here we have to compute the guidance
    #     # I have to compute the gradient of the loss wrt the input

    #     # I think this is the correct way of applying reconstruction guidance
    #     # Euler step
    #     xt = xt + vector_field_pred * (1 / N)

    #     # compute guidance 
    #     (loss, approx_at_time1), grads = guidance_grad_fn(xt, x_test, t)
    #     plt.imshow(approx_at_time1[0].reshape(28, 28), cmap="gray")
    #     plt.show()
    #     # check norm of grads
    #     print(jnp.linalg.norm(grads[0,...]))
    #     # print(jnp.linalg.norm(grads * (1-mask)))

    #     xt = xt - guidance_strength*grads*t
    #     t = t + 1 / N

    # xt_numpy = np.array(xt)
    # torch.save(xt_numpy, saving_dir + "cfm_mnist_samples_reconstruction_guidance.pt")

    # print(xt_numpy.shape)
    # print(xt_numpy[0])

    # plt.imshow(xt_numpy[0], cmap="gray")
    # plt.show()

     # I can plot the samples in a grid
    # fig, axs = plt.subplots(int(jnp.sqrt(n_samples)), int(jnp.sqrt(n_samples)), figsize=(8, 8))
    # for i in range(int(jnp.sqrt(n_samples))):
    #     for j in range(int(jnp.sqrt(n_samples))):
    #         axs[i, j].imshow(xt_numpy[i * int(jnp.sqrt(n_samples)) + j, :, :, 0], cmap="gray")
    #         axs[i, j].axis("off")
    # plt.savefig(img_saving_dir + "cfm_mnist_samples_reconstruction_guidance.png")
    # plt.show()

    ###################################
    #
    #  Inverse problem following Algorithm 3
    #  from https://arxiv.org/abs/2310.04432
    #  work only for the Lipman formulation I guess
    #
    ###################################

    # Let's try to follow what they are telling in
    # Training-free Linear Image Inverses via Flows
    # https://arxiv.org/abs/2310.04432

    # compute initial xt as alpha_t0 * y + sigma_t0 * eps where eps is N(0,1)
    # the measurement matrix in this case is the mask

    n_samples = 1
    t = jnp.zeros((n_samples,)) + 1e-2
    t = t.reshape(-1, *([1] * (len(x_test.shape) - 1))) 
    noise_key, prng_key = split_key(prng_key, num=2)
    eps = sample_gaussian(n_samples, dimension=28 * 28, key=noise_key)
    eps = np.reshape(eps, (n_samples, 28, 28, 1))
    xt = t * (x_test * mask) + (1-t) * eps
    
    print(xt.shape)
    plt.imshow(xt[0].reshape(28, 28), cmap="gray")
    plt.title("xt at time 0")
    plt.show()

    def get_x1hat(xt, t, dnlratio, dlnsigma):
        vector_field_pred = model_apply(params_dict["params"], xt, t)

        # now we have to compute some additional values
        # these are computed using Lipman formulation
        rt2 = ((1-t)**2) / ((1-t)**2 + t**2)

        # convert vector field to xhat1 (kind of Tweedie's formula for getting x1|xt but
        # from the vector field perspective)
        x1hat = 1/(t * dlnratio) * (vector_field_pred - dlnsigma * xt)

        return x1hat, 

    ## now we have to start Euler integration
    N = 200
    for i in tqdm(range(N), desc="Euler integration"):
        # compute constants we need
        dlnratio = 1/(t*(1-t))
        dlnsigma = - 1 / (1-t)
        # we start by predicting the vector field
        vector_field_pred = model_apply(params_dict["params"], xt, t)

        # now we have to compute some additional values
        # these are computed using Lipman formulation
        rt2 = ((1-t)**2) / ((1-t)**2 + t**2)

        # convert vector field to xhat1 (kind of Tweedie's formula for getting x1|xt but
        # from the vector field perspective)
        dlnratio = 1/(t*(1-t))
        dlnsigma = - 1 / (1-t)
        x1hat = 1/(t * dlnratio) * (vector_field_pred - dlnsigma * xt)

        # compute the score of \nabla log p_approx(y|xt)
        score = (x_test * mask - x1hat*mask).T @ (rt2 ) 


    plt.imshow(xt[0].reshape(28, 28), cmap="gray")
    plt.title("xt at time 1")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic interpolants Gaussian-MNIST")
    parser.add_argument("--seed", "-s", type=int, default=1, help="seed")
    parser.add_argument("--dataset_path", "-data_path", type=str, default="data/", help="Folder for the datasets")
    parser.add_argument("--batch_size", "-bs", type=int, default=256, help="Batch size")
    parser.add_argument("--n_samples", "-ns", type=int, default=1, help="Number of samples to generate")

    args = parser.parse_args()
    main(args)




