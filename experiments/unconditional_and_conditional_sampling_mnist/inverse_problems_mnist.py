"""
Here I want to create a small experiment to do image inpainting.
Like missing value inputation of a single MNIST image by using reconstruction guidance.
# Let's try to fix this now
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
from cfm_jax.models.unet_v2.unet import DDPM
from configs.mnist_unet import get_default_configs

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
    yang_song_model = True

    # define the model
    # define the model
    if yang_song_model:
        config = get_default_configs()
        print(config)
        model = DDPM(config)
    else:
        model = Unet(embedding_dim=64, output_channels=1, resnet_block_groups=8, dim_mults=(1, 2))


    data_rng, time_rng, init_rng, prng_key = split_key(prng_key, num=4)

    # function to apply the model
    model_apply = lambda p, x, t: model.apply({"params": p}, x, t)

    ## now I can load the parameters
    ## now I can load the parameters
    if yang_song_model:
        params_dict = pkl.load(open(saving_dir + f"cfm_mnist_weights_{args.method}_best_validation_trial_yangsong_True.pickle", "rb"))
    else:
        params_dict = pkl.load(open(saving_dir + f"cfm_mnist_weights_{args.method}_best_validation_trial.pickle", "rb"))
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
    plt.savefig(img_saving_dir + f"x_test_infilling_{infilling_exp}.png")
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
    # plt.savefig(img_saving_dir + f"mask.png")
    plt.show()

    # now i have to apply the mask to the image
    x_test_masked = (x_test * mask) + (1-mask) * 0.5

    plt.imshow(x_test_masked, cmap="gray")
    plt.savefig(img_saving_dir + f"x_test_masked_infilling_{infilling_exp}.png")
    plt.show()

    ###################################
    #
    #  Inverse problem version new stuff
    #
    ###################################
    # we will still use the approach proposed by https://arxiv.org/abs/2310.04432
    # the Tweedie's formula for xhat is given by (1-t)u_t(x) + x (see derivation here https://hackmd.io/@fedbe/Hyryj7de1l)
    # so we need to define the operator A, compute the er_t^2 and the derivative of xhat wrt x_t (i.e. PIGDM correction) 
    # and the adaprive weights \gamma_t and the data noise \sigma_y
    A_operator = mask.copy()
    def get_alpha_t(t):
        return t

    def get_sigma_t(t):
        return 1-t

    def get_rt2(t):
        sigma_t = get_sigma_t(t)
        alpha_t = get_alpha_t(t)
        return sigma_t**2 / (sigma_t**2 + alpha_t**2)

    def get_x1hat(xt, v_t, t):
        return (1-t) * v_t + xt

    def get_guidance(xt, t, x_true, A, sigma_y):
        sigma_y = sigma_y
        # t = jnp.reshape(t, (-1, *([1] * (len(xt.shape) - 1))))
        # print(f"reshape t: {t.shape}")
        # assert xt.shape == (1, 28, 28, 1), f"xt shape: {xt.shape}, should be (1, 28, 28, 1)"
        # if xt.shape != (1, 28, 28, 1):
        #     xt = jnp.reshape(xt, (1, 28, 28, 1))
        v_t = model_apply(params_dict["params"], xt, t)
        xhat = (1-t) * v_t + xt

        # Now we can approximate p(y|x_t) \approx N(y|A xhat, (sigma_t^2 + r^2(t))I)
        r_t2 = get_rt2(t)

        var_tot = sigma_y**2 + r_t2
        # print(f"Mask A shape: {A.shape}")
        x_true = jnp.reshape(x_true, (1, 28, 28, 1))
        masked_xhat = A * xhat
        masked_x_true = A * x_true
        assert masked_xhat.shape == masked_x_true.shape, f"masked_xhat shape: {masked_xhat.shape}, masked_x_true shape: {masked_x_true.shape}"
        err = -(masked_x_true - masked_xhat) ** 2 
        # print((err / var_tot).shape)
        err = (err / var_tot).sum() / 2
        return err, v_t
    
    def get_gamma_t(t):
        alpha_t = get_alpha_t(t)
        sigma_t = get_sigma_t(t)
        return jnp.sqrt(alpha_t/(sigma_t**2 + alpha_t**2))
    
    ## now I can try to simulate the inverse problem 
    # initialize x_t0 as x_test_masked
    noise_key, prng_key = split_key(prng_key, num=2)
    t = jnp.zeros((1,)) + 1e-3
    # t_reshaped = jnp.reshape(t, (-1, *([1] * (len(x_test.shape) - 1))))
    # print(f"t_reshaped shape: {t_reshaped.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"A_operator shape: {A_operator.shape}")
    print(f"A_operator * x_test  shape: {(A_operator * x_test).shape}")
    eps = sample_gaussian(1, dimension=28 * 28, key=noise_key)
    eps = np.reshape(eps, (28, 28, 1))
    print(f"eps shape: {eps.shape}")
    alpha_t0 = get_alpha_t(t)
    sigma_t0 = get_sigma_t(t)
    alpha_t0 = alpha_t0.reshape(-1, *([1] * (len(eps.shape) - 1)))
    sigma_t0 = sigma_t0.reshape(-1, *([1] * (len(eps.shape) - 1)))
    x_t0 = alpha_t0 * (A_operator * x_test) + sigma_t0 * eps
    plt.imshow(x_t0.reshape(28, 28), cmap="gray")
    plt.show()

    xt = np.reshape(x_t0, (1, 28, 28, 1))
    guidance_func = jax.jit(jax.value_and_grad(get_guidance, argnums=0, has_aux=True)) #jax.jit(jax.value_and_grad(get_x1hat, argnums=0, has_aux=True))

    N = 500
    # since i am not following completely the approach of https://arxiv.org/abs/2310.04432
    # fro now I am adding an additional guidance strength
    my_scale = 5
    for i in tqdm(range(N), desc="Euler integration"):
        (_, v_t), grad = guidance_func(xt, t, x_test, A_operator, sigma_y=0.0)
        
        gamma_t = get_gamma_t(t)
        additional_scale = (1-t) / t
        tot_scale = gamma_t * additional_scale
        tot_scale = jnp.reshape(tot_scale, (-1, *([1] * (len(v_t.shape) - 1))))
        v_t_adapted = v_t + tot_scale * grad * my_scale
        xt = xt + v_t_adapted * (1 / N)
        t = t + 1 / N

    plt.imshow(xt[0].reshape(28, 28), cmap="gray")
    plt.savefig(img_saving_dir + f"x_generated_infilling_{infilling_exp}.png")
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic interpolants Gaussian-MNIST")
    parser.add_argument("--seed", "-s", type=int, default=1, help="seed")
    parser.add_argument("--dataset_path", "-data_path", type=str, default="data/", help="Folder for the datasets")
    parser.add_argument("--batch_size", "-bs", type=int, default=256, help="Batch size")
    parser.add_argument("--n_samples", "-ns", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--method", "-mt", type=str, default="CFMv2", help="Type of Flow matching we use")

    args = parser.parse_args()
    main(args)




