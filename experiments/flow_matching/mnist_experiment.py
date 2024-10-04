"""
File where we train a simple CNN model to learn a velocity filed to go from a single Gaussian 
to the MNIST (or CIFAR) dataset.
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


def main(args):
    seed = args.seed
    prng_key = jax.random.PRNGKey(seed)

    epochs = args.epochs
    batch_size = args.batch_size
    dataset_path = args.dataset_path

    saving_dir = "trained_models/"

    # I can create the datasets
    train_dataset = MNIST(root=dataset_path, transform=image_to_numpy_zero_one_interval, train=True, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)

    # for now we don't need the test since we are not computing the test-log-likelihood
    # test_dataset = MNIST(root=dataset_path, transform=image_to_numpy_zero_one_interval, train=False, download=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)

    # define the model
    model = Unet(embedding_dim=32, output_channels=1, dim_mults=(1, 2, 4))

    data_rng, time_rng, init_rng, prng_key = split_key(prng_key, num=4)

    # now we can initialize the model
    input_shape = (batch_size, 28, 28, 1)

    x = jax.random.normal(data_rng, input_shape)
    time = jax.random.uniform(time_rng, shape=(batch_size,))

    print("initializing the model")
    params_dict = model.init(init_rng, x, time)

    params_vec, unflatten = flatten_util.ravel_pytree(params_dict)
    n_params = len(params_vec)
    print(f"Number of parameters: {n_params}")

    lr = 0.001
    optim = optax.adam(lr)
    opt_state = optim.init(params_dict["params"])

    # # function to apply the model
    model_apply = lambda p, x, t: model.apply({"params": p}, x, t)

    # loss function definition
    conditional_flow_matching_loss = cfm_loss(model_apply)
    loss_grad_fn = jax.jit(jax.value_and_grad(conditional_flow_matching_loss, argnums=0))

    ## conditional flow matching
    flow_matching_model = ConditionalFlowMatchingModel(sigma=0.0, method=args.method)

    # training loop
    for ep in tqdm(range(epochs), desc="Training"):
        ep_loss = 0
        ep_num_elem = 0
        for batch_p1_data, batch_p1_labels in tqdm(train_loader, desc="Batch"):

            key_t, key_p0, key_eps, prng_key = split_key(prng_key, num=4)
            t = jax.random.uniform(key_t, shape=(batch_p1_data.shape[0],))

            # now I have to sample form p0, which in this case is a single Gaussian
            # distribution
            batch_p0 = sample_gaussian(batch_p1_data.shape[0], dimension=28 * 28, key=key_p0)
            # I have to reshape them
            batch_p0 = batch_p0.reshape(batch_p1_data.shape[0], 28, 28, 1)

            ## here I have to assert that the shape of batch_p1_data and batch_p0 are the same
            assert batch_p0.shape == batch_p1_data.shape, "Shapes of the two batches are not the same"

            # sample from the probability path and conditional vector field
            xt = flow_matching_model.sample_xt(batch_p0, batch_p1_data, t, key_epsilon=key_eps)
            ut = flow_matching_model.compute_conditional_flow(batch_p0, batch_p1_data, t, xt)

            loss, grad = loss_grad_fn(params_dict["params"], xt, t, ut)
            params_updates, opt_state = optim.update(grad, opt_state)
            params_dict["params"] = optax.apply_updates(params_dict["params"], params_updates)

            ep_loss += loss * batch_p0.shape[0]
            ep_num_elem += batch_p0.shape[0]

        if ep % 50 == 0:
            print(f"finished {ep} epoch, avg loss trainin: {ep_loss/ep_num_elem}")

    ## at the end we can save the model parameters
    print("Saving model parameters")
    with open(saving_dir + f"cfm_mnist_weights_{args.method}_trial.pickle", "wb") as handle:
        pkl.dump(params_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print("Sampling from the model")
    key_samples, prng_key = split_key(prng_key, num=2)
    x0_samples = sample_gaussian(10, dimension=28 * 28, key=key_samples)
    t = jnp.zeros((10,))

    N = 300
    xt = x0_samples
    for i in range(N):
        pred = model_apply(params_dict["params"], xt, t)
        xt = xt + pred * (1 / N)
        t = t + 1 / N

    ## save results
    xt_numpy = np.array(xt)
    torch.save(xt_numpy, saving_dir + "cfm_mnist_samples.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic interpolants Gaussian-MNIST")
    parser.add_argument("--seed", "-s", type=int, default=1, help="seed")
    parser.add_argument("--epochs", "-ep", type=int, default=500, help="epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=256, help="Batch size")
    parser.add_argument("--method", "-mt", type=str, default="CFM", help="Type of Flow matching we use")
    parser.add_argument("--dataset_path", "-data_path", type=str, default="data/", help="Folder for the datasets")

    args = parser.parse_args()
    main(args)
