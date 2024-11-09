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
from configs.mnist_unet import get_default_configs
from cfm_jax.models.unet_v2.unet import DDPM
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
    yang_song_model = True

    saving_dir = "trained_models/"

    # I can create the datasets
    # train_dataset = MNIST(root=dataset_path, transform=image_to_numpy_zero_one_interval, train=True, download=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    train_dataset = MNIST(root=dataset_path, transform=image_to_numpy_zero_one_interval, train=True, download=True)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000],
                                                   generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)

    # for now we don't need the test since we are not computing the test-log-likelihood
    # test_dataset = MNIST(root=dataset_path, transform=image_to_numpy_zero_one_interval, train=False, download=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)

    # define the model
    if yang_song_model:
        config = get_default_configs()
        print(config)
        model = DDPM(config)
    else:
        model = Unet(embedding_dim=64, output_channels=1, resnet_block_groups=8, dim_mults=(1, 2))

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
    
    init_lr = 1e-4
    lr_schedule = optax.schedules.cosine_decay_schedule(init_lr, epochs, 0.1)
    optim = optax.adam(lr_schedule)
    opt_state = optim.init(params_dict["params"])

    # # function to apply the model
    model_apply = lambda p, x, t: model.apply({"params": p}, x, t)

    # loss function definition
    conditional_flow_matching_loss = cfm_loss(model_apply)
    loss_grad_fn = jax.jit(jax.value_and_grad(conditional_flow_matching_loss, argnums=0))
    # loss_grad_fn = jax.value_and_grad(conditional_flow_matching_loss, argnums=0)

    # trying to imprive speed
    # @jax.jit
    # def train_step(params, opt_state, xt, t, ut):
    #     loss, grad = loss_grad_fn(params, xt, t, ut)
    #     params_updates, new_opt_state = optim.update(grad, opt_state)
    #     new_params = optax.apply_updates(params, params_updates)
    #     return new_params, new_opt_state, loss
    
    # @jax.jit
    # def valid_step(params, xt, t, ut):
    #     return conditional_flow_matching_loss(params, xt, t, ut)


    # @jax.jit
    # def sample_batch(params, x0, t, num_steps):
    #     def body_fn(i, xt):
    #         pred = model_apply(params, xt, t + i/num_steps)
    #         return xt + pred * (1.0 / num_steps)
        
    #     return jax.lax.fori_loop(0, num_steps, body_fn, x0)

    ## conditional flow matching
    flow_matching_model = ConditionalFlowMatchingModel(sigma=0.0, method=args.method)

    # training loop
    best_valid_loss = np.inf
    for ep in range(epochs):#tqdm(range(epochs), desc="Training"):
        ep_loss = 0
        ep_num_elem = 0
        for batch_p1_data, batch_p1_labels in tqdm(train_loader, desc=f"Epoch {ep} Training Batch"):
            # print(f" min: {np.min(batch_p1_data[0,:].reshape(-1))}, max: {np.max(batch_p1_data[0,:].reshape(-1))}")
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
            # params_dict["params"], opt_state, loss = train_step(params_dict["params"], opt_state, xt, t, ut)

            ep_loss += loss * batch_p0.shape[0]
            ep_num_elem += batch_p0.shape[0]
        
        # at the end of the epoch we can compute the validation loss
        valid_loss = 0
        valid_num_elem = 0
        for batch_p1_data_valid, _ in tqdm(valid_loader, desc=f"Epoch {ep} Validation"):
            key_t, key_p0, key_eps, prng_key = split_key(prng_key, num=4)
            t_valid = jax.random.uniform(key_t, shape=(batch_p1_data_valid.shape[0],))
            p0_valid = sample_gaussian(batch_p1_data_valid.shape[0], dimension=28 * 28, key=key_p0)
            p0_valid = p0_valid.reshape(batch_p1_data_valid.shape[0], 28, 28, 1)

            assert p0_valid.shape == batch_p1_data_valid.shape, "Shapes of the two batches are not the same in validation"

            xt_valid = flow_matching_model.sample_xt(p0_valid, batch_p1_data_valid, t_valid, key_epsilon=key_eps)
            ut_valid = flow_matching_model.compute_conditional_flow(p0_valid, batch_p1_data_valid, t_valid, xt_valid)

            loss = conditional_flow_matching_loss(params_dict["params"], xt_valid, t_valid, ut_valid)
            # loss = valid_step(params_dict["params"], xt_valid, t_valid, ut_valid)
            valid_loss += loss * batch_p1_data_valid.shape[0]
            valid_num_elem += batch_p1_data_valid.shape[0]

        # if ep % 50 == 0:
        print(f"finished {ep} epoch, avg loss training: {ep_loss/ep_num_elem}, avg loss validation: {valid_loss/valid_num_elem}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("New best validation loss, saving model")
            with open(saving_dir + f"cfm_mnist_weights_{args.method}_best_validation_trial_yangsong_{yang_song_model}.pickle", "wb") as handle:
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
    parser.add_argument("--epochs", "-ep", type=int, default=100, help="epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=256, help="Batch size")
    parser.add_argument("--method", "-mt", type=str, default="CFMv2", help="Type of Flow matching we use")
    parser.add_argument("--dataset_path", "-data_path", type=str, default="data/", help="Folder for the datasets")

    args = parser.parse_args()
    main(args)
