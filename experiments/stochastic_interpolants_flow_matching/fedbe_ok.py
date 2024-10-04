"""
File where we train a simple model to learn a velocity filed to go from the
distribution fedbe to ok.

This is inspired by https://github.com/helibenhamu/GeMSS_flow_matching/
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax
import torch
import math
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from jax import random
from matplotlib import pyplot as plt
import argparse
from cfm_jax.models.models import MLP
from cfm_jax.datautils import sample_points_from_mask
from cfm_jax.utils import split_key, plot_data
from cfm_jax.losses import cfm_loss
import optax
from cfm_jax.optimal_transport import OTPlanSampler
from tqdm import tqdm


def main(args):
    seed = args.seed
    prng_key = jax.random.PRNGKey(seed)

    OT_couplings = True

    ## other params passed through the argparse
    n_training_points = args.n_training_points
    epochs = args.epochs
    batch_size = args.batch_size

    # Load the mask
    fedbe_mask = Image.open(r"masks/fedbe.png")
    fedbe_mask = jnp.array(fedbe_mask)
    fedbe_mask = (fedbe_mask == 255)[:, :, 0]

    # Load the mask
    ok_mask = Image.open(r"masks/ok.png")
    ok_mask = jnp.array(ok_mask)
    ok_mask = (ok_mask == 255)[:, :, 0]

    # I can create the datasets
    key_p0, key_p1, prng_key = split_key(prng_key, num=3)

    p0_samples = sample_points_from_mask(ok_mask, num_samples=n_training_points, key=key_p0)
    p1_samples = sample_points_from_mask(fedbe_mask, num_samples=n_training_points, key=key_p1)

    ## create dataloader
    p0_samples = torch.from_numpy(np.array(p0_samples)).float()
    p1_samples = torch.from_numpy(np.array(p1_samples)).float()

    # I can plot the data
    plot_data(p0_samples)
    plot_data(p1_samples)

    train_loader_p0 = DataLoader(p0_samples, batch_size=batch_size, shuffle=True)
    train_loader_p1 = DataLoader(p1_samples, batch_size=batch_size, shuffle=True)

    model = MLP(act_fn=flax.linen.swish, output_dim=2, hidden_dim=512, num_layers=5)

    # we have to init the model
    key_random_t, key_random_xt, key_model_init, prng_key = split_key(prng_key, num=4)
    random_t = jax.random.normal(key_random_t, shape=(batch_size, 1))
    random_xt = jax.random.normal(key_random_xt, shape=(batch_size, 2))

    params_dict = model.init(key_model_init, random_xt, random_t)
    params_vec, unflatten = jax.flatten_util.ravel_pytree(params_dict)
    n_params = len(params_vec)
    print(f"Number of parameters: {n_params}")

    # we have to init the optimizer
    lr = 0.001
    optim = optax.adam(lr)
    opt_state = optim.init(params_dict["params"])

    # # function to apply the model
    model_apply = lambda p, x, t: model.apply({"params": p}, x, t)

    # loss function definition
    conditional_flow_matching_loss = cfm_loss(model_apply)
    loss_grad_fn = jax.jit(jax.value_and_grad(conditional_flow_matching_loss, argnums=0))

    if args.method == "OT-CFM":
        OT_coupling = OTPlanSampler(method=args.ot_method)

    ## simple training loop
    for ep in tqdm(range(epochs), desc="Training"):
        ep_loss = 0
        ep_num_elem = 0
        for batch_p0, batch_p1 in zip(train_loader_p0, train_loader_p1):

            if OT_couplings:
                key_t, key_OT_sampler, prng_key = split_key(prng_key, num=3)
            else:
                key_t, prng_key = split_key(prng_key, num=2)

            t = jax.random.uniform(key_t, shape=(batch_p0.shape[0], 1))
            # right now we have independent couplings
            batch_p0 = jnp.array(batch_p0.numpy())
            batch_p1 = jnp.array(batch_p1.numpy())

            if OT_couplings:
                # print("Batch shape before OT couplings")
                # print(batch_p0.shape)
                # print(batch_p1.shape)
                batch_p0, batch_p1 = OT_coupling.sample_plan(
                    x0=batch_p0, x1=batch_p1, key=key_OT_sampler, replace=True
                )
                # print("Batch shape after OT couplings")
                # print(batch_p0.shape)
                # print(batch_p1.shape)
            xt = (1 - t) * batch_p0 + t * batch_p1
            ut = batch_p1 - batch_p0

            ## TODO: don't like this. I will change it
            loss, grad = loss_grad_fn(params_dict["params"], xt, t, ut)
            params_updates, opt_state = optim.update(grad, opt_state)
            params_dict["params"] = optax.apply_updates(params_dict["params"], params_updates)

            ep_loss += loss * batch_p0.shape[0]
            ep_num_elem += batch_p0.shape[0]

        if ep % 50 == 0:
            print(f"finished {ep} epoch, avg loss trainin: {ep_loss/ep_num_elem}")

    ## at the end we cna check if the model learnt something
    key_samples, prng_key = split_key(prng_key, num=2)

    x0_samples = sample_points_from_mask(ok_mask, num_samples=5000, key=key_samples)
    t = jnp.zeros((5000, 1))

    # number of discretization steps
    N = 100
    xt = x0_samples

    # sampling involves discretizing and solving the ODE
    for i in range(N):
        pred = model_apply(params_dict["params"], xt, t)
        xt = xt + pred * (1 / N)
        t = t + 1 / N

    # plt.scatter(xt[:, 0], xt[:, 1])
    # plt.axis("equal")
    # plt.title(f"Samples from x1")
    # plt.show()
    plot_data(xt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic interpolants Fedbe-OK")
    parser.add_argument("--seed", "-s", type=int, default=1, help="seed")
    parser.add_argument("--n_training_points", "-npoints", type=int, default=4000, help="Number of training points")
    parser.add_argument("--epochs", "-ep", type=int, default=500, help="epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=256, help="Batch size")
    parser.add_argument("--method", "-mt", type=str, default="CFMv2", help="Type of Flow matching we use")
    parser.add_argument(
        "--ot_method", "-ot_mt", type=str, default="exact", help="Method to use to compute OT function"
    )
    args = parser.parse_args()
    main(args)
