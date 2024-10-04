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
import torchdyn
from torchdyn.datasets import generate_moons
from cfm_jax.utils import split_key


def sample_points_from_mask(mask, num_samples=1000, sigma=0.05, key=None):
    """
    This function is taken from https://github.com/helibenhamu/GeMSS_flow_matching/
    TODO: add explanation of arguments
    """
    # Get the indices of the pixels that are part of the letters
    letter_indices = jnp.argwhere(mask)
    # print(letter_indices.shape)

    # Uniformly sample points from these indices
    key_index, key_eps = split_key(key, num=2)

    sampled_indices = letter_indices[jax.random.choice(key_index, len(letter_indices), (num_samples,))]
    # print(sampled_indices.shape)
    x = sampled_indices.astype(jnp.float32)
    x = x.at[:, 0].set(-(x[:, 0] - mask.shape[0] / 2) / (mask.shape[0] / 8))
    x = x.at[:, 1].set((x[:, 1] - mask.shape[1] / 2) / (mask.shape[0] / 8))
    x = np.flip(x, 1)
    # x = torch.tensor(x.astype('float32'))
    # x = x + sigma * torch.randn_like(x)

    # I have to sample some noise in jax
    # x = x + sigma * torch.randn_like(x)

    # now I can sample from
    eps = jax.random.normal(key_eps, shape=x.shape)
    x = x + sigma * eps

    return x


def sample_eight_circle_normal_dist(n_points, dimension, scale=1, var=1):
    """
    Taken from https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/utils.py
    """
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dimension), math.sqrt(var) * torch.eye(dimension)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    # centers = centers * scale
    # noise = jax.random.multivariate_normal(key, mean=jnp.zeros(dimension), cov=jnp.sqrt(var)*jnp.eye(dimension), shape=(n_points,))

    noise = m.sample((n_points,))
    multi = torch.multinomial(torch.ones(8), n_points, replacement=True)
    # multi =
    data = []
    for i in range(n_points):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data.numpy()


def generate_moons(n_samples: int = 100, noise: float = 1e-4, key=None):
    """Creates a *moons* dataset of `n_samples` datasets points.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float

    Taken from https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/utils.py
    and adapted ot jax.
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    outer_circ_x = jnp.cos(np.linspace(0, jnp.pi, n_samples_out))
    outer_circ_y = jnp.sin(np.linspace(0, jnp.pi, n_samples_out))
    inner_circ_x = 1 - jnp.cos(np.linspace(0, jnp.pi, n_samples_in))
    inner_circ_y = 1 - jnp.sin(np.linspace(0, jnp.pi, n_samples_in)) - 0.5

    X = np.vstack([np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)])

    if noise is not None:
        # X += np.random.rand(n_samples, 1) * noise
        X += jax.random.uniform(key, shape=(n_samples, 1)) * noise

    # TODO: Y SHOULD BE LONG
    return X, y


def sample_moons(n, key=None):
    x0, _ = generate_moons(n, noise=0.2, key=key)
    return x0 * 3 - 1


def sample_8gaussians(n_points: int):
    return sample_eight_circle_normal_dist(n_points, 2, scale=5, var=0.1)


def sample_gaussian(n_points, dimension, mean=0, var=1, key=None):
    """
    Get samples from a Gaussian distribution. We can choose the mean and the variance
    """

    if dimension == 1:
        # steandard Gaussian in 1D
        noise = jax.random.normal(key, shape=(n_points,))
        samples = mean * jnp.ones((n_points)) + jnp.sqrt(var) * noise
    else:
        samples = jax.random.multivariate_normal(
            key, mean=mean * np.ones((dimension,)), cov=jnp.sqrt(var) * jnp.eye(dimension), shape=(n_points,)
        )

    return samples


def get_dataset(dataset_name, n_points, mask=None, pnrg_key=None):
    assert dataset_name in ["mask", "moons", "8circles", "gaussian"]

    if dataset_name == "mask":
        return sample_points_from_mask(mask, num_samples=n_points, key=pnrg_key)
    elif dataset_name == "moons":
        return sample_moons(n_points, pnrg_key)
    elif dataset_name == "8circles":
        return sample_8gaussians(n_points)
    elif dataset_name == "Gaussian":
        return sample_gaussian(n_points, key=pnrg_key)
    else:
        raise NotImplementedError("dataset_name should be in ['mask', 'moons', '8circles', 'gaussian']")
