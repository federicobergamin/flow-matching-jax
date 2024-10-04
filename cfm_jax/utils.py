import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def split_key(pnrg_key, num=1):
    new_pnrg_key = jax.random.split(pnrg_key, num=num)
    return new_pnrg_key


def plot_data(xt_, xlim=(-6, 6), ylim=(-6, 6)):
    plt.scatter(xt_[:, 0], xt_[:, 1], s=1)
    # plt.axis("equal")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.show()


def euclidian_distance(x0, x1):
    norm_x0 = jnp.sum(x0**2, axis=1, keepdims=True)
    norm_x1 = jnp.sum(x1**2, axis=1, keepdims=True)

    distances = norm_x0 + norm_x1.T - 2 * jnp.dot(x0, x1.T)

    return jnp.sqrt(distances)


#### the next two functions are taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial11/NF_image_modeling.html
def image_to_numpy(img):
    img = np.array(img, dtype=np.int32)
    img = img[..., None]  # Make image [28, 28, 1]
    return img


def image_to_numpy_zero_one_interval(img):
    """
    This return the same values that are return by the torchvision.transforms.ToTensor()
    when we load the MNIST dataset in PyTorch
    """
    img = np.array(img, dtype=np.float32)
    img = img[..., None] / 255  # Make image [28, 28, 1]
    return img


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
