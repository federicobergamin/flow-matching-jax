"""
Maybe I should create a class where I have the different mean
and sigma function of our Gaussian probability path
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax


def reshape_t_as_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (len(x.shape) - 1)))


class ConditionalFlowMatchingModel:

    def __init__(self, sigma=0.0, method="CFM", OT_plan_sampler=None) -> None:
        """
        We will implement the following method for now:
        - CFM: original way proposed by Lipman et al.
        - CFMv2: CFM by Tong without the OT coupling (allows different distributions for p(x0))
        - OT-CFM: minibatches OT sampling proposed by Tong et al.
        """
        ## certain methods require a sigma
        self.sigma = sigma
        if method not in ["CFM", "CFMv2", "OT-CFM", "Riem-CFM"]:
            raise NotImplementedError(f"You are passing {method} as method args but this is not implemented yet")

        self.method = method

        if method == "OT-CFM" and OT_plan_sampler is None:
            raise ValueError(
                "You are selecting an OT based flow matching approach but not providing a OT plan sampler"
            )

        self.OT_plan_sampler = OT_plan_sampler

    def compute_mu_t(self, x0=None, x1=None, t=None):

        if self.method == "CFM":
            t = reshape_t_as_x(t, x1)
            return t * x1
        elif self.method in ["OT-CFM", "CFMv2"]:
            t = reshape_t_as_x(t, x0)
            return (1 - t) * x0 + t * x1
        else:
            # we are at the "Riem-CFM"
            raise NotImplementedError("Will be there")

    def compute_sigma_t(self, t=None):
        if self.method == "CFM":
            return 1 - (1 - self.sigma) * t
        else:
            return self.sigma

    def sample_xt(self, x0, x1, t, key_epsilon=None, key_OT_sampler=None, replace=True):
        """
        Method to compute xt
        """

        if self.method == "OT-CFM":
            # psrint("Here")
            # I have to run the OT coupling first
            x0, x1 = self.OT_plan_sampler.sample_plan(x0=x0, x1=x1, key=key_OT_sampler, replace=replace)

        # sampling the mean_t
        mean_t = self.compute_mu_t(x0, x1, t)

        # sampling sigma
        sigma_t = self.compute_sigma_t(t)

        # sampling xt from N(xt|mean_t, sigma**2 I )
        eps = jax.random.normal(key_epsilon, shape=mean_t.shape)
        # print(f"eps: {eps.shape}")
        # here sigma_t is a scalar, so I have to pad it
        sigma_t = reshape_t_as_x(sigma_t, mean_t)
        # print(f"mean_t: {mean_t.shape}")
        # print(f"sigma_t: {sigma_t.shape}")
        xt = mean_t + sigma_t * eps
        # print(f"xt: {xt.shape}")
        if self.method == "OT-CFM":
            return xt, x0, x1
        else:
            return xt

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Method to compute ut
        """
        if self.method == "CFM":
            t = reshape_t_as_x(t, x1)
            # sigma = reshape_t_as_x(self.sigma, x1)
            return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)
        else:
            return x1 - x0
