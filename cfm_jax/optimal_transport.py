"""
Here we weill translate to JAX the method proposed by Tong et al. 
We will mostly focus on the minibatch-OT coupling.

Mostly inspired by https://github.com/atong01/conditional-flow-matching
just tryng to keep the basic function we need
"""

import jax
import jax.numpy as jnp
import numpy as np
import warnings
from functools import partial
import ot as pot
from cfm_jax.utils import euclidian_distance

# import jax.scipy.spatial


class OTPlanSampler:
    """
    Simplify version from https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py
    """

    def __init__(self, method: str, reg: float = 0.05, reg_m: float = 1.0, normalize_cost: bool = False) -> None:

        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=1)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost

    def get_map(self, x0, x1):
        """
        Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        These corresponds to just compute the OT map. So we have to define uniform weights for
        both x0 and x1. Compute the cost matrix M squared Euclidean cost and then run the OT function.
        """

        weights_x0 = pot.unif(x0.shape[0])
        weights_x1 = pot.unif(x1.shape[0])

        if len(x0.shape) > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if len(x1.shape) > 2:
            x1 = x1.reshape(x1.shape[0], -1)

        # M = jax.scipy.spatial.distance.cdist(x0, x1) ** 2
        M = euclidian_distance(x0, x1) ** 2

        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches

        ## TODO: HERE I AM GOING FROM JNP TO NP --> IS THERE A BETTER WAY?
        ## ALSO SHOULD I DO EVERYTHING IN NUMPY OR JNP?
        p = self.ot_fn(weights_x0, weights_x1, np.array(M))

        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=True, key=None):
        p = pi.flatten()
        p = p / p.sum()

        choices = jax.random.choice(key, pi.shape[0] * pi.shape[1], shape=(batch_size,), p=p, replace=replace)
        return jnp.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=True, key=None):
        ## NOTE: HOW DOES REPLACEMENT AFFECT THE RESULTS?
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace, key=key)
        return x0[i], x1[j]
