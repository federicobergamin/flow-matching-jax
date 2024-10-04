from typing import Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Callable, Optional, Iterable
from einops import rearrange
import math

from flax.linen.linear import canonicalize_padding, _conv_dimension_numbers


class MLP(nn.Module):
    act_fn: Callable
    output_dim: int
    hidden_dim: int = 64
    num_layers: int = 3

    @nn.compact
    def __call__(self, x, t):
        # here we have to concatenate t
        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate((x, t), axis=1)
        # here we put the first and the N-1 hidden layers
        for _ in range(self.num_layers - 1):
            x = nn.Dense(self.hidden_dim)(x)
            x = self.act_fn(x)
        # and in the end the output layer
        x = nn.Dense(self.output_dim)(x)

        return x
    

class ConvNet(nn.Module):

    act_fn: Callable
    channels: list ## these are the channels of each conv layer, e.g. [32, 64, 128]
    num_classes: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels[0], kernel_size=self.kernel_size, padding='SAME')(x)
        x = self.act_fn(x)
        for i in range(1, len(self.channels)):
            x = nn.Conv(self.channels[i], kernel_size=self.kernel_size, padding='SAME')(x)
            x = self.act_fn(x)
        # now we should have the output of the last conv layer
        # we now need to flatten the output
        x = x.reshape((x.shape[0], -1))
        # and then process it with a Dense layer
        x = nn.Dense(self.num_classes)(x) 
        # returning the logits    
        return x
        