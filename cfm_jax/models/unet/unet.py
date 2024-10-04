from typing import Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Callable, Optional, Iterable, Any, Sequence, Union, Tuple
from einops import rearrange
import math

# from cfm_jax.models.unet.nn import get_timestep_embedding

from flax.linen.linear import canonicalize_padding, _conv_dimension_numbers
from cfm_jax.models.unet.nn import SinusoidalPosEmb, ResnetBlock, AttnBlock, Downsample, Upsample
from jax import flatten_util


class Unet(nn.Module):
    embedding_dim: int
    first_conv_channels: Optional[int] = None  # if None, same as dim
    output_channels: Optional[int] = None
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    resnet_block_groups: int = 8
    learned_variance: bool = False
    dtype: Any = jnp.float32
    standartize_conv: bool = False

    """
    embedding_dim: embedding dim
    first_conv_channels: number of channels of the results of first convolution
    output_channels: output channels
    dim_mults: (1,2,4) for FashionMnist --> define the resolutions
    resnet_block_groups: group norm stuff (check what this is doing)
    resnet_block_groups: if we use conv or standardize conv
    """

    @nn.compact
    def __call__(self, x, time):
        B, H, W, C = x.shape

        init_dim = self.embedding_dim if self.first_conv_channels is None else self.first_conv_channels
        hs = []
        h = nn.Conv(features=init_dim, kernel_size=(7, 7), padding=3, name="init.conv_0", dtype=self.dtype)(x)

        hs.append(h)
        # use sinusoidal embeddings to encode timesteps
        # print("Time shape")
        time_emb = SinusoidalPosEmb(self.embedding_dim, dtype=self.dtype)(time)  # [B. dim]
        # print(time_emb.shape)
        time_emb = nn.Dense(features=self.embedding_dim * 4, dtype=self.dtype, name="time_mlp.dense_0")(time_emb)
        # print(time_emb.shape)
        time_emb = nn.Dense(features=self.embedding_dim * 4, dtype=self.dtype, name="time_mlp.dense_1")(
            nn.gelu(time_emb)
        )  # [B, 4*dim]
        # print(time_emb.shape)

        # downsampling
        num_resolutions = len(self.dim_mults)
        for ind in range(num_resolutions):
            dim_in = h.shape[-1]
            h = ResnetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                standardize_conv=self.standartize_conv,
                name=f"down_{ind}.resblock_0",
            )(h, time_emb)
            hs.append(h)

            h = ResnetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                standardize_conv=self.standartize_conv,
                name=f"down_{ind}.resblock_1",
            )(h, time_emb)
            h = AttnBlock(dtype=self.dtype, name=f"down_{ind}.attnblock_0")(h)
            hs.append(h)

            if ind < num_resolutions - 1:
                h = Downsample(
                    dim=self.embedding_dim * self.dim_mults[ind], dtype=self.dtype, name=f"down_{ind}.downsample_0"
                )(h)

        mid_dim = self.embedding_dim * self.dim_mults[-1]
        h = nn.Conv(
            features=mid_dim, kernel_size=(3, 3), padding=1, dtype=self.dtype, name=f"down_{num_resolutions-1}.conv_0"
        )(h)

        # middle
        h = ResnetBlock(
            dim=mid_dim,
            groups=self.resnet_block_groups,
            dtype=self.dtype,
            standardize_conv=self.standartize_conv,
            name="mid.resblock_0",
        )(h, time_emb)
        h = AttnBlock(use_linear_attention=False, dtype=self.dtype, name="mid.attenblock_0")(h)
        h = ResnetBlock(
            dim=mid_dim,
            groups=self.resnet_block_groups,
            dtype=self.dtype,
            standardize_conv=self.standartize_conv,
            name="mid.resblock_1",
        )(h, time_emb)

        # upsampling
        for ind in reversed(range(num_resolutions)):

            dim_in = self.embedding_dim * self.dim_mults[ind]
            dim_out = self.embedding_dim * self.dim_mults[ind - 1] if ind > 0 else init_dim

            assert h.shape[-1] == dim_in
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            assert h.shape[-1] == dim_in + dim_out
            h = ResnetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                standardize_conv=self.standartize_conv,
                name=f"up_{ind}.resblock_0",
            )(h, time_emb)

            h = jnp.concatenate([h, hs.pop()], axis=-1)
            assert h.shape[-1] == dim_in + dim_out
            h = ResnetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                standardize_conv=self.standartize_conv,
                name=f"up_{ind}.resblock_1",
            )(h, time_emb)
            h = AttnBlock(dtype=self.dtype, name=f"up_{ind}.attnblock_0")(h)

            assert h.shape[-1] == dim_in
            if ind > 0:
                h = Upsample(dim=dim_out, dtype=self.dtype, name=f"up_{ind}.upsample_0")(h)

        h = nn.Conv(features=init_dim, kernel_size=(3, 3), padding=1, dtype=self.dtype, name=f"up_0.conv_0")(h)

        # final
        h = jnp.concatenate([h, hs.pop()], axis=-1)
        assert h.shape[-1] == init_dim * 2

        out = ResnetBlock(
            dim=self.embedding_dim,
            groups=self.resnet_block_groups,
            dtype=self.dtype,
            standardize_conv=self.standartize_conv,
            name="final.resblock_0",
        )(h, time_emb)

        default_out_dim = C * (1 if not self.learned_variance else 2)
        out_dim = default_out_dim if self.output_channels is None else self.output_channels

        return nn.Conv(out_dim, kernel_size=(1, 1), dtype=self.dtype, name="final.conv_0")(out)


if __name__ == "__main__":
    ## here I want to check if something works as expected

    # I have to initialize the UNet model we have above
    # assuming that the input is 28x28x1
    rng = jax.random.PRNGKey(0)

    rng, data_rng, time_rng, init_rng = jax.random.split(rng, num=4)

    image_size = 28
    image_channel = 1
    batch_size = 4
    input_shape = (batch_size, image_size, image_size, image_channel)

    ## NOTE: GET ERROR WITH dim_mults=(1, 2, 4, 8)
    model = Unet(dim=32, out_dim=1, dim_mults=(1, 2, 4))
    # I will create a random input
    x = jax.random.normal(data_rng, input_shape)
    time = jax.random.uniform(time_rng, shape=(batch_size,))

    print(x.shape)
    print(time.shape)

    # checking methods
    # print("trial embedding")
    # trial_time_embedding = get_timestep_embedding(time, 32)
    # print(trial_time_embedding.shape)

    # and I will run the model
    print("initializing the model")
    params_dict = model.init(init_rng, x, time)

    params_vec, unflatten = flatten_util.ravel_pytree(params_dict)
    n_params = len(params_vec)
    print(f"Number of parameters: {n_params}")

    print("Running the model")
    out = model.apply({"params": params_dict["params"]}, x, time)
    print(out.shape)
    # I expect the output to be 28x28x1
    # since the input and output have the same shape
    assert out.shape == (4, 28, 28, 1)
