# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""LTX2 Video Autoencoder Architecture."""

from typing import Any

from max import functional as F
from max import random
from max.driver import Device
from max.dtype import DType
from max.graph import TensorType
from max.graph.weights import Weights
from max.nn import (
    Conv3d,
    Dropout,
    Identity,
    LayerNorm,
    Module,
    ModuleList,
    Sequential,
)
from max.pipelines.lib import SupportedEncoding
from max.tensor import Tensor

from ..embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from .model import BaseAutoencoderModel
from .model_config import AutoencoderKLLTX2VideoConfig


class PerChannelRMSNorm(Module[[Tensor], Tensor]):
    """Per-channel RMS normalization layer."""

    def __init__(self, channel_dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean_sq = F.mean(x**2, axis=self.channel_dim)
        rms = F.sqrt(mean_sq + self.eps)
        return x / rms


class LTX2VideoCausalConv3d(Module[[Tensor, bool], Tensor]):
    """Causal or non-causal 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",  # Placeholder for padding logic
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = (
            stride if isinstance(stride, tuple) else (stride, stride, stride)
        )

        # Spatial padding (height and width)
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        # Padding for [depth, height, width]
        padding = (0, height_pad, width_pad)

        self.conv = Conv3d(
            kernel_size=self.kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            num_groups=groups,
            padding=padding,
            permute=True,  # LTX2 uses PyTorch format internally but we refactoring to MAX
        )

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        tk = self.kernel_size[0]

        if causal:
            # Pad left (past) for causality
            # x shape: [B, C, D, H, W]
            pad_left = F.tile(x[:, :, :1, :, :], [1, 1, tk - 1, 1, 1])
            x = F.concat([pad_left, x], axis=2)
        else:
            # Pad both sides for non-causal
            pad_left = F.tile(x[:, :, :1, :, :], [1, 1, (tk - 1) // 2, 1, 1])
            pad_right = F.tile(x[:, :, -1:, :, :], [1, 1, (tk - 1) // 2, 1, 1])
            x = F.concat([pad_left, x, pad_right], axis=2)

        return self.conv(x)


class LTX2VideoResnetBlock3d(Module[[Tensor, Tensor | None, bool], Tensor]):
    """3D ResNet block used in LTX2 Video decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = PerChannelRMSNorm()
        self.conv1 = LTX2VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm2 = PerChannelRMSNorm()
        self.dropout = Dropout(dropout)
        self.conv2 = LTX2VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm3: LayerNorm | None = None
        self.conv_shortcut: Module[[Tensor], Tensor] | None = None
        if in_channels != out_channels:
            self.norm3 = LayerNorm(
                in_channels, eps=eps, elementwise_affine=True, use_bias=True
            )
            # LTX 2.0 uses a normal Conv3d here rather than LTXVideoCausalConv3d
            self.conv_shortcut = Conv3d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                permute=True,
            )

        self.per_channel_scale1: Tensor | None = None
        self.per_channel_scale2: Tensor | None = None
        if inject_noise:
            self.per_channel_scale1 = Tensor.constant(Tensor.zeros((in_channels, 1, 1)))
            self.per_channel_scale2 = Tensor.constant(Tensor.zeros((in_channels, 1, 1)))

        self.scale_shift_table: Tensor | None = None
        if timestep_conditioning:
            self.scale_shift_table = Tensor.constant(random.gaussian((4, in_channels)) / in_channels**0.5)

    def forward(
        self,
        inputs: Tensor,
        temb: Tensor | None = None,
        seed: Tensor | None = None,
        causal: bool = True,
    ) -> Tensor:
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states)

        if self.scale_shift_table is not None:
            # LTX2 uses unflatten(1, (4, -1)) + table[None, ..., None, None, None]
            # In MAX, we reshape and broadcast manually.
            # temb: [B, 4*C] -> [B, 4, C, 1, 1, 1]
            temb = temb.reshape((temb.shape[0], 4, -1, 1, 1, 1))
            # table: [4, C] -> [1, 4, C, 1, 1, 1]
            table = self.scale_shift_table.reshape((1, 4, -1, 1, 1, 1))
            temb = temb + table

            shift_1 = temb[:, 0]
            scale_1 = temb[:, 1]
            shift_2 = temb[:, 2]
            scale_2 = temb[:, 3]

            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states, causal=causal)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = random.gaussian(
                spatial_shape, seed=seed, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]

        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class LTX2VideoUpsample3d(Module[[Tensor], Tensor]):
    """Upsampling block for LTX2 Video decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        spatio_temporal: bool = True,
        residual: bool = True,
        factor: int = 2,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.factor = factor
        self.spatio_temporal = spatio_temporal
        self.residual = residual

        self.conv = Conv3d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            padding=1,
            permute=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Upsample using interpolation
        # x shape: [B, C, D, H, W]
        # if spatio_temporal: [B, C, D*2, H*2, W*2]
        # else: [B, C, D, H*2, W*2]

        if self.spatio_temporal:
            # 3D interpolation
            x = F.interpolate(
                x,
                scale_factor=int(self.factor),
                mode="nearest",
            )
        else:
            # Spatial-only interpolation (keep depth dimension)
            # F.interpolate handles tuple scale_factor
            x = F.interpolate(
                x,
                scale_factor=(1.0, float(self.factor), float(self.factor)),
                mode="nearest",
            )

        return self.conv(x)


class LTX2VideoDecoderBlock3D(Module[[Tensor, Tensor | None, bool], Tensor]):
    """A single block of the LTX2 Video decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        spatio_temporal_scaling: bool = False,
        inject_noise: bool = False,
        upsample_factor: int = 2,
        upsample_residual: bool = True,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        resnets = []
        for i in range(num_layers):
            block_in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=block_in_channels,
                    out_channels=out_channels,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = Sequential(*resnets)

        self.upsampler: LTX2VideoUpsample3d | None = None
        if spatio_temporal_scaling:
            self.upsampler = LTX2VideoUpsample3d(
                in_channels=out_channels,
                out_channels=out_channels,
                spatio_temporal=True,
                residual=upsample_residual,
                factor=upsample_factor,
            )

    def forward(
        self, x: Tensor, temb: Tensor | None = None, causal: bool = True
    ) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x, temb, causal=causal)

        if self.upsampler is not None:
            x = self.upsampler(x)

        return x


class LTX2VideoDecoder3d(Module[[Tensor, Tensor | None, bool], Tensor]):
    """LTX2 Video Decoder module."""

    def __init__(self, config: AutoencoderKLLTX2VideoConfig) -> None:
        super().__init__()
        self.config = config

        # Initial projection
        self.conv_in = Conv3d(
            kernel_size=3,
            in_channels=config.latent_channels,
            out_channels=config.decoder_block_out_channels[0],
            padding=1,
            permute=True,
        )

        # Timestep conditioning embedding
        self.time_embed: PixArtAlphaCombinedTimestepSizeEmbeddings | None = None
        if config.timestep_conditioning:
            # We'll use the one we just created in architectures/embeddings.py
            self.time_embed = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=config.decoder_block_out_channels[0] * 4,
                size_emb_dim=config.decoder_block_out_channels[0],
                use_additional_conditions=False,
            )

        # Middle block
        curr_channels = config.decoder_block_out_channels[0]
        self.mid = Module()
        self.mid.block_1 = LTX2VideoResnetBlock3d(
            in_channels=curr_channels,
            out_channels=curr_channels,
            timestep_conditioning=config.timestep_conditioning,
            spatial_padding_mode=config.decoder_spatial_padding_mode,
        )

        # NOTE: LTX2 Video VAE doesn't typically have attention in the mid block,
        # but the layer hierarchy usually includes it as Identity if using diffusers pattern.
        # Based on user feedback, we'll ensure the hierarchy matches.
        self.mid.attn_1 = Identity()

        self.mid.block_2 = LTX2VideoResnetBlock3d(
            in_channels=curr_channels,
            out_channels=curr_channels,
            timestep_conditioning=config.timestep_conditioning,
            spatial_padding_mode=config.decoder_spatial_padding_mode,
        )

        # Decoder (Upsampling) blocks
        up_blocks = []
        for i, out_mult in enumerate(config.decoder_block_out_channels):
            up_blocks.append(
                LTX2VideoDecoderBlock3D(
                    in_channels=curr_channels,
                    out_channels=out_mult,
                    num_layers=config.decoder_layers_per_block[i],
                    spatio_temporal_scaling=config.decoder_spatio_temporal_scaling[
                        i
                    ]
                    if i < len(config.decoder_spatio_temporal_scaling)
                    else False,
                    inject_noise=config.decoder_inject_noise[i]
                    if i < len(config.decoder_inject_noise)
                    else False,
                    upsample_factor=config.upsample_factor[i]
                    if i < len(config.upsample_factor)
                    else 2,
                    upsample_residual=config.upsample_residual[i]
                    if i < len(config.upsample_residual)
                    else True,
                    timestep_conditioning=config.timestep_conditioning,
                    spatial_padding_mode=config.decoder_spatial_padding_mode,
                )
            )
            curr_channels = out_mult
        self.up_blocks = ModuleList(up_blocks)

        # Final output projection
        self.norm_out = PerChannelRMSNorm(eps=config.resnet_norm_eps)
        self.conv_out = Conv3d(
            kernel_size=3,
            in_channels=curr_channels,
            out_channels=config.out_channels,
            padding=1,
            permute=True,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the decoder model."""
        latent_type = TensorType(
            self.config.dtype,
            shape=[
                "batch_size",
                self.config.latent_channels,
                "latent_num_frames",
                "latent_height",
                "latent_width",
            ],
            device=self.config.device,
        )

        if self.config.timestep_conditioning:
            timestep_type = TensorType(
                DType.float32,
                shape=["batch_size"],
                device=self.config.device,
            )
            return (latent_type, timestep_type)

        return (latent_type,)

    def forward(
        self, z: Tensor, timestep: Tensor | None = None, causal: bool = True
    ) -> Tensor:
        x = self.conv_in(z)

        temb = None
        if self.time_embed is not None and timestep is not None:
            # Simplified conditioning for autoencoder
            # In LTX2, conditioning is often just zeros or based on fake resolution
            fake_res = Tensor.constant(0.0, (z.shape[0],))
            temb = self.time_embed(
                timestep, fake_res, fake_res, z.shape[0], z.dtype
            )

        x = self.mid.block_1(x, temb, causal=causal)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x, temb, causal=causal)

        for block in self.up_blocks:
            x = block(x, temb, causal=causal)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


class AutoencoderKLLTX2Video(Module[[Tensor, Tensor | None, bool], Tensor]):
    """Refactored LTX2 Video Autoencoder (Decode-only)."""

    def __init__(self, config: AutoencoderKLLTX2VideoConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = LTX2VideoDecoder3d(config)

    def forward(
        self,
        z: Tensor,
        timestep: Tensor | None = None,
        causal: bool | None = None,
    ) -> Tensor:
        if causal is None:
            causal = self.config.decoder_causal
        return self.decoder(z, timestep, causal=causal)


class AutoencoderKLLTX2VideoModel(BaseAutoencoderModel):
    """ComponentModel wrapper for LTX2 Video Autoencoder."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config=config,
            encoding=encoding,
            devices=devices,
            weights=weights,
            config_class=AutoencoderKLLTX2VideoConfig,
            autoencoder_class=AutoencoderKLLTX2Video,
        )
