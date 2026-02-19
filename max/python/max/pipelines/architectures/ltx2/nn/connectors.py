# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import math

import max.functional as F
from max import nn, random
from max.driver import Device
from max.dtype import DType
from max.graph import TensorType
from max.nn.activations import FeedForward
from max.tensor import Tensor

from ..ltx2 import LTX2Attention
from ..model_config import LTX2TextConnectorsConfig


class LTX2RotaryPosEmbed1d(
    nn.Module[[int, int, Device], tuple[Tensor, Tensor]]
):
    """
    1D rotary positional embeddings (RoPE) for the LTX 2.0 text encoder connectors.
    """

    def __init__(
        self,
        dim: int,
        base_seq_len: int = 4096,
        theta: float = 10000.0,
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ):
        if rope_type not in ("interleaved", "split"):
            raise ValueError(
                f"{rope_type=} not supported. Choose between 'interleaved' and 'split'."
            )

        self.dim = dim
        self.base_seq_len = base_seq_len
        self.theta = theta
        self.double_precision = double_precision
        self.rope_type = rope_type
        self.num_attention_heads = num_attention_heads

    def forward(
        self,
        batch_size: int,
        pos: int,
        device: Device,
    ) -> tuple[Tensor, Tensor]:
        # 1. Get 1D position ids
        grid_1d = Tensor.arange(pos, dtype=DType.float32, device=device)
        # Get fractional indices relative to self.base_seq_len
        grid_1d = grid_1d / self.base_seq_len
        grid = F.tile(
            grid_1d.unsqueeze(0), (batch_size, 1)
        )  # [batch_size, seq_len]

        # 2. Calculate 1D RoPE frequencies
        num_rope_elems = 2  # 1 (because 1D) * 2 (for cos, sin) = 2
        freqs_dtype = DType.float64 if self.double_precision else DType.float32
        steps = self.dim // num_rope_elems
        linspace = Tensor.arange(steps, dtype=freqs_dtype, device=device)
        if steps > 1:
            linspace = linspace / float(steps - 1)

        pow_indices = F.pow(self.theta, linspace)
        freqs = (pow_indices * math.pi / 2.0).cast(DType.float32)

        # 3. Matrix-vector outer product between pos ids of shape (batch_size, seq_len) and freqs vector of shape
        # (self.dim // 2,).
        freqs = (
            grid.unsqueeze(-1) * 2 - 1
        ) * freqs  # [B, seq_len, self.dim // 2]

        # 4. Get real, interleaved (cos, sin) frequencies, padded to self.dim
        if self.rope_type == "interleaved":
            cos_freqs = F.repeat_interleave(F.cos(freqs), 2, axis=-1)
            sin_freqs = F.repeat_interleave(F.sin(freqs), 2, axis=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = Tensor.ones_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                sin_padding = Tensor.zeros_like(
                    sin_freqs[:, :, : self.dim % num_rope_elems]
                )
                cos_freqs = F.concat([cos_padding, cos_freqs], axis=-1)
                sin_freqs = F.concat([sin_padding, sin_freqs], axis=-1)

        elif self.rope_type == "split":
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = F.cos(freqs)
            sin_freq = F.sin(freqs)

            if pad_size != 0:
                cos_padding = Tensor.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = Tensor.zeros_like(sin_freq[:, :, :pad_size])

                cos_freq = F.concat([cos_padding, cos_freq], axis=-1)
                sin_freq = F.concat([sin_padding, sin_freq], axis=-1)

            # Reshape freqs to be compatible with multi-head attention
            b = cos_freq.shape[0]
            t = cos_freq.shape[1]

            cos_freq = cos_freq.reshape((b, t, self.num_attention_heads, -1))
            sin_freq = sin_freq.reshape((b, t, self.num_attention_heads, -1))

            cos_freqs = cos_freq.transpose(1, 2)  # (B,H,T,D//2)
            sin_freqs = sin_freq.transpose(1, 2)  # (B,H,T,D//2)

        return cos_freqs, sin_freqs


class LTX2TransformerBlock1d(
    nn.Module[[Tensor, Tensor | None, Tensor | None], Tensor]
):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        activation_fn: str = "gelu-approximate",
        eps: float = 1e-6,
        rope_type: str = "interleaved",
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps, elementwise_affine=False)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            rope_type=rope_type,
        )

        self.norm2 = nn.RMSNorm(dim, eps, elementwise_affine=False)
        self.ff = FeedForward(dim, activation_fn=activation_fn)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        rotary_emb: Tensor | None = None,
    ) -> Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_hidden_states = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        ff_hidden_states = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_hidden_states

        return hidden_states


class LTX2ConnectorTransformer1d(
    nn.Module[[Tensor, Tensor | None, float], tuple[Tensor, Tensor]]
):
    """
    A 1D sequence transformer for modalities such as text.

    In LTX 2.0, this is used to process the text encoder hidden states for each of the video and audio streams.
    """

    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 128,
        num_layers: int = 2,
        num_learnable_registers: int | None = 128,
        rope_base_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        eps: float = 1e-6,
        causal_temporal_positioning: bool = False,
        rope_type: str = "interleaved",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning

        self.num_learnable_registers = num_learnable_registers
        self.learnable_registers = None
        if num_learnable_registers is not None:
            init_registers = (
                random.uniform((num_learnable_registers, self.inner_dim)) * 2.0
                - 1.0
            )
            self.learnable_registers = Tensor.constant(init_registers)

        self.rope = LTX2RotaryPosEmbed1d(
            self.inner_dim,
            base_seq_len=rope_base_seq_len,
            theta=rope_theta,
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTX2TransformerBlock1d(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rope_type=rope_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.RMSNorm(
            self.inner_dim, eps, elementwise_affine=False
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_binarize_threshold: float = -9000.0,
    ) -> tuple[Tensor, Tensor]:
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        # attention_mask shape: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
        batch_size = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])

        # 1. Replace padding with learned registers, if using
        if self.learnable_registers is not None:
            num_register_repeats = seq_len // self.num_learnable_registers
            registers = F.tile(
                self.learnable_registers, (num_register_repeats, 1)
            )  # [seq_len, inner_dim]

            # Graph-safe register replacement: use masked blending
            # instead of dynamic boolean indexing / per-batch loops.
            binary_attn_mask = (
                attention_mask >= attn_mask_binarize_threshold
            ).cast(DType.int32)
            if binary_attn_mask.rank == 4:
                binary_attn_mask = binary_attn_mask.squeeze(1).squeeze(
                    1
                )  # [B, 1, 1, L] --> [B, L]

            # Expand mask for broadcasting: [B, L] -> [B, L, 1]
            mask_3d = binary_attn_mask.unsqueeze(-1).cast(DType.float32)

            # Blend: keep valid tokens where mask=1, use registers where mask=0
            hidden_states = (
                mask_3d * hidden_states + (1.0 - mask_3d) * registers
            )

            # Overwrite attention_mask with an all-zeros mask if using registers.
            attention_mask = Tensor.zeros_like(attention_mask)

        # 2. Calculate 1D RoPE positional embeddings
        rotary_emb = self.rope(batch_size, seq_len, device=hidden_states.device)

        # 3. Run 1D transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                rotary_emb=rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states)

        return hidden_states, attention_mask


class LTX2TextConnectors(
    nn.Module[[Tensor, Tensor, bool], tuple[Tensor, Tensor, Tensor]]
):
    """
    Text connector stack used by LTX 2.0 to process the packed text encoder hidden states for both the video and audio
    streams.
    """

    def __init__(
        self,
        config: LTX2TextConnectorsConfig,
    ):
        super().__init__()
        self.config = config

        self.text_proj_in = nn.Linear(
            config.caption_channels * config.text_proj_in_factor, config.caption_channels, bias=False
        )
        self.video_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=config.video_connector_num_attention_heads,
            attention_head_dim=config.video_connector_attention_head_dim,
            num_layers=config.video_connector_num_layers,
            num_learnable_registers=config.video_connector_num_learnable_registers,
            rope_base_seq_len=config.connector_rope_base_seq_len,
            rope_theta=config.rope_theta,
            rope_double_precision=config.rope_double_precision,
            causal_temporal_positioning=config.causal_temporal_positioning,
            rope_type=config.rope_type,
        )
        self.audio_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=config.audio_connector_num_attention_heads,
            attention_head_dim=config.audio_connector_attention_head_dim,
            num_layers=config.audio_connector_num_layers,
            num_learnable_registers=config.audio_connector_num_learnable_registers,
            rope_base_seq_len=config.connector_rope_base_seq_len,
            rope_theta=config.rope_theta,
            rope_double_precision=config.rope_double_precision,
            causal_temporal_positioning=config.causal_temporal_positioning,
            rope_type=config.rope_type,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model."""
        text_encoder_hidden_states_type = TensorType(
            self.config.dtype,
            shape=[
                "batch_size",
                "text_seq_len",
                self.config.caption_channels * self.config.text_proj_in_factor,
            ],
            device=self.config.device,
        )
        attention_mask_type = TensorType(
            self.config.dtype,
            shape=["batch_size", "text_seq_len"],
            device=self.config.device,
        )
        return (text_encoder_hidden_states_type, attention_mask_type)

    def forward(
        self,
        text_encoder_hidden_states: Tensor,
        attention_mask: Tensor,
        additive_mask: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Convert to additive attention mask, if necessary
        if not additive_mask:
            text_dtype = text_encoder_hidden_states.dtype
            # Use float32 for arithmetic to avoid promotion issues with bool/uint8 masks
            mask_float = attention_mask.cast(DType.float32)
            attention_mask = (mask_float - 1.0).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            )
            attention_mask = (
                attention_mask.cast(text_dtype) * DType.finfo(text_dtype).max
            )

        text_encoder_hidden_states = self.text_proj_in(
            text_encoder_hidden_states
        )

        video_text_embedding, new_attn_mask = self.video_connector(
            text_encoder_hidden_states, attention_mask
        )

        attn_mask = (new_attn_mask < 1e-6).cast(DType.int64)
        attn_mask = attn_mask.reshape(
            (video_text_embedding.shape[0], video_text_embedding.shape[1], 1)
        )
        video_text_embedding = video_text_embedding * attn_mask
        new_attn_mask = attn_mask.squeeze(-1)

        audio_text_embedding, _ = self.audio_connector(
            text_encoder_hidden_states, attention_mask
        )

        return video_text_embedding, audio_text_embedding, new_attn_mask
