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

import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

import max.experimental.functional as F
import numpy as np
import numpy.typing as npt
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import random
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines import PixelContext
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    PixelModelInputs,
)
from max.pipelines.lib.interfaces.diffusion_pipeline import max_compile
from tqdm import tqdm
from transformers import Gemma3ForConditionalGeneration

from ..autoencoders import (
    AutoencoderKLLTX2AudioModel,
    AutoencoderKLLTX2VideoModel,
)
from .model import (
    LTX2TextConnectorsModel,
    LTX2TransformerModel,
    LTX2VocoderModel,
)

logger = logging.getLogger("max.pipelines")


@dataclass(kw_only=True)
class LTX2ModelInputs(PixelModelInputs):
    """A class representing inputs for the LTX2 model.

    This class encapsulates the input tensors required for the LTX2 model execution
    and extends the generic PixelModelInputs used by other diffusion pipelines.

    Core scalar and array fields such as height, width, timesteps, sigmas, and
    latents are inherited from PixelModelInputs and populated via
    `from_context(PixelContext)` in `prepare_inputs`.

    Only LTX2-specific optional fields are added here.
    """

    width: int = 768
    height: int = 512
    guidance_scale: float = 4.0
    num_inference_steps: int = 40
    num_frames: int = 121
    frame_rate: float = 24.0
    num_visuals_per_prompt: int = 1
    mask: npt.NDArray[np.bool_] | None = None
    """Attention mask for the text encoder (True = attend, False = ignore)."""
    extra_params: dict[str, npt.NDArray[Any]] | None = None
    """LTX2-specific preprocessed arrays (e.g. ltx2_video_latents_5d, ltx2_audio_latents)."""

    @property
    def do_true_cfg(self) -> bool:
        return self.negative_tokens is not None


@dataclass
class LTX2PipelineOutput:
    """Output class for LTX2 video+audio generation pipelines.

    Args:
        frames: Generated video tensor of shape ``(batch, frames, height, width, channels)``
            with values in ``[0, 1]``.
        audio: Generated audio waveform tensor of shape ``(batch, channels, samples)``.
    """

    frames: Tensor
    audio: Tensor


class LTX2Pipeline(DiffusionPipeline):
    """A LTX2 pipeline for text-to-video and image-to-video generation."""

    vae: AutoencoderKLLTX2VideoModel
    audio_vae: AutoencoderKLLTX2AudioModel
    transformer: LTX2TransformerModel
    connectors: LTX2TextConnectorsModel
    vocoder: LTX2VocoderModel

    components = {
        "vae": AutoencoderKLLTX2VideoModel,
        "audio_vae": AutoencoderKLLTX2AudioModel,
        "transformer": LTX2TransformerModel,
        "connectors": LTX2TextConnectorsModel,
        "vocoder": LTX2VocoderModel,
    }

    def init_remaining_components(self) -> None:
        """Initialize non-ComponentModel parts of the LTX2 pipeline.

        - Using pre-loaded component models (vae, audio_vae, transformer, connectors, vocoder)
        - Setting basic VAE/audio compression ratios
        - Pre-compiling hot-path functions via max_compile
        """

        # VAE compression ratios: fall back to known LTX2 defaults if not present.
        # LTX2 uses (t, h, w) scale factors of (8, 32, 32) for the video VAE.
        self.vae_temporal_compression_ratio = 8
        self.vae_spatial_compression_ratio = 32

        # Audio VAE configuration (matching AutoencoderKLLTX2AudioConfig defaults).
        self.audio_vae_mel_compression_ratio = 4
        self.audio_hop_length = 160
        self.audio_sampling_rate = 16000

        # Instantiate Gemma3 text encoder (PyTorch) from the same model path.
        # NOTE: text encoder init is intentionally left as-is (uses PyTorch, not MAX).
        import torch

        model_id = str(self.pipeline_config.model.model_path)
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )

        # Prefer CUDA device if available, otherwise leave on CPU.
        if torch.cuda.is_available():
            self.text_encoder.to("cuda")

        # Cache VAE latent statistics as Tensors for use in compiled postprocess functions.
        # Must be set up BEFORE build_decode_video_latents / build_decode_audio_latents.
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype
        vae_mean = getattr(self.vae, "latents_mean", None) or getattr(
            self.vae.config, "latents_mean", None
        )
        vae_std = getattr(self.vae, "latents_std", None) or getattr(
            self.vae.config, "latents_std", None
        )
        if vae_mean is not None and vae_std is not None:
            self._vae_latents_mean: Tensor | None = Tensor.constant(
                np.array(vae_mean, dtype=np.float32), dtype=dtype, device=device
            )
            self._vae_latents_std: Tensor | None = Tensor.constant(
                np.array(vae_std, dtype=np.float32), dtype=dtype, device=device
            )
        else:
            self._vae_latents_mean = None
            self._vae_latents_std = None

        audio_mean = getattr(self.audio_vae, "latents_mean", None) or getattr(
            self.audio_vae.config, "latents_mean", None
        )
        audio_std = getattr(self.audio_vae, "latents_std", None) or getattr(
            self.audio_vae.config, "latents_std", None
        )
        if audio_mean is not None and audio_std is not None:
            self._audio_latents_mean: Tensor | None = Tensor.constant(
                np.array(audio_mean, dtype=np.float32),
                dtype=dtype,
                device=device,
            )
            self._audio_latents_std: Tensor | None = Tensor.constant(
                np.array(audio_std, dtype=np.float32),
                dtype=dtype,
                device=device,
            )
        else:
            self._audio_latents_mean = None
            self._audio_latents_std = None

        # Pre-compile hot-path tensor functions via max_compile (Flux2-style).
        self.build_pack_latents()
        self.build_pack_audio_latents()
        self.build_prepare_scheduler()
        self.build_scheduler_step_video()
        self.build_scheduler_step_audio()
        self.build_decode_video_latents()
        self.build_decode_audio_latents()
        self.build_prepare_cfg_latents()
        self.build_apply_cfg_guidance()

        # A workaround to remove overhead from `functional.wrapped`.
        if unwrapped_transformer := self.transformer.unwrap_model():
            self.transformer = cast(Any, unwrapped_transformer)

        self._joint_attention_kwargs: dict[str, Any] | None = None
        self._num_timesteps: int = 0
        self._cached_sigmas: dict[str, Tensor] = {}

    def prepare_inputs(self, context: PixelContext) -> LTX2ModelInputs:  # type: ignore[override]
        return LTX2ModelInputs.from_context(context)

    # -----------------------------------------------------------------------
    # build_* methods: compile hot-path functions via max_compile (Flux2 style)
    # -----------------------------------------------------------------------

    def build_pack_latents(self) -> None:
        device = self.transformer.devices[0]
        _channels = self.transformer.config.in_channels
        _latent_num_frames = 16  # (121-1)//8+1
        _latent_height = 16
        _latent_width = 24
        input_types = [
            TensorType(
                DType.float32,
                shape=[
                    1,
                    _channels,
                    _latent_num_frames,
                    _latent_height,
                    _latent_width,
                ],
                device=device,
            ),
        ]
        self.__dict__["_pack_video_latents"] = max_compile(
            self._pack_video_latents,
            input_types=input_types,
        )

    def build_pack_audio_latents(self) -> None:
        device = self.transformer.devices[0]
        _latent_mel_bins = 64 // self.audio_vae_mel_compression_ratio  # 16
        _audio_channels = (
            self.transformer.config.audio_in_channels // _latent_mel_bins
        )  # 8
        _audio_num_frames = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                DType.float32,
                shape=[1, _audio_channels, _audio_num_frames, _latent_mel_bins],
                device=device,
            ),
        ]
        self.__dict__["_pack_audio_latents_packed"] = max_compile(
            self._pack_audio_latents_packed,
            input_types=input_types,
        )

    def build_prepare_scheduler(self) -> None:
        """Compile prepare_scheduler: pre-compute timesteps and per-step dts."""
        input_types = [
            TensorType(
                DType.float32,
                shape=["num_sigmas"],
                device=self.transformer.devices[0],
            ),
        ]
        self.__dict__["prepare_scheduler"] = max_compile(
            self.prepare_scheduler,
            input_types=input_types,
        )

    def build_scheduler_step_video(self) -> None:
        """Compile _scheduler_step_video: Euler update for video latents.

        batch=1, seq=6144 (16*16*24), channels=128
        """
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        _channels = self.transformer.config.in_channels  # 128
        _video_seq_len = 6144  # 16 * 16 * 24
        input_types = [
            TensorType(
                dtype, shape=[1, _video_seq_len, _channels], device=device
            ),
            TensorType(
                dtype, shape=[1, _video_seq_len, _channels], device=device
            ),
            TensorType(DType.float32, shape=[1], device=device),
        ]
        self.__dict__["_scheduler_step_video"] = max_compile(
            self.scheduler_step,
            input_types=input_types,
        )

    def build_scheduler_step_audio(self) -> None:
        """Compile _scheduler_step_audio: Euler update for audio latents.

        batch=1, seq=126 (round((121/24)*25)=126), channels=128
        """
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        _channels = self.transformer.config.audio_in_channels  # 128
        _audio_seq_len = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                dtype, shape=[1, _audio_seq_len, _channels], device=device
            ),
            TensorType(
                dtype, shape=[1, _audio_seq_len, _channels], device=device
            ),
            TensorType(DType.float32, shape=[1], device=device),
        ]
        self.__dict__["_scheduler_step_audio"] = max_compile(
            self.scheduler_step,
            input_types=input_types,
        )

    def build_decode_video_latents(self) -> None:
        """Compile _postprocess_video_latents if VAE latent statistics are available.

        Mirrors Flux2's build_decode_latents -> _postprocess_latents pattern.
        """
        if self._vae_latents_mean is None or self._vae_latents_std is None:
            return
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        num_channels = int(self._vae_latents_mean.shape[0])  # 128
        _latent_num_frames = 16  # (121-1)//8+1
        _latent_height = 16  # 512//32
        _latent_width = 24  # 768//32
        input_types = [
            TensorType(
                dtype,
                shape=[
                    1,
                    num_channels,
                    _latent_num_frames,
                    _latent_height,
                    _latent_width,
                ],
                device=device,
            ),
            TensorType(dtype, shape=[num_channels], device=device),
            TensorType(dtype, shape=[num_channels], device=device),
        ]
        self.__dict__["_postprocess_video_latents"] = max_compile(
            self._postprocess_video_latents,
            input_types=input_types,
        )

    def build_decode_audio_latents(self) -> None:
        """Compile _postprocess_audio_latents if audio VAE statistics are available.

        Mirrors Flux2's build_decode_latents -> _postprocess_latents pattern.
        """
        if self._audio_latents_mean is None or self._audio_latents_std is None:
            return
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        num_channels = int(self._audio_latents_mean.shape[0])  # C_audio (8)
        _latent_mel_bins = 64 // self.audio_vae_mel_compression_ratio  # 16
        _audio_num_frames = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                dtype,
                shape=[1, num_channels, _audio_num_frames, _latent_mel_bins],
                device=device,
            ),
            TensorType(dtype, shape=[num_channels], device=device),
            TensorType(dtype, shape=[num_channels], device=device),
        ]
        self.__dict__["_postprocess_audio_latents"] = max_compile(
            self._postprocess_audio_latents,
            input_types=input_types,
        )

    def build_prepare_cfg_latents(self) -> None:
        """Compile _prepare_cfg_latents: concat+cast for CFG latent doubling.

        Fuses two F.concat calls and two casts into a single compiled graph,
        eliminating per-step Python dispatch overhead.
          video: [1, 6144, 128] bfloat16 -> [2, 6144, 128] bfloat16
          audio: [1, 126, 128]   bfloat16 -> [2, 126, 128]   bfloat16
        """
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        _channels = self.transformer.config.in_channels  # 128
        _video_seq_len = 6144  # 16 * 16 * 24
        _audio_channels = self.transformer.config.audio_in_channels  # 128
        _audio_seq_len = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                dtype, shape=[1, _video_seq_len, _channels], device=device
            ),
            TensorType(
                dtype, shape=[1, _audio_seq_len, _audio_channels], device=device
            ),
        ]
        self.__dict__["_prepare_cfg_latents"] = max_compile(
            self._prepare_cfg_latents,
            input_types=input_types,
        )

    def build_apply_cfg_guidance(self) -> None:
        """Compile _apply_cfg_guidance: CFG formula for video+audio noise preds.

        Fuses cast + split + guidance arithmetic into a single compiled graph:
          video in:  [2, 6144, 128] bfloat16 -> [1, 6144, 128] bfloat16
          audio in:  [2, 126, 128]   bfloat16 -> [1, 126, 128]   bfloat16
          guidance:  [1]             float32
        """
        dtype = self.transformer.config.dtype
        device = self.transformer.devices[0]
        _channels = self.transformer.config.in_channels  # 128
        _video_seq_len = 6144  # 16 * 16 * 24
        _audio_channels = self.transformer.config.audio_in_channels  # 128
        _audio_seq_len = 126  # round((121/24)*25.0)=126
        input_types = [
            TensorType(
                dtype, shape=[2, _video_seq_len, _channels], device=device
            ),
            TensorType(
                dtype, shape=[2, _audio_seq_len, _audio_channels], device=device
            ),
            TensorType(DType.float32, shape=[1], device=device),
        ]
        self.__dict__["_apply_cfg_guidance"] = max_compile(
            self._apply_cfg_guidance,
            input_types=input_types,
        )

    def _encode_tokens(
        self, token_ids: np.ndarray, device: str = "cuda"
    ) -> Tensor:
        """Encode token_ids using transformers Gemma3ForConditionalGeneration.

        The token_ids come from PixelGenerationTokenizer which already handles
        tokenization. This method just runs them through the text encoder to
        get hidden states.

        Args:
            token_ids: Token IDs from PixelGenerationTokenizer (via model_inputs.token_ids).
            device: Device to run encoding on.

        Returns:
            Hidden states tensor from the text encoder, stacked across all layers.
        """
        import torch

        # Convert MAX Tensor to PyTorch tensor
        input_ids = torch.from_dlpack(token_ids).to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Stack all hidden states: [batch_size, seq_len, hidden_dim, num_layers]
            hidden_states = torch.stack(outputs.hidden_states, axis=-1)

        # Free GPU and CPU memory used by the text encoder after encoding.
        self.text_encoder.to("cpu")
        del self.text_encoder
        self.text_encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert to MAX Tensor
        return Tensor.from_dlpack(hidden_states.to(torch.bfloat16))

    @staticmethod
    def _pack_text_embeds(
        text_hidden_states: Tensor,
        sequence_lengths: Tensor,
        device: Device,
        padding_side: str = "left",
        scale_factor: int = 8,
        eps: float = 1e-6,
    ) -> Tensor:
        """
        Packs and normalizes text encoder hidden states, respecting padding. Normalization is performed per-batch and
        per-layer in a masked fashion (only over non-padded positions).

        Args:
            text_hidden_states (`Tensor` of shape `(batch_size, seq_len, hidden_dim, num_layers)`):
                Per-layer hidden_states from a text encoder (e.g. `Gemma3ForConditionalGeneration`).
            sequence_lengths (`Tensor of shape `(batch_size,)`):
                The number of valid (non-padded) tokens for each batch instance.
            device: (`Device`, *optional*):
                Device to place the resulting embeddings on
            padding_side: (`str`, *optional*, defaults to `"left"`):
                Whether the text tokenizer performs padding on the `"left"` or `"right"`.
            scale_factor (`int`, *optional*, defaults to `8`):
                Scaling factor to multiply the normalized hidden states by.
            eps (`float`, *optional*, defaults to `1e-6`):
                A small positive value for numerical stability when performing normalization.

        Returns:
            `Tensor` of shape `(batch_size, seq_len, hidden_dim * num_layers)`:
                Normed and flattened text encoder hidden states.
        """
        import torch

        # Convert MAX Tensors to PyTorch: MAX Tensor lacks masked_fill, amin,
        # amax, and multi-axis sum with keepdim.
        ths = torch.from_dlpack(text_hidden_states)
        sl = torch.from_dlpack(sequence_lengths).view(
            -1
        )  # ensure 1D [batch_size]
        torch_device = ths.device

        batch_size, seq_len, hidden_dim, num_layers = ths.shape
        original_dtype = ths.dtype

        # Create padding mask
        token_indices = torch.arange(seq_len, device=torch_device).unsqueeze(0)
        if padding_side == "right":
            # For right padding, valid tokens are from 0 to sequence_length-1
            mask = token_indices < sl[:, None]  # [batch_size, seq_len]
        elif padding_side == "left":
            # For left padding, valid tokens are from (T - sequence_length) to T-1
            start_indices = seq_len - sl[:, None]  # [batch_size, 1]
            mask = token_indices >= start_indices  # [B, T]
        else:
            raise ValueError(
                f"padding_side must be 'left' or 'right', got {padding_side}"
            )
        mask = mask[
            :, :, None, None
        ]  # [batch_size, seq_len] --> [batch_size, seq_len, 1, 1]

        # Compute masked mean over non-padding positions
        masked_ths = ths.masked_fill(~mask, 0.0)
        num_valid_positions = (sl * hidden_dim).view(batch_size, 1, 1, 1)
        masked_mean = masked_ths.sum(dim=(1, 2), keepdim=True) / (
            num_valid_positions + eps
        )

        # Compute min/max over non-padding positions
        x_min = ths.masked_fill(~mask, float("inf")).amin(
            dim=(1, 2), keepdim=True
        )
        x_max = ths.masked_fill(~mask, float("-inf")).amax(
            dim=(1, 2), keepdim=True
        )

        # Normalization
        normalized = (ths - masked_mean) / (x_max - x_min + eps)
        normalized = normalized * scale_factor

        # Pack the hidden states to a 3D tensor (batch_size, seq_len, hidden_dim * num_layers)
        normalized = normalized.flatten(2)
        mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
        normalized = normalized.masked_fill(~mask_flat, 0.0)
        normalized = normalized.to(dtype=original_dtype)
        return Tensor.from_dlpack(normalized.contiguous())

    @staticmethod
    def _pack_latents(
        latents: Tensor, patch_size: int = 1, patch_size_t: int = 1
    ) -> Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, _num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            (
                batch_size,
                -1,
                post_patch_num_frames,
                patch_size_t,
                post_patch_height,
                patch_size,
                post_patch_width,
                patch_size,
            )
        )
        latents = latents.permute([0, 2, 4, 6, 1, 3, 5, 7])
        latents = latents.rebind(
            (
                batch_size,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                _num_channels,
                patch_size_t,
                patch_size,
                patch_size,
            )
        )
        # Flatten (F_post, H_post, W_post) -> S
        # Indices: 0(B), 1(F), 2(H), 3(W), 4(C), 5(pt), 6(p), 7(p)
        latents = latents.flatten(1, 3)
        # Flatten (C, pt, p, p) -> D
        # Indices after first flatten: 0(B), 1(S), 2(C), 3(pt), 4(p), 5(p)
        latents = latents.flatten(2, 5)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.shape[0]
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size

        latents = latents.reshape(
            (
                batch_size,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                -1,
                patch_size_t,
                patch_size,
                patch_size,
            )
        )
        _num_channels = latents.shape[4]
        latents = latents.permute((0, 4, 1, 5, 2, 6, 3, 7))
        latents = latents.rebind(
            (
                batch_size,
                _num_channels,
                post_patch_num_frames,
                patch_size_t,
                post_patch_height,
                patch_size,
                post_patch_width,
                patch_size,
            )
        )
        # Flatten (F_post, pt) -> F
        # Indices: 0(B), 1(C), 2(F), 3(pt), 4(H), 5(p), 6(W), 7(p)
        latents = latents.flatten(2, 3)
        # Flatten (H_post, p) -> H
        # Indices: 0(B), 1(C), 2(F), 3(H), 4(p), 5(W), 6(p)
        latents = latents.flatten(3, 4)
        # Flatten (W_post, p) -> W
        # Indices: 0(B), 1(C), 2(F), 3(H), 4(W), 5(p)
        latents = latents.flatten(4, 5)
        return latents

    @staticmethod
    def _pack_audio_latents(
        latents: Tensor,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> Tensor:
        # Audio latents shape: [B, C, L, M], where L is the latent audio length and M is the number of mel bins
        if patch_size is not None and patch_size_t is not None:
            # Packs the latents into a patch sequence of shape [B, L // p_t * M // p, C * p_t * p] (a ndim=3 tnesor).
            # dim=1 is the effective audio sequence length and dim=2 is the effective audio input feature size.
            batch_size, _num_channels, latent_length, latent_mel_bins = (
                latents.shape
            )
            post_patch_latent_length = latent_length // patch_size_t
            post_patch_mel_bins = latent_mel_bins // patch_size
            latents = latents.reshape(
                (
                    batch_size,
                    -1,
                    post_patch_latent_length,
                    patch_size_t,
                    post_patch_mel_bins,
                    patch_size,
                )
            )
            latents = latents.permute((0, 2, 4, 1, 3, 5))
            latents = latents.rebind(
                (
                    batch_size,
                    post_patch_latent_length,
                    post_patch_mel_bins,
                    _num_channels,
                    patch_size_t,
                    patch_size,
                )
            )
            # Flatten (L_post, M_post) -> S
            # Indices: 0(B), 1(L), 2(M), 3(C), 4(pt), 5(p)
            latents = latents.flatten(1, 2)
            # Flatten (C, pt, p) -> D
            # Indices: 0(B), 1(S), 2(C), 3(pt), 4(p)
            latents = latents.flatten(2, 4)
        else:
            # Packs the latents into a patch sequence of shape [B, L, C * M]. This implicitly assumes a (mel)
            # patch_size of M (all mel bins constitutes a single patch) and a patch_size_t of 1.
            latents = F.flatten(
                latents.transpose(1, 2), 2, 3
            )  # [B, C, L, M] --> [B, L, C * M]
        return latents

    @staticmethod
    def _unpack_audio_latents(
        latents: Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> Tensor:
        # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor of shape [B, C, L, M],
        # where L is the latent audio length and M is the number of mel bins.
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.shape[0]
            post_patch_latent_length = latent_length // patch_size_t
            post_patch_mel_bins = num_mel_bins // patch_size
            latents = latents.reshape(
                (
                    batch_size,
                    post_patch_latent_length,
                    post_patch_mel_bins,
                    -1,
                    patch_size_t,
                    patch_size,
                )
            )
            _num_channels = latents.shape[3]
            latents = latents.permute((0, 3, 1, 4, 2, 5))
            latents = latents.rebind(
                (
                    batch_size,
                    _num_channels,
                    post_patch_latent_length,
                    patch_size_t,
                    post_patch_mel_bins,
                    patch_size,
                )
            )
            # Flatten (L_post, pt) -> L
            # Indices: 0(B), 1(C), 2(L), 3(pt), 4(M), 5(p)
            latents = latents.flatten(2, 3)
            # Flatten (M_post, p) -> M
            # Indices: 0(B), 1(C), 2(L), 3(M), 4(p)
            latents = latents.flatten(3, 4)
        else:
            # Assume [B, S, D] = [B, L, C * M], which implies that patch_size = M and patch_size_t = 1.
            latents = latents.reshape(
                (latents.shape[0], latents.shape[1], -1, num_mel_bins)
            ).transpose(1, 2)
        return latents

    # -----------------------------------------------------------------------
    # Compiled instance methods (overridden in __dict__ by build_* at startup)
    # -----------------------------------------------------------------------

    def _prepare_cfg_latents(
        self, video_latents: Tensor, audio_latents: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Concat+cast video and audio latents for CFG [1,S,D]->[2,S,D].

        Compiled via build_prepare_cfg_latents. Called once per denoising step
        when do_classifier_free_guidance is True.
        """
        dtype = self.transformer.config.dtype
        video = F.concat([video_latents, video_latents], axis=0).cast(dtype)
        audio = F.concat([audio_latents, audio_latents], axis=0).cast(dtype)
        return video, audio

    def _apply_cfg_guidance(
        self,
        noise_pred_video: Tensor,
        noise_pred_audio: Tensor,
        guidance_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply classifier-free guidance formula to video and audio noise preds.

        Compiled via build_apply_cfg_guidance. Fuses cast + split + arithmetic
        for both modalities into a single compiled graph, called once per step.

        guidance: uncond + scale * (cond - uncond)
        """
        dtype = DType.float32
        scale = guidance_scale.cast(dtype)

        video = noise_pred_video.cast(dtype)
        v_uncond = F.slice_tensor(video, [slice(0, 1)])
        v_cond = F.slice_tensor(video, [slice(1, 2)])
        guided_video = v_uncond + scale * (v_cond - v_uncond)

        audio = noise_pred_audio.cast(dtype)
        a_uncond = F.slice_tensor(audio, [slice(0, 1)])
        a_cond = F.slice_tensor(audio, [slice(1, 2)])
        guided_audio = a_uncond + scale * (a_cond - a_uncond)

        return guided_video.cast(
            self.transformer.config.dtype
        ), guided_audio.cast(self.transformer.config.dtype)

    def _pack_video_latents(self, latents: Tensor) -> Tensor:
        """Cast and pack video latents [B,C,F,H,W] -> [B,S,C] (patch_size=1).

        With patch_size=patch_size_t=1 the pack is a simple permute + flatten:
        [B,C,F,H,W] -> [B,F,H,W,C] -> [B,F*H*W,C].
        """
        latents = latents.cast(self.transformer.config.dtype)
        # [B,C,F,H,W] -> [B,F,H,W,C]
        latents = latents.permute([0, 2, 3, 4, 1])
        # [B,F,H,W,C] -> [B,S,C]  where S = F*H*W
        latents = latents.flatten(1, 3)
        return latents

    def _pack_audio_latents_packed(self, latents: Tensor) -> Tensor:
        """Cast and pack audio latents [B,C,L,M] -> [B,L,C*M] (no spatial patch).

        Equivalent to the no-patch branch of _pack_audio_latents:
        [B,C,L,M] -> [B,L,C,M] -> [B,L,C*M].
        """
        latents = latents.cast(self.transformer.config.dtype)
        # [B,C,L,M] -> [B,L,C,M]
        latents = latents.permute((0, 2, 1, 3))
        # [B,L,C,M] -> [B,L,C*M]
        latents = latents.flatten(2, 3)
        return latents

    def prepare_scheduler(self, sigmas: Tensor) -> tuple[Tensor, Tensor]:
        """Pre-compute timesteps and per-step dt values from sigmas.

        Returns:
            (all_timesteps, all_dts) where timesteps = sigmas[:-1] as float32,
            and dts = sigmas[1:] - sigmas[:-1] (float32).
        """
        sigmas_curr = F.slice_tensor(sigmas, [slice(0, -1)])
        sigmas_next = F.slice_tensor(sigmas, [slice(1, None)])
        all_dt = F.sub(sigmas_next, sigmas_curr)
        all_timesteps = sigmas_curr.cast(self.transformer.config.dtype)
        return all_timesteps, all_dt

    def scheduler_step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        dt: Tensor,
    ) -> Tensor:
        """Apply a single Euler update step in sigma space.

        Shared for both video and audio latents.
        """
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        latents = latents + dt * noise_pred
        return latents.cast(latents_dtype)

    def _postprocess_video_latents(
        self,
        latents: Tensor,
        latents_mean: Tensor,
        latents_std: Tensor,
    ) -> Tensor:
        """Denormalize video latents [B,C,F,H,W] using per-channel stats.

        Mirrors Flux2's _postprocess_latents: the compiled inner step of
        decode_video_latents.
        """
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        c = latents_mean.shape[0]
        mean_r = F.reshape(latents_mean, (1, c, 1, 1, 1))
        std_r = F.reshape(latents_std, (1, c, 1, 1, 1))
        return latents * std_r / scaling_factor + mean_r

    def _postprocess_audio_latents(
        self,
        latents: Tensor,
        latents_mean: Tensor,
        latents_std: Tensor,
    ) -> Tensor:
        """Denormalize audio latents [B,C,L,M] using per-channel stats.

        Mirrors Flux2's _postprocess_latents for the audio stream.
        """
        c = latents_mean.shape[0]
        mean_r = F.reshape(latents_mean, (1, c, 1, 1))
        std_r = F.reshape(latents_std, (1, c, 1, 1))
        return latents * std_r + mean_r

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self) -> dict[str, Any] | None:
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    def decode_video_latents(
        self,
        latents: Tensor,
        num_frames: int,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        """Decode packed video latents into a float32 [B,F,H,W,C] NumPy array.

        Mirrors Flux2's decode_latents: shape-dependent unpack (not compiled) is
        performed here, then the compiled _postprocess_video_latents handles
        denormalization, the VAE decoder runs, and finally the pixels are scaled
        to [0, 1] and permuted to channel-last.

        Args:
            latents: Packed latents [B, S, C].
            num_frames: Latent frame count.
            height: Latent height.
            width: Latent width.
            output_type: "latent" returns latents as-is; otherwise decodes to NumPy.

        Returns:
            float32 NumPy array [B, F, H, W, C] or raw latent Tensor.
        """
        if output_type == "latent":
            return latents
        # Shape-dependent unpack: [B,S,C] -> [B,C,F,H,W]  (not compiled).
        latents = self._unpack_latents(latents, num_frames, height, width)
        # Compiled postprocess: denormalize using per-channel stats.
        if (
            self._vae_latents_mean is not None
            and self._vae_latents_std is not None
        ):
            latents = self._postprocess_video_latents(
                latents, self._vae_latents_mean, self._vae_latents_std
            )
        else:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latents = latents / scaling_factor
        # VAE decode: [B,C,F,H,W].
        video = self.vae.decode(latents.cast(DType.bfloat16))
        # Scale pixels to [0, 1] and permute to channel-last [B,F,H,W,C].
        video = (video / 2.0 + 0.5).clip(min=0.0, max=1.0)
        video = video.permute((0, 2, 3, 4, 1))
        return self._to_numpy(video)

    def decode_audio_latents(
        self,
        latents: Tensor,
        audio_num_frames: int,
        latent_mel_bins: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        """Decode packed audio latents into a waveform Tensor (or return latents).

        Args:
            latents: Packed audio latents [B, L, C*M].
            audio_num_frames: Latent audio length used for unpacking.
            latent_mel_bins: Number of mel frequency bins in the latent space.
            output_type: "latent" returns latents as-is; otherwise decodes via vocoder.

        Returns:
            Waveform Tensor or raw latent Tensor.
        """
        if output_type == "latent":
            return latents
        # Shape-dependent unpack: [B,L,C*M] -> [B,C,L,M]  (not compiled).
        latents = self._unpack_audio_latents(
            latents, audio_num_frames, latent_mel_bins
        )
        # Compiled postprocess: denormalize.
        if (
            self._audio_latents_mean is not None
            and self._audio_latents_std is not None
        ):
            latents = self._postprocess_audio_latents(
                latents, self._audio_latents_mean, self._audio_latents_std
            )
        mel_spectrograms = self.audio_vae.decode(latents.cast(DType.bfloat16))
        return self.vocoder(mel_spectrograms)

    def _to_numpy(self, image: Tensor) -> np.ndarray:
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    def execute(  # type: ignore[override]
        self,
        model_inputs: LTX2ModelInputs,
    ) -> LTX2PipelineOutput:
        r"""
        Executes the LTX2 model with the prepared inputs.

        Args:
            model_inputs: A LTX2Inputs instance containing all image generation parameters
                including prompt, dimensions, guidance scale, etc.

        Returns:
            ModelOutputs containing the generated images.
        """
        # Extract parameters from model_inputs
        height = model_inputs.height or 512
        width = model_inputs.width or 768
        num_frames = model_inputs.num_frames or 121
        frame_rate = model_inputs.frame_rate or 24
        num_inference_steps = model_inputs.num_inference_steps
        guidance_scale = model_inputs.guidance_scale
        device = self.devices[0]

        # Optional LTX2-specific precomputed latents from PixelGenerationTokenizer
        extra_params = model_inputs.extra_params or {}
        video_latents_5d_np: np.ndarray | None = extra_params.get(
            "ltx2_video_latents_5d"
        )
        audio_latents_np: np.ndarray | None = extra_params.get(
            "ltx2_audio_latents"
        )
        # Positional embedding coordinates precomputed on CPU by
        # PixelGenerationTokenizer (avoids tensor compilation in the pipeline).
        video_coords_np: np.ndarray | None = extra_params.get(
            "ltx2_video_coords"
        )
        audio_coords_np: np.ndarray | None = extra_params.get(
            "ltx2_audio_coords"
        )
        # When the tokenizer pre-doubled the coords for CFG, skip F.concat below.
        coords_cfg_doubled: bool = bool(
            extra_params.get("ltx2_coords_cfg_doubled", False)
        )

        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps

        # 1. Resolve latent dimensions.
        # Prefer values precomputed by the tokenizer; fall back to deriving
        # them from the request parameters when running without a tokenizer.
        if "ltx2_latent_num_frames" in extra_params:
            latent_num_frames = int(extra_params["ltx2_latent_num_frames"])
            latent_height = int(extra_params["ltx2_latent_height"])
            latent_width = int(extra_params["ltx2_latent_width"])
        else:
            latent_num_frames = (
                num_frames - 1
            ) // self.vae_temporal_compression_ratio + 1
            latent_height = height // self.vae_spatial_compression_ratio
            latent_width = width // self.vae_spatial_compression_ratio

        # 2. Get sigmas from PixelModelInputs, cache, and pre-compute scheduler tensors.
        sigmas_np: np.ndarray = model_inputs.sigmas
        sigmas_key = str(num_inference_steps)
        if sigmas_key in self._cached_sigmas:
            sigmas = self._cached_sigmas[sigmas_key]
        else:
            sigmas = Tensor.from_dlpack(sigmas_np).to(device)
            self._cached_sigmas[sigmas_key] = sigmas
        all_timesteps, all_dts = self.prepare_scheduler(sigmas)

        # For faster tensor slicing inside the denoising loop.
        timesteps_seq = all_timesteps
        dts_seq = all_dts
        if hasattr(timesteps_seq, "driver_tensor"):
            timesteps_seq = timesteps_seq.driver_tensor
        if hasattr(dts_seq, "driver_tensor"):
            dts_seq = dts_seq.driver_tensor

        # 3. Encode text with Gemma3 (via transformers) using TokenBuffer data
        token_ids_np: np.ndarray = model_inputs.tokens.array
        if token_ids_np.ndim == 1:
            token_ids_np = np.expand_dims(token_ids_np, axis=0)

        mask_np: npt.NDArray[np.bool_] | None = model_inputs.mask
        if mask_np is None:
            mask_np = np.ones_like(token_ids_np, dtype=np.bool_)
        if mask_np.ndim == 1:
            mask_np = np.expand_dims(mask_np, axis=0)
        prompt_attention_mask = (
            Tensor.from_dlpack(
                mask_np.astype(np.bool_),
            )
            .to(device)
            .cast(DType.bool)
        )

        text_encoder_hidden_states = self._encode_tokens(
            token_ids_np, device="cuda"
        )
        # Reduce [B, seq_len] bool mask → [B] integer counts, matching
        sequence_lengths = prompt_attention_mask.cast(DType.int64).sum(axis=-1)
        prompt_embeds = self._pack_text_embeds(
            text_encoder_hidden_states, sequence_lengths, device
        )

        # Encode negative prompt if doing CFG
        if self.do_classifier_free_guidance:
            if model_inputs.negative_tokens is not None:
                negative_ids_np: np.ndarray = model_inputs.negative_tokens.array
                if negative_ids_np.ndim == 1:
                    negative_ids_np = np.expand_dims(negative_ids_np, axis=0)
                negative_attention_mask = prompt_attention_mask
                negative_hidden_states = self._encode_tokens(
                    negative_ids_np, device="cuda"
                )
                negative_sequence_lengths = negative_attention_mask.cast(
                    DType.int64
                ).sum(axis=-1)
                negative_prompt_embeds = self._pack_text_embeds(
                    negative_hidden_states,
                    negative_sequence_lengths,
                    device,
                )
            else:
                # Use zeros for negative prompt if not provided
                negative_prompt_embeds = Tensor.zeros_like(prompt_embeds)
            # Concatenate for CFG: [negative, positive]
            prompt_embeds = F.concat(
                [negative_prompt_embeds, prompt_embeds], axis=0
            )
            prompt_attention_mask = F.concat(
                [prompt_attention_mask, prompt_attention_mask], axis=0
            )

        # 4. Process text embeddings through connectors.
        # valid_length was pre-computed by the tokenizer (pure numpy, outside
        # any compiled graph) so the MHA padded kernel gets si64 stride metadata.
        # Fallback handles the case where no tokenizer is used (e.g. tests).
        valid_length_np = extra_params.get("ltx2_valid_length")
        if valid_length_np is not None:
            prompt_valid_length = (
                Tensor.from_dlpack(
                    valid_length_np.astype(np.uint32),
                )
                .to(device)
                .cast(DType.uint32)
            )
        else:
            # Fallback: recompute from the bool mask tensor (CFG-doubled already).
            prompt_valid_length = prompt_attention_mask.cast(DType.uint32).sum(
                axis=-1
            )
        (
            connector_prompt_embeds,
            connector_audio_prompt_embeds,
            _,
        ) = self.connectors(prompt_embeds, prompt_valid_length)

        # 5. Prepare video latents
        # Prefer precomputed 5D latents from PixelGenerationTokenizer when available.
        if video_latents_5d_np is not None:
            if video_latents_5d_np.ndim != 5:
                raise ValueError(
                    "ltx2_video_latents_5d must have shape [B, C, F, H, W]"
                )
            batch_size, _, latent_num_frames, latent_height, latent_width = (
                video_latents_5d_np.shape
            )
            latents_5d = video_latents_5d_np.astype(np.float32)
        else:
            # Fallback: expand 4D latents [B, C, H, W] along temporal dimension.
            latents_np_4d: np.ndarray = model_inputs.latents
            if latents_np_4d.ndim != 4:
                raise ValueError(
                    "Expected 4D latents [B, C, H, W] when ltx2_video_latents_5d is not provided"
                )
            batch_size = int(latents_np_4d.shape[0])
            latents_5d = latents_np_4d[:, :, None, :, :]
            latents_5d = np.repeat(latents_5d, latent_num_frames, axis=2)
            latents_5d = latents_5d.astype(np.float32)

        latents = Tensor.from_dlpack(latents_5d).to(device)
        # Pack latents: [B, C, F, H, W] -> [B, S, D]
        latents = self._pack_video_latents(latents)

        # 6. Prepare audio latents
        if audio_latents_np is not None:
            if audio_latents_np.ndim != 4:
                raise ValueError(
                    "ltx2_audio_latents must have shape [B, C, L, M]"
                )
            (
                batch_size_audio,
                _audio_channels,
                audio_num_frames,
                latent_mel_bins,
            ) = audio_latents_np.shape
            if batch_size_audio != batch_size:
                raise ValueError(
                    "Mismatch between video and audio batch sizes in LTX2 latents"
                )
            # Prefer tokenizer-precomputed scalars to avoid shape extraction.
            if "ltx2_audio_num_frames" in extra_params:
                audio_num_frames = int(extra_params["ltx2_audio_num_frames"])
                latent_mel_bins = int(extra_params["ltx2_latent_mel_bins"])
            audio_latents_arr = audio_latents_np.astype(np.float32)
        else:
            # Fallback: sample audio latents matching the original pipeline logic.
            num_mel_bins = 64  # From audio VAE config
            latent_mel_bins = (
                num_mel_bins // self.audio_vae_mel_compression_ratio
            )
            duration_s = num_frames / frame_rate
            audio_latents_per_second = 16_000 / 160 / 4.0
            audio_num_frames = round(duration_s * audio_latents_per_second)
            audio_shape = (batch_size, 8, audio_num_frames, latent_mel_bins)
            audio_latents_arr = random.gaussian(
                audio_shape, dtype=DType.float32
            ).to(device)

        audio_latents = (
            audio_latents_arr
            if isinstance(audio_latents_arr, Tensor)
            else Tensor.from_dlpack(audio_latents_arr)
        )
        audio_latents = audio_latents.to(device)
        audio_latents = self._pack_audio_latents_packed(audio_latents)

        # 7. Pre-compute positional embeddings.
        # Prefer numpy arrays precomputed by PixelGenerationTokenizer (zero
        # compilation cost).
        video_coords = Tensor.from_dlpack(video_coords_np).to(device)

        audio_coords_np_f32 = audio_coords_np.astype(np.float32)
        audio_coords = Tensor.from_dlpack(audio_coords_np_f32).to(device)

        # Duplicate positional coords for CFG (batch dim doubles).
        # Skip when the tokenizer already pre-doubled them on the CPU fast path.
        if self.do_classifier_free_guidance and not coords_cfg_doubled:
            video_coords = F.concat([video_coords, video_coords], axis=0)
            audio_coords = F.concat([audio_coords, audio_coords], axis=0)

        # Pre-create the guidance scale tensor once so it can be passed to
        # the compiled _apply_cfg_guidance graph without recreation per step.
        guidance_scale_tensor = Tensor.from_dlpack(
            np.array([guidance_scale], dtype=np.float32),
        ).to(device)

        # 8. Denoising loop
        for i in tqdm(range(num_inference_steps), desc="Denoising"):
            timestep = timesteps_seq[i : i + 1]
            dt = dts_seq[i : i + 1]

            # Prepare CFG inputs (compiled: concat+cast for both modalities).
            if self.do_classifier_free_guidance:
                latent_model_input, audio_latent_model_input = (
                    self._prepare_cfg_latents(latents, audio_latents)
                )
                timestep_t = Tensor.from_dlpack(timestep)
                timestep = F.concat([timestep_t, timestep_t], axis=0)
            else:
                latent_model_input = latents
                audio_latent_model_input = audio_latents

            latent_model_input = latent_model_input.cast(prompt_embeds.dtype)
            audio_latent_model_input = audio_latent_model_input.cast(
                prompt_embeds.dtype
            )

            # After unwrap_model(), the transformer is a compiled engine
            # expecting exactly 8 positional tensor inputs matching
            # input_types(): hidden_states, audio_hidden_states,
            # encoder_hidden_states, audio_encoder_hidden_states,
            # timestep, audio_timestep, video_coords, audio_coords.
            # Non-tensor kwargs (num_frames, height, etc.) are only used
            # during graph tracing and must not be passed at inference.
            noise_pred_video, noise_pred_audio = self.transformer(
                latent_model_input,
                audio_latent_model_input,
                connector_prompt_embeds,
                connector_audio_prompt_embeds,
                timestep,
                timestep,  # audio_timestep = timestep
                video_coords,
                audio_coords,
            )

            if self.do_classifier_free_guidance:
                # Compiled: cast + split + guidance formula for both modalities.
                noise_pred_video, noise_pred_audio = self._apply_cfg_guidance(
                    noise_pred_video, noise_pred_audio, guidance_scale_tensor
                )
            else:
                noise_pred_video = noise_pred_video.cast(DType.float32)
                noise_pred_audio = noise_pred_audio.cast(DType.float32)

            # Euler scheduler step (separate compiled functions for video and audio).
            latents = self._scheduler_step_video(latents, noise_pred_video, dt)
            audio_latents = self._scheduler_step_audio(
                audio_latents, noise_pred_audio, dt
            )

        print("End of the denoising loop.")
        # 9. Decode video latents -> float32 [B,F,H,W,C] NumPy array.
        # decode_video_latents: unpack (shape-dependent) + compiled _postprocess_video_latents
        # + VAE decode + scale to [0,1] + permute to channel-last.
        frames = self.decode_video_latents(
            latents, latent_num_frames, latent_height, latent_width
        )

        print("End of the video decoding.")
        # 10. Decode audio latents (unpack then compiled postprocess then audio VAE + vocoder).
        audio = self.decode_audio_latents(
            audio_latents,
            audio_num_frames,
            latent_mel_bins,
        )
        print("End of the audio decoding.")
        return LTX2PipelineOutput(
            frames=frames,
            audio=audio,
        )
