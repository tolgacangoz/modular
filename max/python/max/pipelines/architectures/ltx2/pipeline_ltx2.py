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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

import max.functional as F
import numpy as np
import numpy.typing as npt
from max import random
from max.driver import CPU, Device
from max.dtype import DType
from max.pipelines import PixelContext
from max.pipelines.lib import (
    ModelInputs,
)
from max.pipelines.lib.interfaces import (
    DiffusionPipeline,
    PixelModelInputs,
)
from max.tensor import Tensor
from tqdm.auto import tqdm
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

        Follows the Flux1-style pattern by:
        - Using pre-loaded component models (vae, audio_vae, transformer, connectors, vocoder)
        - Setting basic VAE/audio compression ratios
        """

        # VAE compression ratios: fall back to known LTX2 defaults if not present.
        # LTX2 uses (t, h, w) scale factors of (4, 32, 32) for the video VAE.
        self.vae_temporal_compression_ratio = 4
        self.vae_spatial_compression_ratio = 32

        # Audio VAE configuration (matching AutoencoderKLLTX2AudioConfig defaults).
        self.audio_vae_mel_compression_ratio = 4
        self.audio_hop_length = 160
        self.audio_sampling_rate = 16000

        # Instantiate Gemma3 text encoder (PyTorch) from the same model path.
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

        self._joint_attention_kwargs: dict[str, Any] | None = None
        self._num_timesteps: int = 0

    def prepare_inputs(self, context: PixelContext) -> LTX2ModelInputs:
        return LTX2ModelInputs.from_context(context)

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
        sl = torch.from_dlpack(sequence_lengths)
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

    def _denormalize_latents(self, latents: Tensor) -> Tensor:
        """Denormalize video latents."""
        # Prefer full stats if available on the video VAE model; otherwise
        # fall back to a simple scaling_factor-only scheme.
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)

        latents_mean = getattr(self.vae, "latents_mean", None)
        latents_std = getattr(self.vae, "latents_std", None)

        if latents_mean is not None and latents_std is not None:
            latents = latents * latents_std / scaling_factor + latents_mean
        else:
            latents = latents / scaling_factor
        return latents

    def _denormalize_audio_latents(self, latents: Tensor) -> Tensor:
        """Denormalize audio latents."""
        latents_mean = getattr(self.audio_vae, "latents_mean", None)
        latents_std = getattr(self.audio_vae, "latents_std", None)

        if latents_mean is not None and latents_std is not None:
            return (latents * latents_std) + latents_mean
        return latents

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

    def _decode_latents(
        self,
        latents: Tensor,
        num_frames: int,
        height: int,
        width: int,
        output_type: Literal["np", "latent", "pil"] = "np",
    ) -> Tensor | np.ndarray:
        if output_type == "latent":
            return latents
        latents = Tensor.from_dlpack(latents)
        latents = self._unpack_latents(latents, num_frames, height, width)
        # Denormalize
        latents = self._denormalize_latents(latents)

        return self._to_numpy(self.vae.decode(latents.cast(DType.bfloat16)))

    def _to_numpy(self, image: Tensor) -> np.ndarray:
        cpu_image: Tensor = image.cast(DType.float32).to(CPU())
        return np.from_dlpack(cpu_image)

    def _scheduler_step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        sigmas: Tensor,
        step_index: int,
    ) -> Tensor:
        latents_dtype = latents.dtype
        latents = latents.cast(DType.float32)
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        latents = latents + dt * noise_pred
        latents = latents.cast(latents_dtype)
        return latents

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> LTX2PipelineOutput:
        r"""
        Executes the LTX2 model with the prepared inputs.

        Args:
            model_inputs: A LTX2Inputs instance containing all image generation parameters
                including prompt, dimensions, guidance scale, etc.

        Returns:
            ModelOutputs containing the generated images.
        """
        # Use cast for type safety
        model_inputs = cast(LTX2ModelInputs, model_inputs)

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

        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps

        # 1. Compute latent dimensions (may be overridden by precomputed latents)
        latent_num_frames = (
            num_frames - 1
        ) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        # 2. Get timesteps and sigmas from PixelModelInputs (numpy arrays)
        timesteps_np: np.ndarray = model_inputs.timesteps
        sigmas_np: np.ndarray = model_inputs.sigmas
        timesteps = Tensor.constant(
            timesteps_np.astype(np.float32, copy=False),
            dtype=DType.float32,
            device=device,
        )
        sigmas = Tensor.constant(
            sigmas_np.astype(np.float32, copy=False),
            dtype=DType.float32,
            device=device,
        )

        # 3. Encode text with Gemma3 (via transformers) using TokenBuffer data
        token_ids_np: np.ndarray = model_inputs.tokens.array
        if token_ids_np.ndim == 1:
            token_ids_np = np.expand_dims(token_ids_np, axis=0)
        # token_ids = Tensor.constant(
        #     token_ids_np.astype(np.int64, copy=False),
        #     dtype=DType.int64,
        #     device=device,
        # )

        mask_np: npt.NDArray[np.bool_] | None = model_inputs.mask
        if mask_np is None:
            mask_np = np.ones_like(token_ids_np, dtype=np.bool_)
        if mask_np.ndim == 1:
            mask_np = np.expand_dims(mask_np, axis=0)
        prompt_attention_mask = Tensor.constant(
            mask_np.astype(np.bool_, copy=False),
            dtype=DType.bool,
            device=device,
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
                negative_token_ids = Tensor.constant(
                    negative_ids_np.astype(np.int64, copy=False),
                    dtype=DType.int64,
                    device=device,
                )
                negative_attention_mask = prompt_attention_mask

                negative_hidden_states = self._encode_tokens(
                    negative_token_ids, device="cuda"
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

        # 4. Process text embeddings through connectors
        mask = prompt_attention_mask.cast(DType.float32)
        additive_attention_mask = ((1.0 - mask) * -1000000.0).cast(
            prompt_embeds.dtype
        )
        (
            connector_prompt_embeds,
            connector_audio_prompt_embeds,
            connector_attention_mask,
        ) = self.connectors(
            prompt_embeds, additive_attention_mask, additive_mask=True
        )

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
            latents_5d = video_latents_5d_np.astype(np.float32, copy=False)
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
            latents_5d = latents_5d.astype(np.float32, copy=False)

        latents = Tensor.constant(
            latents_5d,
            dtype=DType.bfloat16,
            device=device,
        )
        # Pack latents: [B, C, F, H, W] -> [B, S, D]
        latents = self._pack_latents(latents)

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
            audio_latents_arr = audio_latents_np.astype(np.float32, copy=False)
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
            else Tensor.constant(
                audio_latents_arr,
                dtype=DType.float32,
                device=device,
            )
        )
        audio_latents = self._pack_audio_latents(audio_latents)

        # 7. Pre-compute positional embeddings
        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0],
            latent_num_frames,
            latent_height,
            latent_width,
            device,
            fps=frame_rate,
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, device
        )

        # Duplicate positional coords for CFG (batch dim doubles).
        if self.do_classifier_free_guidance:
            video_coords = F.concat([video_coords, video_coords], axis=0)
            audio_coords = F.concat([audio_coords, audio_coords], axis=0)

        num_warmup_steps = model_inputs.num_warmup_steps

        # 8. Denoising loop
        with tqdm(total=num_inference_steps, desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                # Prepare CFG inputs
                if self.do_classifier_free_guidance:
                    latent_model_input = F.concat([latents, latents], axis=0)
                    audio_latent_model_input = F.concat(
                        [audio_latents, audio_latents], axis=0
                    )
                else:
                    latent_model_input = latents
                    audio_latent_model_input = audio_latents

                latent_model_input = latent_model_input.cast(
                    prompt_embeds.dtype
                )
                audio_latent_model_input = audio_latent_model_input.cast(
                    prompt_embeds.dtype
                )

                # Broadcast timestep
                if self.do_classifier_free_guidance:
                    timestep = F.concat(
                        [t.unsqueeze(0), t.unsqueeze(0)], axis=0
                    )
                else:
                    timestep = t.unsqueeze(0)

                noise_pred_video, noise_pred_audio = self.transformer(
                    hidden_states=latent_model_input,
                    audio_hidden_states=audio_latent_model_input,
                    encoder_hidden_states=connector_prompt_embeds,
                    audio_encoder_hidden_states=connector_audio_prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=connector_attention_mask,
                    audio_encoder_attention_mask=connector_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=frame_rate,
                    audio_num_frames=audio_num_frames,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                )
                noise_pred_video = noise_pred_video.cast(DType.float32)
                noise_pred_audio = noise_pred_audio.cast(DType.float32)

                if self.do_classifier_free_guidance:
                    # Split into uncond and cond predictions
                    noise_pred_video_uncond = noise_pred_video[0:1]
                    noise_pred_video_text = noise_pred_video[1:2]
                    noise_pred_video = (
                        noise_pred_video_uncond
                        + guidance_scale
                        * (noise_pred_video_text - noise_pred_video_uncond)
                    )

                    noise_pred_audio_uncond = noise_pred_audio[0:1]
                    noise_pred_audio_text = noise_pred_audio[1:2]
                    noise_pred_audio = (
                        noise_pred_audio_uncond
                        + guidance_scale
                        * (noise_pred_audio_text - noise_pred_audio_uncond)
                    )

                # Scheduler step
                audio_latents = self._scheduler_step(
                    audio_latents, noise_pred_audio, sigmas, i
                )
                latents = self._scheduler_step(
                    latents, noise_pred_video, sigmas, i
                )

                progress_bar.update()

        # 9. Decode latents to video
        # Unpack latents: [B, S, D] -> [B, C, F, H, W]
        latents = self._unpack_latents(
            latents, latent_num_frames, latent_height, latent_width
        )
        latents = self._denormalize_latents(latents)

        video = self.vae.decode(latents.cast(DType.bfloat16))

        # Scale to [0, 1] and permute to [B, F, H, W, C]
        video = (video / 2.0 + 0.5).clip(min=0.0, max=1.0)
        video = video.permute((0, 2, 3, 4, 1))

        # 10. Decode audio
        audio_latents = self._unpack_audio_latents(
            audio_latents, audio_num_frames, latent_mel_bins
        )
        audio_latents = self._denormalize_audio_latents(audio_latents)
        mel_spectrograms = self.audio_vae.decode(
            audio_latents.cast(DType.bfloat16)
        )
        audio = self.vocoder(mel_spectrograms)

        return LTX2PipelineOutput(
            frames=video,
            audio=audio,
        )
