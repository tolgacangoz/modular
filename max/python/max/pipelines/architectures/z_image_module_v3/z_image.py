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
"""Implements the ZImage multimodal model architecture."""

from __future__ import annotations

from dataclasses import asdict

from max.driver import Device
from max.dtype import DType
from max.experimental.tensor import Tensor
import max.nn.module_v3 as nn
from max.pipelines.architectures.qwen3.qwen3 import Qwen3

from .model_config import ZImageConfig
from .nn.autoencoder_kl import AutoencoderKL
from .nn.transformer_z_image import ZImageTransformer2DModel
from .scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class ZImage(nn.Module):
    """The overall interface to the ZImage model."""

    def __init__(
        self, config: ZImageConfig, device: Device | None = None
    ) -> None:
        self.config = config
        self.device = device
        self.scheduler = self.build_scheduler()
        self.vae = self.build_vae()
        self.text_encoder = self.build_text_encoder()
        self.transformer = self.build_transformer()

        self.vae_scale_factor = 2 ** (len(self.config.vae_config.block_out_channels) - 1)

    def build_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        """Build the scheduler component."""
        return FlowMatchEulerDiscreteScheduler(
            **asdict(self.config.scheduler_config)
        )

    def build_vae(self) -> AutoencoderKL:
        """Build the VAE component."""
        return AutoencoderKL(**asdict(self.config.vae_config))

    def build_text_encoder(self) -> Qwen3:
        """Build the text encoder component.

        Uses native Qwen3 with return_hidden_states=ALL configured.
        The hidden states are extracted in model.py's _encode_prompt(),
        specifically hidden_states[-2] (second-to-last layer output).
        """
        return Qwen3(self.config.text_encoder_config)

    def build_transformer(self) -> ZImageTransformer2DModel:
        """Build the transformer component."""
        # Pass device for RoPE precomputation on GPU
        transformer_kwargs = asdict(self.config.transformer_config)
        transformer_kwargs["device"] = self.device
        return ZImageTransformer2DModel(**transformer_kwargs)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: Device | None = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: list[Tensor] | None = None,
        negative_prompt_embeds: Tensor | None = None,
        max_sequence_length: int = 512,
    ) -> tuple[list[Tensor], list[Tensor]]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            prompt_embeds,
            max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = (
                    [negative_prompt]
                    if isinstance(negative_prompt, str)
                    else negative_prompt
                )
            if len(prompt) != len(negative_prompt):
                raise ValueError(
                    "The lists of prompt and negative prompt must have the same length"
                )
            negative_prompt_embeds = self._encode_prompt(
                negative_prompt,
                device,
                negative_prompt_embeds,
                max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: str | list[str],
        device: Device | None = None,
        prompt_embeds: list[Tensor] | None = None,
        max_sequence_length: int = 512,
    ) -> list[Tensor]:
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

            if isinstance(prompt, str):
                prompt = [prompt]

            for i, prompt_item in enumerate(prompt):
                messages = [
                    {"role": "user", "content": prompt_item},
                ]
                prompt_item = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                prompt[i] = prompt_item

            text_inputs = self.tokenizer(
                prompt,
                # padding="max_length",
                # max_length=max_sequence_length,
                truncation=True,
                # return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids.to(device)
            prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: DType,
        device: Device,
        seed: int | None = None,
        latents: Tensor | None = None,
    ) -> Tensor:
        """Prepare latents for the diffusion process.

        Args:
            batch_size: Number of images to generate.
            num_channels_latents: Number of channels in the latent space.
            height: Target image height.
            width: Target image width.
            dtype: Data type for the latents.
            device: Device to place the latents on.
            seed: Optional seed for reproducible random generation.
                If None, uses random seed (non-deterministic).
            latents: Optional pre-generated latents to use instead.

        Returns:
            Latent tensor of shape (batch_size, num_channels_latents, height, width).
        """
        import torch

        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            # Use PyTorch for random generation to match diffusers exactly
            # This ensures identical results when using the same seed
            generator = torch.Generator("cpu")
            if seed is not None:
                generator.manual_seed(seed)
            latents_torch = torch.randn(
                shape, generator=generator, dtype=torch.float32
            )
            # Convert to MAX tensor and move to device
            latents = Tensor.from_dlpack(latents_torch.numpy()).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)
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

    @property
    def interrupt(self) -> Callable[[Tensor], bool] | None:
        return self._interrupt

    def progress_bar(self, total: int) -> tqdm:
        return tqdm(total=total)

    def __call__(
        self,
        # prompt: str | list[str] = None,
        prompt: Tensor | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 0.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int | None = 1,
        latents: Tensor | None = None,
        prompt_embeds: list[Tensor] | None = None,
        negative_prompt_embeds: list[Tensor] | None = None,
        output_type: str | None = "pil",
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str] = "latents",
        max_sequence_length: int = 512,
        cu_seqlens: list[Tensor] | None = None,
        max_seqlen: list[Tensor] | None = None,
    ) -> Tensor:
        r"""
        Executes the Z-Image model with the prepared inputs.

        Args:
            model_inputs: A ZImageInputs instance containing all image generation parameters
                including prompt, dimensions, guidance scale, etc.

        Returns:
            ModelOutputs containing the generated images.
        """
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        device = self.device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        batch_size = 1
        # 2. Define call parameters
        # if prompt is not None and isinstance(prompt, str):
        #     batch_size = 1
        # elif prompt is not None and isinstance(prompt, list):
        #     batch_size = len(prompt)
        # else:
        #     batch_size = len(prompt_embeds)

        # # If prompt_embeds is provided and prompt is None, skip encoding
        # if prompt_embeds is not None and prompt is None:
        #     if (
        #         self.do_classifier_free_guidance
        #         and negative_prompt_embeds is None
        #     ):
        #         raise ValueError(
        #             "When `prompt_embeds` is provided without `prompt`, "
        #             "`negative_prompt_embeds` must also be provided for classifier-free guidance."
        #         )
        # else:
        #     (
        #         prompt_embeds,
        #         negative_prompt_embeds,
        #     ) = self.encode_prompt(
        #         prompt=prompt,
        #         negative_prompt=negative_prompt,
        #         do_classifier_free_guidance=self.do_classifier_free_guidance,
        #         prompt_embeds=prompt_embeds,
        #         negative_prompt_embeds=negative_prompt_embeds,
        #         device=device,
        #         max_sequence_length=max_sequence_length,
        #     )
        # from safetensors.torch import load_file

        # # Load prompt embeddings from safetensors file
        # # Note: This is temporary! The text encoder should be used instead.
        # data = load_file("/root/prompt_embeds.safetensors")
        # prompt_embeds_torch = data["prompt_embeds"]
        # prompt_embeds_np = (
        #     prompt_embeds_torch.float().numpy()
        # )  # bfloat16 -> float32 -> numpy
        # prompt_embeds = (
        #     Tensor.from_dlpack(prompt_embeds_np).to(device).cast(DType.bfloat16)
        # )
        prompt_embeds = prompt
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels

        # Extract seed for reproducible latent generation
        seed = 0

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            DType.float32,
            device,
            seed=seed,
            latents=latents,
        )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [
                pe for pe in prompt_embeds for _ in range(num_images_per_prompt)
            ]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [
                    npe
                    for npe in negative_prompt_embeds
                    for _ in range(num_images_per_prompt)
                ]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (int(latents.shape[2]) // 2) * (
            int(latents.shape[3]) // 2
        )

        # 5. Prepare timesteps
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.base_image_seq_len,
            self.scheduler.max_image_seq_len,
            self.scheduler.base_shift,
            self.scheduler.max_shift,
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(
            int(timesteps.shape[0])
            - num_inference_steps * self.scheduler.order,
            0,
        )
        self._num_timesteps = int(timesteps.shape[0])

        # Pre-set step index to avoid expensive lookup on each step
        self.scheduler._step_index = 0

        # 6. Denoising loop
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i in range(self._num_timesteps):
            t = timesteps[i]
            # if self.interrupt:
            #     continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = F.broadcast_to(t, (int(latents.shape[0]),))
            timestep = (1000 - timestep) / 1000
            # Normalized time for time-aware config (0 at start, 1 at end)
            # Use loop index to avoid GPU sync from .item()
            t_norm = i / max(1, self._num_timesteps - 1)

            # Handle cfg truncation
            current_guidance_scale = self.guidance_scale
            if (
                self.do_classifier_free_guidance
                and self._cfg_truncation is not None
                and float(self._cfg_truncation) <= 1
            ):
                if t_norm > self._cfg_truncation:
                    current_guidance_scale = 0.0

            # Run CFG only if configured AND scale is non-zero
            apply_cfg = (
                self.do_classifier_free_guidance
                and current_guidance_scale > 0
            )

            if apply_cfg:
                latents_typed = latents.cast(DType.bfloat16)
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = (
                    prompt_embeds + negative_prompt_embeds
                )
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents.cast(DType.bfloat16)
                prompt_embeds_model_input = prompt_embeds
                timestep_model_input = timestep

            latent_model_input = F.unsqueeze(latent_model_input, 2)

            x_in = F.squeeze(latent_model_input, 0)

            model_out = self.transformer(
                x_in,
                timestep_model_input,
                prompt_embeds_model_input,
            )

            if apply_cfg:
                # Perform CFG
                pos_out = model_out_list[:actual_batch_size]
                neg_out = model_out_list[actual_batch_size:]

                noise_pred = []
                for j in range(actual_batch_size):
                    pos = pos_out[j].cast(DType.float32)
                    neg = neg_out[j].cast(DType.float32)

                    pred = pos + current_guidance_scale * (pos - neg)

                    # Renormalization
                    if (
                        self._cfg_normalization
                        and float(self._cfg_normalization) > 0.0
                    ):
                        # ori_pos_norm = torch.linalg.vector_norm(pos)
                        # new_pos_norm = torch.linalg.vector_norm(pred)
                        ori_pos_norm = F.sqrt(F.sum(pos * pos))
                        new_pos_norm = F.sqrt(F.sum(pred * pred))
                        max_new_norm = ori_pos_norm * float(
                            self._cfg_normalization
                        )
                        if new_pos_norm > max_new_norm:
                            pred = pred * (max_new_norm / new_pos_norm)

                    noise_pred.append(pred)

                noise_pred = F.stack(noise_pred, axis=0)
            else:
                # model_out is a single tensor [C, F, H, W], add batch dim
                noise_pred = F.unsqueeze(model_out.cast(DType.float32), 0)

            noise_pred = -F.squeeze(noise_pred, 2)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred.cast(DType.float32),
                t,
                latents,
            ).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(
                    self, i, t, callback_kwargs
                )

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop(
                    "prompt_embeds", prompt_embeds
                )
                negative_prompt_embeds = callback_outputs.pop(
                    "negative_prompt_embeds", negative_prompt_embeds
                )

            # call the callback, if provided
            # if i == int(timesteps.shape[0]) - 1 or (
            #     (i + 1) > num_warmup_steps
            #     and (i + 1) % self.scheduler.order == 0
            # ):
            #     progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = latents.cast(DType.bfloat16)
            latents = (
                latents / self.vae.scaling_factor
            ) + self.vae.shift_factor

            image = self.vae.decoder(latents)  # .sample

        # Offload all models
        # self.maybe_free_model_hooks()

        return image

