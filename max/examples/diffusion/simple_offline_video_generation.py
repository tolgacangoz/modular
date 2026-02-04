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

"""Offline video generation example using LTX-2 diffusion model.

This module demonstrates end-to-end video generation using:
- PixelGenerationRequest: Create generation requests with prompts
- PixelGenerationTokenizer: Tokenize prompts and prepare model context
- LTX2Pipeline: Execute the LTX-2 diffusion model to generate video frames

Usage:
    python ltx2_offline_generation.py \
        --model Lightricks/LTX-2 \
        --prompt "A cat walking in a garden" \
        --num-frames 24 \
        --output output.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import os

import numpy as np
from max.driver import CPU, Device, DeviceSpec, load_devices
from max.engine import InferenceSession
from max.interfaces import (
    PixelGenerationRequest,
    RequestID,
)
from max.pipelines import PipelineConfig
from max.pipelines.architectures.ltx2.pipeline_ltx2 import LTX2Pipeline
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.pipeline_variants.utils import get_weight_paths
from max.tensor import Tensor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pixel generation example.

    Args:
        argv: Optional explicit list of argument strings. If None, arguments
            are read from sys.argv[1:].

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate videos with LTX-2 diffusion model.",
    )
    parser.add_argument(
        "--model",
        default="Lightricks/LTX-2",
        help="Identifier of the model to use for generation (default: Lightricks/LTX-2).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt describing the image to generate.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt to guide what NOT to generate.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of generated image in pixels. None uses model's native resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of generated video in pixels (default: 512).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=24,
        help="Number of video frames to generate (default: 24).",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=24,
        help="Frame rate for the output video (default: 24 fps).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps. More steps = higher quality but slower.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale for classifier-free guidance. Set to 1.0 to disable CFG.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output filename for the generated video.",
    )

    args = parser.parse_args(argv)

    # Validate arguments
    assert args.prompt, "Prompt must be a non-empty string."
    if args.height is not None:
        assert args.height > 0, "Height must be a positive integer."
    if args.width is not None:
        assert args.width > 0, "Width must be a positive integer."
    assert args.num_inference_steps > 0, (
        "num-inference-steps must be a positive integer."
    )
    assert args.guidance_scale > 0.0, "guidance-scale must be positive."

    return args

# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fractions import Fraction
from typing import Optional

import av


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    samples: Tensor,
    audio_sample_rate: int,
) -> None:
    if samples.rank == 1:
        samples = samples[:, None]

    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != DType.int16:
        samples = Tensor.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).cast(DType.int16)

    frame_in = av.AudioFrame.from_ndarray(
        np.from_dlpack(samples.reshape(1, -1).to(CPU())),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)


def video_postprocess(
    video: Tensor, fps: int, audio: Tensor | None = None, audio_sample_rate: int | None = None, output_path: str = "./output.mp4"
) -> None:
    video_np = np.from_dlpack(video.to(CPU()).cast(DType.float32))
    # Scale from [0, 1] to [0, 255] uint8
    video_np = (video_np * 255.0).round().astype(np.uint8)

    # Take first sample from batch
    if video_np.ndim == 5:
        video_np = video_np[0]

    height, width = video_np.shape[1], video_np.shape[2]

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")

        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    for frame_array in video_np:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    container.close()




def save_video(
    pixel_data: np.ndarray, output_path: str, frame_rate: int = 24
) -> None:
    """Save generated pixel data as a video file.

    Args:
        pixel_data: Numpy array of shape (F, H, W, C) with values in [0, 1]
        output_path: Path where the video should be saved
        frame_rate: Frame rate for the output video
    """
    try:
        import imageio

        # Convert from float [0, 1] to uint8 [0, 255]
        pixel_data = (pixel_data * 255).clip(0, 255).astype(np.uint8)

        # Save as video
        imageio.mimwrite(output_path, pixel_data, fps=frame_rate)
        print(f"Video saved to: {output_path}")
    except ImportError:
        print("WARNING: imageio not available, saving as numpy array instead")
        np.save(output_path.replace(".mp4", ".npy"), pixel_data)
        print(f"Pixel data saved to: {output_path.replace('.mp4', '.npy')}")


def save_image(pixel_data: np.ndarray, output_path: str) -> None:
    """Save generated pixel data as an image file.

    Args:
        pixel_data: Numpy array of shape (H, W, C) with values in [0, 1]
        output_path: Path where the image should be saved
    """
    try:
        from PIL import Image

        # Convert from float [0, 1] to uint8 [0, 255]
        pixel_data = (pixel_data * 255).clip(0, 255).astype(np.uint8)

        # Create and save image
        image = Image.fromarray(pixel_data)
        image.save(output_path)
        print(f"Image saved to: {output_path}")
    except ImportError:
        print("WARNING: PIL not available, saving as numpy array instead")
        np.save(output_path.replace(".png", ".npy"), pixel_data)
        print(f"Pixel data saved to: {output_path.replace('.png', '.npy')}")


async def generate_video(args: argparse.Namespace) -> None:
    """Main video generation logic.

    Args:
        args: Parsed command-line arguments
    """
    print(f"Loading model: {args.model}")

    # Step 1: Initialize pipeline configuration
    config = PipelineConfig(
        model_path=args.model,
        device_specs=[DeviceSpec.accelerator()],
        use_legacy_module=False,
    )

    # Step 2: Initialize the tokenizer
    # The tokenizer handles prompt encoding and context preparation
    # LTX-2 uses Gemma3 text encoder, so we use the text_encoder subfolder
    tokenizer = PixelGenerationTokenizer(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",  # Gemma3 tokenizer subfolder
        max_length=256,  # Gemma3 max length
    )

    # Step 3: Initialize devices, session, weights, and the LTX2 pipeline
    devices = load_devices(config.model.device_specs)
    session = InferenceSession(devices=devices)
    config.configure_session(session)
    weight_paths = get_weight_paths(config.model)

    pipeline = LTX2Pipeline(
        pipeline_config=config,
        session=session,
        devices=devices,
        weight_paths=weight_paths,
    )

    print(f"Generating video for prompt: '{args.prompt}'")
    print(f"Video settings: {args.num_frames} frames at {args.frame_rate} fps")

    # Step 4: Create a PixelGenerationRequest
    request = PixelGenerationRequest(
        request_id=RequestID(),
        model_name=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
    )

    print(
        f"Parameters: steps={args.num_inference_steps}, guidance={args.guidance_scale}"
    )

    # Step 5: Create a PixelContext object from the request
    # The tokenizer handles prompt tokenization, timestep scheduling,
    # latent initialization, and all other preprocessing
    context = await tokenizer.new_context(request)

    print(
        f"Context created: {context.height}x{context.width}, {context.num_inference_steps} steps"
    )

    # Step 6: Prepare inputs and execute the LTX2 pipeline directly
    print("Running diffusion model...")
    model_inputs = pipeline.prepare_inputs(context)
    pipeline_output = pipeline.execute(model_inputs)

    print("Generation complete!")

    # Step 7: Post-process the pixel and audio data
    # LTX2Pipeline now returns a Flux-style LTX2PipelineOutput with Max tensors for video and audio
    video_tensor = pipeline_output.video
    audio_tensor = pipeline_output.audio

    # LTX2 usage: 24kHz audio sample rate
    video_postprocess(
        video_tensor,
        fps=args.frame_rate,
        audio=audio_tensor,
        audio_sample_rate=24000,
        output_path=args.output
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the pixel generation example.

    Args:
        argv: Optional explicit list of argument strings. If None, arguments
            are read from sys.argv[1:].

    Returns:
        Process exit code. 0 indicates success.
    """
    args = parse_args(argv)

    try:
        asyncio.run(generate_video(args))
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    raise SystemExit(main())
