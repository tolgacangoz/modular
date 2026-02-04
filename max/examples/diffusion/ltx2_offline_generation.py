#!/usr/bin/env python3
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

import argparse
import asyncio
from pathlib import Path

import numpy as np
from max.driver import DeviceSpec
from max.pipelines import (
    PipelineConfig,
    PixelContext,
    PixelGenerationPipeline,
    PixelGenerationTokenizer,
)


async def generate_video(args: argparse.Namespace) -> None:
    """Generate a video using the LTX2 pipeline."""

    # Step 1: Initialize pipeline configuration
    # LTX2 requires many weights, so ensure max_batch_size is reasonable
    config = PipelineConfig(
        model_path=args.model,
        device_specs=[DeviceSpec.accelerator()],
        use_legacy_module=False,
        max_batch_size=1,
    )

    # Step 2: Initialize the tokenizer
    print("Initializing tokenizer...")
    tokenizer = PixelGenerationTokenizer.from_config(config)

    # Step 3: Initialize the pipeline
    print("Initializing LTX2 pipeline (this may take a while)...")
    pipeline = PixelGenerationPipeline.from_config(config)

    # Step 4: Create a PixelContext object with video parameters
    context = PixelContext(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
    )

    # Step 5: Pre-process inputs (tokenization, latent initialization, etc.)
    print("Preparing inputs...")
    inputs = tokenizer.tokenize_and_prepare_inputs(context)

    # Step 6: Execute the pipeline
    print(f"Running LTX2 for {args.num_frames} frames...")
    outputs = pipeline.execute(inputs)

    # Step 7: Process and save the output
    pixel_data = outputs.hidden_states.to_numpy() # [B, C, F, H, W] or similar

    print(f"Result shape: {pixel_data.shape}")

    # Save the result
    output_path = Path(args.output)
    np.save(output_path.with_suffix(".npy"), pixel_data)
    print(f"Saved raw output to {output_path.with_suffix('.npy')}")


def main():
    parser = argparse.ArgumentParser(description="LTX2 Offline Generation")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to LTX2 model weights"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output file name (saves as .npy for now)",
    )
    parser.add_argument("--height", type=int, default=1024, help="Video height")
    parser.add_argument("--width", type=int, default=1024, help="Video width")
    parser.add_argument(
        "--num_frames", type=int, default=1, help="Number of frames (1 for image)"
    )
    parser.add_argument(
        "--frame_rate", type=int, default=24, help="Frame rate for video"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="CFG guidance scale"
    )

    args = parser.parse_args()
    asyncio.run(generate_video(args))


if __name__ == "__main__":
    main()
