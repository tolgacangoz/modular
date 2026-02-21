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

"""Simple offline pixel generation example using diffusion models.

This module demonstrates end-to-end video generation using:
- OpenResponsesRequest: Create generation requests with prompts
- PixelGenerationTokenizer: Tokenize prompts and prepare model context
- PixelGenerationPipeline: Execute the diffusion model to generate pixels

Usage:
    ./bazelw run //max/examples/diffusion:simple_offline_video_generation -- \
        --model Lightricks/LTX-2 \
        --prompt "A cat walking in a garden" \
        --num-frames 24 \
        --output output.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os

from max.driver import DeviceSpec
from max.interfaces import (
    PixelGenerationInputs,
    RequestID,
)
from max.interfaces.provider_options import (
    ProviderOptions,
    VideoProviderOptions,
)
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    OpenResponsesRequestBody,
    OutputAudioContent,
    OutputVideoContent,
)
from max.pipelines import PipelineConfig
from max.pipelines.architectures.ltx2.pipeline_ltx2 import (
    LTX2Pipeline,
)
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.config_enums import SupportedEncoding
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)


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
    from typing import get_args
    parser.add_argument(
        "--encoding",
        type=str,
        default="bfloat16",
        choices=list(get_args(SupportedEncoding)),
        help="Weight encoding type (default: bfloat16).",
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


def save_video(video_data: str, output_path: str) -> None:
    """Save base64-encoded video data to a file.

    Args:
        video_data: Base64-encoded video data string
        output_path: Path where the video should be saved
    """
    try:
        video_bytes = base64.b64decode(video_data)
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        print(f"Video saved to: {output_path}")
    except Exception as e:
        print(f"WARNING: Cannot save video: {e}")
        print(f"Base64 data length: {len(video_data)} chars")


def save_audio(audio_data: str, output_path: str) -> None:
    """Save base64-encoded audio data to a file.

    Args:
        audio_data: Base64-encoded audio data string
        output_path: Path where the audio should be saved
    """
    try:
        audio_bytes = base64.b64decode(audio_data)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"Audio saved to: {output_path}")
    except Exception as e:
        print(f"WARNING: Cannot save audio: {e}")
        print(f"Base64 data length: {len(audio_data)} chars")


async def generate_video(args: argparse.Namespace) -> None:
    """Main video generation logic.

    Args:
        args: Parsed command-line arguments
    """
    print(f"Loading model: {args.model}")

    import max.driver

    if max.driver.accelerator_count() > 0:
        device_specs = [DeviceSpec.accelerator()]
    else:
        print("No accelerator found, using CPU.")
        device_specs = [DeviceSpec.cpu()]

    config = PipelineConfig(
        model_path=args.model,
        device_specs=device_specs,
        quantization_encoding=args.encoding,
        use_legacy_module=False,
    )

    # Step 2: Initialize the tokenizer
    # The tokenizer handles prompt encoding and context preparation
    tokenizer = PixelGenerationTokenizer(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",  # Tokenizer is in a subfolder for diffusion models
        max_length=1024,
    )

    # Step 3: Initialize the pipeline
    # The pipeline executes the diffusion model
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=LTX2Pipeline,
    )

    print(f"Generating video for prompt: '{args.prompt}'")

    # Step 4: Create a OpenResponsesRequest
    body = OpenResponsesRequestBody(
        model=args.model,
        input=args.prompt,
        seed=args.seed,
        provider_options=ProviderOptions(
            video=VideoProviderOptions(
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
        ),
    )
    request = OpenResponsesRequest(request_id=RequestID(), body=body)

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

    # Step 6: Prepare inputs for the pipeline
    # Create a batch with a single context
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )

    # Step 7: Execute the pipeline
    print("Running diffusion model...")
    responses = pipeline.execute(inputs)

    # Step 8: Get the output for our request
    response = responses[context.request_id]

    # Check if generation completed successfully
    if not response.is_done:
        print(f"WARNING: Generation status: {response.final_status}")
        return

    print("Generation complete!")

    # Step 9: Extract and save content
    if not response.output:
        print("ERROR: No content generated")
        return

    # Save each generated item
    video_count = 0
    audio_count = 0

    for content_item in response.output:
        if isinstance(content_item, OutputVideoContent):
            # Determine output filename
            base_name, ext = os.path.splitext(args.output)
            if video_count > 0:
                output_path = f"{base_name}_{video_count}{ext}"
            else:
                output_path = args.output

            if content_item.video_data:
                save_video(content_item.video_data, output_path)
            elif content_item.video_url:
                print(f"Video available at URL: {content_item.video_url}")
            video_count += 1

        elif isinstance(content_item, OutputAudioContent):
            # Determine output filename (replace extension with .wav or .raw)
            base_name, _ = os.path.splitext(args.output)
            # Simple heuristic for audio extension
            ext = ".wav"
            if audio_count > 0:
                output_path = f"{base_name}_{audio_count}{ext}"
            else:
                output_path = f"{base_name}{ext}"

            if content_item.audio_data:
                save_audio(content_item.audio_data, output_path)
            elif content_item.audio_url:
                print(f"Audio available at URL: {content_item.audio_url}")
            audio_count += 1
        else:
            print(f"Unknown content type: {content_item.type}")


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
