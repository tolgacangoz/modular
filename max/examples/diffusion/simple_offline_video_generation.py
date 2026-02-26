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
    PipelineTask,
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
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.architectures.ltx2.pipeline_ltx2 import (
    LTX2Pipeline,
)
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.config.config_enums import SupportedEncoding
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


def mux_and_save(
    video_data: str,
    audio_data: str | None,
    output_path: str,
    frame_rate: int,
) -> None:
    """Mux base64-encoded video and audio into a single MP4 file.

    When audio is provided the video (H.264 MP4) and audio (WAV) streams are
    combined into one container with PyAV, mirroring the behaviour of the
    diffusers ``encode_video`` helper.  When no audio is available the video
    is saved as-is.

    Args:
        video_data: Base64-encoded MP4 video data string.
        audio_data: Base64-encoded WAV audio data string, or None.
        output_path: Path where the merged video should be saved.
        frame_rate: Frames per second - used for the video stream time-base.
    """
    import io

    try:
        import av
    except ImportError:
        # PyAV not installed - fall back to saving the video stream only.
        av = None  # type: ignore[assignment]

    video_bytes = base64.b64decode(video_data)

    if av is None or audio_data is None:
        # No PyAV or no audio: just write the raw video bytes.
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        print(f"Video saved to: {output_path}")
        return

    audio_bytes = base64.b64decode(audio_data)

    try:
        in_video = av.open(io.BytesIO(video_bytes), mode="r")
        in_audio = av.open(io.BytesIO(audio_bytes), mode="r")
        out = av.open(output_path, mode="w", format="mp4")

        # Remap video stream with copy codec.
        in_v_stream = in_video.streams.video[0]
        out_v_stream = out.add_stream(template=in_v_stream)

        # Add AAC audio stream.
        in_a_stream = in_audio.streams.audio[0]
        sample_rate = in_a_stream.codec_context.sample_rate or 24000
        out_a_stream = out.add_stream("aac", rate=sample_rate)
        out_a_stream.codec_context.sample_rate = sample_rate
        out_a_stream.codec_context.layout = "stereo"

        # Remux video packets directly (no re-encode).
        for packet in in_video.demux(in_v_stream):
            if packet.dts is None:
                continue
            packet.stream = out_v_stream
            out.mux(packet)

        # Re-encode audio through the AAC encoder.
        audio_next_pts = 0
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="stereo",
            rate=sample_rate,
        )
        for packet in in_audio.demux(in_a_stream):
            for frame in packet.decode():
                for rframe in resampler.resample(frame):
                    if rframe.pts is None:
                        rframe.pts = audio_next_pts
                    audio_next_pts += rframe.samples
                    for out_packet in out_a_stream.encode(rframe):
                        out.mux(out_packet)

        # Flush audio encoder.
        for out_packet in out_a_stream.encode():
            out.mux(out_packet)

        in_video.close()
        in_audio.close()
        out.close()
        print(f"Video (with audio) saved to: {output_path}")
    except Exception as e:
        # Muxing failed - fall back to writing video-only.
        print(
            f"WARNING: Audio muxing failed ({e}); saving video without audio."
        )
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        print(f"Video (no audio) saved to: {output_path}")


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
        model=MAXModelConfig(
            model_path=args.model,
            device_specs=[DeviceSpec.accelerator()],
        ),
        prefer_module_v3=True,
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        prefer_module_v3=config.prefer_module_v3,
        task=PipelineTask.PIXEL_GENERATION,
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
                num_frames=args.num_frames,
                frames_per_second=args.frame_rate,
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

    # Step 8: Get the output for our request and post-process
    response = responses[context.request_id]
    response = await tokenizer.postprocess(response)

    # Check if generation completed successfully
    if not response.is_done:
        print(f"WARNING: Generation status: {response.final_status}")
        return

    print("Generation complete!")

    # Step 9: Extract and save content
    if not response.output:
        print("ERROR: No content generated")
        return

    # Separate video and audio items so they can be muxed together.
    video_items: list[OutputVideoContent] = []
    audio_items: list[OutputAudioContent] = []
    unknown_items = []

    for content_item in response.output:
        if isinstance(content_item, OutputVideoContent):
            video_items.append(content_item)
        elif isinstance(content_item, OutputAudioContent):
            audio_items.append(content_item)
        else:
            unknown_items.append(content_item)

    for idx, video_item in enumerate(video_items):
        base_name, ext = os.path.splitext(args.output)
        output_path = args.output if idx == 0 else f"{base_name}_{idx}{ext}"

        # Pair each video with the corresponding audio track (if present).
        audio_item = audio_items[idx] if idx < len(audio_items) else None
        audio_data = audio_item.audio_data if audio_item is not None else None

        if video_item.video_data:
            mux_and_save(
                video_item.video_data,
                audio_data,
                output_path,
                args.frame_rate,
            )
        elif video_item.video_url:
            print(f"Video available at URL: {video_item.video_url}")

    # Save any audio tracks that had no matching video.
    for idx in range(len(video_items), len(audio_items)):
        audio_item = audio_items[idx]
        base_name, _ = os.path.splitext(args.output)
        audio_path = f"{base_name}_{idx}.wav" if idx > 0 else f"{base_name}.wav"
        if audio_item.audio_data:
            audio_bytes = base64.b64decode(audio_item.audio_data)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            print(f"Audio saved to: {audio_path}")
        elif audio_item.audio_url:
            print(f"Audio available at URL: {audio_item.audio_url}")

    for item in unknown_items:
        print(f"Unknown content type: {item.type}")


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
