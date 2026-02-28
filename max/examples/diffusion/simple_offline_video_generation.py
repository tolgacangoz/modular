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
from typing import cast

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
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.config.config_enums import SupportedEncoding
from max.pipelines.lib.interfaces import DiffusionPipeline
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


def encode_video(
    video: list[PIL.Image.Image] | np.ndarray | torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    audio: torch.Tensor,
    audio_sample_rate: int,
    output_path: str,
    video_chunks_number: int = 1,
) -> None:
    """
    Encodes a video with audio using the PyAV library. Based on code from the original LTX-2 repo:
    https://github.com/Lightricks/LTX-2/blob/4f410820b198e05074a1e92de793e3b59e9ab5a0/packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py#L182

    Args:
        video (`List[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`):
            A video tensor of shape [frames, height, width, channels] with integer pixel values in [0, 255]. If the
            input is a `np.ndarray`, it is expected to be a float array with values in [0, 1] (which is what pipelines
            usually return with `output_type="np"`).
        fps (`int`)
            The frames per second (FPS) of the encoded video.
        audio (`torch.Tensor`, *optional*):
            An audio waveform of shape [audio_channels, samples].
        audio_sample_rate: (`int`, *optional*):
            The sampling rate of the audio waveform. For LTX 2, this is typically 24000 (24 kHz).
        output_path (`str`):
            The path to save the encoded video to.
        video_chunks_number (`int`, *optional*, defaults to `1`):
            The number of chunks to split the video into for encoding. Each chunk will be encoded separately. The
            number of chunks to use often depends on the tiling config for the video VAE.
    """
    if isinstance(video, list) and isinstance(video[0], PIL.Image.Image):
        # Pipeline output_type="pil"; assumes each image is in "RGB" mode
        video_frames = [np.array(frame) for frame in video]
        video = np.stack(video_frames, axis=0)
        video = torch.from_numpy(video)
    elif isinstance(video, np.ndarray):
        # Pipeline output_type="np"
        is_denormalized = np.logical_and(np.zeros_like(video) <= video, video <= np.ones_like(video))
        if np.all(is_denormalized):
            video = (video * 255).round().astype("uint8")
        else:
            logger.warning(
                "Supplied `numpy.ndarray` does not have values in [0, 1]. The values will be assumed to be pixel "
                "values in [0, ..., 255] and will be used as is."
            )
        video = torch.from_numpy(video)

    if isinstance(video, torch.Tensor):
        # Split into video_chunks_number along the frame dimension
        video = torch.tensor_split(video, video_chunks_number, dim=0)
        video = iter(video)

    first_chunk = next(video)

    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")

        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    for video_chunk in tqdm(chain([first_chunk], video), total=video_chunks_number, desc="Encoding video chunks"):
        video_chunk_cpu = video_chunk.to("cpu").numpy()
        for frame_array in video_chunk_cpu:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    container.close()


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
            quantization_encoding=args.encoding,
            device_specs=device_specs,
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
    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
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
            encode_video(
                video_item.video_data,
                args.frame_rate,
                audio_data,
                24_000,
                output_path,
            )
        elif video_item.video_url:
            print(f"Video available at URL: {video_item.video_url}")


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
