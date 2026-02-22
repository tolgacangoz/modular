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
"""Provider-specific options for MAX platform and modalities."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .max import MaxProviderOptions
from .modality import ImageProviderOptions, VideoProviderOptions


class ProviderOptions(BaseModel):
    """Container for all provider-specific options.

    Includes both universal MAX options and modality-specific options.
    All options are validated at the API layer.

    Example:
        {
            "max": {"target_endpoint": "instance-123"},
            "image": {"width": 1024, "height": 768}
        }
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max: MaxProviderOptions | None = Field(
        None,
        description="Universal MAX platform options.",
    )

    image: ImageProviderOptions | None = Field(
        None,
        description="Image generation modality options.",
    )

    video: VideoProviderOptions | None = Field(
        None,
        description="Video generation modality options.",
    )

    @model_validator(mode="after")
    def _merge_video_into_image(self) -> "ProviderOptions":
        """When only video options are provided, backfill image options from them.

        The pixel-generation tokenizer reads from ``provider_options.image`` for
        parameters that are shared between image and video pipelines (guidance_scale,
        height, width, steps, negative_prompt).  When a caller supplies only
        ``video`` options we transparently propagate those shared fields so the
        tokenizer can find them, while preserving any explicitly-set ``image``
        options.
        """
        if self.video is not None and self.image is None:
            merged = ImageProviderOptions(
                negative_prompt=self.video.negative_prompt,
                height=self.video.height,
                width=self.video.width,
                steps=self.video.steps if self.video.steps is not None else 50,
                guidance_scale=(
                    self.video.guidance_scale
                    if self.video.guidance_scale is not None
                    else 3.5
                ),
            )
            return self.model_copy(update={"image": merged})
        return self

    # Add more modality fields here as needed:
    # text: TextModalityOptions | None = None
    # tts: TTSModalityOptions | None = None


__all__ = [
    "ImageProviderOptions",
    "MaxProviderOptions",
    "ProviderOptions",
    "VideoProviderOptions",
]
