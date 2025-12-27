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
"""Image generation interface definitions for Modular's MAX API.

This module provides data structures and interfaces for handling image generation
responses, including status tracking and image data encapsulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

import msgspec
import numpy as np
import numpy.typing as npt
from max.interfaces.context import BaseContext, SamplingParams
from max.interfaces.pipeline import PipelineInputs, PipelineOutput
from max.interfaces.request import Request, RequestID
from max.interfaces.status import GenerationStatus
from typing_extensions import TypeVar


@dataclass(frozen=True)
class ImageGenerationRequest(Request):
    model: str = field()
    """The name of the model to be used for generating images. This should match
    the available models on the server and determines the behavior and
    capabilities of the response generation.
    """

    input: str | None = None
    """The text to generate images for. The maximum length is 4096 characters."""

    image_prompt_tokens: list[int] = field(default_factory=list)
    """The prompt image IDs to use for image generation."""

    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    """Request sampling configuration options."""

    prompt: list[int] | str | None = field(default=None)
    """Optionally provide a preprocessed list of token ids or a prompt string to pass as input directly into the model.
    This replaces automatically generating TokenGeneratorRequestMessages given the input, image prompt tokens fields."""

    streaming: bool = False
    """Whether to stream the image generation."""

    guidance_scale: float = 0.0
    """Guidance scale for classifier-free guidance. Set to 0 to disable CFG."""

    height: int | None = 1024
    """Height of generated image in pixels. Defaults to model's default (typically 1024)."""

    width: int | None = 1024
    """Width of generated image in pixels. Defaults to model's default (typically 1024)."""

    num_inference_steps: int = 50
    """Number of denoising steps. More steps = higher quality but slower."""

    negative_prompt: str | None = None
    """Negative prompt to guide what NOT to generate."""

    num_images_per_prompt: int = 1
    """Number of images to generate per prompt."""

    def __post_init__(self) -> None:
        if self.prompt is None and self.input is None:
            raise RuntimeError("either token_ids or input must be provided.")


class ImageGenerationMetadata(
    msgspec.Struct, tag=True, omit_defaults=True, kw_only=True
):
    """
    Represents metadata associated with image generation.

    This class will eventually replace the metadata dictionary used throughout
    the ImageGenerationOutput object, providing a structured and type-safe
    alternative for image generation metadata.

    Configuration:
        model_name: Name of the model used for generation.
        request_id: Unique identifier for the generation request.
    """

    model_name: str | None = None
    request_id: RequestID | None = None

    def to_dict(self) -> dict[str, int | float | str | bool]:
        """
        Convert the metadata to a dictionary format.

        Returns:
            dict[str, any]: Dictionary representation of the metadata.
        """
        result = {}
        for attr in self.__annotations__:
            if value := getattr(self, attr, None):
                result[attr] = value
        return result


ImageGenerationContextType = TypeVar(
    "ImageGenerationContextType", bound=BaseContext
)
"""Type variable for image generation context types.

This type variable is bound to BaseContext and represents the specific context
type used in image generation pipelines. It allows for type-safe generic
programming while ensuring that all context types inherit from BaseContext
and maintain the required interface for image generation operations.
"""


@dataclass
class ImageGenerationContext:
    """Context for image generation requests.

    This is a simple context that implements BaseContext protocol for diffusion
    model pipelines. It includes fields for both image generation parameters
    and compatibility with the text generation scheduler.

    Attributes:
        request_id: Unique identifier for this request.
        prompt: Text prompt for image generation.
        max_length: Maximum sequence length for tokenization (required for msgspec).
        height: Height of generated image in pixels.
        width: Width of generated image in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale (0 to disable CFG).
        negative_prompt: Negative prompt for what NOT to generate.
        num_images_per_prompt: Number of images to generate.
        model_name: Name of the model (for scheduler compatibility).
        status: Current generation status.
    """
    request_id: RequestID
    prompt: str
    max_length: int = 4096  # Default max sequence length for text encoder
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 0.0
    negative_prompt: str | None = None
    num_images_per_prompt: int = 1
    model_name: str = ""  # For scheduler compatibility
    _status: GenerationStatus = field(default=GenerationStatus.ACTIVE)

    @property
    def status(self) -> GenerationStatus:
        """Current generation status of the request."""
        return self._status

    @status.setter
    def status(self, value: GenerationStatus) -> None:
        """Update the generation status."""
        self._status = value

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self._status.is_done

    @property
    def needs_ce(self) -> bool:
        """Whether this context needs context encoding.

        For image generation, we never need context encoding since
        we process the full prompt at once through the text encoder.
        """
        return False

    @property
    def active_length(self) -> int:
        """Current sequence length for batch constructor compatibility."""
        return 1

    @property
    def current_length(self) -> int:
        """Current length for batch constructor compatibility."""
        return 1

    def compute_num_available_steps(self, max_seq_len: int) -> int:
        """Compute number of available steps for scheduler compatibility.

        For image generation, this returns the number of inference steps.
        """
        return self.num_inference_steps


@dataclass(frozen=True)
class ImageGenerationInputs(
    PipelineInputs, Generic[ImageGenerationContextType]
):
    """Input data structure for image generation pipelines.

    This class represents the input data required for image generation operations
    within the pipeline framework. It extends PipelineInputs and provides type-safe
    generic support for different image generation context types.
    """

    batch: dict[RequestID, ImageGenerationContextType]
    """A dictionary mapping RequestID to ImageGenerationContextType instances.
    This batch structure allows for processing multiple image generation
    requests simultaneously while maintaining request-specific context
    and configuration data.
    """


class ImageGenerationOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """Represents a response from the image generation API.

    This class encapsulates the result of an image generation request, including
    the final status, generated image data, and optional buffered speech tokens.
    """

    final_status: GenerationStatus
    """The final status of the generation process."""
    steps_executed: int
    """The number of steps previously executed."""
    image_data: npt.NDArray[np.float32] = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """The generated image data, if available."""
    metadata: ImageGenerationMetadata = msgspec.field(
        default_factory=ImageGenerationMetadata
    )
    """Metadata associated with the image generation, such as chunk information, prompt details, or other relevant context."""

    @property
    def is_done(self) -> bool:
        """Indicates whether the audio generation process is complete.

        Returns:
            :class:`bool`: ``True`` if generation is done, ``False`` otherwise.
        """
        return self.final_status.is_done


def _check_image_generator_output_implements_pipeline_output(
    x: ImageGenerationOutput,
) -> PipelineOutput:
    return x
