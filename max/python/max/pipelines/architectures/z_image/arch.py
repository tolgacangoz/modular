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

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask, PixelContext
from max.pipelines.lib import SupportedArchitecture, SupportedEncoding, PixelGenerationTokenizer

from .model import ZImagePipeline

z_image_arch = SupportedArchitecture(
    name="ZImagePipeline",
    task=PipelineTask.PIXEL_GENERATION,
    example_repo_ids=["Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={SupportedEncoding.bfloat16: []},
    pipeline_model=ZImagePipeline,
    tokenizer=PixelGenerationTokenizer,
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
)
