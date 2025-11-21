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
from max.interfaces import PipelineTask
from max.pipelines.lib import SupportedArchitecture, SupportedEncoding

from .model import QwenImageEditPlusModel
from .tokenizer import Qwen2_5_VLTokenizer
from .weight_adapters import convert_qwenimage_edit_plus_model_state_dict

qwenimage_edit_plus_arch = SupportedArchitecture(
    name="QwenImageEditPlusPipeline",
    task=PipelineTask.IMAGE_GENERATION,
    example_repo_ids=[
        "Qwen/Qwen-Image-Edit-2509",
    ],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: None,
        SupportedEncoding.bfloat16: None,
    },
    weight_adapters={
        WeightsFormat.safetensors: convert_qwenimage_edit_plus_model_state_dict,
    },
    pipeline_model=QwenImageEditPlusModel,
    tokenizer=Qwen2_5_VLTokenizer,
    required_arguments={
        "enable_chunked_prefill": False,
    },
)
