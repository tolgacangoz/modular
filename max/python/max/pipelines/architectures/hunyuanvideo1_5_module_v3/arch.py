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
from max.interfaces import ImageGenerationContext, PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.architectures.llama3 import weight_adapters
from max.pipelines.architectures.qwen2_5vl.qwen2_5vl import Qwen2_5VL
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .model import HunyuanVideo15Model
from .nn.autoencoder_kl import AutoencoderKLHunyuanVideo15
from .nn.transformer_hunyuan_video15 import HunyuanVideo15Transformer3DModel
from .scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

hunyuan_video15_arch = SupportedArchitecture(
    name="HunyuanVideo15Pipeline",
    task=PipelineTask.IMAGE_GENERATION,
    example_repo_ids=["Tongyi-MAI/HunyuanVideo15"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED]},
    pipeline_model=HunyuanVideo15Model,
    scheduler=FlowMatchEulerDiscreteScheduler,
    vae=AutoencoderKLHunyuanVideo15,
    text_encoder=Qwen2_5VL,
    tokenizer=TextTokenizer,
    transformer=HunyuanVideo15Transformer3DModel,
    context_type=ImageGenerationContext,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict
    },
)
