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

from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)
from max.pipelines.core import TextContext

from .model import ZImageModel
from .qwen3_encoder import Qwen3Encoder
from .nn.transformer_z_image import ZImageTransformer2DModel
from .nn.autoencoder_kl import AutoencoderKL
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

z_image_arch = SupportedArchitecture(
    name="ZImagePipeline",
    task=PipelineTask.IMAGE_GENERATION,
    example_repo_ids=[
        "Tongyi-MAI/Z-Image-Base",
        "Tongyi-MAI/Z-Image-Turbo",
    ],
    # Z-Image uses standard safetensors layout for Qwen3 text encoder and
    # diffusers-style layouts for VAE/transformer, so no custom weight adapter
    # is required at the architecture level.
    default_weights_format=None,
    multi_gpu_supported=True,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: KVCacheStrategy.PAGED,
        SupportedEncoding.float32: KVCacheStrategy.PAGED,
    },
    weight_adapters={},
    pipeline_model=ZImageModel,
    scheduler=FlowMatchEulerDiscreteScheduler,
    vae=AutoencoderKL,
    text_encoder=Qwen3Encoder,
    tokenizer=TextTokenizer,
    transformer=ZImageTransformer2DModel,
    rope_type=RopeType.normal,
    required_arguments={
        "enable_chunked_prefill": False,
    },
    context_type=TextContext,
)
