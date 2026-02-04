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

from max.driver import Device
from max.nn import Module

from ..autoencoders import (
    AutoencoderKLLTX2AudioModel,
    AutoencoderKLLTX2VideoModel,
)
from ..gemma3multimodal.model import Gemma3_MultiModalModel
from ..lib.diffusion_schedulers import FlowMatchEulerDiscreteScheduler
from .nn.transformer_ltx2 import LTX2Transformer2DModel


class LTX2(Module):
    """
    LTX-2 container module that groups the transformer, VAE, and text encoder.
    """

    def __init__(self, config, device: Device):
        super().__init__()
        self.transformer = LTX2Transformer2DModel(config.transformer_config)
        self.vae = AutoencoderKLLTX2VideoModel(config.vae_config)
        self.vae_audio = AutoencoderKLLTX2AudioModel(config.vae_audio_config)
        self.text_encoder = Gemma3_MultiModalModel(config.text_encoder_config)
        self.scheduler = FlowMatchEulerDiscreteScheduler()
