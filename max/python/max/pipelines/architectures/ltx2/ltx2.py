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

import torch
from max.driver import Device
from max.nn import Module
from transformers import Gemma3ForConditionalGeneration

from ..autoencoders import (
    AutoencoderKLLTX2AudioModel,
    AutoencoderKLLTX2VideoModel,
)
from ..lib.diffusion_schedulers import FlowMatchEulerDiscreteScheduler
from .nn.connectors import LTX2TextConnectors
from .nn.transformer_ltx2 import LTX2Transformer2DModel
from .nn.vocoder import LTX2Vocoder


class LTX2(Module):
    """
    LTX-2 container module that groups the transformer, VAE, and text encoder.
    """

    def __init__(self, config, device: Device):
        super().__init__()
        self.transformer = LTX2Transformer2DModel(config.transformer_config)
        self.vae = AutoencoderKLLTX2VideoModel(config.vae_config)
        self.vae_audio = AutoencoderKLLTX2AudioModel(config.vae_audio_config)
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            "Lightricks/LTX-2",
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        self.connectors = LTX2TextConnectors(config.connectors_config)
        self.vocoder = LTX2Vocoder(**config.vocoder_config)
        # self.scheduler = FlowMatchEulerDiscreteScheduler()
