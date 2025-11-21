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
"""Implements the QwenImageEditPlus image editing model architecture."""

from max.nn import (
    Module,
)

from .model_config import QwenImageEditPlusConfig
from .nn.autoencoderkl_qwenimage import AutoencoderKLQwenImage


class QwenImageEditPlus(Module):
    """The overall interface to the QwenImageEditPlus model."""

    def __init__(self, config: QwenImageEditPlusConfig) -> None:
        self.config = config
        self.vae = self.build_VAE()
        self.transformer = self.build_transformer()
        self.text_encoder = self.build_text_encoder()

    def build_VAE(self) -> AutoencoderKLQwenImage:
        return AutoencoderKLQwenImage(
            config=self.config.vae_config,
        )

    def build_transformer(self) -> Module:
        """Return the language model component."""
        raise NotImplementedError(
            "QwenImageEditPlus transformer is not yet implemented."
        )

    def __call__(self, *args, **kwargs):
        """This class is not meant to be called directly. Use the component models instead."""
        raise NotImplementedError(
            "QwenImageEditPlus is a container class. Use vision_encoder() or language_model() instead"
        )
