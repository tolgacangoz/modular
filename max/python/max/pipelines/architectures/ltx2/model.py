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

from collections.abc import Callable
from typing import Any

from max import functional as F
from max.driver import Device
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.tensor import Tensor

from .ltx2 import LTX2VideoTransformer3DModel
from .model_config import (
    LTX2TransformerConfig,
    LTX2TextConnectorsConfig,
    LTX2VocoderConfig,
)
from .nn.connectors import LTX2TextConnectors
from .nn.vocoder import LTX2Vocoder


class LTX2TransformerModel(ComponentModel):
    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
        )
        self.config = LTX2TransformerConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        with F.lazy():
            ltx2transformer = LTX2VideoTransformer3DModel(self.config)
            ltx2transformer.to(self.devices[0])
        self.model = ltx2transformer.compile(*ltx2transformer.input_types(), weights=state_dict)
        return self.model

    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor,
    ) -> Any:
        return self.model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )


class LTX2VocoderModel(ComponentModel):
    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
        )
        self.config = LTX2VocoderConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        with F.lazy():
            vocoder = LTX2Vocoder(self.config)
            vocoder.to(self.devices[0])
        self.model = vocoder.compile(*vocoder.input_types(), weights=state_dict)
        return self.model

    def __call__(self, mel_spectrogram: Tensor) -> Any:
        return self.model(mel_spectrogram)


class LTX2TextConnectorsModel(ComponentModel):
    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
        )
        self.config = LTX2TextConnectorsConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        with F.lazy():
            connectors = LTX2TextConnectors(self.config)
            connectors.to(self.devices[0])
        self.model = connectors.compile(
            *connectors.input_types(), weights=state_dict
        )
        return self.model

    def __call__(
        self,
        text_encoder_hidden_states: Tensor,
        attention_mask: Tensor,
        additive_mask: bool = False,
    ) -> Any:
        return self.model(
            text_encoder_hidden_states,
            attention_mask,
            additive_mask=additive_mask,
        )
