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
    LTX2TextConnectorsConfig,
    LTX2TransformerConfig,
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
        state_dict = {
            key: value.data().astype(self.config.dtype)
            for key, value in self.weights.items()
        }
        with F.lazy():
            ltx2transformer = LTX2VideoTransformer3DModel(self.config)
            ltx2transformer.load_state_dict(state_dict)
            ltx2transformer.to(self.devices[0])
        self.model = ltx2transformer.compile(
            *ltx2transformer.input_types(),# weights=state_dict
        )
        return self.model

    def __call__(self, **kwargs: Any) -> Any:
        # 1. Provide fallbacks for optional tensors matching ltx2.py forward() logic.
        # This ensures the compiled engine always receives all required inputs.
        if "audio_timestep" not in kwargs or kwargs["audio_timestep"] is None:
            kwargs["audio_timestep"] = kwargs.get("timestep")

        # 2. Filter to only pass tensors that are part of input_types
        # and match the forward signature of the compiled graph.
        tensor_keys = {
            "hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
            "audio_encoder_hidden_states",
            "timestep",
            "audio_timestep",
            "encoder_attention_mask",
            "audio_encoder_attention_mask",
            "video_coords",
            "audio_coords",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in tensor_keys}
        return self.model(**filtered_kwargs)


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
        state_dict = {
            key: value.data().astype(self.config.dtype)
            for key, value in self.weights.items()
        }
        with F.lazy():
            vocoder = LTX2Vocoder(self.config)
            vocoder.load_state_dict(state_dict)
            vocoder.to(self.devices[0])
        self.model = vocoder.compile(*vocoder.input_types(),
                                     # weights=state_dict
                                     )
        return self.model

    def __call__(self, mel_spectrogram: Tensor, **kwargs: Any) -> Any:
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
        state_dict = {
            key: value.data().astype(self.config.dtype)
            for key, value in self.weights.items()
        }
        with F.lazy():
            connectors = LTX2TextConnectors(self.config)
            connectors.load_state_dict(state_dict)
            connectors.to(self.devices[0])
        self.model = connectors.compile(
            *connectors.input_types(),# weights=state_dict
        )
        return self.model

    def __call__(
        self,
        text_encoder_hidden_states: Tensor,
        attention_mask: Tensor,
        **kwargs: Any,
    ) -> Any:
        return self.model(text_encoder_hidden_states, attention_mask)
