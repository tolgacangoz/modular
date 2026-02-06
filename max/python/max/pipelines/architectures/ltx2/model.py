from collections.abc import Callable
from typing import Any

from max import functional as F
from max.driver import Device
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.tensor import Tensor

from .ltx2 import LTX2Transformer3DModel
from .model_config import LTX2Config


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
        self.config = LTX2Config.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        with F.lazy():
            ltx2 = LTX2Transformer3DModel(self.config)
            ltx2.to(self.devices[0])
        self.model = ltx2.compile(*ltx2.input_types(), weights=state_dict)
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
