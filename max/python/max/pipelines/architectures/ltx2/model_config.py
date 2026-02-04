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

from typing import Any, ClassVar

from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase


class LTX2ConfigBase(MAXModelConfigBase):
    activation_fn: str = "gelu-approximate"
    attention_bias: bool = True
    attention_head_dim: int = 128
    attention_out_bias: bool = True
    audio_attention_head_dim: int = 64
    audio_cross_attention_dim: int = 2048
    audio_hop_length: int = 160
    audio_in_channels: int = 128
    audio_num_attention_heads: int = 32
    audio_out_channels: int = 128
    audio_patch_size: int = 1
    audio_patch_size_t: int = 1
    audio_pos_embed_max_pos: int = 20
    audio_sampling_rate: int = 16000
    audio_sampling_rate_audio: int = 16000  # Added for audio path
    audio_scale_factor: int = 4
    base_height: int = 2048
    base_width: int = 2048
    caption_channels: int = 3840
    causal_offset: int = 1
    cross_attention_dim: int = 4096
    cross_attn_timestep_scale_multiplier: int = 1000
    in_channels: int = 128
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-06
    num_attention_heads: int = 32
    num_layers: int = 48
    out_channels: int = 128
    patch_size: int = 1
    patch_size_t: int = 1
    pos_embed_max_pos: int = 20
    qk_norm: str = "rms_norm_across_heads"
    rope_double_precision: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "split"
    timestep_scale_multiplier: int = 1000
    vae_scale_factors: tuple[int, int, int] = (8, 32, 32)

    # Added to store nested configs if provided
    transformer_config: dict[str, Any] = {}
    vae_config: dict[str, Any] = {}
    vae_audio_config: dict[str, Any] = {}
    text_encoder_config: dict[str, Any] = {}
    connectors_config: dict[str, Any] = {}


class LTX2Config(LTX2ConfigBase):
    config_name: ClassVar[str] = "config.json"

    @staticmethod
    def generate(
        pipeline_config: Any,
        **kwargs: Any,
    ) -> LTX2ConfigBase:
        # Extract generic config from pipeline_config if available
        config_dict = {}
        if hasattr(pipeline_config, "model") and hasattr(
            pipeline_config.model, "config"
        ):
            config_dict = pipeline_config.model.config

        # Merge with transformer_config or other kwargs if they contain base params
        if "transformer_config" in kwargs:
            config_dict.update(kwargs["transformer_config"])

        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in LTX2ConfigBase.__annotations__
        }

        # Handle specific nested configs
        for cfg_name in [
            "transformer_config",
            "vae_config",
            "vae_audio_config",
            "text_encoder_config",
            "connectors_config",
        ]:
            if cfg_name in kwargs:
                init_dict[cfg_name] = kwargs[cfg_name]

        # Handle mandatory fields from pipeline_config/kwargs
        if "dtype" in kwargs:
            init_dict["dtype"] = kwargs["dtype"]
        elif hasattr(pipeline_config, "model"):
            init_dict["dtype"] = (
                pipeline_config.model.quantization_encoding.dtype
            )

        if "device" in kwargs:
            init_dict["device"] = kwargs["device"]
        elif (
            hasattr(pipeline_config, "device_specs")
            and pipeline_config.device_specs
        ):
            init_dict["device"] = DeviceRef.from_device(
                pipeline_config.device_specs[0]
            )

        return LTX2ConfigBase(**init_dict)
