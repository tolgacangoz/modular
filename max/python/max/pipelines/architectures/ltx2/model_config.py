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

from typing import TYPE_CHECKING, Any, ClassVar

from max.driver import Device, load_devices
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from typing_extensions import Self

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


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
    vocoder_config: dict[str, Any] = {}


class LTX2Config(LTX2ConfigBase):
    config_name: ClassVar[str] = "config.json"

    @classmethod
    def initialize(cls, pipeline_config: "PipelineConfig") -> Self:
        """Initialize LTX2Config from a PipelineConfig.

        Args:
            pipeline_config: The pipeline configuration.

        Returns:
            An initialized LTX2Config instance.
        """
        if pipeline_config.model.quantization_encoding is None:
            raise ValueError("Quantization encoding is required for LTX2Config")

        # Get the huggingface config if available
        hf_config = pipeline_config.model.huggingface_config
        config_dict = hf_config.to_dict() if hf_config is not None else {}

        # Convert device specs to devices
        devices = load_devices(pipeline_config.model.device_specs)

        # Generate config using the existing generate method
        config_base = cls.generate(
            config_dict,
            pipeline_config.model.quantization_encoding,
            devices,
        )

        # Convert to LTX2Config (which is just LTX2ConfigBase with extra methods)
        return cls(**config_base.model_dump())

    def get_max_seq_len(self) -> int:
        """Get the maximum sequence length.

        For pixel generation models, this returns a placeholder value
        as sequence length is not applicable.

        Returns:
            A placeholder sequence length value.
        """
        # Pixel generation models don't have a text sequence length constraint
        # Return a reasonable default
        return 128  # Standard Gemma3ForConditionalGeneration text encoder max length used in diffusion models

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> LTX2ConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in LTX2ConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": encoding.dtype,
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return LTX2ConfigBase(**init_dict)
