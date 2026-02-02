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

from collections.abc import Sequence
from typing import Any, ClassVar

from max.driver import Device
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from pydantic import Field


class ZImageTransformer2DModelConfigBase(MAXModelConfigBase):
    all_patch_size: Sequence[int] = ((2,),)
    all_f_patch_size: Sequence[int] = ((1,),)
    in_channels: int = (16,)
    dim: int = (3840,)
    n_layers: int = (30,)
    n_refiner_layers: int = (2,)
    n_heads: int = (30,)
    n_kv_heads: int = (30,)
    norm_eps: float = (1e-5,)
    qk_norm: bool = (True,)
    cap_feat_dim: int = (2560,)
    rope_theta: float = (256.0,)
    t_scale: float = (1000.0,)
    axes_dims: Sequence[int] = ((32, 48, 48),)
    axes_lens: Sequence[int] = ((1536, 512, 512),)
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)


class ZImageTransformer2DModelConfig(ZImageTransformer2DModelConfigBase):
    config_name: ClassVar[str] = "config.json"

    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> ZImageTransformer2DModelConfigBase:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in ZImageTransformer2DModelConfigBase.__annotations__
        }
        init_dict.update(
            {
                "dtype": encoding.dtype,
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return ZImageTransformer2DModelConfigBase(**init_dict)
