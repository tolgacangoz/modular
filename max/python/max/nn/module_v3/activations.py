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
"""Activation function modules for module_v3."""

import max.experimental.functional as F
from max.experimental.tensor import Tensor

from .module import Module


class SiLU(Module[[Tensor], Tensor]):
    """Sigmoid Linear Unit (SiLU/Swish) activation function module.

    Applies the SiLU activation function element-wise:
        SiLU(x) = x * sigmoid(x)
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply SiLU activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with SiLU applied element-wise.
        """
        return F.silu(x)


class Identity(Module[[Tensor], Tensor]):
    """Identity module that returns the input unchanged.

    Useful as a placeholder or when conditionally disabling layers.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Return input unchanged.

        Args:
            x: Input tensor.

        Returns:
            The same input tensor.
        """
        return x


class GELU(Module[[Tensor], Tensor]):
    """GELU activation function with optional tanh approximation.

    Args:
        approximate: If ``"tanh"``, uses the tanh approximation of GELU.
            Otherwise uses the exact formulation. Default: ``"none"``
    """

    def __init__(self, approximate: str = "none"):
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with GELU applied element-wise.
        """
        return F.gelu(x, approximate=self.approximate)
