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
"""Activation function modules for MAX neural networks."""

from max import functional as F
from max.tensor import Tensor

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


class Dropout(Module[[Tensor], Tensor]):
    """Dropout module for regularization.

    During inference (which is the primary use case in MAX), this module
    acts as a pass-through and returns the input unchanged.

    Args:
        p: Probability of an element to be zeroed during training.
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize Dropout module.

        Args:
            p: Dropout probability (unused during inference).
        """
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """Apply dropout (no-op during inference).

        Args:
            x: Input tensor.

        Returns:
            The input tensor unchanged during inference.
        """
        # During inference, dropout is a no-op
        return x
