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
"""Qwen3Encoder for extracting text embeddings from Qwen3 models.

This encoder is specifically designed for Z-Image text encoding, where we need
to extract the second-to-last layer's hidden states (similar to how diffusers
uses `hidden_states[-2]` from the text encoder).

Unlike the standard Qwen3 which returns logits for text generation, this encoder:
- Returns hidden states from the second-to-last layer
- Skips the lm_head projection
- Maintains KV cache interface for Modular infrastructure compatibility
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.architectures.qwen3.qwen3 import Qwen3


class Qwen3Encoder(Qwen3):
    """Qwen3-based text encoder that returns hidden states instead of logits.

    This class extends Qwen3 to provide text embeddings for the Z-Image pipeline.
    Instead of returning logits from the final layer, it returns the hidden states
    from the second-to-last layer (matching diffusers' `hidden_states[-2]`).

    The encoder keeps the standard Qwen3 interface with KV cache support for
    compatibility with Modular's infrastructure.

    Example:
        ```python
        encoder = Qwen3Encoder(config)

        # Forward pass returns hidden states from layer n-2
        # (where n is the total number of layers)
        hidden_states = encoder(
            tokens,
            kv_collection,
            return_n_logits,
            input_row_offsets,
        )
        ```

    Note:
        The output hidden states are NOT normalized. The caller should apply
        normalization if needed for their use case.
    """

    def __init__(self, config: Qwen3Config) -> None:
        """Initialize the Qwen3Encoder.

        Args:
            config: Qwen3 configuration object. Note that `return_logits` and
                `return_hidden_states` settings are ignored since this encoder
                always returns the second-to-last layer's hidden states.
        """
        super().__init__(config)
        # Store the layer index for the second-to-last layer
        # e.g., if num_hidden_layers=36, we want layer 34 (0-indexed)
        self._encoder_output_layer_idx = config.num_hidden_layers - 2

    def _process_hidden_states(
        self,
        h: TensorValue,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Process embeddings through transformer layers and return hidden states.

        Unlike the base Qwen3._process_hidden_states which returns logits,
        this method returns the hidden states from the second-to-last layer.

        Args:
            h: Input embeddings of shape (total_seq_len, hidden_size).
            kv_collection: Paged KV cache values.
            return_n_logits: Number of logits to return (ignored for encoder).
            input_row_offsets: Row offsets for batch processing.

        Returns:
            Tuple containing a single tensor: the hidden states from the
            second-to-last layer with shape (total_seq_len, hidden_size).
        """
        freqs_cis = self.rope.freqs_cis

        # Process through transformer layers up to second-to-last only.
        # We use static slicing instead of early return to maintain
        # static control flow required for graph compilation.
        encoder_layers = self.layers[: self._encoder_output_layer_idx + 1]

        for idx, layer in enumerate(encoder_layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )

        # Return the hidden states from the second-to-last layer
        # Shape: (total_seq_len, hidden_size)
        return (h,)
