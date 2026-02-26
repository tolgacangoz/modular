import max.nn.module_v3 as nn
from max.experimental.tensor import Tensor


class Dropout(nn.Module[[Tensor], Tensor]):
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
