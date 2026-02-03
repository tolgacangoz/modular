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

from __future__ import annotations

from collections.abc import Iterable

import max.driver as md
from max.dtype import DType
<<<<<<< HEAD
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.graph.type import FilterLayout

from .layer import Module, Shardable
=======
from max.graph import DeviceRef
from max.graph.type import ConvInputLayout, FilterLayout
from max.nn import Module
from max.tensor import Tensor
>>>>>>> 49761efa58 (asd)


class Conv2d(Module, Shardable):
    """A 2D convolution over an input signal composed of several input
    planes.

    Example:
        .. code-block:: python

            conv = nn.Conv2d(
                kernel_size=3,
                in_channels=64,
                out_channels=128,
                dtype=DType.float32,
                stride=1,
                padding=0,
                has_bias=False,
                name="conv2d_weight",
                device=DeviceRef.GPU(),
            )
    """

    device: DeviceRef | None
    """The device where matrix operations are performed."""

    filter: Weight
    """The weight matrix stored on CPU with shape (height, width, in_channels / num_groups, out_channels).
    Model init moves the weight to :obj:`device`."""

    stride: tuple[int, int]
    """Controls the stride for the cross-correlation."""

    padding: tuple[int, int, int, int]
    """Controls the amount of padding applied before and after the input for height and width dimensions.

    Format: (pad_top, pad_bottom, pad_left, pad_right)."""

    dilation: tuple[int, int]
    """Controls the dilation rate."""

    num_groups: int
    """Number of blocked connections from input channels to output channels."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_channels,).
    Model init moves the bias to :obj:`device` if present."""

    permute: bool = False
    """bool controls whether self.filter is permuted from PyTorch order to max order.
    PyTorch order is: (out_channels, in_channels / num_groups, height, width)
    Max API order: (height, width, in_channels / num_groups, out_channels)."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ) -> None:
        """Initializes the Conv2d layer with weights and optional bias.

        Args:
            kernel_size: Size of the convolving kernel. Can be a single int (square kernel) or tuple (height, width).
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution for height and width dimensions.
                Can be int (applied to both dimensions) or tuple (stride_h, stride_w). Default: 1
            padding: Padding added to input. Can be int (applied to all sides),
                tuple of 2 ints (pad_h, pad_w), or tuple of 4 ints (pad_top, pad_bottom, pad_left, pad_right) to support asymmetric padding. Default: 0
            dilation: Spacing between kernel elements for height and width dimensions.
                Can be int (applied to both dimensions) or tuple (dilation_h, dilation_w). Default: 1
            num_groups: Number of blocked connections from input channels to output channels.
                Input channels and output channels are divided into groups. Default: 1
            device: The target device for computation. If None, defaults to CPU.
                Weights are initially stored on CPU and moved to target device during computation.
            name: Base name for weights. If provided, weights are named ``{name}.weight`` and
                ``{name}.bias`` (if bias is enabled). If None, uses "weight" and "bias".
            has_bias: If true, adds a learnable bias vector to the layer.
                Defaults to :obj:`False`.
            permute: If true, permutes weights from PyTorch format to MAX format.
                PyTorch order: (out_channels, in_channels / num_groups, height, width).
                MAX API order: (height, width, in_channels / num_groups, out_channels).
                Defaults to :obj:`False`.
        """
        super().__init__()

        # Store configuration for easy reconstruction
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.num_groups = num_groups
        self.has_bias = has_bias
        self.name = name

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            kernel_height = kernel_width = kernel_size
            self.kernel_size = (kernel_size, kernel_size)
        else:
            kernel_height, kernel_width = kernel_size
            self.kernel_size = kernel_size

        self.filter = Weight(
            name=f"{name}.weight" if name else "weight",
            dtype=dtype,
            shape=(
                [
                    out_channels,
                    in_channels // num_groups,
                    kernel_height,
                    kernel_width,
                ]
                if self.permute
                else [
                    kernel_height,
                    kernel_width,
                    in_channels // num_groups,
                    out_channels,
                ]
            ),
            device=self.device or DeviceRef.CPU(),
        )

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=dtype,
                shape=(out_channels,),
                device=self.device or DeviceRef.CPU(),
            )

        # Convert scalar parameters to tuples as needed
        self.stride = (stride, stride) if isinstance(stride, int) else stride

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            # Convert (pad_h, pad_w) to (pad_top, pad_bottom, pad_left, pad_right)
            pad_h, pad_w = padding
            padding = (pad_h, pad_h, pad_w, pad_w)

        self.padding = padding

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError("Conv2d not implemented with weight quantization.")

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the Conv2d sharding strategy."""
        # Always take the sharding strategy of the conv filter.
        return self.filter.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the conv layer.

        Args:
            strategy: The strategy describing the conv's sharding.
        """
        if not strategy.is_replicate:
            raise ValueError(
                "only replicate is supported for Conv2d, currently"
            )

        self.filter.sharding_strategy = strategy
        if self.bias:
            self.bias.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Conv2d]:
        """Creates sharded views of this Conv2d layer across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Conv2d instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "Conv2d layer cannot be sharded because no sharding strategy was provided."
            )
        assert self.sharding_strategy.is_replicate

        # Get sharded weights
        sharded_filters = self.filter.shard(devices)
        sharded_biases = self.bias.shard(devices) if self.bias else None

        shards = []
        for idx, (device, filter_shard) in enumerate(
            zip(devices, sharded_filters, strict=True)
        ):
            # Create new Conv2d with same configuration.
            sharded = Conv2d(
                kernel_size=self.kernel_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                dtype=self.dtype,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                num_groups=self.num_groups,
                device=device,
                has_bias=self.has_bias,
                permute=self.permute,
                name=self.name,
            )

            # Replace the weights with sharded versions.
            sharded.filter = filter_shard
            if sharded_biases:
                sharded.bias = sharded_biases[idx]

            shards.append(sharded)

        return shards

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply 2D convolution to input `x`. Permutes pytorch weights to match max API if permute=True.

        Args:
            x: a tensor of shape [batch_size, height, width, in_channels]
            if self.permute, then input is of shape: [batch_size, in_channels, height, width]
            and will be permuted to match max's expected input shape.

        Returns:
            a tensor of shape [batch_size, new_height, new_width, out_channels]
            if self.permute, then output shape will be [batch_size, out_channels, new_height, new_width]
        """
        weight: TensorValue = self.filter

        is_nvidia_gpu = (
            isinstance(self.device, DeviceRef)
            and self.device.is_gpu()
            and md.accelerator_api() == "cuda"
        )

        if self.permute:
            # Input: [batch_size, in_channels, height, width] -> [batch_size, height, width, in_channels]
            x = ops.permute(x, [0, 2, 3, 1])

            # GPU supports FCRS but CPU doesn't. On CPU, permute from
            # FCRS to RSCF format.
            if not is_nvidia_gpu:
                # Permute weight from [out_channels, in_channels // num_groups, height, width]
                # to [height, width, in_channels // num_groups, out_channels] (RSCF)
                weight = ops.permute(weight, [2, 3, 1, 0])

        output = ops.conv2d(
            x,
            weight,
            self.stride,
            self.dilation,
            self.padding,
            self.num_groups,
            self.bias,
            filter_layout=FilterLayout.FCRS
            if (self.permute and is_nvidia_gpu)
            else FilterLayout.RSCF,
        )

        if self.permute:
            # Output: [batch_size, new_height, new_width, out_channels] -> [batch_size, out_channels, new_height, new_width]
            output = ops.permute(output, [0, 3, 1, 2])

        return output


<<<<<<< HEAD
class Conv1D(Module):
    """A 1D convolution over an input signal composed of several input
    planes.
=======
class Conv1d(Module[[Tensor], Tensor]):
    """A 1D convolution layer.

    This is a Conv1d implementation that uses Tensor instead of Weight objects.
    It internally uses Conv2d by unsqueezing the input to 4D and squeezing the output back.
>>>>>>> 49761efa58 (asd)

    Example:
        .. code-block:: python

<<<<<<< HEAD
            conv = nn.Conv1D(
                kernel_size=3,
                in_channels=64,
                out_channels=128,
                dtype=DType.float32,
                stride=1,
                padding=0,
                has_bias=False,
                name="conv1d_weight",
                device=DeviceRef.GPU(),
            )
    """

    device: DeviceRef | None
    """The device where matrix operations are performed."""

    filter: Weight
    """The weight matrix stored on CPU with shape (kernel_size, in_channels / num_groups, out_channels).
    Model init moves the weight to :obj:`device`."""

    stride: int
    """Controls the stride for the cross-correlation."""

    padding: int | tuple[int, int]
    """Controls the amount of padding applied to the input.

    If int: symmetric padding applied to both sides (pad_left = pad_right = padding).
    If tuple[int, int]: asymmetric padding as (pad_left, pad_right).
    """

    dilation: int
    """Controls the dilation rate."""

    num_groups: int
    """Number of blocked connections from input channels to output channels."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_channels,).
    Model init moves the bias to :obj:`device` if present."""

    permute: bool = False
    """bool controls whether self.filter is permuted from PyTorch order to max order.
    PyTorch order is: (out_channels, in_channels / num_groups, kernel_size)
    Max API order: (kernel_size, in_channels / num_groups, out_channels)."""

    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
=======
            from max.nn import Conv1d
            from max.tensor import Tensor

            conv = Conv1d(
                kernel_size=3,
                in_channels=3,
                out_channels=64,
                has_bias=True,
                permute=True,
            )

            x = Tensor.ones([1, 3, 32])  # [batch, channels, length]
            result = conv(x)
    """

    weight: Tensor
    """The weight tensor with shape [out_channels, in_channels // num_groups, kernel_size]."""

    bias: Tensor | Literal[0]
    """The bias tensor with shape [out_channels] (or 0 if bias is disabled)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dtype: DType | None = None,
        stride: int = 1,
        padding: int | str | tuple[int, int] = 0,
>>>>>>> 49761efa58 (asd)
        dilation: int = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
<<<<<<< HEAD
        permute: bool = False,
        name: str | None = None,
    ) -> None:
        """Initializes the Conv1D layer with weights and optional bias.

        Args:
            kernel_size: Size of the convolving kernel (width dimension).
            in_channels: Number of channels in the input signal.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution. Controls the step size when sliding the kernel. Default: 1
            padding: Padding added to the input sequence. Can be:
                - int: symmetric padding applied to both sides (pad_left = pad_right = padding). Default: 0
                - tuple[int, int]: asymmetric padding as (pad_left, pad_right) for causal convolutions.
            dilation: Spacing between kernel elements. Controls the kernel dilation rate. Default: 1
            num_groups: Number of blocked connections from input channels to output channels.
                Input channels and output channels are divided into groups. Default: 1
            device: The target device for computation. If None, defaults to CPU.
                Weights are initially stored on CPU and moved to target device during computation.
            name: Base name for weights. If provided, weights are named ``{name}.weight`` and
                ``{name}.bias`` (if bias is enabled). If None, uses "weight" and "bias".
            has_bias: If true, adds a learnable bias vector to the layer.
                Defaults to :obj:`False`.
            permute: If true, permutes weights from PyTorch format to MAX format.
                PyTorch order: (out_channels, in_channels / num_groups, kernel_size).
                MAX API order: (kernel_size, in_channels / num_groups, out_channels).
                Defaults to :obj:`False`.
        """
        super().__init__()

        self.device = device
        self.permute = permute

        if self.permute:
            self.filter = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[out_channels, in_channels // num_groups, kernel_size],
                device=self.device or DeviceRef.CPU(),
            )
        else:
            self.filter = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[kernel_size, in_channels // num_groups, out_channels],
                device=self.device or DeviceRef.CPU(),
            )

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=dtype,
                shape=(out_channels,),
                device=self.device or DeviceRef.CPU(),
            )

        self.kernel_size = kernel_size
        self.stride = stride
        # Normalize padding to tuple format: (pad_left, pad_right)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = dilation
        self.num_groups = num_groups

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError("Conv1D not implemented with weight quantization.")

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applied 1D convolution to input `x`. Permutes pytorch weights to match max API if permute=True.

        Args:
            x: a tensor of shape [batch_size, length, in_channels]
            if self.permute, then input is of shape: [batch_size, in_channels, length]
            and will be permuted to match max's expected input shape.

        Returns:
            a tensor of shape [batch_size, new_length, out_channels]
            if self.permute, then output shape will be [batch_size, out_channels, new_length]
            new_length = ((length + 2 * padding - (kernel_size - 1) - 1) / stride) + 1
        """
        weight: TensorValue = self.filter

        is_nvidia_gpu = (
            isinstance(self.device, DeviceRef)
            and self.device.is_gpu()
            and md.accelerator_api() == "cuda"
        )

        if self.permute:
            x = ops.permute(x, [0, 2, 1])  # [batch_size, length, in_channels]

            # GPU supports FCRS but CPU doesn't. On CPU, permute from
            # FCS to SCF, then add dummy dim to become RSCF.
            if not is_nvidia_gpu:
                weight = ops.unsqueeze(ops.permute(weight, [2, 1, 0]), 0)
            # on GPU, unsqueeze FCS to FCRS
            else:
                weight = ops.unsqueeze(weight, 2)
        # No permute, filer is SCF and unsqueeze to RSCF.
        else:
            weight = ops.unsqueeze(weight, 0)

        # Reshape for Conv2dV1
        x = ops.unsqueeze(x, 1)  # [batch_size, height=1, length, in_channels]

        # Convert padding tuple (pad_left, pad_right) to conv2d format (pad_top, pad_bottom, pad_left, pad_right)
        pad_left, pad_right = self.padding  # type: ignore[misc]
        output = ops.conv2d(
=======
        permute: bool = True,
        name: str | None = None,
    ):
        """Initialize Conv1d layer.

        Args:
            in_channels: Number of channels in the input signal.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution. Default: 1
            padding: Padding added to input. Can be int (applied to both sides),
                tuple of 2 ints (pad_left, pad_right), or "same" for same-size output. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            num_groups: Number of blocked connections from input channels to output channels. Default: 1
            device: The target device for computation.
            has_bias: If true, adds a learnable bias vector to the layer. Defaults to False.
            permute: If true, expects PyTorch format (batch, channels, length).
                Defaults to True for PyTorch compatibility.
            name: Base name for weights. Defaults to None.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.num_groups = num_groups
        self.has_bias = has_bias
        self.name = name
        self.stride = stride
        self.dilation = dilation

        # Handle padding
        if isinstance(padding, str):
            if padding == "same":
                # For "same" padding, we calculate it dynamically
                self.padding_mode = "same"
                self.padding = (0, 0)
            else:
                raise ValueError(f"Unsupported padding mode: {padding}")
        elif isinstance(padding, int):
            self.padding_mode = None
            self.padding = (padding, padding)
        else:
            self.padding_mode = None
            self.padding = padding

        self.weight = random.normal(
            [out_channels, in_channels // num_groups, kernel_size]
            if self.permute
            else [kernel_size, in_channels // num_groups, out_channels],
            dtype=self.dtype,
            device=self.device.to_device() if self.device is not None else None,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels],
                dtype=self.dtype,
                device=self.device.to_device()
                if self.device is not None
                else None,
            )
        else:
            self.bias = 0

    def forward(self, x: Tensor) -> Tensor:
        """Apply 1D convolution to input.

        Args:
            x: Input tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, in_channels, length]
                - If permute=False: [batch_size, length, in_channels]

        Returns:
            Output tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, out_channels, new_length]
                - If permute=False: [batch_size, new_length, out_channels]
        """
        weight = self.weight.to(x.device)

        is_nvidia_gpu = (
            isinstance(x.device, Accelerator) and accelerator_api() == "cuda"
        )

        # Handle "same" padding by calculating padding dynamically
        if self.padding_mode == "same":
            input_length = x.shape[-1] if self.permute else x.shape[1]
            effective_kernel = (self.kernel_size - 1) * self.dilation + 1
            total_padding = max(effective_kernel - 1, 0)
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left
            padding = (pad_left, pad_right)
        else:
            padding = self.padding

        if self.permute:
            # Input: [batch_size, in_channels, length] -> [batch_size, in_channels, 1, length]
            x = x.unsqueeze(2)
            # Weight: [out_channels, in_channels // num_groups, kernel_size]
            #      -> [out_channels, in_channels // num_groups, 1, kernel_size]
            weight = weight.unsqueeze(2)

            if not is_nvidia_gpu:
                # Permute for CPU: [out_channels, in_channels // num_groups, 1, kernel_size]
                #               -> [1, kernel_size, in_channels // num_groups, out_channels]
                weight = F.permute(weight, [2, 3, 1, 0])
                # Input: [batch, channels, 1, length] -> [batch, 1, length, channels]
                x = F.permute(x, [0, 2, 3, 1])
        else:
            # Input: [batch_size, length, in_channels] -> [batch_size, 1, length, in_channels]
            x = x.unsqueeze(1)
            # Weight: [kernel_size, in_channels // num_groups, out_channels]
            #      -> [1, kernel_size, in_channels // num_groups, out_channels]
            weight = weight.unsqueeze(0)

        # Convert 1D padding to 2D: (pad_left, pad_right) -> (0, 0, pad_left, pad_right)
        pad_left, pad_right = padding
        padding_2d = (0, 0, pad_left, pad_right)

        output = F.conv2d(
>>>>>>> 49761efa58 (asd)
            x,
            weight,
            (1, self.stride),
            (1, self.dilation),
<<<<<<< HEAD
            (0, 0, pad_left, pad_right),
            self.num_groups,
            self.bias,
=======
            padding_2d,
            self.num_groups,
            self.bias if isinstance(self.bias, Tensor) else None,
>>>>>>> 49761efa58 (asd)
            filter_layout=FilterLayout.FCRS
            if (self.permute and is_nvidia_gpu)
            else FilterLayout.RSCF,
        )

<<<<<<< HEAD
        # Reshape back from Conv2dV1
        output = ops.squeeze(
            output, 1
        )  # [batch_size, new_length, out_channels]

        if self.permute:
            output = ops.permute(
                output, [0, 2, 1]
            )  # [batch_size, out_channels, new_length]
=======
        if self.permute:
            if not is_nvidia_gpu:
                # Output: [batch, 1, new_length, out_channels] -> [batch, out_channels, 1, new_length]
                output = F.permute(output, [0, 3, 1, 2])
            # Remove dummy height dimension: [batch, out_channels, 1, new_length] -> [batch, out_channels, new_length]
            output = output.squeeze(2)
        else:
            # Remove dummy height dimension: [batch, 1, new_length, out_channels] -> [batch, new_length, out_channels]
            output = output.squeeze(1)
>>>>>>> 49761efa58 (asd)

        return output


<<<<<<< HEAD
class Conv3D(Module):
    """A 3D convolution over an input signal composed of several input
    planes.
=======
class ConvTranspose1d(Module[[Tensor], Tensor]):
    """A 1D transposed convolution layer.

    This is a ConvTranspose1d implementation that uses Tensor instead of Weight objects.
    It internally uses conv2d_transpose by unsqueezing the input to 4D and squeezing the output back.
>>>>>>> 49761efa58 (asd)

    Example:
        .. code-block:: python

<<<<<<< HEAD
            conv = nn.Conv3D(
                depth=3,
                height=3,
                width=3,
                in_channels=64,
                out_channels=128,
                dtype=DType.float32,
                stride=1,
                padding=0,
                has_bias=False,
                name="conv3d_weight",
                device=DeviceRef.GPU(),
            )
    """

    device: DeviceRef | None
    """The device where matrix operations are performed."""

    filter: Weight
    """The weight matrix stored on CPU with shape (depth, height, width, in_channels / num_groups, out_channels).
    Model init moves the weight to :obj:`device`."""

    stride: tuple[int, int, int]
    """Controls the stride for the cross-correlation."""

    padding: tuple[int, int, int, int, int, int]
    """Controls the amount of padding applied before and after the input for depth, height, and width dimensions.

    Format: (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right)."""

    dilation: tuple[int, int, int]
    """Controls the dilation rate for depth, height, and width dimensions."""

    num_groups: int
    """Number of blocked connections from input channels to output channels."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_channels,).
    Model init moves the bias to :obj:`device` if present."""

    permute: bool = False
    """bool controls whether self.filter is permuted from PyTorch order to max order.
    PyTorch order is: (out_channels, in_channels / num_groups, depth, height, width)
    Max API order: (depth, height, width, in_channels / num_groups, out_channels)."""

    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: int | tuple[int, int, int] = 1,
        padding: int
        | tuple[int, int, int]
        | tuple[int, int, int, int, int, int] = 0,
=======
            from max.nn import ConvTranspose1d
            from max.tensor import Tensor

            conv = ConvTranspose1d(
                kernel_size=4,
                in_channels=64,
                out_channels=32,
                stride=2,
                has_bias=True,
                permute=True,
            )

            x = Tensor.ones([1, 64, 16])  # [batch, channels, length]
            result = conv(x)  # [batch, 32, 32]
    """

    weight: Tensor
    """The weight tensor with shape [in_channels, out_channels, kernel_size]."""

    bias: Tensor | None
    """The bias tensor with shape [out_channels] (or None if bias is disabled)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dtype: DType | None = None,
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int = 0,
        dilation: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = True,
        name: str | None = None,
    ):
        """Initialize ConvTranspose1d layer.

        Args:
            in_channels: Number of channels in the input signal.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution. Default: 1
            padding: Padding subtracted from output. Default: 0
            output_padding: Additional size added to one side of the output. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            device: The target device for computation.
            has_bias: If true, adds a learnable bias vector to the layer. Defaults to False.
            permute: If true, expects PyTorch format (batch, channels, length).
                Defaults to True for PyTorch compatibility.
            name: Base name for weights. Defaults to None.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.has_bias = has_bias
        self.name = name
        self.stride = stride
        self.dilation = dilation
        self.output_padding = output_padding

        # Handle padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.weight = random.normal(
            [in_channels, out_channels, kernel_size]
            if self.permute
            else [kernel_size, out_channels, in_channels],
            dtype=self.dtype,
            device=self.device.to_device() if self.device is not None else None,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels],
                dtype=self.dtype,
                device=self.device.to_device()
                if self.device is not None
                else None,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply 1D transposed convolution to input.

        Args:
            x: Input tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, in_channels, length]
                - If permute=False: [batch_size, length, in_channels]

        Returns:
            Output tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, out_channels, new_length]
                - If permute=False: [batch_size, new_length, out_channels]
        """
        weight = self.weight.to(x.device)

        if self.permute:
            # Input: [batch, in_channels, length] -> [batch, in_channels, 1, length]
            x = x.unsqueeze(2)
            # Weight: [in_channels, out_channels, kernel_size]
            #      -> [in_channels, out_channels, 1, kernel_size]
            weight = weight.unsqueeze(2)
        else:
            # Input: [batch, length, in_channels] -> [batch, 1, length, in_channels]
            x = x.unsqueeze(1)
            # Weight: [kernel_size, out_channels, in_channels]
            #      -> [1, kernel_size, out_channels, in_channels]
            weight = weight.unsqueeze(0)

        # Convert 1D parameters to 2D
        pad_left, pad_right = self.padding
        padding_2d = (0, 0, pad_left, pad_right)

        output = F.conv2d_transpose(
            x,
            weight,
            stride=(1, self.stride),
            dilation=(1, self.dilation),
            padding=padding_2d,
            output_paddings=(0, self.output_padding),
            bias=self.bias,
            input_layout=ConvInputLayout.NCHW
            if self.permute
            else ConvInputLayout.NHWC,
            filter_layout=FilterLayout.CFRS
            if self.permute
            else FilterLayout.RSCF,
        )

        if self.permute:
            # Remove dummy height dimension: [batch, out_channels, 1, new_length] -> [batch, out_channels, new_length]
            output = output.squeeze(2)
        else:
            # Remove dummy height dimension: [batch, 1, new_length, out_channels] -> [batch, new_length, out_channels]
            output = output.squeeze(1)

        return output


class Conv3d(Module[[Tensor], Tensor]):
    """A 3D convolution layer.

    Example:
        .. code-block:: python

            from max.nn import Conv3d
            from max.tensor import Tensor

            conv = Conv3d(
                kernel_size=3,
                in_channels=3,
                out_channels=64,
                has_bias=True,
                permute=True,
            )

            x = Tensor.ones([1, 8, 32, 32, 3])
            result = conv(x)
    """

    weight: Tensor
    """The weight tensor with shape [depth, height, width, in_channels / num_groups, out_channels]."""

    bias: Tensor | Literal[0]
    """The bias tensor with shape [out_channels] (or 0 if bias is disabled)."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        dtype: DType | None = None,
        stride: int | tuple[int, int, int] = 1,
<<<<<<< HEAD
        padding: int | tuple[int, int, int] | tuple[int, int, int, int, int, int] = 0,
>>>>>>> 49761efa58 (asd)
=======
        padding: int
        | tuple[int, int, int]
        | tuple[int, int, int, int, int, int] = 0,
>>>>>>> febb773bfe (`pre-commit run --all-files`)
        dilation: int | tuple[int, int, int] = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
<<<<<<< HEAD
    ) -> None:
        """Initializes the Conv3D layer with weights and optional bias.

        Args:
            depth: Depth dimension of the convolution kernel (kernel_size[0]).
            height: Height dimension of the convolution kernel (kernel_size[1]).
            width: Width dimension of the convolution kernel (kernel_size[2]).
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution for depth, height, and width dimensions.
                Can be int (applied to all dimensions) or tuple of 3 ints. Default: 1
            padding: Padding added to the input in order:
                (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right).
                Can be int (applied to all sides), tuple of 3 ints (pad_d, pad_h, pad_w) expanded symmetrically,
                or tuple of 6 ints (fully asymmetric). Default: 0
            dilation: Spacing between kernel elements for depth, height, and width dimensions.
                Can be int (applied to all dimensions) or tuple of 3 ints. Default: 1
            num_groups: Number of blocked connections from input channels to output channels.
                Input channels and output channels are divided into groups. Default: 1.
            device: The target device for computation. If None, defaults to CPU.
                Weights are initially stored on CPU and moved to target device during computation.
            name: Base name for weights. If provided, weights are named ``{name}.weight`` and
                ``{name}.bias`` (if bias is enabled). If None, uses "weight" and "bias".
            has_bias: If true, adds a learnable bias vector to the layer.
                Defaults to :obj:`False`.
            permute: If true, permutes weights from PyTorch format to MAX format.
                PyTorch order: (out_channels, in_channels / num_groups, depth, height, width).
                MAX API order: (depth, height, width, in_channels / num_groups, out_channels).
                Defaults to :obj:`False`.
        """
        super().__init__()

        self.device = device

        self.permute = permute

        if self.permute:
            self.filter = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[
                    out_channels,
                    in_channels // num_groups,
                    depth,
                    height,
                    width,
                ],
                device=self.device or DeviceRef.CPU(),
            )
        else:
            self.filter = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[
                    depth,
                    height,
                    width,
                    in_channels // num_groups,
                    out_channels,
                ],
                device=self.device or DeviceRef.CPU(),
            )

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=dtype,
                shape=(out_channels,),
                device=self.device or DeviceRef.CPU(),
            )
        # These need to be casted as the underlying ops.conv3d call
        # expects them to only be tuple types.
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (
                padding,
                padding,
                padding,
                padding,
                padding,
                padding,
            )
        elif len(padding) == 3:
            pad_d, pad_h, pad_w = padding
            padding = (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)

        self.padding = padding

        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.dilation = dilation

        self.num_groups = num_groups

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError("Conv3D not implemented with weight quantization.")

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applied 3D convolution to input `x`. Permutes pytorch weights to match max API if permute=True.

        Args:
            x: a tensor of shape (batch_size, depth, height, width, in_channels)
            if self.permute, then input is of shape: (batch_size, in_channels, depth, height, width)
            and will be permuted to match max's expected input shape.

        Returns:
            a tensor of shape (batch_size, new_depth, new_height, new_width, out_channels).
            if self.permute, then the output shape will be (batch_size, out_channels, new_depth, new_height, new_width)
        """
        weight: TensorValue = self.filter
        if self.permute:
            weight = ops.permute(self.filter, [2, 3, 4, 1, 0])
            x = ops.permute(x, [0, 2, 3, 4, 1])

        res = ops.conv3d(
=======
    ):
        """Initialize Conv3d layer.

        Args:
            kernel_size: Size of the convolving kernel.
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for weights and bias.
            stride: Stride of the convolution. Default: 1
            padding: Padding added to input. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            num_groups: Number of groups. Default: 1
            device: The target device.
            has_bias: If true, adds a learnable bias vector. Defaults to False.
            permute: If true, permutes weights from PyTorch format to MAX format.
                Defaults to False.
            name: Base name for weights.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.num_groups = num_groups
        self.has_bias = has_bias
        self.name = name

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        kd, kh, kw = self.kernel_size

        self.weight = random.normal(
            [out_channels, in_channels // num_groups, kd, kh, kw]
            if self.permute
            else [kd, kh, kw, in_channels // num_groups, out_channels],
            dtype=self.dtype,
            device=self.device.to_device() if self.device is not None else None,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels],
                dtype=self.dtype,
                device=self.device.to_device()
                if self.device is not None
                else None,
            )
        else:
            self.bias = 0

        self.stride = (
            (stride, stride, stride) if isinstance(stride, int) else stride
        )

        if isinstance(padding, int):
            self.padding = (
                padding,
                padding,
                padding,
                padding,
                padding,
                padding,
            )
        elif len(padding) == 3:
            pd, ph, pw = padding
            self.padding = (pd, pd, ph, ph, pw, pw)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation, dilation)
        else:
            self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        """Apply 3D convolution to input.

        Args:
            x: Input tensor. Shape depends on `permute`:
                - If permute=True: [batch, in_channels, depth, height, width]
                - If permute=False: [batch, depth, height, width, in_channels]

        Returns:
            Output tensor.
        """
        weight = self.weight.to(x.device)

        if self.permute:
            # PyTorch format: NCDHW -> NDHWC
            x = F.permute(x, [0, 2, 3, 4, 1])
            # Weight: FCDHW -> DHWCF
            weight = F.permute(weight, [2, 3, 4, 1, 0])

        output = F.conv3d(
>>>>>>> 49761efa58 (asd)
            x,
            weight,
            self.stride,
            self.dilation,
            self.padding,
            self.num_groups,
<<<<<<< HEAD
            self.bias,
        )
        # permute output from (batch_size, depth, height, width, out_channels) to (batch_size, out_channels, depth, height, width).
        if self.permute:
            res = ops.permute(res, [0, 4, 1, 2, 3])
        return res
=======
            self.bias if isinstance(self.bias, Tensor) else None,
            filter_layout=self.weight.quantization_encoding
            if hasattr(self.weight, "quantization_encoding")
            else None,  # Placeholder for layout logic if needed
        )

        if self.permute:
            # NDHWC -> NCDHW
            output = F.permute(output, [0, 4, 1, 2, 3])

        return output
>>>>>>> 49761efa58 (asd)
