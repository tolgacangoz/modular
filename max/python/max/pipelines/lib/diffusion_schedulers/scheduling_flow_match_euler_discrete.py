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


import numpy as np
import torch

from max import functional as F
from max import random
from max.driver import CPU, Device
from max.dtype import DType
from max.random import set_seed
from max.tensor import Tensor


if is_scipy_available():
    import scipy.stats

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FlowMatchEulerDiscreteScheduler:
    """Minimal stub for FlowMatchEulerDiscreteScheduler."""

    def __init__(self, **kwargs) -> None:
        self.config = type("Config", (), {"use_flow_sigmas": False})()
        self.timesteps = np.array([], dtype=np.float32)
        self.sigmas = np.array([], dtype=np.float32)
        self.order = 1

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        sigmas: npt.NDArray[np.float32] | None = None,
        **kwargs,
    ) -> None:
        """Set the timesteps and sigmas for the diffusion process.

        Args:
            num_inference_steps: Number of inference steps. Used to generate
                timesteps if sigmas is not provided.
            sigmas: Custom sigma schedule. If provided, timesteps are derived
                from sigmas.
            **kwargs: Additional arguments (accepted for compatibility).
        """
        if sigmas is not None:
            # Use provided sigmas and derive timesteps
            # Append final sigma of 0.0 for the last scheduler step
            # (scheduler step accesses sigmas[i+1], so we need n+1 elements)
            self.sigmas = np.append(sigmas, np.float32(0.0))
            self.timesteps = sigmas * 1000.0
        elif num_inference_steps is not None:
            # Generate default timesteps
            self.timesteps = np.linspace(
                1000.0, 0.0, num_inference_steps, dtype=np.float32
            )
            self.sigmas = self.timesteps / 1000.0

    def step(
        self,
        model_output: Any,
        timestep: Any,
        sample: Any,
        return_dict: bool = True,
    ) -> Any:
        """Step function for FlowMatchEulerDiscreteScheduler."""
        # For flow matching, model_output is the predicted velocity
        # x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * velocity

        # In a real implementation we would look up sigma_t and sigma_{t-1}
        # For a stub, we'll assume a linear step based on fixed num_inference_steps
        # Or just use the timestep difference if they are sigmas.

        # Simplified Euler step:
        # Assuming timestep is sigma here or can be mapped to it.
        # Let's assume timestep is from 1000 down to 0.
        dt = 1.0 / len(self.timesteps) if len(self.timesteps) > 0 else 0.02
        prev_sample = sample - dt * model_output

        if not return_dict:
            return (prev_sample,)

        class SchedulerOutput:
            def __init__(self, prev_sample):
                self.prev_sample = prev_sample

        return SchedulerOutput(prev_sample)
