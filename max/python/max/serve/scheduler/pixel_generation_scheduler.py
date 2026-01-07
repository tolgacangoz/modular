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
"""Pixel generation scheduler for diffusion model pipelines."""

import logging
import queue
from dataclasses import dataclass

from max.interfaces import (
    MAXPullQueue,
    MAXPushQueue,
    Pipeline,
    PixelGenerationInputs,
    PixelGenerationOutput,
    RequestID,
    Scheduler,
    SchedulerResult,
)
from max.interfaces.queue import BackgroundQueueDrainer
from max.pipelines.core import PixelContext
from max.pipelines.lib import PipelineConfig
from max.profiler import traced

from .base import SchedulerProgress

logger = logging.getLogger("max.serve")


@dataclass
class PixelGenerationSchedulerConfig:
    """Pixel Generation Scheduler configuration."""

    # The maximum number of requests that can be in the generation batch.
    max_batch_size: int = 1


class PixelGenerationScheduler(Scheduler):
    """Scheduler for pixel generation (diffusion model) pipelines.

    This scheduler handles batching and execution of pixel generation requests,
    such as image generation from diffusion models.
    """

    def __init__(
        self,
        scheduler_config: PixelGenerationSchedulerConfig,
        pipeline: Pipeline[
            PixelGenerationInputs[PixelContext], PixelGenerationOutput
        ],
        request_queue: MAXPullQueue[PixelContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[PixelGenerationOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        offload_queue_draining: bool = False,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        # Background queue drainer for async queue processing
        self._queue_drainer: BackgroundQueueDrainer[PixelContext] | None = None
        if offload_queue_draining:
            self._queue_drainer = BackgroundQueueDrainer[PixelContext](
                self.request_queue,
                max_items_per_drain=self.scheduler_config.max_batch_size * 2,
            )

    @traced
    def _create_batch_to_execute(self) -> dict[RequestID, PixelContext]:
        """Create a batch of requests to process."""
        max_batch_size = self.scheduler_config.max_batch_size
        batch: dict[RequestID, PixelContext] = {}

        if self._queue_drainer is not None:
            # Start draining the queue in the background
            self._queue_drainer.start_draining()

            # Process items from the drainer
            while len(batch) < max_batch_size:
                try:
                    item = self._queue_drainer.retrieve_item()
                    batch[item.request_id] = item
                except queue.Empty:
                    break
        else:
            # Synchronous draining
            while len(batch) < max_batch_size:
                try:
                    item = self.request_queue.get_nowait()
                except queue.Empty:
                    break

                batch[item.request_id] = item

        return batch

    def run_iteration(self) -> SchedulerProgress:
        """Run one iteration of the scheduler loop.

        Returns:
            SchedulerProgress: Indicates whether work was performed.
        """
        batch_to_execute = self._create_batch_to_execute()
        if len(batch_to_execute) == 0:
            return SchedulerProgress.NO_PROGRESS

        self._schedule_generation(batch_to_execute)
        return SchedulerProgress.MADE_PROGRESS

    @traced
    def _handle_terminated_responses(
        self,
        batch_executed: dict[RequestID, PixelContext],
        batch_response: dict[RequestID, PixelGenerationOutput],
    ) -> None:
        """Handle responses for terminated requests."""
        already_terminated = set()
        terminated = batch_executed.keys() - batch_response.keys()
        for req_id in terminated:
            if req_id in already_terminated:
                continue
            del batch_executed[req_id]
            already_terminated.add(req_id)

    @traced
    def _schedule_generation(
        self, batch_to_execute: dict[RequestID, PixelContext]
    ) -> None:
        """Execute a batch of pixel generation requests."""
        # Execute the batch through the pipeline
        batch_responses = self.pipeline.execute(
            PixelGenerationInputs(batch=batch_to_execute)
        )

        # Remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)

        # Send responses to the API process
        self.response_queue.put_nowait(
            {
                request_id: SchedulerResult.create(response)
                for request_id, response in batch_responses.items()
            }
        )


def load_pixel_generation_scheduler(
    pipeline: Pipeline[
        PixelGenerationInputs[PixelContext], PixelGenerationOutput
    ],
    pipeline_config: PipelineConfig,
    request_queue: MAXPullQueue[PixelContext],
    response_queue: MAXPushQueue[
        dict[RequestID, SchedulerResult[PixelGenerationOutput]]
    ],
    cancel_queue: MAXPullQueue[list[RequestID]],
) -> PixelGenerationScheduler:
    """Load and configure a pixel generation scheduler.

    Args:
        pipeline: The pixel generation pipeline to use.
        pipeline_config: Configuration for the pipeline.
        request_queue: Queue for incoming generation requests.
        response_queue: Queue for outgoing generation responses.
        cancel_queue: Queue for cancellation requests.

    Returns:
        A configured PixelGenerationScheduler instance.
    """
    scheduler_config = PixelGenerationSchedulerConfig(
        max_batch_size=pipeline_config.max_batch_size
        if pipeline_config.max_batch_size is not None
        else 1
    )

    return PixelGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        offload_queue_draining=pipeline_config.experimental_background_queue,
    )
