# Documentation: `torch/distributed/pipelining/schedules.py`

## File Metadata

- **Path**: `torch/distributed/pipelining/schedules.py`
- **Size**: 142,217 bytes (138.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import copy
import csv
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from enum import Enum
from functools import lru_cache
from typing import Any, cast, NamedTuple, Optional, Protocol, Union

import torch
import torch.distributed as dist
from torch._dynamo import OptimizedModule
from torch.distributed.fsdp import FSDPModule, UnshardHandle
from torch.nn.modules.loss import _Loss
from torch.profiler import record_function

from ._utils import generate_rank_to_stage_mapping, generate_stage_to_rank_mapping
from .microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
from .stage import _PipelineStageBase


__all__ = [
    "get_schedule_class",
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
    "ScheduleInterleavedZeroBubble",
    "ScheduleZBVZeroBubble",
    "ScheduleDualPipeV",
]

logger = logging.getLogger(__name__)


class _ComputationType(Enum):
    # TODO(whc) rename to _ActType?
    FORWARD = 1
    BACKWARD_INPUT = 2
    BACKWARD_WEIGHT = 3
    UNSHARD = 4
    RESHARD = 5
    SEND_F = 6
    RECV_F = 7
    SEND_B = 8
    RECV_B = 9
    FULL_BACKWARD = 10
    OVERLAP_F_B = 11
    REDUCE_GRAD = 12

    def __str__(self):
        str_map = {
            _ComputationType.FORWARD: "F",
            _ComputationType.BACKWARD_INPUT: "I",
            _ComputationType.BACKWARD_WEIGHT: "W",
            _ComputationType.UNSHARD: "UNSHARD",
            _ComputationType.RESHARD: "RESHARD",
            _ComputationType.SEND_F: "SEND_F",
            _ComputationType.RECV_F: "RECV_F",
            _ComputationType.SEND_B: "SEND_B",
            _ComputationType.RECV_B: "RECV_B",
            _ComputationType.FULL_BACKWARD: "B",
            _ComputationType.OVERLAP_F_B: "OVERLAP_F_B",
            _ComputationType.REDUCE_GRAD: "REDUCE_GRAD",
        }
        return str_map[self]

    @staticmethod
    def from_str(action):
        if action == "F":
            return _ComputationType.FORWARD
        elif action == "I":
            return _ComputationType.BACKWARD_INPUT
        elif action == "W":
            return _ComputationType.BACKWARD_WEIGHT
        elif action == "UNSHARD":
            return _ComputationType.UNSHARD
        elif action == "RESHARD":
            return _ComputationType.RESHARD
        elif action == "SEND_F":
            return _ComputationType.SEND_F
        elif action == "RECV_F":
            return _ComputationType.RECV_F
        elif action == "SEND_B":
            return _ComputationType.SEND_B
        elif action == "RECV_B":
            return _ComputationType.RECV_B
        elif action == "B":
            return _ComputationType.FULL_BACKWARD
        elif action == "OVERLAP_F_B":
            return _ComputationType.OVERLAP_F_B
        elif action == "REDUCE_GRAD":
            return _ComputationType.REDUCE_GRAD
        else:
            raise RuntimeError(f"Invalid computation type {action}")


FORWARD = _ComputationType.FORWARD
BACKWARD_INPUT = _ComputationType.BACKWARD_INPUT
BACKWARD_WEIGHT = _ComputationType.BACKWARD_WEIGHT
UNSHARD = _ComputationType.UNSHARD
RESHARD = _ComputationType.RESHARD
SEND_F = _ComputationType.SEND_F
RECV_F = _ComputationType.RECV_F
SEND_B = _ComputationType.SEND_B
RECV_B = _ComputationType.RECV_B
FULL_BACKWARD = _ComputationType.FULL_BACKWARD
OVERLAP_F_B = _ComputationType.OVERLAP_F_B
REDUCE_GRAD = _ComputationType.REDUCE_GRAD

# Convenience shorthand for compute actions only since they are used in 'simple schedule format'
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD

# Helper to parse an action string like 1F0 into a tuple of (stage_index, computation_type, microbatch_index)
_action_regex = re.compile(
    r"(\d+)(F|I|B|W|UNSHARD|RESHARD|REDUCE_GRAD|SEND_F|RECV_F|SEND_B|RECV_B)(\d*)"
)


class _Action(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: Optional[int] = None
    sub_actions: Optional[tuple["_Action", ...]] = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.sub_actions is not None:
            # Use recursive repr for sub_actions
            sub_action_reprs = [repr(sub_action) for sub_action in self.sub_actions]
            return f"({';'.join(sub_action_reprs)}){self.computation_type}"
        else:
            repr_str = str(self.stage_index)
            repr_str += str(self.computation_type)
            if self.microbatch_index is not None:
                repr_str += str(self.microbatch_index)
            return repr_str

    @property
    def is_compute_op(self) -> bool:
        return self.computation_type in (
            FORWARD,
            FULL_BACKWARD,
            BACKWARD_INPUT,
            BACKWARD_WEIGHT,
            OVERLAP_F_B,
        )

    @staticmethod
    def from_str(action_string: str):
        """
        Reverse of __repr__

        String should be formatted as [stage][action type][(microbatch)]
            e.g. `2F0`, `1UNSHARD`, `3SEND_F1`
        """
        action_string = action_string.strip()
        if action_string == "":
            return None

        # Check for sub_actions format: [sub_action1;sub_action2;...]ComputationType
        if action_string.startswith("(") and ")" in action_string:
            # Find the closing bracket to separate sub_actions from computation type
            bracket_end = action_string.find(")")
            sub_part = action_string[
                1:bracket_end
            ]  # Remove '[' and get content before ']'
            computation_type_part = action_string[
                bracket_end + 1 :
            ]  # Get part after ']'

            # Parse sub_actions
            sub_actions = []
            if sub_part.strip():
                for sub_str in sub_part.split(";"):
                    sub_action = _Action.from_str(sub_str.strip())
                    if sub_action is not None:
                        sub_actions.append(sub_action)

            # For sub_actions format, we create an action with just the computation type
            # The stage_index and microbatch_index are not meaningful for the container action
            return _Action(
                stage_index=-1,  # Placeholder, not meaningful for sub_actions container
                computation_type=_ComputationType.from_str(computation_type_part),
                microbatch_index=None,
                sub_actions=tuple(sub_actions) if sub_actions else None,
            )

        # Handle regular single action format
        if match := _action_regex.match(action_string):
            stage_index, computation_type, microbatch_index = match.groups()
            return _Action(
                int(stage_index),
                _ComputationType.from_str(computation_type),
                int(microbatch_index) if len(microbatch_index) else None,
            )
        elif action_string == "":
            return None
        raise RuntimeError(
            f"Invalid action string: {action_string}, should be formatted as [stage][action type][(microbatch)] e.g. 2F0"
        )


@lru_cache
def _get_profiler_function_name(action: _Action) -> str:
    return f"PP:{str(action)}"


def _format_pipeline_order(
    pipeline_order: dict[int, list[Optional[_Action]]],
    error_step_number: Optional[int] = None,
) -> str:
    """
    Formats the pipeline order in a timestep (row) x rank (column) grid of actions
    and returns the formatted string.

    If `error_step_number` is passed in, an additional label will be added to signify which step
    that it is erroring on.
    """

    # don't mutate the original
    pipeline_order = copy.deepcopy(pipeline_order)

    # Replace None with ""
    for rank in pipeline_order:
        for i in range(len(pipeline_order[rank])):
            if pipeline_order[rank][i] is None:
                # TODO make a real 'None action' that prints as empty string and make mypy happy
                pipeline_order[rank][i] = ""  # type: ignore[call-overload]

    # Calculate the maximum number of steps across all ranks
    num_steps = max(len(actions) for actions in pipeline_order.values())
    step_labels = [
        "Step " + str(i).zfill(len(str(num_steps - 1))) for i in range(num_steps)
    ]
    # Sorting the dictionary by keys and retrieving values in that order
    rank_actions = [
        pipeline_order.get(key, [""] * num_steps) for key in sorted(pipeline_order)
    ]
    # Transpose the list of lists (rows to columns)
    # pyrefly: ignore [no-matching-overload]
    transposed_actions = list(itertools.zip_longest(*rank_actions, fillvalue=""))
    # Generate column labels for ranks
    num_ranks = len(pipeline_order)
    rank_labels = ["Rank " + str(i) for i in range(num_ranks)]
    # Calculate the maximum length of each column, considering labels
    max_lengths = [
        max(len(str(item)) if item is not None else 0 for item in col)
        for col in zip(step_labels, *transposed_actions)
    ]
    # Format the header row with rank labels
    header_row = " " * (len(step_labels[0]) + 2) + " ".join(
        f"{label:<{max_lengths[i]}}" for i, label in enumerate(rank_labels)
    )
    # Format each row with its corresponding label
    formatted_rows = [
        f"{label}: "
        + " ".join(f"{str(item):<{max_lengths[i]}}" for i, item in enumerate(row))
        + (
            " <-- ERROR HERE"
            if error_step_number is not None
            and int(label.split()[1]) == error_step_number
            else ""
        )
        for label, row in zip(step_labels, transposed_actions)
    ]
    # Join the rows into a single string
    formatted_table = header_row + "\n" + "\n".join(formatted_rows) + "\n"
    return formatted_table


class _PipelineSchedule(ABC):
    def __init__(
        self,
        n_microbatches: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        args_chunk_spec: Optional[tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[dict[str, Any], tuple[Any]]] = None,
        scale_grads: bool = True,
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = loss_fn

        # See documentation in `PipelineScheduleSingle` / `PipelineScheduleMulti`
        self.scale_grads = scale_grads

        # Chunking specification for positional inputs. (default: `None`)
        self._args_chunk_spec = args_chunk_spec
        # Chunking specification for keyword inputs. (default: `None`)
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        """
        # args_chunk_spec and kwargs_chunk_spec specify how to chunk inputs.
        # They are used to convert batch to microbatches in `step(x)`.  See
        # `TensorChunkSpec` for helper methods for creating them.
        """

        # Derived
        self._has_backward = self._loss_fn is not None

        # Holds the losses for each microbatch.
        self._internal_losses: list[torch.Tensor] = []
        logger.info("Using %s", self.__class__.__name__)

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        if stage.is_last and self._loss_fn is not None:
            loss = self._compute_loss(output, target_mbs[mb_index])  # type: ignore[index]
            self._internal_losses.append(loss)

    def _maybe_get_loss(self, stage, mb_index):
        valid_index = 0 <= mb_index < len(self._internal_losses)
        if stage.is_last and self._loss_fn is not None and valid_index:
            return self._internal_losses[mb_index]
        elif len(self._internal_losses) != 0 and not valid_index:
            raise RuntimeError(
                f"Loss for microbatch {mb_index} is not available. "
                f"Available losses for microbatches: {self._internal_losses}"
            )
        else:
            return None

    def _update_losses(self, stages, losses):
        """
        Update the losses to those in the internal state
        """
        # if stages not a list turn into a list
        if not isinstance(stages, list):
            stages = [stages]
        contains_last_stage = any(stage.is_last for stage in stages)

        # Return losses if there is a container passed in
        if contains_last_stage and losses is not None:
            if len(self._internal_losses) != self._n_microbatches:
                raise RuntimeError(
                    f"Expecting {self._n_microbatches} losses but got {len(self._internal_losses)}"
                )

            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.extend(self._internal_losses)

        self._internal_losses.clear()

    @abstractmethod
    def _step_microbatches(
        self,
        arg_mbs: Optional[list] = None,
        kwarg_mbs: Optional[list] = None,
        target_mbs: Optional[list] = None,
        losses: Optional[list] = None,
        return_outputs: bool = True,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
            return_outputs: whether to return the outputs from the last stage.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        *args,
        target=None,
        losses: Optional[list] = None,
        return_outputs=True,
        **kwargs,
    ):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        return_outputs: whether to return the outputs from the last stage.
        """
        raise NotImplementedError

    def eval(self, *args, target=None, losses: Optional[list] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches, calling forward only.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target values for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        # Save the original has_backward state
        original_has_backward = self._has_backward
        try:
            self._has_backward = False
            return self.step(*args, target=target, losses=losses, **kwargs)
        finally:
            # Restore the original state
            self._has_backward = original_has_backward

    def _check_inputs(
        self,
        arg_mbs: Optional[list] = None,
        kwarg_mbs: Optional[list] = None,
        target_mbs: Optional[list] = None,
        losses: Optional[list] = None,
    ) -> tuple[list, list]:
        """
        Pre-process/check inputs
        """

        def check_type_and_len(mbs, name: str):
            if not isinstance(mbs, list):
                raise TypeError(f"{name} must be a list but got a {type(mbs)}")
            if len(mbs) != self._n_microbatches:
                raise ValueError(
                    f"Expecting {self._n_microbatches} {name} but got {len(mbs)}"
                )

        if arg_mbs is not None:
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None:
            if not isinstance(losses, list):
                raise TypeError(f"losses must be a list but got a {type(losses)}")

        return arg_mbs, kwarg_mbs

    def _compute_loss(self, output, target):
        return self._loss_fn(output, target)  # type: ignore[misc]

    def _split_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                self._args_chunk_spec,
                self._kwargs_chunk_spec,
            )
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def _merge_outputs(self, output_chunks: list[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )


def _batch_p2p(
    p2p_ops: list[dist.P2POp], desc: Optional[str] = None
) -> list[dist.Work]:
    """
    Simple wrapper over batch_isend_irecv from torch.distributed, which just adds a descriptive logger on top.
    """
    if len(p2p_ops) == 0:
        return []
    desc_str = f"{desc}, " if desc else ""
    logger.debug("batch_p2p %s%s", desc_str, p2p_ops)
    return dist.batch_isend_irecv(p2p_ops)


def _sorted_batch_p2p(
    p2p_ops: list[dist.P2POp], desc: Optional[str] = None
) -> dict[int, list[dist.Work]]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: dict[int, list[dist.P2POp]] = defaultdict(list)
    work_by_peer: dict[int, list[dist.Work]] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


def _wait_batch_p2p(work: list[dist.Work]):
    """
    Waits for a list of dist.Work (typically from _batch_p2p / _sorted_batch_p2p).
    """
    for w in work:
        w.wait()


class PipelineScheduleSingle(_PipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.

    Gradients are scaled by num_microbatches depending on the `scale_grads` argument, defaulting to True.  This setting
    should match the configuration of your loss_fn, which may either average losses (scale_grads=True)
    or sum losses (scale_grads=False).
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[dict[str, Any], tuple[Any]]] = None,
        scale_grads: bool = True,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        self._stage_forward_initialized = False
        self._stage_backward_initialized = False

        if n_microbatches < self._num_stages:
            raise ValueError(
                f"Number of microbatches ({n_microbatches}) must be greater than \
or equal to the number of stages ({self._num_stages})."
            )

        self.pipeline_order: Optional[dict[int, list[Optional[_Action]]]] = (
            self._get_pipeline_order()
        )

    def _initialize_stage(self, args, kwargs):
        if not self._stage_forward_initialized:
            # Prepare the communication needed for the pipeline schedule execution
            # This is needed because during execution we always perform a series of batch P2P ops
            # The first call of the batched P2P needs to involve the global group
            all_ops: list[dist.P2POp] = []
            all_ops.extend(self._stage._get_init_p2p_neighbors_ops())
            _wait_batch_p2p(_batch_p2p(all_ops))

            self._stage._prepare_forward_infra(self._n_microbatches, args, kwargs)
            self._stage_forward_initialized = True

        if self._has_backward and not self._stage_backward_initialized:
            self._stage._prepare_backward_infra(self._n_microbatches)
            self._stage_backward_initialized = True

    def step(
        self,
        *args,
        target=None,
        losses: Optional[list] = None,
        return_outputs: bool = True,
        **kwargs,
    ):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        return_outputs: whether to return the outputs from the last stage.
        """
        if self._has_backward and not torch.is_grad_enabled():
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation; "
                "it should not be used under torch.no_grad() context. "
                "Please call eval() instead."
            )

        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward

        # Clean per iteration
        self._stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(
            args_split, kwargs_split, targets_split, losses, return_outputs
        )

        # Return merged results per original format
        if self._stage.is_last and return_outputs:
            return self._merge_outputs(self._stage.output_chunks)
        else:
            return None

    def _get_pipeline_order(self) -> Optional[dict[int, list[Optional[_Action]]]]:
        """
        Returns the pipeline execution order as a schedule IR.

        The returned IR is a dictionary mapping rank IDs to lists of actions.
        Each action is either an _Action object representing computation to perform,
        or None representing a deliberate idle step.

        The None values are used to represent pipeline bubbles where a rank
        must wait for dependencies from other ranks before proceeding. However
        during execution, with  the _PipelineScheduleRuntime, these Nones are
        skipped since the relevant communication (send/recv) will be scheduled and waited on.

        Returns:
            A dictionary mapping rank -> list of actions
        """
        return None


class _ScheduleForwardOnly(PipelineScheduleSingle):
    """
    The forward-only schedule.
    Will go through all the microbatches and perform only the forward pass
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[list] = None,
        kwarg_mbs: Optional[list] = None,
        target_mbs: Optional[list] = None,
        losses: Optional[list] = None,
        return_outputs: bool = True,
    ):
        """
        Run one iteration of the pipeline schedule
        """
        if target_mbs is not None or losses is not None:
            raise RuntimeError(
                "Forward-only schedule does not support loss computation"
            )

        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Delay send waits
        fwd_sends_to_wait: list[list[dist.Work]] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    _wait_batch_p2p(work)

                self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Forwarded microbatch %s", self._stage.stage_index, i)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            _wait_batch_p2p(work)


class ScheduleGPipe(PipelineScheduleSingle):
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[list] = None,
        kwarg_mbs: Optional[list] = None,
        target_mbs: Optional[list] = None,
        losses: Optional[list] = None,
        return_outputs: bool = True,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
            return_outputs: whether to return the outputs from the last stage.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Delay send waits
        fwd_sends_to_wait: list[list[dist.Work]] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    _wait_batch_p2p(work)

                output = self._stage.forward_one_chunk(
                    i, arg_mbs[i], kwarg_mbs[i], save_forward_output=return_outputs
                )  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Forwarded microbatch %s", self._stage.stage_index, i)

            self._maybe_compute_loss(self._stage, output, target_mbs, i)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            _wait_batch_p2p(work)

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: list[list[dist.Work]] = []
        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for work in works.values():
                    _wait_batch_p2p(work)

                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(
                    i,
                    loss=loss,
                    last_backward=i == self._n_microbatches - 1,
                )

                ops = self._stage.get_bwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Backwarded microbatch %s", self._stage.stage_index, i)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            _wait_batch_p2p(work)

        # Update losses if there is a container passed in
        self._update_losses(self._stage, losses)

        self._stage.perform_reduce_grad(self._n_microbatches if self.scale_grads else 1)

    def _get_pipeline_order(self) -> Optional[dict[int, list[Optional[_Action]]]]:
        """
        Returns the pipeline order for GPipe schedule.

        See base method in PipelineScheduleSingle for details on the schedule IR format.
        """
        pipeline_order = {}
        pp_group_size = self._num_stages

        for rank in range(pp_group_size):
            actions: list[Optional[_Action]] = []

            # 1. Initial delay based on rank position
            warmup_delay = rank
            actions.extend([None] * warmup_delay)

            # 2. Forward passes for all microbatches
            for mb_idx in range(self._n_microbatches):
                actions.append(_Action(rank, _ComputationType.FORWARD, mb_idx))

            # 3. Wait period before backward passes can begin
            backward_delay = 3 * (pp_group_size - 1 - rank)
            actions.extend([None] * backward_delay)

            # 4. Backward passes for all microbatches
            for mb_idx in range(self._n_microbatches):
                actions.append(_Action(rank, _ComputationType.FULL_BACKWARD, mb_idx))

            pipeline_order[rank] = _add_reduce_grad(actions, self._n_microbatches)

        return pipeline_order  # type: ignore[return-value]


class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[list] = None,
        kwarg_mbs: Optional[list] = None,
        target_mbs: Optional[list] = None,
        losses: Optional[list] = None,
        return_outputs: bool = True,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
            return_outputs: whether to return the outputs from the last stage.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        # Warmup phase
        send_work: list[dist.Work] = []
        fwd_sends = []
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            _wait_batch_p2p(_batch_p2p(fwd_recvs, desc="fwd_recv"))

            # Compute
            output = self._stage.forward_one_chunk(
                fwd_mb_index,
                arg_mbs[fwd_mb_index],
                kwarg_mbs[fwd_mb_index],
                save_forward_output=return_outputs,
            )  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            _wait_batch_p2p(send_work)

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last forward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)
            fwd_mb_index += 1

        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.

        # 1B1F phase
        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)

            # Now, we need to fire the fwd_sends and bwd_recvs together
            _wait_batch_p2p(_batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv"))

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)

            # Fuse it with bwd_sends above
            _wait_batch_p2p(_batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv"))

            # Now do the fwd
            output = self._stage.forward_one_chunk(
                fwd_mb_index,
                arg_mbs[fwd_mb_index],
                kwarg_mbs[fwd_mb_index],
                save_forward_output=return_outputs,
            )  # type: ignore[index]

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)

            # Get the fwd send ops, but don't fire, leave it for the next iter (wrap-around)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            fwd_mb_index += 1

        # Remember we still have some bwd_sends left over after the break? Now it is time to fire it
        send_work = _batch_p2p(bwd_sends, desc="bwd_send")

        # Cooldown
        while bwd_mb_index < self._n_microbatches:
            # prepare bwd recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            _wait_batch_p2p(_batch_p2p(bwd_recvs, desc="bwd_recv"))

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )

            # Clear previous chunk's backward sends (hopefully they have well finished)
            _wait_batch_p2p(send_work)

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            bwd_mb_index += 1

        # Wait for the last backward send to finish
        _wait_batch_p2p(send_work)

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

        self._stage.perform_reduce_grad(self._n_microbatches if self.scale_grads else 1)

    def _get_pipeline_order(self) -> Optional[dict[int, list[Optional[_Action]]]]:
        """
        Returns the pipeline order for 1F1B schedule.

        See base method in PipelineScheduleSingle for details on the schedule IR format.
        """
        pipeline_order = {}
        pp_group_size = self._num_stages

        for rank in range(pp_group_size):
            actions: list[Optional[_Action]] = []

            # 1. Warmup phase: initial delay based on rank
            actions.extend([None] * rank)

            # 2. Initial forward passes before 1F1B phase
            num_forward = (pp_group_size - 1) - rank
            forward_mb = 0
            for i in range(num_forward):
                actions.append(_Action(rank, _ComputationType.FORWARD, i))
                forward_mb = i

            # 3. Wait for backward to be ready
            wait_for_1f1b = max(0, 2 * (pp_group_size - 1 - rank))
            actions.extend([None] * wait_for_1f1b)

            # 4. 1F1B steady state phase
            backward_mb = 0
            remaining_forward = self._n_microbatches - num_forward

            while remaining_forward > 0:
                # One forward
                forward_mb += 1
                actions.append(_Action(rank, _ComputationType.FORWARD, forward_mb))
                remaining_forward -= 1

                # One backward
                actions.append(
                    _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                )
                backward_mb += 1

            # 5. Cooldown phase: remaining backward passes
            remaining_backward = self._n_microbatches - backward_mb

            while remaining_backward > 0:
                # Add None and backward actions in alternating pattern
                # based on distance from the last stage
                if (pp_group_size - rank) > 0:
                    actions.append(None)
                    # Decrement the wait counter only if we still have backward passes to do
                    if remaining_backward > 0:
                        actions.append(
                            _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                        )
                        backward_mb += 1
                        remaining_backward -= 1
                else:
                    # If we're at the last stage, just add backward actions without None
                    actions.append(
                        _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                    )
                    backward_mb += 1
                    remaining_backward -= 1

            pipeline_order[rank] = _add_reduce_grad(actions, self._n_microbatches)
        return pipeline_order


def _requires_reduce_grad(action_type: _ComputationType) -> bool:
    return action_type in (W, B)


def _add_reduce_grad(
    actions: list[Optional[_Action]], n_microbatches: int
) -> list[Optional[_Action]]:
    """
    REDUCE_GRAD refers to joint across minibatches grad reduction.
    reduce_grad frees memory and we want to schedule it just after the last "backward"-like stage.
    """
    actions_with_reduce_grad: list[Optional[_Action]] = []
    cnt: dict[int, int] = defaultdict(int)

    def _leaf_action(a, to_schedule):
        if _requires_reduce_grad(a.computation_type):
            stage_index = a.stage_index
            cnt[stage_index] += 1
            if cnt[stage_index] == n_microbatches:
                to_schedule.append(stage_index)

    for a in actions:
        if a is None:
            continue
        actions_with_reduce_grad.append(a)
        schedule_reduce_grad_stage_idxs: list[int] = []
        if a.computation_type == OVERLAP_F_B and a.sub_actions is not None:
            for sub_action in a.sub_actions:
                _leaf_action(sub_action, schedule_reduce_grad_stage_idxs)
        else:
            _leaf_action(a, schedule_reduce_grad_stage_idxs)

        for stage_idx in schedule_reduce_grad_stage_idxs:
            actions_with_reduce_grad.append(_Action(stage_idx, REDUCE_GRAD, None))
    return actions_with_reduce_grad


def _add_unshard_reshard(
    compute_actions: list[Optional[_Action]],
    max_active_stages: int = 3,
) -> list[_Action]:
    """Given a basic schedule involving only compute actions (F,B,W,OVERLAP_F_B), add UNSHARD/RESHARD actions for FSDP.

    UNSHARD refers to fetching the full contents of an FSDP-sharded layer, requiring an all-gather operation.
    RESHARD does the opposite, releasing memory (but doing no communication)

    We abandon the "timestep lock"  during lowering

    max_active_stages controls how many prefetches we allow. It should be measured in mb and tuneable but in practice
    3 stages is probably the thing we want?
    (to account for having one f and one b active, and something else prefetching?)
    """

    def next_stage_indices(
        count: int, next_actions: list[Optional[_Action]]
    ) -> list[int]:
        """Remove duplicates (same stage, different microbatch), find next 'count' stages that will do compute."""
        seen: set[int] = set()
        ret: list[int] = []

        for a in next_actions:
            if a is not None:
                # Handle OVERLAP_F_B actions by checking their sub_actions
                if a.computation_type == OVERLAP_F_B and a.sub_actions is not None:
                    for sub_action in a.sub_actions:
                        if sub_action.stage_index not in seen:
                            seen.add(sub_action.stage_index)
                            ret.append(sub_action.stage_index)
                    if len(ret) >= count:
                        break
                else:
                    # Regular action
                    if a.stage_index not in seen:
                        seen.add(a.stage_index)
                        ret.append(a.stage_index)
                        if len(ret) == count:
                            break
        return ret

    active_stages: set[int] = set()
    fsdp_aware_actions: list[_Action] = []

    def _unshard(stage_index: int):
        active_stages.add(stage_index)
        fsdp_aware_actions.append(_Action(stage_index, UNSHARD, None))

    def _reshard(stage_index: int):
        active_stages.remove(stage_index)
        fsdp_aware_actions.append(_Action(stage_index, RESHARD, None))

    for i, action in enumerate(compute_actions):
        if action is None:
            continue

        # We prefetch the next N stages we'll see, dropping existing stages to make room
        next_n = next_stage_indices(max_active_stages, compute_actions[i:])
        # Fetch needs to be ordered correctly, so don't use a set
        fetch = list(filter(lambda s: s not in active_stages, next_n))
        # Unclear what the best policy is for eviction, but we can maintain order so we do
        evict = list(filter(lambda s: s not in next_n, active_stages))

        # logger.debug(
        #     "_add_unshard_reshard Step %d active: %s fetch %s, evict %s",
        #     i,
        #     active_stages,
        #     fetch,
        #     evict,
        # )

        for stage in evict:
            _reshard(stage)
        for stage in fetch:
            _unshard(stage)
        fsdp_aware_actions.append(action)

    # Reshard all remaining active stages after processing all operations
    for stage in list(active_stages):
        _reshard(stage)

    return fsdp_aware_actions


def _merge_bw(
    compute_actions: list[Optional[_Action]],
) -> list[_Action]:
    """Given a basic schedule involving only compute actions (F,I,W), merge adjacent I and W ops into B ops.
    (note: I = BACKWARD_INPUT, W = BACKWARD_WEIGHT, B = FULL_BACKWARD)

    B refers to running the whole backward (not separating grad_input and grad_weight), which can be more efficient
    in some cases.
    """
    merged_actions = []
    while compute_actions:
        action = compute_actions.pop(0)
        if action is None:
            continue

        # Remove any None actions and find the next non-None action
        while len(compute_actions) and compute_actions[0] is None:
            compute_actions.pop(0)

        # Get the next action if it exists
        next_action = compute_actions[0] if len(compute_actions) > 0 else None

        if (
            action.computation_type == BACKWARD_INPUT
            and next_action is not None
            and next_action.computation_type == BACKWARD_WEIGHT
            and action.stage_index == next_action.stage_index
            and action.microbatch_index == next_action.microbatch_index
        ):
            merged_actions.append(
                _Action(action.stage_index, FULL_BACKWARD, action.microbatch_index)
            )
            compute_actions.pop(0)
        else:
            merged_actions.append(action)
    return merged_actions


def _add_send_recv(
    compute_actions: dict[int, list[_Action]],
    stage_to_rank: Callable[[int], int],
    num_stages: int,
) -> dict[int, list[_Action]]:
    """
    Transforms a compute-only schedule into a complete schedule with communication actions.

    For actions with sub-actions (OVERLAP_F_B) we ensure that all the subactions have been
    computed and the communication is ready
    """
    comm_actions: dict[int, list[_Action]] = {rank: [] for rank in compute_actions}
    prev_actions: dict[int, set[_Action]] = {rank: set() for rank in compute_actions}

    def _has_comms(action: _Action) -> bool:
        if action.computation_type == F:
            return action.stage_index != num_stages - 1 and stage_to_rank(
                action.stage_index + 1
            ) != stage_to_rank(action.stage_index)
        elif action.computation_type in (BACKWARD_INPUT, FULL_BACKWARD):
            return action.stage_index != 0 and stage_to_rank(
                action.stage_index - 1
            ) != stage_to_rank(action.stage_index)
        return False

    def _get_comms(action: _Action) -> tuple[_Action, _Action]:
        assert _has_comms(action), f"{action} is not a valid comm action"
        stage_idx = action.stage_index
        ctype = action.computation_type
        mb_idx = action.microbatch_index
        send = _Action(stage_idx, SEND_F if ctype == F else SEND_B, mb_idx)
        recv_stage_idx = stage_idx + 1 if ctype == F else stage_idx - 1
        recv = _Action(recv_stage_idx, RECV_F if ctype == F else RECV_B, mb_idx)
        return send, recv

    def _ready_to_schedule(
        action: Optional[_Action], prev_actions: set[_Action]
    ) -> bool:
        """We don't put our own recv ops in the schedule, we let a sender on another rank put our recv ops in place.
        This helps ensure a sane (non-hanging) ordering of sends and recvs.
        But it also means we might not be able to schedule our next compute action yet.
        """
        if action is None:
            return True
        elif action.computation_type == F and action.stage_index != 0:
            if (
                _Action(action.stage_index, RECV_F, action.microbatch_index)
                in prev_actions
            ):
                return True
            elif (
                _Action(action.stage_index - 1, F, action.microbatch_index)
                in prev_actions
            ):
                return True
            return False
        elif (
            action.computation_type in (BACKWARD_INPUT, FULL_BACKWARD)
            and action.stage_index != num_stages - 1
        ):
            if (
                _Action(action.stage_index, RECV_B, action.microbatch_index)
                in prev_actions
            ):
                return True
            elif (
                _Action(action.stage_index + 1, BACKWARD_INPUT, action.microbatch_index)
                in prev_actions
            ):
                return True
            elif (
                _Action(action.stage_index + 1, FULL_BACKWARD, action.microbatch_index)
                in prev_actions
            ):
                return True
            return False
        else:
            return True

    while compute_actions:
        progress = False
        # go in order of ranks even if dict keys aren't ordered
        for rank in sorted(compute_actions):
            assert len(compute_actions[rank]) > 0, (
                f"{rank=}, {len(compute_actions[rank])=}"
            )
            action = compute_actions[rank][0]
            # handle case where parent action (e.g. OVERLAP_F_B) can be comprised of subactions
            if action is not None and action.sub_actions is not None:
                all_actions = action.sub_actions
            else:
                all_actions = (action,)

            if not all(_ready_to_schedule(a, prev_actions[rank]) for a in all_actions):
                continue

            # The action's dependencies are satisfied, so add to schedule
            if action is not None:
                comm_actions[rank].append(action)
                for a in all_ac
```



## High-Level Overview


This Python file contains 20 class(es) and 93 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_ComputationType`, `_Action`, `_PipelineSchedule`, `PipelineScheduleSingle`, `_ScheduleForwardOnly`, `ScheduleGPipe`, `Schedule1F1B`, `PipelineScheduleMulti`, `_PipelineContext`, `_CustomFunctionProtocol`, `_PipelineScheduleRuntime`, `ScheduleLoopedBFS`, `ScheduleInterleaved1F1B`, `ScheduleInterleavedZeroBubble`, `ScheduleZBVZeroBubble`, `ScheduleDualPipeV`

**Functions defined**: `__str__`, `from_str`, `__str__`, `__repr__`, `is_compute_op`, `from_str`, `_get_profiler_function_name`, `_format_pipeline_order`, `__init__`, `_maybe_compute_loss`, `_maybe_get_loss`, `_update_losses`, `_step_microbatches`, `step`, `eval`, `_check_inputs`, `check_type_and_len`, `_compute_loss`, `_split_inputs`, `_merge_outputs`

**Key imports**: copy, csv, itertools, logging, re, ABC, abstractmethod, Counter, defaultdict, Callable, Enum, lru_cache


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/pipelining`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `csv`
- `itertools`
- `logging`
- `re`
- `abc`: ABC, abstractmethod
- `collections`: Counter, defaultdict
- `collections.abc`: Callable
- `enum`: Enum
- `functools`: lru_cache
- `typing`: Any, cast, NamedTuple, Optional, Protocol, Union
- `torch`
- `torch.distributed as dist`
- `torch._dynamo`: OptimizedModule
- `torch.distributed.fsdp`: FSDPModule, UnshardHandle
- `torch.nn.modules.loss`: _Loss
- `torch.profiler`: record_function
- `._utils`: generate_rank_to_stage_mapping, generate_stage_to_rank_mapping
- `.microbatch`: merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
- `.stage`: _PipelineStageBase
- `json`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/pipelining`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_backward.py_docs.md`](./_backward.py_docs.md)
- [`_unflatten.py_docs.md`](./_unflatten.py_docs.md)
- [`microbatch.py_docs.md`](./microbatch.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`stage.py_docs.md`](./stage.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`_IR.py_docs.md`](./_IR.py_docs.md)
- [`_debug.py_docs.md`](./_debug.py_docs.md)


## Cross-References

- **File Documentation**: `schedules.py_docs.md`
- **Keyword Index**: `schedules.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
