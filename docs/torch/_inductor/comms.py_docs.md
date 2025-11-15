# Documentation: `torch/_inductor/comms.py`

## File Metadata

- **Path**: `torch/_inductor/comms.py`
- **Size**: 102,529 bytes (100.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# pyre-strict
from __future__ import annotations

import heapq
import importlib
import itertools
import logging
import operator
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch._logging import trace_structured
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet

from . import config, config_comms, ir
from .dependencies import WeakDep


if TYPE_CHECKING:
    from .ir import IRNode, Operation

from .memory import (
    estimate_peak_memory,
    estimate_peak_memory_allocfree,
    FreeableInputBuffer,
    get_freeable_input_buf,
    SNodeMemory,
)
from .utils import (
    contains_collective,
    contains_wait,
    find_recursive_deps_of_node,
    find_recursive_users_of_node,
    is_collective,
    is_fallback_op,
    is_wait,
)
from .virtualized import V


log = logging.getLogger(__name__)
overlap_log = torch._logging.getArtifactLogger(__name__, "overlap")

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode


def align_runtime_estimations_across_all_distributed_ranks(
    snodes: list[BaseSchedulerNode],
):
    runtime_estimations = {}
    for snode in snodes:
        runtime_estimations[snode] = snode.get_estimated_runtime()
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group

    world_size = dist.get_world_size()
    pg = _get_default_group()
    gathered_runtime_estimations: list[list[float]] = [[] for _ in range(world_size)]
    dist.all_gather_object(
        gathered_runtime_estimations, list(runtime_estimations.values()), pg
    )
    median_runtime_estimations = torch.median(
        torch.tensor(gathered_runtime_estimations), dim=0
    ).values.tolist()
    for i in range(len(snodes)):
        snodes[i].override_estimated_runtime = median_runtime_estimations[i]


def sink_waits(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Greedily schedules waits as late as possible.
    """
    return _schedule_for_comm(
        snodes, raise_comms=False, sink_waits=True, reorder_for_overlap=False
    )


def raise_comms(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Greedily schedules comms as early as possible.
    """
    return _schedule_for_comm(
        snodes, raise_comms=True, sink_waits=False, reorder_for_overlap=False
    )


def reorder_compute_for_overlap(
    snodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    This achieves the following overall scheduling procedure:
        Step 1: Given that we've currently scheduled comm N, we now schedule all compute nodes
            that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.
        Step 2: If all those compute nodes are sufficient to overlap comm N, we're done.
            Otherwise, we now need to look elsewhere to find compute that overlaps with comm N.
            We prioritize compute nodes that are needed sooner.
        Step 3: We schedule the compute nodes dependent on comm N and required for comm N + 1.
        Step 4: We schedule comm N + 1.
        Repeat this for subsequent comm nodes.
    """
    return _schedule_for_comm(
        snodes, raise_comms=True, sink_waits=True, reorder_for_overlap=True
    )


def reorder_communication_preserving_peak_memory(
    snodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    Reorders communication ops relative to computation ops to improve communication-compute overlapping and hide comm
    latency.  Stops moving a particular op if it reaches a point that would have increased the peak memory footprint.

    Currently, follows these heuristics (subject to change or tune):
    - never reorders collectives relative to one another, for SPMD safety
    - has an option for per-collective prefetch limit, but does not enable it by default
    - limits the total number of reorder steps to some factor of the graph size to prevent worst-case quadratic
      performance

    Prerequisite: sink_comms_and_waits - ensure comm and wait nodes are scheduled as late as possible, respecting data
    dependencies.  That allows reorder_communication_preserving_peak_memory to take a best case peak-memory snapshot,
    and then monotonically improve latency by moving collectives backward in time.

    Peak memory impact is computed in an iterative fashion.  First, memory use at each timestep is computed, and global
    peak memory is computed as a max over timesteps.  Then, when swapping any two adjacent nodes, only the curr-memory
    for the earlier of the nodes after the swap is affected.  This enables checking step by step whether a swap is
    peak-memory-safe, and bailing out if not.  Example:

    0   n0      C0
    1   n1      C0 + Allocs(n1) - Frees(n1)
    2   n2      C0 + Allocs(n1) - Frees(n1) + Allocs(n2) - Frees(n2)

    0   n0      C0
    1   n2      C0 + Allocs(n2) - Frees(n2)    <-- After moving n2 to Time 1, only time1 memory changes
    2   n1      C0 + Allocs(n2) - Frees(n2) + Allocs(n1) - Frees(n1)

    """
    reordered_snodes, node_stats = (
        _reorder_communication_preserving_peak_memory_internal(snodes)
    )

    return reordered_snodes


@dataclass
class ReorderInfo:
    """
    Debug info describing how an individual snode was reordered
    """

    limiting_factor: str = "None"
    moves: int = 0
    grouped: int = 0
    grouped_info: str = ""
    comm_time: float = -1.0
    comp_time: float = -1.0
    initial_exposed: float = -1.0
    final_exposed: float = -1.0
    overlap_info: str = "None"

    @property
    def improvement(self):
        return self.initial_exposed - self.final_exposed


def is_gemm_like(node: Optional[Union[IRNode, Operation]]) -> bool:
    if node is None:
        return False

    if is_fallback_op(
        node,  # type: ignore[arg-type]
        torch.ops.aten._scaled_dot_product_flash_attention.default,
    ):
        return True

    if (
        python_kernel_name := getattr(node, "python_kernel_name", None)
    ) and "extern_kernels" in python_kernel_name:
        return True
    return False


def contains_gemm_like(snode: BaseSchedulerNode) -> bool:
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode):
        return any(contains_gemm_like(x) for x in snode.snodes)
    else:
        return is_gemm_like(snode.node)


def _temp_group_visit_leaves(snode: BaseSchedulerNode, fn):
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode) and snode.temp_grouping:
        for _snode in snode.snodes:
            fn(_snode)
    else:
        fn(snode)


def wait_exposed_communication_time(
    snodes_to_wait: list[BaseSchedulerNode], runtimes: dict[BaseSchedulerNode, float]
) -> tuple[float, float, str]:
    """
    Calculate exposed communication time for a wait operation by finding its corresponding
    collective and accumulating overlapping compute time between them.

    The Wait node must be the last in snodes_to_wait.
    Compute time between corresponding Collective and Wait is accumulated.
    If there is another pair of Collective and Wait inside,
    Only compute before first such Wait' is considered as overlapping.

    Multiple process groups are not modeled so far.
    """
    wait_snode = snodes_to_wait[-1]
    assert is_wait(wait_snode.node)
    assert len(snodes_to_wait) > 1
    idx = len(snodes_to_wait) - 2
    comm_time = 0.0
    comp_time = 0.0
    overlap_info = ""
    waits_found = []
    for i in range(idx, -1, -1):
        c = snodes_to_wait[i]
        if contains_wait(c):
            waits_found.append(c)
        if contains_collective(c):
            if is_corresponding_collective_wait(c, wait_snode):
                comm_time = runtimes[c]
                overlap_info += f"->C[{c.get_name()}]"
                break

            if not contains_async_collective(c):
                # Sync Collective
                comp_time = 0.0
                continue
            else:
                for w in waits_found:
                    if is_corresponding_collective_wait(c, w):
                        # Similar to Sync Collective
                        # If after our Collective exist another Collective-Wait,
                        # All compute after it will not be overlapping
                        comp_time = 0.0
                        continue

        comp_time_before = comp_time

        def accumulate_time(_snode: BaseSchedulerNode) -> None:
            nonlocal comp_time
            comp_time += runtimes[_snode]

        _temp_group_visit_leaves(c, accumulate_time)
        comp_time_after = comp_time
        overlap_info += f"+{c.get_name()}[{comp_time_after - comp_time_before}]"

    return comm_time, comp_time, overlap_info


def coll_exposed_communication_time(
    snodes: list[BaseSchedulerNode],
    runtimes: dict[BaseSchedulerNode, float],
) -> tuple[float, float, str]:
    """
    Calculate exposed communication time for a collective operation by finding its corresponding
    wait and accumulating compute time that can overlap with communication.

    The Collective node must be the first in snodes.
    Compute time between corresponding Collective and Wait is accumulated.
    If there is another pair of Collective and Wait inside,
    Only compute before first such Wait' is considered as overlapping.

    Multiple process groups are not modeled so far.
    """
    collective_snode = snodes[0]
    comm_time = runtimes[collective_snode]
    comp_time = 0.0
    collective_outs: OrderedSet[str] = OrderedSet(
        o.get_name() for o in collective_snode.get_outputs()
    )
    overlap_info = ""
    collectives_found: list[BaseSchedulerNode] = []
    for snode in snodes[1:]:
        # We may have some ops without Wait,
        # e.g. DTensor torch.ops._dtensor.shard_dim_alltoall
        unmet_deps = OrderedSet(
            d.name for d in snode.unmet_dependencies if not _is_fake_dep(d)
        )

        if unmet_deps & collective_outs:
            overlap_info += f"->W[{snode.get_name()}]"
            break

        if contains_collective(snode):
            if not contains_async_collective(snode):
                break
            else:
                collectives_found.append(snode)
                continue
        if contains_wait(snode):
            has_wait_for_collectives_found = False
            for _coll in collectives_found:
                if is_corresponding_collective_wait(collective_snode, snode):
                    has_wait_for_collectives_found = True
                    break
            if has_wait_for_collectives_found:
                # Any compute after not overlapping original Collective
                break

        comp_time_before = comp_time

        def accumulate_time(_snode: BaseSchedulerNode) -> None:
            nonlocal comp_time
            comp_time += runtimes[_snode]

        _temp_group_visit_leaves(snode, accumulate_time)
        comp_time_after = comp_time
        overlap_info += f"+{snode.get_name()}[{comp_time_after - comp_time_before}]"
    return comm_time, comp_time, overlap_info


def _group_name(snode, with_bufs=False) -> str:
    ret = ""
    for n in snode.snodes:
        if ret:
            ret += "_"
        ret += n.get_name()
        if with_bufs:
            ret += f"{list(snode.get_buffer_names())}"
    return ret


def _is_fake_dep(d):
    return isinstance(d, WeakDep) and d.is_fake


def _group_names(gns: list[BaseSchedulerNode]) -> str:
    return "~".join([gn.get_name() for gn in gns])


def _initialize_memory_tracking(snodes, graph_inputs, graph_outputs):
    """Initialize memory tracking data structures"""
    name_to_freeable_input_buf = get_freeable_input_buf(snodes, graph_inputs)
    peak_memory, snodes_curr_memory, snodes_allocfree, buf_to_snode_last_use = (
        estimate_peak_memory_allocfree(
            snodes, name_to_freeable_input_buf, graph_outputs
        )
    )
    _curr_memory = dict(zip(snodes, snodes_curr_memory))
    _curr_memory[None] = (0, 0)
    return (
        peak_memory,
        _curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
        name_to_freeable_input_buf,
    )


def _initialize_double_linked_list(
    snodes: list[BaseSchedulerNode],
) -> tuple[
    dict[BaseSchedulerNode, Optional[BaseSchedulerNode]],
    dict[BaseSchedulerNode, Optional[BaseSchedulerNode]],
    BaseSchedulerNode,
]:
    """Create double-linked list structure from snodes"""
    _prev = {}
    _next = {}
    for i, snode in enumerate(snodes):
        _prev[snode] = snodes[i - 1] if i > 0 else None
        _next[snode] = snodes[i + 1] if i < len(snodes) - 1 else None
    _head = snodes[0]
    return _prev, _next, _head


def is_corresponding_collective_wait(
    collective_snode: BaseSchedulerNode, wait_snode: BaseSchedulerNode
) -> bool:
    """
    Check if a wait node corresponds to a given collective node by verifying if the wait
    depends on outputs from the collective.
    """
    collective_outs = OrderedSet(o.get_name() for o in collective_snode.get_outputs())
    unmet_deps = OrderedSet(d.name for d in wait_snode.unmet_dependencies)
    return bool(unmet_deps & collective_outs)


def _op_runtime_estimate_mult(snode):
    # Apply multipliers for faster experimentation.
    # TODO(ivankobzarev): Remove after confirmation that runtime estimations are correct.
    if contains_collective(snode):
        return config_comms.reorder_sink_runtime_estimations_comm_mult

    return config_comms.reorder_sink_runtime_estimations_non_comm_mult


def is_async_collective(snode):
    """
    Filtering out ops that contain Collective and Wait inside and considered as Collectives.
    See contains_collective function.
    If the op contains Wait inside - consider as Synchronous compute.
    """
    if python_kernel_name := getattr(snode.node, "python_kernel_name", None):
        if "torch.ops._dtensor.shard_dim_alltoall.default" in python_kernel_name:
            return False

    return True


def contains_async_collective(snode):
    return contains_collective(snode, is_async_collective)


def _group_nodes_from_linked_list(
    head: Optional[BaseSchedulerNode],
    tail: Optional[BaseSchedulerNode],
    next_dict: dict[BaseSchedulerNode, Optional[BaseSchedulerNode]],
) -> list[BaseSchedulerNode]:
    """
    Traverse doubly-linked list from head to tail and return nodes as a list.

    Args:
        head: Starting node of the segment
        tail: Ending node of the segment (inclusive)
        next_dict: Dictionary mapping each node to its next node

    Returns:
        List of nodes from head to tail (inclusive)
    """
    ret = []
    n = head
    while True:
        if n is not None:
            ret.append(n)
        if n == tail:
            break
        n = next_dict[n]  # type: ignore[index]
    return ret


def _perform_double_linked_list_swap(
    candidate: BaseSchedulerNode,
    group_head: BaseSchedulerNode,
    group_tail: BaseSchedulerNode,
    prev_dict: dict[BaseSchedulerNode, Optional[BaseSchedulerNode]],
    next_dict: dict[BaseSchedulerNode, Optional[BaseSchedulerNode]],
    head: BaseSchedulerNode,
) -> BaseSchedulerNode:
    """
    Swap positions of candidate and group in doubly-linked list.

    Transforms:
    candidate_prev -> candidate -> group_head...group_tail -> group_tail_next
    Into:
    candidate_prev -> group_head...group_tail -> candidate -> group_tail_next

    Args:
        candidate: Node to swap with group
        group_head: First node of group
        group_tail: Last node of group
        prev_dict: Dictionary mapping nodes to their previous nodes
        next_dict: Dictionary mapping nodes to their next nodes
        head: Current head of the linked list

    Returns:
        New head of the linked list (may change if candidate was the head)
    """
    # 0: Update candidate's previous node
    candidate_prev = prev_dict[candidate]
    if candidate_prev:
        next_dict[candidate_prev] = group_head
    prev_dict[group_head] = candidate_prev

    # 2: Update group_tail's next node
    group_tail_next = next_dict[group_tail]
    if group_tail_next:
        prev_dict[group_tail_next] = candidate
    next_dict[candidate] = group_tail_next

    # 1: Link group_tail to candidate
    prev_dict[candidate] = group_tail
    next_dict[group_tail] = candidate

    # Update head if candidate was the head
    if head == candidate:
        return group_head
    return head


def _calculate_potential_peak_memory_reorder(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    group_tail: BaseSchedulerNode,
    group_peak_memory: int,
    candidate_delta_mem: int,
    candidate_allocfree: SNodeMemory,
    group_n_to_bufs_after_swap_dealloc_by_candidate: dict,
    curr_memory: dict,
) -> tuple[int, dict[BaseSchedulerNode, int]]:
    """
    Calculate potential peak memory after swapping candidate with group (reorder version).

    Computes new memory levels for all affected nodes and returns the potential
    peak memory along with cached post-allocation memory values for each node.

    Args:
        candidate: Node being moved
        gns: Group nodes
        group_tail: Last node of group
        group_peak_memory: Current peak memory within the group
        candidate_delta_mem: Net memory change from candidate (alloc - free)
        candidate_allocfree: Candidate's allocation/free info
        group_n_to_bufs_after_swap_dealloc_by_candidate: Buffers whose deallocation moves to candidate
        curr_memory: Current memory state dict

    Returns:
        Tuple of (potential_peak_memory, post_alloc_update_dict)
    """
    # Caching calculations of memory for group nodes and candidate,
    # to apply without recalculation after swap.
    _post_alloc_update: dict[BaseSchedulerNode, int] = {}
    potential_peak: int = 0
    if not group_n_to_bufs_after_swap_dealloc_by_candidate:
        # Not accounting for buffers last use change
        potential_peak = max(
            group_peak_memory - candidate_delta_mem,
            curr_memory[group_tail][1]
            - candidate_delta_mem
            + candidate_allocfree.size_alloc,
        )
        return potential_peak, _post_alloc_update

    # If candidate will be after group, the starting memory level of group nodes
    # changes to the -(candidate.size_alloc - candidate.size_free)
    mem_after_reorder_delta: int = -candidate_delta_mem
    for gn in gns:
        gn_post_alloc_mem = curr_memory[gn][0] + mem_after_reorder_delta
        _post_alloc_update[gn] = gn_post_alloc_mem
        potential_peak = max(potential_peak, gn_post_alloc_mem)

        bufs = group_n_to_bufs_after_swap_dealloc_by_candidate.get(gn)
        if bufs is not None:
            for buf in bufs:
                # Candidate will deallocate those buffers
                mem_after_reorder_delta += buf.mpi_buffer.size_free

    candidate_mem_post_alloc = (
        curr_memory[group_tail][1]
        + mem_after_reorder_delta
        + candidate_allocfree.size_alloc
    )
    _post_alloc_update[candidate] = candidate_mem_post_alloc
    potential_peak = max(potential_peak, candidate_mem_post_alloc)
    return potential_peak, _post_alloc_update


def _update_memory_tracking_after_swap_reorder(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    group_tail: BaseSchedulerNode,
    candidate_delta_mem: int,
    candidate_allocfree: SNodeMemory,
    group_n_to_bufs_after_swap_dealloc_by_candidate: dict,
    post_alloc_update: dict[BaseSchedulerNode, int],
    curr_memory: dict,
    buf_to_snode_last_use: dict,
    snodes_allocfree: dict,
) -> None:
    """
    Update memory tracking structures after swap (reorder version).

    Updates curr_memory, buf_to_snode_last_use, and snodes_allocfree dictionaries
    to reflect the new memory state after swapping candidate with group.

    Args:
        candidate: Node that was moved
        gns: Group nodes
        group_tail: Last node of group
        candidate_delta_mem: Net memory change from candidate (alloc - free)
        candidate_allocfree: Candidate's allocation/free info
        group_n_to_bufs_after_swap_dealloc_by_candidate: Buffers whose deallocation moves to candidate
        post_alloc_update: Cached post-allocation memory values
        curr_memory: Current memory state dict (mutated)
        buf_to_snode_last_use: Buffer to last-use node mapping (mutated)
        snodes_allocfree: Node allocation/free info dict (mutated)
    """
    if not group_n_to_bufs_after_swap_dealloc_by_candidate:
        for gn in gns:
            cm = curr_memory[gn]
            curr_memory[gn] = (
                cm[0] - candidate_delta_mem,
                cm[1] - candidate_delta_mem,
            )
        _candidate_post_alloc_mem = (
            curr_memory[group_tail][1] + candidate_allocfree.size_alloc
        )
        _candidate_post_free_mem = (
            _candidate_post_alloc_mem - candidate_allocfree.size_free
        )
        curr_memory[candidate] = (
            _candidate_post_alloc_mem,
            _candidate_post_free_mem,
        )
        return

    # Candidate becomes last use of some bufs
    for bufs in group_n_to_bufs_after_swap_dealloc_by_candidate.values():
        for buf in bufs:
            buf_to_snode_last_use[buf] = candidate

    size_free_to_move_to_candidate_sum: int = 0
    for n in gns:
        _gn_post_alloc_mem: int = post_alloc_update[n]
        size_free_to_move_to_candidate: int = sum(
            buf.mpi_buffer.size_free
            for buf in group_n_to_bufs_after_swap_dealloc_by_candidate[n]
        )
        size_free_to_move_to_candidate_sum += size_free_to_move_to_candidate
        # group node does not deallocate this after swap
        snodes_allocfree[n].size_free -= size_free_to_move_to_candidate
        gn_post_free_mem: int = _gn_post_alloc_mem - snodes_allocfree[n].size_free
        curr_memory[n] = (_gn_post_alloc_mem, gn_post_free_mem)
    _candidate_post_alloc_mem = post_alloc_update[candidate]
    snodes_allocfree[candidate].size_free += size_free_to_move_to_candidate_sum
    candidate_post_free_mem = (
        _candidate_post_alloc_mem - snodes_allocfree[candidate].size_free
    )
    curr_memory[candidate] = (
        _candidate_post_alloc_mem,
        candidate_post_free_mem,
    )


def _find_buffers_with_changed_last_use(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    buf_to_snode_last_use: dict,
) -> dict[BaseSchedulerNode, list[Union[FreeableInputBuffer, Any]]]:
    """
    Find buffers whose last use will change after swapping candidate with group.

    When we swap [candidate [group]] to [[group] candidate], some buffers that
    were last used by a group node will now be last used by candidate instead.
    This affects memory deallocation timing.

    Args:
        candidate: The node being moved
        gns: Group nodes being swapped with candidate
        buf_to_snode_last_use: Mapping of buffers to their current last-use nodes

    Returns:
        Dict mapping group nodes to buffers that will change their last-use node
    """
    group_n_to_bufs_after_swap_dealloc_by_candidate: dict[
        BaseSchedulerNode, list[Union[FreeableInputBuffer, Any]]
    ] = defaultdict(list)
    for (
        buf,
        snode_last_use,
    ) in buf_to_snode_last_use.items():
        succ_nodes = buf.mpi_buffer.succ_nodes
        if candidate not in succ_nodes:
            continue

        if not any(gn == snode_last_use for gn in gns):
            continue

        group_n_to_bufs_after_swap_dealloc_by_candidate[snode_last_use].append(buf)

    return group_n_to_bufs_after_swap_dealloc_by_candidate


def _is_node_groupable_for_reorder(
    candidate: BaseSchedulerNode,
) -> tuple[bool, Optional[str]]:
    """
    Check if a candidate node can be grouped with collective during reordering.

    This pass processes collectives left to right, so we avoid grouping with
    already-processed collectives based on configuration.

    Args:
        candidate: Node to check for groupability

    Returns:
        Tuple of (is_groupable, reason_if_not_groupable)
    """
    # This pass processes collectives left to right,
    # Do not group with processed collectives.
    # Leaving config for experimentation in 2D
    if not config_comms.reorder_iterative_group_with_collectives:
        if contains_async_collective(candidate):
            return (
                False,
                f"candidate contains_collective {candidate.get_name()}",
            )
    if not config_comms.reorder_iterative_use_runtime_estimations:
        if contains_gemm_like(candidate):
            return False, "contains_gemm_like"
    return True, None


def _format_and_log_reordering_stats(
    stats: dict[BaseSchedulerNode, ReorderInfo],
    head: BaseSchedulerNode,
    next_dict: dict[BaseSchedulerNode, Optional[BaseSchedulerNode]],
    original_snodes_num: int,
    peak_memory: int,
    name_to_freeable_input_buf: dict,
    graph_outputs: OrderedSet[str],
) -> list[BaseSchedulerNode]:
    """
    Format reordering statistics, log them, and return final node list.

    Computes improvement metrics, creates a formatted table (using tabulate if
    available), validates the reordered node count, recalculates peak memory,
    and logs all information.

    Args:
        stats: Per-node reordering statistics
        head: Head of the reordered linked list
        next_dict: Linked list next pointers
        original_snodes_num: Original number of nodes (for validation)
        peak_memory: Initial peak memory before reordering
        name_to_freeable_input_buf: Buffer memory tracking info
        graph_outputs: Graph output names

    Returns:
        Final reordered list of scheduler nodes
    """
    node_stats = stats
    improvement = {snode: node_stats[snode].improvement for snode in node_stats}
    total_improvement = sum([improvement[snode] for snode in improvement])
    total_moves = sum([node_stats[snode].moves for snode in node_stats])

    reorder_log_str = (
        f"reorder_communication_preserving_peak_memory improved overlap by {total_improvement} ns"
        f" after {total_moves} reorders.\n"
    )
    headers = [
        "Collective node",
        "comm_time(us)",
        "comp_time(us)",
        "initial exposed(us)",
        "final exposed(us)",
        "improvement(us)",
        "limiting factor",
        "moves",
        "grouped",
        "grouped_info",
        "overlap_info",
    ]
    rows = [
        [
            node_summary(snode),
            node_info.comm_time / 1e3,
            node_info.comp_time / 1e3,
            node_info.initial_exposed / 1e3,
            node_info.final_exposed / 1e3,
            node_info.improvement / 1e3,
            node_info.limiting_factor,
            node_info.moves,
            node_info.grouped,
            node_info.grouped_info,
            node_info.overlap_info,
        ]
        for snode, node_info in node_stats.items()
    ]
    if importlib.util.find_spec("tabulate"):
        # pyrefly: ignore[import-error]
        from tabulate import tabulate

        reorder_log_str += tabulate(
            rows,
            headers=headers,
        )
    else:
        reorder_log_str += (
            "Please `pip install tabulate` to nicely render overlap stats.\n"
        )
        reorder_log_str += str(headers) + "\n"
        reorder_log_str += "\n".join(map(str, rows))

    new_snodes = _group_nodes_from_linked_list(head, None, next_dict)
    assert len(new_snodes) == original_snodes_num
    new_peak_memory, _, _, _ = estimate_peak_memory_allocfree(
        new_snodes, name_to_freeable_input_buf, graph_outputs
    )
    reorder_log_str += f"\n peak_memory_before:{peak_memory}"
    reorder_log_str += f"\n peak_memory_after:{new_peak_memory}"

    overlap_log.info(reorder_log_str)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "reorder_communication_preserving_peak_memory",
            "encoding": "string",
        },
        payload_fn=lambda: reorder_log_str,
    )

    return new_snodes


def _reorder_communication_preserving_peak_memory_internal(
    snodes: list[BaseSchedulerNode],
) -> tuple[list[BaseSchedulerNode], dict[BaseSchedulerNode, ReorderInfo]]:
    """
    Internal testing helper that also returns debug info.
    Returns:
        - reordered snodes list
        - dict {snode: ReorderInfo}
    """
    has_collectives = False
    for snode in snodes:
        if contains_collective(snode):
            has_collectives = True
            break
    if not has_collectives:
        return snodes, {}

    from torch._inductor.scheduler import GroupedSchedulerNode

    original_snodes_num = len(snodes)
    # heuristic to avoid degenerating to quadratic time
    graph_inputs: OrderedSet[str] = OrderedSet(V.graph.graph_inputs.keys())
    graph_outputs: OrderedSet[str] = OrderedSet(V.graph.get_output_names())
    (
        peak_memory,
        _curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
        name_to_freeable_input_buf,
    ) = _initialize_memory_tracking(snodes, graph_inputs, graph_outputs)

    runtimes: dict[BaseSchedulerNode, float] = {
        snode: estimate_op_runtime(snode) * _op_runtime_estimate_mult(snode)
        for snode in snodes
    }
    # debug stats
    stats: dict[BaseSchedulerNode, ReorderInfo] = {}

    total_moves = 0

    _prev, _next, _head = _initialize_double_linked_list(snodes)

    debug_num_collectives_to_reorder: Optional[int] = (
        config_comms.reorder_iterative_debug_limit_to_reorder
    )

    num_processed_collectives: int = 0
    curr: Optional[BaseSchedulerNode] = _head
    debug_iterative_memory_recompute = (
        config_comms.reorder_iterative_debug_memory_recompute
    )
    iterative_recompute_error = False

    while curr is not None and _next[curr] is not None:
        _next_curr = _next[curr]
        if iterative_recompute_error:
            break
        # pyrefly: ignore [bad-argument-type]
        if not contains_async_collective(curr):
            curr = _next_curr
            continue

        if debug_num_collectives_to_reorder is not None and (
            num_processed_collectives >= debug_num_collectives_to_reorder
        ):
            break
        num_processed_collectives += 1

        info = stats[curr] = ReorderInfo()
        comm_time, comp_time, overlap_info = coll_exposed_communication_time(
            _group_nodes_from_linked_list(curr, None, _next), runtimes
        )
        info.comm_time = comm_time
        info.comp_time = comp_time
        info.initial_exposed = info.final_exposed = comm_time - comp_time
        info.overlap_info = overlap_info

        candidate = _prev[curr]
        group_head = curr
        group_tail = curr
        group_waits = {}
        group_runtime = 0.0
        group_peak_memory = _curr_memory[curr][0]  # post_alloc memory

        while candidate is not None:
            if config_comms.reorder_iterative_use_runtime_estimations and (
                info.final_exposed
                < -config_comms.reorder_iterative_extra_comm_comp_overlap
                * info.comm_time
            ):
                info.limiting_factor = "unexposed by runtime estimations"
                break

            if (
                not config_comms.reorder_iterative_unsafe_collectives_reorder
                and contains_collective(candidate)
            ):
                info.limiting_factor = "collective ordering"
                break

            gns: list[BaseSchedulerNode] = _group_nodes_from_linked_list(
                group_head, group_tail, _next
            )
            group = GroupedSchedulerNode(
                curr.scheduler,
                gns,
                temp_grouping=True,
            )

            # We can have multiple deps with the same name.
            # As we ignore WeakDep(is_fake=True) =>
            # filter them out first to avoid overwriting  of real dep.
            data_deps = {
                d.name: d for d in group.unmet_dependencies if not _is_fake_dep(d)
            }

            candidate_outs = candidate.get_outputs()
            data_dep = None
            for o in candidate_outs:
                if d := data_deps.get(o.get_name(), None):
                    data_dep = d
                    break

            if data_dep is not None:
                is_groupable_result, grouping_reason = _is_node_groupable_for_reorder(
                    candidate
                )
                if is_groupable_result:
                    group_head = candidate
                    # pyrefly: ignore[unbound-name]
                    if config_comms.reorder_iterative_use_runtime_estimations:
                        if contains_wait(candidate):
                            comm_time, comp_time, _ = wait_exposed_communication_time(
                                _group_nodes_from_linked_list(_head, candidate, _next),
                                runtimes,
                            )
                            group_waits[candidate] = comm_time, comp_time
                        if not contains_async_collective(candidate):
                            group_runtime += runtimes[candidate]

                    group_peak_memory = max(
                        group_peak_memory, _curr_memory[candidate][0]
                    )
                    info.grouped += 1
                    info.grouped_info = _group_names(gns)
                    candidate = _prev[candidate]
                    continue
                else:
                    msg = (
                        f"data dependency {data_dep}(dep_names:{list(data_deps.keys())})"
                        f"\n candidate:{candidate.get_name()}(outs:{[candidate.get_buffer_names()]})"
                        f"dep on {_group_names(gns)}"
                        f"\n non_group_reason:{grouping_reason}"
                    )
                    info.limiting_factor = msg
                    break

            # pyrefly: ignore[unbound-name]
            if config_comms.reorder_iterative_use_runtime_estimations:
                # Check if candidate has sync runtime
                if not contains_async_collective(candidate):
                    c_runtime = runtimes[candidate]

                    if c_runtime > 0 and len(group_waits) > 0:
                        # pyrefly: ignore[no-matching-overload]
                        exposed_before = max(0, info.comm_time - info.comp_time)
                        # pyrefly: ignore[no-matching-overload]
                        exposed_after = max(
                            0, info.comm_time - info.comp_time - c_runtime
                        )
                        exposed_delta = exposed_after - exposed_before
                        for gw_comm_time, gw_comp_time in group_waits.values():
                            gw_exposed_before = max(0, gw_comm_time - gw_comp_time)
                            gw_exposed_after = max(
                                0, gw_comm_time - gw_comp_time + c_runtime
                            )

                            exposed_delta += gw_exposed_after - gw_exposed_before

                        if exposed_delta > 0:
                            info.limiting_factor = (
                                f"candidate has compute {c_runtime},"
                                f" group contains waits, total_exposed_delta {exposed_delta}"
                            )
                            break
                        else:
                            # Update all group_colls comm_time, comp_time
                            for gw, (
                                gw_comm_time,
                                gw_comp_time,
                            ) in group_waits.items():
                                group_waits[gw] = (
                                    gw_comm_time,
                                    gw_comp_time - c_runtime,
                                )
                else:
                    # Candidate is async_collective

                    # Unsafe collectives reordering
                    # Cj -> [...group_runtime..., Ci] -> Wj
                    # Checking that we are not increasing exposed time of Cj
                    if group_runtime > 0:
                        comm_time, comp_time, _ = coll_exposed_communication_time(
                            _group_nodes_from_linked_list(candidate, None, _next),
                            runtimes,
                        )
                        # pyrefly: ignore[no-matching-overload]
                        exposed_before = max(0, comm_time - comp_time)
                        # pyrefly: ignore[no-matching-overload]
                        exposed_after = max(0, comm_time - comp_time + group_runtime)
                        exposed_delta = exposed_after - exposed_before
                        if exposed_delta > 0:
                            info.limiting_factor = (
                                f"candidate {candidate.get_name()} is collective,"
                                f" group_runtime:{group_runtime},"
                                f" exposed_delta:{exposed_delta} c_comm_time:{comm_time} c_comp_time:{comp_time}"
                            )
                            break

            candidate_allocfree: SNodeMemory = snodes_allocfree[candidate]
            candidate_delta_mem: int = (
                candidate_allocfree.size_alloc - candidate_allocfree.size_free
            )
            # candidate and one of group nodes are successors of the same buffer
            # and last use of the buffer happen in group nodes.
            # This last use deallocates it.
            # If we swap [candidate [group]] to [[group] candidate],
            # candidate becomes the last use
            # and deallocated this buffer instead of group node.
            # we need to update size_free accordingly to group_node and candidate,
            # and recalculate post_alloc, post_free for them.
            #
            # Buf that changes its last use snode,
            # after swap will be deallocated only by candidate,
            # while before it was deallocated by group node.
            group_n_to_bufs_after_swap_dealloc_by_candidate = (
                _find_buffers_with_changed_last_use(
                    candidate, gns, buf_to_snode_last_use
                )
            )

            potential_peak, _post_alloc_update = (
                _calculate_potential_peak_memory_reorder(
                    candidate,
                    gns,
                    group_tail,
                    group_peak_memory,
                    candidate_delta_mem,
                    candidate_allocfree,
                    group_n_to_bufs_after_swap_dealloc_by_candidate,
                    _curr_memory,
                )
            )

            if (
                potential_peak - peak_memory
                # pyrefly: ignore[unbound-name]
                > peak_memory * config_comms.reorder_iterative_peak_memory_budget
            ):
                info.limiting_factor = (
                    f"peak memory new:{potential_peak} vs base:{peak_memory}"
                )
                break
            info.moves += 1
            total_moves += 1

            _head = _perform_double_linked_list_swap(
                candidate, group_head, group_tail, _prev, _next, _head
            )

            comm_time, comp_time, overlap_info = coll_exposed_communication_time(
                _group_nodes_from_linked_list(curr, None, _next), runtimes
            )
            info.comm_time = comm_time
            info.comp_time = comp_time
            info.overlap_info = overlap_info
            info.final_exposed = comm_time - comp_time

            _update_memory_tracking_after_swap_reorder(
                candidate,
                gns,
                group_tail,
                candidate_delta_mem,
                candidate_allocfree,
                group_n_to_bufs_after_swap_dealloc_by_candidate,
                _post_alloc_update,
                _curr_memory,
                buf_to_snode_last_use,
                snodes_allocfree,
            )

            if debug_iterative_memory_recompute:
                # Compare iteratively recomputed memory data
                # with full run of estimate_peak_memory

                from .comms_debug import _debug_iterative_memory_recompute

                iterative_recompute_error = _debug_iterative_memory_recompute(
                    candidate,
                    gns,
                    _group_names(gns),
                    _group_nodes_from_linked_list(_head, None, _next),
                    name_to_freeable_input_buf,
                    graph_outputs,
                    peak_memory,
                    _curr_memory,
                    snodes_allocfree,
                    "reorder_communication_preserving_peak_memory",
                    group_n_to_bufs_after_swap_dealloc_by_candidate,
                )
                if iterative_recompute_error:
                    break
            candidate = _prev[group_head]
        curr = _next_curr

    new_snodes = _format_and_log_reordering_stats(
        stats,
        _head,
        _next,
        original_snodes_num,
        peak_memory,
        name_to_freeable_input_buf,
        graph_outputs,
    )

    return new_snodes, stats


def _schedule_for_comm(
    snodes: list[BaseSchedulerNode],
    raise_comms: bool,
    sink_waits: bool,
    reorder_for_overlap: bool,
) -> list[BaseSchedulerNode]:
    """
    Schedule `snodes` for various comm optimization objectives.

    Args:
        snodes: the nodes to be scheduled.
        raise_comms: whether to greedily schedule collectives as early as possible
        sink_wait: whether to greedily schedule waits as late as possible
        reorder_compute_for_overlap: whether to reorder compute nodes to
            optimize for compute/communication overlapping.

    Returns:
        The new schedule order.

    Some notes on the synergy between different options:
        - `raise_comms` provides more overlapping oppurtunies for `reorder_compute_for_overlap`.
        - When both `raise_comms` and `sink_waits` is `True`, `raise_comms` is prioritized.
    """
    # We assign each node a tuple of scores (score_0, score_1, score_2),
    # decreasing in importance, with a lower value indicating a higher ranking:
    #
    # - score_0: the lowest comm_idx among the comm nodes that the node blocks.
    # If a node doesn't block any comm nodes, its score_0 is set to
    # sys.maxsize. This score ensures that comm nodes get scheduled as early as
    # possible.
    # - score_1: 1 if the node is a wait node, 0 otherwise. This score ensures
    # that wait nodes are deferred as late as possible.
    # - score_2: the index of the node in the original topological order. This
    # score provides stability in case of ties.
    #
    # When only raise_comms is True, only score_0 and score_2 are considered.
    # When only sink_waits is True, only score_1 and score_2 are considered.
    # When neither is True, the original order is yielded.
    buf_name_to_snode = {}
    name_to_fused_node = {}
    scores_0, scores_1, scores_2 = {}, {}, {}
    for idx, snode in enumerate(snodes):
        for buf_name in snode.get_buffer_names():
            buf_name_to_snode[buf_name] = snode

        for op_name in snode.get_operation_names():
            name_to_fused_node[op_name] = snode
        name_to_fused_node[snode.get_name()] = snode

        node_name = snode.get_name()
        scores_0[node_name] = sys.maxsize
        scores_1[node_name] = 0
        scores_2[node_name] = idx

    comm_idx = 0
    for snode in snodes:
        if raise_comms and contains_collective(snode):
            scores_0[snode.get_name()] = comm_idx
            for ancestor in snode.ancestors:
                anc_fused_name = name_to_fused_node[ancestor].get_name()
                scores_0[anc_fused_name] = min(scores_0[anc_fused_name], comm_idx)
            comm_idx += 1
        elif sink_waits and contains_wait(snode):
            scores_1[snode.get_name()] = 1

    class Runnable:
        def __init__(self, snode) -> None:
            self.snode = snode
            name = next(iter(snode.get_operation_names()))
            fused_name = name_to_fused_node[name].get_name()
            self.score = (
                scores_0[fused_name],
                scores_1[fused_name],
                scores_2[fused_name],
            )

        def __lt__(self, other):
            return self.score < other.score

    unmet_deps: dict[BaseSchedulerNode, OrderedSet[str]] = {
        snode: OrderedSet(dep.name for dep in snode.unmet_dependencies)
        for snode in snodes
    }

    ready: list[Runnable] = []
    buffer_users: dict[str, OrderedSet[BaseSchedulerNode]] = defaultdict(OrderedSet)
    snode_to_cost = {snode: estimate_op_runtime(snode) for snode in snodes}

    for snode, deps in unmet_deps.items():
        if len(deps) == 0:
            heapq.heappush(ready, Runnable(snode))
        for dep in deps:
            buffer_users[dep].add(snode)

    scheduled = []

    def schedule(snode):
        """
        Schedules `snode` and put all unblocked nodes onto the ready queue.
        """
        scheduled.append(snode)
        for buf_name in snode.get_buffer_names():
            for snode in buffer_users[buf_name]:
                unmet_deps[snode].remove(buf_name)
                if len(unmet_deps[snode]) == 0:
                    heapq.heappush(ready, Runnable(snode))

    def get_overlapping_candidate():
        """
        Return the next node in the ready queue that's neither a collective or
        a wait.
        """
        candidates = [
            x
            for x in ready
            if not contains_collective(x.snode) and not contains_wait(x.snode)
        ]
        if len(candidates) == 0:
            return None
        return min(candidates, key=lambda x: x.score)

    def schedule_collective_for_overlap(snode):
        """
        Schedules collective node `snode`, along with one or more compute nodes
        to overlap with it. The strategy is described in the comment of
        `reorder_compute_for_overlap`.
        """
        assert contains_collective(snode)
        schedule(snode)

        collective_cost = snode_to_cost[snode]
        while (
            collective_cost > 0
            and (candidate := get_overlapping_candidate()) is not None
        ):
            ready.remove(candidate)

            schedule(candidate.snode)

            collective_cost -= snode_to_cost[candidate.snode]
        heapq.heapify(ready)

    while ready:
        snode = heapq.heappop(ready).snode
        if reorder_for_overlap and contains_collective(snode):
            schedule_collective_for_overlap(snode)
        else:
            schedule(snode)

    for deps in unmet_deps.values():
        assert len(deps) == 0, (
            f"Detected unscheduled nodes. Nodes with unmet dependencies: {unmet_deps}"
        )
    return scheduled


def decide_global_ordering_of_comms(
    nodes: list[BaseSchedulerNode], name_to_buf, name_to_fused_node
) -> list[BaseSchedulerNode]:
    """
    Decide global ordering of comms, by just enforcing the ordering that's in the input graph
    (might not be the same ordering as the eager mode program).
    TODO: Come up with a better approach
    """
    if not torch.distributed.is_available():
        return nodes

    comm_nodes = [n for n in nodes if contains_collective(n)]

    for i in range(1, len(comm_nodes)):
        # Enforce ordering by making previous comm a `WeakDep` dependency of the next comm
        mutating_buf = next(iter(comm_nodes[i].get_buffer_names()))
        for buf in comm_nodes[i - 1].get_buffer_names():
            comm_nodes[i].add_fake_dep(
                WeakDep(buf, mutating_buf=mutating_buf, is_fake=True)
            )

    return nodes


@dataclass
class SinkWaitInfo:
    grouped: int = 0
    grouped_info: str = ""
    moves: int = 0
    moves_info: str = ""
    limiting_factor: str = "None"
    comm_time: float = -1.0
    comp_time: float = -1.0
    initial_exposed: float = -1.0
    final_exposed: float = -1.0
    overlap_info: str = "None"

    @property
    def improvement(self):
        return self.initial_exposed - self.final_exposed


def _is_node_groupable_for_sink_waits(
    candidate: BaseSchedulerNode,
) -> tuple[bool, Optional[str]]:
    """
    Check if a candidate node can be grouped during sink_waits pass.

    Sink Waits traverses waits right to left, so we don't group with
    processed waits on the right or with async collectives.

    Args:
        candidate: Node to check for groupability

    Returns:
        Tuple of (is_groupable, reason_if_not_groupable)
    """
    # Sink Waits traverse Waits right to left,
    # => we do not group with processed Waits on the right.
    if contains_wait(candidate):
        return False, f"candidate contains wait {candidate.get_name()}"
    if contains_async_collective(candidate):
        return (
            False,
            f"candidate contains_async_collective {candidate.get_name()}",
        )

    # pyrefly: ignore[unbound-name]
    if not config_comms.sink_iterative_use_runtime_estimations:
        # Heuristics pre-use_runtime_estimations:
        # TODO(ivankobzarev): Remove them after confirming,
        # that using runtime estimations always give better results.
        # We do not want to group with collectives to not reorder them forward.
        if contains_collective(candidate):
   
```



## High-Level Overview


This Python file contains 4 class(es) and 62 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ReorderInfo`, `Runnable`, `SinkWaitInfo`

**Functions defined**: `align_runtime_estimations_across_all_distributed_ranks`, `sink_waits`, `raise_comms`, `reorder_compute_for_overlap`, `reorder_communication_preserving_peak_memory`, `improvement`, `is_gemm_like`, `contains_gemm_like`, `_temp_group_visit_leaves`, `wait_exposed_communication_time`, `accumulate_time`, `coll_exposed_communication_time`, `accumulate_time`, `_group_name`, `_is_fake_dep`, `_group_names`, `_initialize_memory_tracking`, `_initialize_double_linked_list`, `is_corresponding_collective_wait`, `_op_runtime_estimate_mult`

**Key imports**: annotations, heapq, importlib, itertools, logging, operator, sys, time, defaultdict, dataclass


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `heapq`
- `importlib`
- `itertools`
- `logging`
- `operator`
- `sys`
- `time`
- `collections`: defaultdict
- `dataclasses`: dataclass
- `typing`: Any, Optional, TYPE_CHECKING, Union
- `torch`
- `torch._logging`: trace_structured
- `torch.multiprocessing.reductions`: StorageWeakRef
- `torch.utils._ordered_set`: OrderedSet
- `.`: config, config_comms, ir
- `.dependencies`: WeakDep
- `.ir`: IRNode, Operation
- `.virtualized`: V
- `torch._inductor.scheduler`: BaseSchedulerNode
- `torch.distributed as dist`
- `torch.distributed.distributed_c10d`: _get_default_group
- `tabulate`: tabulate
- `.comms_debug`: _debug_iterative_memory_recompute


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `comms.py_docs.md`
- **Keyword Index**: `comms.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
