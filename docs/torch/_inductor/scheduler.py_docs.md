# Documentation: `torch/_inductor/scheduler.py`

## File Metadata

- **Path**: `torch/_inductor/scheduler.py`
- **Size**: 248,172 bytes (242.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import inspect
import itertools
import logging
import math
import operator
import os
import pprint
import textwrap
import traceback
import typing
from collections import Counter, defaultdict
from typing import Any, Generic, Optional, TYPE_CHECKING, TypeAlias, TypeVar, Union
from typing_extensions import ParamSpec

from torch.utils._ordered_set import OrderedSet

from .ir import ComputedBuffer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType

import sympy

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.ir import TritonTemplateCallerBase
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT
from torch.utils._triton import has_triton

from . import comms, config, config_comms, dependencies, ir, metrics
from .analyze_preserves_zero_mask import can_codegen_without_upcasts
from .codegen.common import BackendFeature, get_scheduling_for_device, Kernel
from .comm_analysis import (
    estimate_nccl_collective_runtime,
    estimate_nccl_collective_runtime_nccl_estimator,
)
from .dependencies import Dep, MemoryDep, StarDep, WeakDep
from .exc import GPUTooOldForTriton, TritonMissing
from .fx_utils import count_flops_fx
from .ir import (
    assign_origin_node,
    get_device_type,
    GraphPartitionSignature,
    MultiOutput,
    MultiOutputLayout,
    NoneLayout,
)
from .loop_body import LoopBody
from .memory import MemoryPlanningInfoForBuffer, MemoryPlanningInfoForNode
from .runtime.hints import ReductionHint
from .runtime.runtime_utils import green_text, red_text
from .sizevars import SimplifyIndexing
from .utils import (
    _unstable_customized_partition_wrapper,
    cache_on_self,
    cmp,
    device_need_guard,
    get_current_backend,
    get_device_tflops,
    get_dtype_size,
    get_gpu_dram_gbps,
    GraphPartitionMap,
    IndentedBuffer,
    is_collective,
    is_cudagraph_unsafe_op,
    is_gpu,
    is_multi_outputs_template,
    is_output_of_multi_outputs_template,
    is_wait,
    maybe_log_cudagraph_partition,
    sympy_product,
)
from .virtualized import V


log = logging.getLogger(__name__)
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")
loop_ordering_log = torch._logging.getArtifactLogger(__name__, "loop_ordering")
compute_dependencies_log = torch._logging.getArtifactLogger(
    __name__, "compute_dependencies"
)

PartitionType: TypeAlias = list["BaseSchedulerNode"]
_T = TypeVar("_T")
_P = ParamSpec("_P")


class MixOrderReduction:
    """
    This class contains utility functions to decide if we should fuse reductions
    reducing across different dimensions of the same input tensor.
    """

    @staticmethod
    def is_split_reduction(node: BaseSchedulerNode) -> bool:
        return node.is_reduction() and all(
            subnode.node._split_size is not None
            for subnode in node.get_nodes()
            if isinstance(subnode, SchedulerNode)
            and subnode.is_reduction()
            and isinstance(subnode.node, ComputedBuffer)
        )

    @classmethod
    def get_numel_rnumel(cls, node: BaseSchedulerNode) -> tuple[sympy.Expr, sympy.Expr]:
        if cls.is_split_reduction(node):
            xnumel = None
            rnumel = None
            for subnode in node.get_nodes():
                if not (
                    isinstance(subnode, SchedulerNode)
                    and subnode.is_reduction()
                    and isinstance(subnode.node, ComputedBuffer)
                ):
                    continue

                assert subnode.node._original_ranges is not None
                curxnumel = V.graph.sizevars.simplify(
                    sympy_product(subnode.node._original_ranges)
                )
                assert subnode.node._original_reduction_ranges is not None
                currnumel = V.graph.sizevars.simplify(
                    sympy_product(subnode.node._original_reduction_ranges)
                )

                if xnumel is None:
                    xnumel = curxnumel
                    rnumel = currnumel
                else:
                    assert V.graph.sizevars.statically_known_equals(
                        xnumel, curxnumel
                    ), f"{xnumel} v.s. {curxnumel}"
                    assert V.graph.sizevars.statically_known_equals(
                        rnumel, currnumel
                    ), f"{rnumel} v.s. {currnumel}"

            assert xnumel is not None
            return (xnumel, rnumel)
        else:
            return node.group[1]  # type: ignore[return-value]

    @classmethod
    def has_mix_reduction_orders(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        g1 = cls.get_numel_rnumel(node1)
        g2 = cls.get_numel_rnumel(node2)

        if len(g1) != 2 or len(g2) != 2 or g1 == g2:
            return False

        return tuple(g1) == tuple(reversed(g2))

    @classmethod
    def _is_full_access(cls, buf: str, node: BaseSchedulerNode) -> bool:
        """
        The access to 'buf' is not a broadcast access.
        """
        found_dep = None
        for dep in node.read_writes.reads:
            if isinstance(dep, MemoryDep) and dep.name == buf:
                found_dep = dep
                break

        if not found_dep:
            return False

        index = found_dep.index
        var_ranges = node.read_writes.var_ranges

        if not var_ranges:
            assert isinstance(node, FusedSchedulerNode), f"{type(node)}"
            var_ranges = node.snodes[0].read_writes.var_ranges

        assert var_ranges
        if not (OrderedSet(var_ranges) - OrderedSet(index.free_symbols)):
            return True

        # cases that happen after merging loops:
        #   MemoryDep('arg0_1', c0, {c0: 25165824})])
        #   var_ranges={d0: 32768, d1: 768}
        if V.graph.sizevars.statically_known_equals(
            sympy_product(found_dep.size), sympy_product(var_ranges.values())
        ):
            return True
        return False

    @classmethod
    def get_common_read(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> list[str]:
        out = []
        common_reads = node1.used_buffer_names() & node2.used_buffer_names()
        for buf in common_reads:
            if cls._is_full_access(buf, node1) and cls._is_full_access(buf, node2):
                out.append(buf)

        return out

    @classmethod
    def has_common_read(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return len(cls.get_common_read(node1, node2)) > 0

    # TODO add a cache
    @classmethod
    def can_fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """
        Check whether we can fuse two reductions with mix loop orders.
        """
        if not config.triton.mix_order_reduction:
            return False

        if not node1.is_gpu() or not node2.is_gpu():
            return False
        device_type = node1.get_device().type  # type: ignore[union-attr]
        if (
            device_type not in ("cuda", "xpu")
            or get_current_backend(device_type) != "triton"
        ):
            return False
        if not node1.is_reduction() or not node2.is_reduction():
            return False

        # check for mix reduction orders
        if not cls.has_mix_reduction_orders(node1, node2):
            return False

        # check common buffer accesses
        common_reads = MixOrderReduction.get_common_read(node1, node2)
        if len(common_reads) == 0:
            return False

        g1 = cls.get_numel_rnumel(node1)
        nrow = sympy.Max(g1[0], g1[1])
        ncol = sympy.Min(g1[0], g1[1])

        # the fused version has worse perf than non-fused version for
        # small workload. When a workload is small enough, data can be
        # fully cached by L2
        size_thres = 5 * 2**20
        if not V.graph.sizevars.statically_known_geq(nrow * ncol, size_thres):
            return False

        # We require more more row than columns since
        # 1, we prefer doing persistent reduction for each row
        # 2, we will split the reduction across the rows
        if not V.graph.sizevars.statically_known_geq(nrow, ncol * 2):
            return False

        # When nrow is small, ncol should also be small (due to the check
        # above). Thus the entire tensor should be well cached in L2.
        # Mix order reduction is less beneficial.
        if not V.graph.sizevars.statically_known_geq(nrow, 4096):
            return False

        contiguous_node, other_node = (
            (node1, node2) if g1[1] == ncol else (node2, node1)
        )

        # We previously only check the contiguous_node has contiguous
        # access to common_reads. But that turns out to be not enough.
        # The contiguous node may access a buffer that's node use by
        # other_ndoe. If that ascess is non-contiugous, generating
        # mix-order reduction can be inefficient especially when we
        # force XBLOCK to be 1
        # if not all(
        #     cls.is_contiguous_load(buf, contiguous_node) for buf in common_reads
        # ):
        #     return False
        if not all(
            cls.is_contiguous_load(dep.name, contiguous_node)
            for dep in contiguous_node.read_writes.reads
        ):
            return False

        # Make sure a persistent reduction will be generated
        if any(
            subnode.node.data.reduction_hint  # type: ignore[union-attr]
            not in (
                ReductionHint.INNER,
                ReductionHint.DEFAULT,
            )
            for subnode in contiguous_node.get_nodes()
            if subnode.is_reduction()
        ):
            return False

        # rnumel so large that we will not generated persistent reduction
        if not V.graph.sizevars.statically_known_leq(ncol, 1024 * 16):
            return False

        # Other reduction types like max/min is not supported yet.
        # There are no real use case as well.
        out = all(
            subnode.node.get_reduction_type()  # type: ignore[union-attr]
            in {
                "sum",
                "prod",
            }
            for subnode in other_node.get_nodes()
            if subnode.is_reduction()
        )
        return out

    @classmethod
    def are_mix_order_reductions(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return cls.can_fuse(node1, node2)

    @classmethod
    def is_contiguous_load(cls, buf: str, parent_node: BaseSchedulerNode) -> bool:
        from torch._inductor.loop_body import MemoryUsageType

        for node in parent_node.get_nodes():
            assert isinstance(node, SchedulerNode)
            loop_body = node._body
            entries = loop_body.memory_usage[MemoryUsageType.LOAD]
            index_names = [e.index_name for e in entries if e.buffer_name == buf]

            if len(index_names) == 0:
                continue

            # there can be multiple index_names some times
            for index_name in index_names:
                index_expr = loop_body.indexing_exprs[index_name]
                var_ranges = loop_body.var_ranges

                # assumes the final symbol is for reduction
                var_symbols = list(var_ranges.keys())
                stride_vars = V.graph.sizevars.stride_vars(
                    index_expr,
                    var_symbols,
                    var_symbols,
                )

                # stride==0 means a broadcast
                if not (stride_vars[-1] == 0 or stride_vars[-1] == 1):
                    return False
        return True


@dataclasses.dataclass
class SchedulerBuffer:
    scheduler: Scheduler
    node: ir.Buffer
    defining_op: Optional[BaseSchedulerNode]
    users: list[NodeUser] = dataclasses.field(default_factory=list)
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def defining_op_name(self) -> str:
        op = self.defining_op
        assert op is not None
        return op.get_name()

    def __hash__(self) -> int:
        return hash(self.node.name)

    def debug_str(self) -> str:
        result = IndentedBuffer()
        name = self.get_name()
        result.writeline(f"{name}: {type(self.node).__name__}")
        result.writeline(f"{name}.layout = {self.node.layout}")
        if self.get_aliases():
            result.writeline(f"{name}.aliases = {pformat(self.get_aliases())}")
        if self.get_mutations():
            result.writeline(f"{name}.mutations = {pformat(self.get_mutations())}")

        if len(self.users) <= 1:
            result.writeline(f"{name}.users = {self.users}")
        else:
            result.writeline(f"{name}.users = [")
            with result.indent(1):
                for user in self.users:
                    result.writeline(f"{user},")
            result.writeline("]")
        return result.getrawvalue()

    def get_name(self) -> str:
        return self.node.get_name()

    def allocate(self) -> None:
        assert self.node is not None
        if not self.node.should_allocate():
            return

        if (
            self.node.get_inputs_that_alias_output()
            or self.node.get_mutation_names()
            or isinstance(self.node.get_output_spec(), ir.CommBufferLayout)
        ):
            V.graph.wrapper_code.codegen_allocation(self.node)
            return

        # hacky check for if V.kernel is a real kernel or NullHandler
        if (
            hasattr(V.kernel, "args")
            and self.get_name() in V.kernel.inplace_update_buffers
        ):
            input_buffer: Union[ir.DonatedBuffer, ir.Buffer]
            input_buffer_name = V.kernel.inplace_update_buffers[self.get_name()]
            if input_buffer_name in self.scheduler.name_to_donated_buffer:
                input_buffer = self.scheduler.name_to_donated_buffer[
                    input_buffer_name
                ].node
            else:
                input_buffer = self.scheduler.name_to_buf[input_buffer_name].node
            V.graph.wrapper_code.codegen_inplace_reuse(
                input_buffer,
                self.node,
            )
        else:
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self) -> bool:
        # There's no real allocated buffer, no need to free it
        assert self.node is not None
        if isinstance(self.node.layout, ir.NoneLayout) or is_multi_outputs_template(
            self.node
        ):
            return False
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
        return True

    def set_users(self, users: list[NodeUser]) -> None:
        # deduplicate
        result: dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = use.merge(result[id(use.node)])
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def get_aliases(self) -> Sequence[str]:
        assert self.node is not None
        return self.node.get_inputs_that_alias_output()

    def get_mutations(self) -> Sequence[str]:
        assert self.node is not None
        return self.node.get_mutation_names()

    def get_device(self) -> Optional[torch.device]:
        return self.node.get_output_spec().get_device()


@dataclasses.dataclass
class SchedulerDonatedBuffer(SchedulerBuffer):
    defining_op: Optional[BaseSchedulerNode] = None


class BaseSchedulerNode:
    ancestors: OrderedSet[str]
    group: tuple[torch.device, tuple[tuple[sympy.Expr, ...], ...]]
    last_usage: OrderedSet[str]
    # .min_order and .max_order are only relevant for "grouped" nodes such as FusedSchedulerNode.
    # e.g. if the FusedSchedulerNode includes nodes (op_1, op_2, op_3), and op_X is X-th node
    # in `self.scheduler.nodes`, then for this FusedSchedulerNode, .min_order is 1 and .max_order is 3.
    # For non-"grouped" nodes (i.e. regular SchedulerNode),
    # .min_order = .max_order = X if this node is X-th node in `self.scheduler.nodes`.
    min_order: int
    max_order: int
    mpi_node: MemoryPlanningInfoForNode
    mutation_renames: dict[str, str]
    node: Optional[ir.Operation] = None
    outputs: list[SchedulerBuffer]
    outputs_by_name: dict[str, SchedulerBuffer]
    override_estimated_runtime: Optional[float] = None
    read_writes: dependencies.ReadWrites
    unmet_dependencies: OrderedSet[Dep]
    written: bool = False

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler: Scheduler = scheduler
        self.debug_device_str: Callable[[BaseSchedulerNode], list[str]] = (
            lambda *args, **kwargs: []
        )

    def _init_from_node(self, node: ir.Operation) -> None:
        self.node = node
        self.ancestors = OrderedSet()
        self.last_usage = OrderedSet[
            str
        ]()  # buffers that won't be used after this kernel
        self.written = False
        self.outputs = [
            SchedulerBuffer(
                scheduler=self.scheduler,
                node=output,
                defining_op=self,
            )
            for output in node.get_outputs()
        ]
        self.outputs_by_name = {buf.get_name(): buf for buf in self.outputs}

        # mutation_renames for the current node. Due to potential
        # more mutations happening later, this can be different
        # to Scheduler.mutation_renames. Also this dict should be small
        # since only mutation information relevant to the deps for this
        # node is stored here.
        self.mutation_renames = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.get_name()!r})"

    def debug_str(self) -> str:
        """Longer form printout for trace logs"""
        name = self.get_name()
        buf = IndentedBuffer()
        buf.splice(
            f"""\
{name}: {type(self).__name__}({type(getattr(self, "node", None)).__name__})
{name}.writes = {pformat(self.read_writes.writes)}
{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}
{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}
{name}.outputs = [
        """
        )
        with buf.indent():
            for out in self.get_outputs():
                buf.splice(out.debug_str())
        buf.writeline("]")

        try:
            buf.splice(self.debug_str_extra())
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)

        return buf.getrawvalue().rstrip()

    def debug_str_extra(self) -> str:
        return ""

    def _debug_str_for_device(self) -> list[str]:
        return self.debug_device_str(self)

    def debug_str_short(self) -> str:
        maybe_data = getattr(self.node, "data", None)
        data_str = ""
        if isinstance(maybe_data, torch._inductor.ir.Pointwise):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_size()], shorten=False, multiline=False
            )
        elif isinstance(maybe_data, torch._inductor.ir.Reduction):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_reduction_size(), maybe_data.get_reduction_type()],
                shorten=False,
                multiline=False,
            )
        return f"{self}{data_str}"

    def log_details(self) -> None:
        log.info(
            "%s: unmet_dependencies = %s, writes = %s",
            self,
            self.unmet_dependencies,
            self.read_writes.writes,
        )

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> bool:
        return False

    def update_mutated_names(self, renames: dict[str, str]) -> None:
        self.mutation_renames = {
            name: renames[name]
            for name in (dep.name for dep in self.read_writes.reads_and_writes())
            if name in renames
        }
        self.set_read_writes(self.read_writes.rename(self.mutation_renames))

    def add_fake_dep(self, dep: Dep) -> None:
        self.set_read_writes(self.read_writes.with_read(dep))

    def has_aliasing_or_mutation(self) -> bool:
        return any(
            buf.get_aliases() or buf.get_mutations() for buf in self.get_outputs()
        )

    def set_read_writes(self, rw: dependencies.ReadWrites) -> None:
        self.read_writes = rw
        self.unmet_dependencies = self.read_writes.reads
        self.prune_deps()

    def set_last_usage(
        self, future_used_buffers: OrderedSet[str], mutation_real_name: dict[str, str]
    ) -> None:
        used_buffers = self.used_or_aliased_buffer_names()
        used_buffers = OrderedSet(mutation_real_name.get(k, k) for k in used_buffers)
        self.last_usage = used_buffers - future_used_buffers

    def mark_run(self) -> None:
        for buf in self.outputs:
            buf.allocate()

    def used_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet(
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        )

    def used_or_aliased_buffer_names(self) -> OrderedSet[str]:
        used_names: OrderedSet[str] = OrderedSet()

        deps = [
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        ]
        while len(deps) > 0:
            dep = deps.pop()
            used_names.add(dep)
            if V.graph.name_to_buffer.get(dep):
                deps.extend(
                    alias
                    for alias in V.graph.name_to_buffer[
                        dep
                    ].get_inputs_that_alias_output()
                    if alias not in used_names
                )
        return used_names

    def prune_deps(self) -> None:
        self.unmet_dependencies = OrderedSet(
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        )

    def prune_weak_deps(self) -> None:
        # Prune weak dependencies on operations that have been removed
        def should_prune(dep: Dep) -> bool:
            if not isinstance(dep, WeakDep):
                return False
            op_name = self.scheduler.name_to_buf[dep.name].defining_op_name()
            return op_name in V.graph.removed_operations

        to_remove = OrderedSet(
            dep for dep in self.read_writes.reads if should_prune(dep)
        )
        self.set_read_writes(self.read_writes.remove_reads(to_remove))

    def prune_redundant_deps(
        self, name_to_fused_node: dict[str, BaseSchedulerNode]
    ) -> None:
        _prune_redundant_deps(self, name_to_fused_node, self.scheduler.name_to_buf)

    def get_name(self) -> str:
        assert self.node is not None
        return self.node.get_operation_name()

    def get_first_name(self) -> str:
        return self.get_name()

    @cache_on_self
    def get_operation_names(self) -> OrderedSet[str]:
        return OrderedSet(node.get_name() for node in self.get_nodes())

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet(out.get_name() for out in self.outputs)

    @cache_on_self
    def can_codegen_in_low_precision(self) -> bool:
        return all(
            isinstance(n, SchedulerNode)
            and can_codegen_without_upcasts(n, disallow_fp32_ops=True)
            for n in self.get_nodes()
        )

    @cache_on_self
    def can_codegen_without_upcasts(self) -> bool:
        return all(
            isinstance(n, SchedulerNode) and can_codegen_without_upcasts(n)
            for n in self.get_nodes()
        )

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        return [self]

    def get_outputs(self) -> Sequence[SchedulerBuffer]:
        return self.outputs

    def get_output(self, buf_name: str) -> SchedulerBuffer:
        return self.outputs_by_name[buf_name]

    def get_device(self) -> Optional[torch.device]:
        assert self.node is not None
        return self.node.get_device()

    def is_cpu(self) -> bool:
        device = self.get_device()
        return device is not None and device.type == "cpu"

    def is_gpu(self) -> bool:
        device = self.get_device()
        return device is not None and is_gpu(device.type)

    def is_reduction(self) -> bool:
        return False

    def is_native_matmul(self) -> bool:
        return False

    def is_split_scan(self) -> bool:
        return False

    def is_template(self) -> bool:
        return False

    def is_extern(self) -> bool:
        return False

    def is_foreach(self) -> bool:
        return False

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        return False

    def has_side_effects(self) -> bool:
        return False

    def decide_inplace_update(self) -> None:
        """
        Decide if there should be inplace updates for the node
        and record the decision in the active kernel.
        """
        from .codegen.wrapper import can_match_buffer_size

        if not (
            isinstance(self, SchedulerNode)
            and config.inplace_buffers
            and V.graph.has_feature(self.get_device(), BackendFeature.INPLACE_BUFFERS)
            and (
                not isinstance(V.kernel, torch._inductor.codegen.simd.SIMDKernel)
                or getattr(V.kernel, "mutations", None) is not None
            )
            # hacky check for if V.kernel is a real kernel or NullHandler
            and hasattr(V.kernel, "args")
        ):
            return

        # NOTE remove V.graph.removed_operations once deps issue is fixed
        inconsequential_nodes = (
            self.ancestors
            | V.graph.removed_operations
            | self.scheduler.completed_operations
        )

        def single_index_in_fused_node(buf_to_be_inplaced: SchedulerBuffer) -> bool:
            # Inside of NodeUser, we track that the read and write are equivalent
            # before deciding if the use can be inplace.
            # But if that use is fused into a larger kernel, we need to check equivalence
            # of other accesses in fused scheduler node as well.
            fused_node = buf_to_be_inplaced.scheduler.get_fused_node(self)
            buf_name = buf_to_be_inplaced.get_name()
            # Dedup read/writes with equivalent indices
            # TODO - would be nice if we could just cache accesses on ReadWrites,
            # and enforce variant that this class & members are functional..
            deps: OrderedSet[Dep] = OrderedSet()
            for user in buf_to_be_inplaced.users:
                user_node = user.node
                if not isinstance(user_node, BaseSchedulerNode):
                    continue

                if (
                    user_node.get_first_name()
                    not in buf_to_be_inplaced.scheduler.name_to_fused_node
                    or buf_to_be_inplaced.scheduler.get_fused_node(user_node)
                    is not fused_node
                ):
                    continue

                deps |= (
                    o
                    for o in user_node.read_writes.reads_and_writes()
                    if o.name == buf_name
                )
                if len(deps) > 1:
                    return False

            return True

        for buf in self.get_outputs():
            buf_node = buf.node
            assert buf_node is not None
            if (
                not buf_node.should_allocate()
                or buf_node.get_inputs_that_alias_output()
                or buf_node.get_mutation_names()
                or buf.get_name() in V.graph.removed_buffers
            ):
                continue

            for read in self.read_writes.reads:
                input_buf: Optional[Union[SchedulerBuffer, SchedulerDonatedBuffer]]
                if read.name in self.scheduler.name_to_donated_buffer:
                    input_buf = self.scheduler.name_to_donated_buffer[read.name]
                else:
                    input_buf = self.scheduler.name_to_buf.get(read.name)

                if (
                    input_buf
                    and V.graph.wrapper_code.can_reuse(input_buf, self)
                    and not isinstance(input_buf.defining_op, NopKernelSchedulerNode)
                ):
                    assert input_buf.users is not None
                    remaining_uses = [
                        x
                        for x in input_buf.users
                        if x.node.get_name() not in inconsequential_nodes
                    ]
                    if (
                        len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                        and input_buf.node is not None
                        and not isinstance(
                            input_buf.node.get_output_spec(),
                            (
                                ir.NoneLayout,
                                ir.MultiOutputLayout,
                                ir.MutationLayoutSHOULDREMOVE,
                            ),
                        )
                        and not (
                            input_buf.defining_op
                            and isinstance(
                                input_buf.defining_op.node,
                                (ir.FallbackKernel, ir.MultiOutput),
                            )
                            and len(input_buf.node.get_inputs_that_alias_output()) > 0
                        )
                        and can_match_buffer_size(input_buf.node, buf.node)
                        and single_index_in_fused_node(input_buf)
                    ):
                        # if there isn't a triton kernel, then we don't need to call triton-specific things.
                        # but TODO this might be a convenient place to signal to the Collective kernels to inplace
                        # (and, can we make "kernel" less generic of a name?)
                        V.kernel.args.make_inplace(input_buf.get_name(), buf.get_name())
                        # mutations not tracked in cpp kernels
                        if isinstance(
                            V.kernel, torch._inductor.codegen.simd.SIMDKernel
                        ):
                            V.kernel.mutations.add(input_buf.get_name())
                            V.kernel.mutations.add(buf.get_name())

                        V.kernel.inplace_update_buffers[buf.get_name()] = (
                            input_buf.get_name()
                        )
                        break

    def codegen_originating_info(
        self, buffer: IndentedBuffer, only_once: bool = True
    ) -> None:
        if not config.comment_origin:
            return

        if only_once and self.written:
            return
        assert self.node is not None
        origins = self.node.get_origins()
        out_lines = []

        for o in origins:
            if o.op == "output":
                # These are boring and samey
                continue

            out_lines.append("")
            # TODO(voz): Should the pragma be constant somewhere?
            out_lines.append("#pragma CMT ORIGIN:")
            op_info_str = f"#pragma CMT {o.op} {o.target}"
            if "seq_nr" in o.meta:
                op_info_str = op_info_str + f" seq_nr:{o.meta['seq_nr']}"
            out_lines.append(op_info_str)
            if "stack_trace" in o.meta:
                stack_trace = f"{o.meta['stack_trace']}"
                stack_trace_last_line = stack_trace.rsplit("|", maxsplit=1)[-1]
                out_lines.append(
                    "#pragma CMT "
                    + stack_trace_last_line.replace("{", "{{")
                    .replace("}", "}}")
                    .replace("\n", "\\")
                    .replace(
                        "\\", "\\\\"
                    )  # For windows safe path, avoid for example \x, \U.
                )
                out_lines.append("#pragma CMT END ORIGIN")
                out_lines.append("")

        if len(out_lines) == 0:
            return

        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        buffer.writelines(out_lines)
        self.written = True

    @cache_on_self
    def get_read_write_buffers_sizes(self) -> int:
        return self.get_read_write_buffers_sizes_impl(
            include_reads=True, include_writes=True
        )

    @cache_on_self
    def get_read_buffer_sizes(self) -> int:
        return self.get_read_write_buffers_sizes_impl(
            include_reads=True, include_writes=False
        )

    @cache_on_self
    def get_write_buffer_sizes(self) -> int:
        return self.get_read_write_buffers_sizes_impl(
            include_reads=False, include_writes=True
        )

    def get_read_write_buffers_sizes_impl(
        self, include_reads: bool, include_writes: bool
    ) -> int:
        return sum(
            self.get_read_write_buffer_accesses(
                include_reads=include_reads, include_writes=include_writes
            ).values(),
            start=0,
        )

    def get_read_write_buffer_accesses(
        self, include_reads: bool, include_writes: bool
    ) -> dict[str, int]:
        """
        Counting the number of bytes accessed for a kernel is
        surprisingly tricky. In particular, there is a differentiation
        between 'theoretical' memory accesses and practical memory
        accesses. For example, a layernorm kernel may actually access an
        input 3 times, but in theory, it only needs to access its input
        once (and may be optimized to do so through say, persistent
        reductions)

        Another example is that even though a buffer is passed in, we may
        not access the entire buffer. This may occur if we are accessing
        a slice of the buffer. Another tricky case is for indirect
        indexing, where the amount of bytes accessed depends on the
        values of the input.

        What this function aims to compute is the memory accesses for
        worst-case inputs, best-case optimization. What this means is
        that for each buffer we compute the amount of potential accesses in two ways and take the minimum.

        1. Numel in ranges multiplied by number of deps the buffer has
        2. The buffer size

        Returns memory accesses per buffer.
        """
        if isinstance(self, NopKernelSchedulerNode):
            return {}
        if isinstance(self, ExternKernelSchedulerNode) and isinstance(
            self.node, MultiOutput
        ):
            # todo: Calculate this - it's kinda annoying.
            return {}
        if (
            isinstance(self, ExternKernelSchedulerNode)
            and isinstance(self.node, ir.FallbackKernel)
            and self.node.op_overload
            is torch._prims.rng_prims.graphsafe_run_with_rng_state
        ):
            return {}

        def try_size_hint(s: sympy.Expr) -> int:
            return V.graph.sizevars.size_hint(s, fallback=0)

        if isinstance(self, SchedulerNode):
            node_numel = try_size_hint(
                sympy_product(self.get_ranges()[0])
                * sympy_product(self.get_ranges()[1]),
            )
        else:
            node_numel = int(1e9)
        buf_accesses = collections.defaultdict(list)

        if include_reads:
            for dep in self.read_writes.reads:
                buf_accesses[dep.name].append(dep)

        if include_writes:
            for dep in self.read_writes.writes:
                buf_accesses[dep.name].append(dep)

        reads = (
            OrderedSet(dep.name for dep in self.read_writes.reads)
            if include_reads
            else OrderedSet()
        )
        writes = (
            OrderedSet(dep.name for dep in self.read_writes.writes)
            if include_writes
            else OrderedSet()
        )

        def is_materialized(buf: str, snodes: Sequence[BaseSchedulerNode]) -> bool:
            users = self.scheduler.name_to_buf[buf].users
            buf_uses = OrderedSet(user.node for user in users)
            return len(buf_uses - OrderedSet(snodes)) > 0

        if isinstance(self, FusedSchedulerNode):
            removed_buffers = OrderedSet(
                dep for dep in writes if not is_materialized(dep, self.snodes)
            )
            writes = writes - removed_buffers
            reads = reads - removed_buffers

        buf_byte_accesses: dict[str, int] = {}

        for buf_name in reads | writes:
            buf_accessed_elems = sum(node_numel for dep in buf_accesses[buf_name])
            buf: Union[ir.Buffer, ir.TensorBox, ir.TorchBindObject]
            if buf_name in V.graph.name_to_buffer:
                buf = V.graph.name_to_buffer[buf_name]
            elif buf_name in V.graph.graph_inputs:
                buf = V.graph.graph_inputs[buf_name]
            else:
                continue

            def get_buf_bytes(
                buf: Optional[Union[ir.Buffer, ir.TensorBox, ir.TorchBindObject]],
            ) -> int:
                if not buf:
                    return 0

                if isinstance(buf, ir.TorchBindObject):
                    return buf.get_buf_bytes()
                elif isinstance(buf.layout, MultiOutputLayout):
                    # Kind of a lazy way to get the MultiOutput nodes corresponding to
                    # a MultiOutputLayout
                    users = self.scheduler.name_to_buf[buf.get_name()].users
                    tot = 0
                    for user in users:
                        assert isinstance(user.node, BaseSchedulerNode)
                        if isinstance(user.node.node, MultiOutput):
                            for sched_buf in user.node.get_outputs():
                                tot += get_buf_bytes(sched_buf.node)
                        else:
                            # Buf is a MultiOutputLayout but not all of its
                            # users are MultiOutputs...
                            # TODO: Figure out what's going on
                            return 0
                    return tot
                elif isinstance(buf.layout, ir.NoneLayout):
                    return sum(
                        get_buf_bytes(V.graph.get_buffer(mut_name))
                        for mut_name in buf.get_mutation_names()
                    )
                else:
                    buf_elems = try_size_hint(sympy_product(buf.get_size()))
                    return get_dtype_size(buf.get_dtype()) * min(
                        buf_accessed_elems, buf_elems
                    )

            buf_bytes = get_buf_bytes(buf)
            if buf_name not in buf_byte_accesses:
                buf_byte_accesses[buf_name] = buf_bytes
            else:
                buf_byte_accesses[buf_name] += buf_bytes

        return buf_byte_accesses

    @cache_on_self
    def estimate_flops(self) -> int | None:
        if self.node is None:
            return None
        fx_node = self.node.get_origin_node()
        if fx_node is None:
            return None

        flops = count_flops_fx(fx_node)
        if flops is None:
            return None

        resolved_flops = V.graph.sizevars.size_hint(flops, fallback=0)
        counters["inductor"]["flop_count"] += resolved_flops
        return resolved_flops

    def get_estimated_runtime(self) -> float:
        if self.override_estimated_runtime is not None:
            return self.override_estimated_runtime

        return self._get_estimated_runtime()

    @cache_on_self
    def _get_estimated_runtime(self) -> float:
        """
        Returns estimated op runtime in milliseconds (ms)
        """
        buf = self.get_nodes()[0].get_outputs()[0]
        layout = buf.node.get_output_spec()
        if not is_gpu(get_device_type(layout)):
            # default to no reordering based on runtime
            return 0

        # Collective kernels
        if is_collective(self.node):
            assert isinstance(self.node, ir.IRNode)
            try:
                if config_comms.runtime_estimations_use_nccl_lib_estimations:
                    cache_key = get_estimate_runtime_cache_key_from_snode(self)
                    cache = get_estimate_runtime_cache()
                    cache_val = cache.lookup(cache_key)
                    if cache_val is not None:
                        assert isinstance(cache_val, float)
                        return cache_val

                    ms = estimate_nccl_collective_runtime_nccl_estimator(self)
                    if ms is None:
                        # NCCL estimations fail: fallback to in-tree algorithmic estimation.
                        ms = estimate_nccl_collective_runtime(self.node)

                    cache.set_value(cache_key, value=ms)
                    return ms
                return estimate_nccl_collective_runtime(self.node)
            except ValueError as e:
                # We don't know how to estimate runtime for this collective,
                # falling back to 0
                log.info(e)  # noqa: G200
                return 0
            except TypeError as e:
                # this happens when the collective is not of type ir._CollectiveKernel
                log.info(e)  # noqa: G200
                return 0

        elif is_wait(self.node):
            # ir.Wait is only used for collective ops.
            # The time needed for the collective op is already estimated and considered
            # when we are processing the collective op IR node, so ir.Wait takes 0 time
            # since it doesn't take extra time to get the result after the collective is completed.
            return 0

        ret = maybe_estimate_runtime_benchmark(self)
        if ret is not None:
            return ret

        dtype = buf.node.maybe_get_dtype()
        try:
            gpu_memory_bandwidth = get_gpu_dram_gbps()
            gpu_flops = get_device_tflops(dtype) * 10**12
            # If cudaGetDeviceProperties returns 0 for gpu_memory_bandwidth or gpu_flops
            # there is a chance to continue execution successfully. Otherwise, it would fail with
            # ZeroDivisionError below.
            if gpu_memory_bandwidth <= 0:
                raise AssertionError(
                    f"gpu_memory_bandwidth cannot be <= 0, but got {gpu_memory_bandwidth}"
                )
            if gpu_flops <= 0:
                raise AssertionError(f"gpu_flops cannot be <= 0, but got {gpu_flops}")
        except Exception:
            return 0

        flops_est = self.estimate_flops()

        if flops_est == 0 or flops_est is None:
            # no flops estimate, so fall back to memory estimate
            ns = self.get_read_write_buffers_sizes() / gpu_memory_bandwidth
            ms = ns / 1e6
            return ms

        # TODO(xmfan): find a better heuristic to model FLOPS/latency relationship
        factor = 1.0
        counted_bytes = self.get_read_write_buffers_sizes()
        counted_bytes = 0 if counted_bytes is None else counted_bytes
        compute_time = (factor * flops_est / gpu_flops) * 1e9
        transfer_time = counted_bytes / gpu_memory_bandwidth

        # Return estimated runtime in milliseconds
        ns = max(compute_time, transfer_time)
        ms = ns / 1e6
        return ms

    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        return None

    def get_template_node_or_throw(self) -> ir.TemplateBuffer:
        template = self.get_template_node()
        assert template is not None
        return template

    @staticmethod
    def get_prologue_template_epilogue(
        nodes: list[BaseSchedulerNode],
    ) -> tuple[list[BaseSchedulerNode], BaseSchedulerNode, list[BaseSchedulerNode]]:
        """
        For the list of nodes, get the prologue, template, and epilogue
        """
        template_index = next(i for i, n in enumerate(nodes) if n.is_template())

        prologue = nodes[:template_index]
        template_node = nodes[template_index]
        epilogue = nodes[template_index + 1 :]
        return prologue, template_node, epilogue


@functools.cache
def get_estimate_runtime_cache() -> torch._inductor.codecache.LocalCache:
    return torch._inductor.codecache.LocalCache()


def get_estimate_runtime_cache_key_from_snode(snode: BaseSchedulerNode) -> str:
    python_kernel_name = getattr(snode.node, "python_kernel_name", "")
    args = snode.node.inputs  # type: ignore[union-attr]
    args = snode.node.fill_non_provided_args(  # type: ignore[union-attr]
        [*args, *snode.node.constant_args],  # type: ignore[union-attr]
        snode.node.kwargs,  # type: ignore[union-attr]
    )
    kwargs = snode.node.kwargs  # type: ignore[union-attr]
    flat_args, flat_args_pytree_spec = pytree.tree_flatten((args, kwargs))

    def _is_tensor_ir(x) -> bool:  # type: ignore[no-untyped-def]
        return isinstance(x, ir.IRNode) and not isinstance(x, ir.GeneratorState)

    cache_key = str(
        (python_kernel_name,)
        + tuple(tuple(a.get_size()) if _is_tensor_ir(a) else None for a in flat_args)
    )
    return cache_key


def _get_mm_like_fn(snode: BaseSchedulerNode) -> Optional[Callable[[Any], Any]]:
    if not isinstance(snode, ExternKernelSchedulerNode):
        return None
    mms_fns = {
        "extern_kernels.mm": torch.ops.aten.mm,
        "extern_kernels.bmm": torch.ops.aten.bmm,
        "extern_kernels.addmm": torch.ops.aten.addmm,
    }
    python_kernel_name = getattr(snode.node, "python_kernel_name", "")
    if python_kernel_name not in mms_fns:
        return None
    if not isinstance(snode.node, ir.ExternKernel):
        return None
    return mms_fns[python_kernel_name]


def maybe_estimate_runtime_benchmark(snode: BaseSchedulerNode) -> Optional[float]:
    bench_fn = None
    args_kwargs_fn = None
    if config.runtime_estimations_mms_benchmark:
        mm_fn = _get_mm_like_fn(snode)
        if mm_fn is None:
            return None
        bench_fn = mm_fn
        # pyrefly: ignore [unbound-name]
        args_kwargs_fn = lambda: snode_args_kwargs(snode)  # noqa: E731
    else:
        return None

    cache_key = get_estimate_runtime_cache_key_from_snode(snode)
    cache = get_estimate_runtime_cache()
    cache_val = cache.lookup(cache_key)
    if cache_val is not None:
        assert isinstance(cache_val, float)
        return cache_val

    from .utils import snode_args_kwargs

    args, kwargs = args_kwargs_fn()
    from torch._inductor.runtime.benchmarking import benchmarker

    ms = benchmarker.benchmark(bench_fn, args, kwargs)  # type: ignore[arg-type]

    cache.set_value(cache_key, value=ms)
    return ms


@dataclasses.dataclass(slots=True)
class WhyNoFuse:
    name1: str
    name2: str
    reason: str
    args: tuple[Any, ...]

    def __init__(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> None:
        self.name1 = node1.get_name()
        self.name2 = node2.get_name()

    def __call__(self, reason: str, *args: Any) -> None:
        self.reason = reason
        self.args = args
        fusion_log.debug(self)

    def __str__(self) -> str:
        return f"cannot fuse {self.name1} with {self.name2}: " + (
            self.reason % self.args
        )


def pformat(obj: Any) -> str:
    if isinstance(obj, (OrderedSet, set)):  # noqa: set_linter
        # pformat has trouble with sets of sympy exprs
        obj = sorted(obj, key=str)
    result = pprint.pformat(obj, indent=4)
    if "\n" in result:
        return f"\n{textwrap.indent(result, ' ' * 4)}"
    return result


class OutputNode:
    def __init__(self, dep: StarDep) -> None:
        self.unmet_dependencies = OrderedSet([dep])

    def is_reduction(self) -> bool:
        return False

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return ()

    def get_name(self) -> str:
        return "OUTPUT"

    __repr__ = get_name


def _prune_redundant_deps(
    node: BaseSchedulerNode,
    name_to_fused_node: dict[str, BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
) -> None:
    """
    Prunes weakdeps intended for mutation ordering
    on an upstream fused node if after fusion there is another dependency
    on the fused upstream node, making the weakdep redundant

    In essence this enforces an ordering on fusions. As fusions occur, weakdeps will
    be incrementally removed, enabling other fusions, ensuring they are fused in order.
    """
    name_to_dep_count: Counter[str] = collections.Counter()

    for dep in node.unmet_dependencies:
        if not isinstance(dep, WeakDep):
            op_name = name_to_buf[dep.name].defining_op_name()
            name_to_dep_count[name_to_fused_node[op_name].get_name()] += 1

    def should_prune(dep: Dep) -> bool:
        if isinstance(dep, WeakDep):
            op_name = name_to_buf[dep.name].defining_op_name()
            is_redundant = name_to_dep_count[
                name_to_fused_node[op_name].get_name()
            ] > 0 and node.scheduler.fusable_weak_dep(
                dep, name_to_fused_node[op_name], node
            )
            # These can occur becaus
```



## High-Level Overview


This Python file contains 19 class(es) and 322 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MixOrderReduction`, `SchedulerBuffer`, `SchedulerDonatedBuffer`, `BaseSchedulerNode`, `WhyNoFuse`, `OutputNode`, `ExternKernelSchedulerNode`, `NopKernelSchedulerNode`, `SchedulerNode`, `FusedSchedulerNode`, `FusedMixOrderReductions`, `ForeachKernelSchedulerNode`, `GroupedSchedulerNode`, `NodeUser`, `Scheduler`, `DedupList`, `BaseScheduling`

**Functions defined**: `is_split_reduction`, `get_numel_rnumel`, `has_mix_reduction_orders`, `_is_full_access`, `get_common_read`, `has_common_read`, `can_fuse`, `are_mix_order_reductions`, `is_contiguous_load`, `defining_op_name`, `__hash__`, `debug_str`, `get_name`, `allocate`, `can_free`, `set_users`, `get_aliases`, `get_mutations`, `get_device`, `__init__`

**Key imports**: annotations, collections, contextlib, dataclasses, functools, inspect, itertools, logging, math, operator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `collections`
- `contextlib`
- `dataclasses`
- `functools`
- `inspect`
- `itertools`
- `logging`
- `math`
- `operator`
- `os`
- `pprint`
- `textwrap`
- `traceback`
- `typing`
- `typing_extensions`: ParamSpec
- `torch.utils._ordered_set`: OrderedSet
- `.ir`: ComputedBuffer
- `collections.abc`: Callable, Iterator, Sequence
- `types`: ModuleType
- `sympy`
- `torch`
- `torch._inductor.async_compile  `
- `torch.utils._pytree as pytree`
- `torch._dynamo.utils`: counters, dynamo_timed
- `torch._inductor.codecache`: LambdaFuture, PyCodeCache
- `torch._inductor.ir`: TritonTemplateCallerBase
- `torch._inductor.metrics`: get_metric_table, is_metric_table_enabled


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

- **File Documentation**: `scheduler.py_docs.md`
- **Keyword Index**: `scheduler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
