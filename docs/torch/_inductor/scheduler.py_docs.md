# Documentation: scheduler.py

## File Metadata
- **Path**: `torch/_inductor/scheduler.py`
- **Size**: 248172 bytes
- **Lines**: 6364
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
            # These can occur because fused nodes always gather deps from their snodes
            # If B has a weakdep on A
            # B gets fused with C, then any time BC is fused, the weakdep will reappear
            is_self_dep = name_to_fused_node[op_name] == node
            return is_redundant or is_self_dep
        else:
            return False

    deps_to_prune = OrderedSet(
        dep for dep in node.unmet_dependencies if should_prune(dep)
    )

    if deps_to_prune:
        node.unmet_dependencies = node.unmet_dependencies - deps_to_prune
        node.set_read_writes(node.read_writes.remove_reads(deps_to_prune))


class ExternKernelSchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: Scheduler, node: ir.Operation) -> None:
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())

    def debug_str_extra(self) -> str:
        return f"{self.get_name()}.node.kernel = {getattr(self.node, 'python_kernel_name', None)}"

    def is_extern(self) -> bool:
        return True

    def has_side_effects(self) -> bool:
        assert self.node is not None
        return hasattr(self.node, "has_side_effects") and self.node.has_side_effects()


class NopKernelSchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: Scheduler, node: ir.Operation) -> None:
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())


class SchedulerNode(BaseSchedulerNode):
    """
    A SchedulerNode is a node for scheduling that encapsulates either
    a ComputedBuffer or a TemplateBuffer.
    """

    _sizes: tuple[Sequence[sympy.Expr], ...]
    _body: LoopBody

    def __init__(
        self,
        scheduler: Scheduler,
        node: Union[ir.ComputedBuffer, ir.TemplateBuffer],
    ) -> None:
        super().__init__(scheduler)
        self._init_from_node(node)
        self._compute_attrs()

    def _compute_attrs(
        self,
        extra_indexing_constraints: Optional[tuple[dict[Any, Any], list[Any]]] = None,
        recompute_sizes_body_func: Optional[Callable[_P, _T]] = None,
    ) -> None:
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer))
        self._sizes, body = self.node.simplify_and_reorder(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )
        self._body = body  # type: ignore[assignment]

        device = self.node.get_device_or_error()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        # Don't normalize since normalization will merge loops which
        # makes it hard to decide new loop orders.
        should_normalize = not config.loop_ordering_after_fusion or not is_gpu(
            device.type
        )

        if isinstance(self.node, ir.TemplateBuffer):
            self.set_read_writes(
                self.node.extract_read_writes(normalize=should_normalize)
            )
        else:
            self.set_read_writes(
                dependencies.extract_read_writes(
                    self._body, *self._sizes, normalize=should_normalize
                )
            )

    def recompute_size_and_body(
        self,
        extra_indexing_constraints: Optional[tuple[dict[Any, Any], list[Any]]] = None,
        recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._compute_attrs(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )

    def refresh_dependencies(
        self, normalize: bool, need_clear_tiling_cache: bool
    ) -> None:
        # Fake dependencies are added manually. They can not be analyzed from
        # extract_read_writes. Find them out and apply manually.
        fake_deps: OrderedSet[Dep] = OrderedSet(
            dep for dep in self.read_writes.reads if isinstance(dep, (WeakDep, StarDep))
        )

        # don't normalize since the loop order may need to be further changed
        # later
        self.set_read_writes(
            dependencies.extract_read_writes(
                self._body, *self._sizes, normalize=normalize
            )
            .with_read(fake_deps)
            .rename(self.mutation_renames)
        )

        self.pointwise_read_writes.clear_cache(self)

        if need_clear_tiling_cache:
            from .codegen.simd import SIMDScheduling

            # TODO(shunting) if this cause compilation time increase when
            # enabling LOAF by default, try just clearing the specific cache
            # entry by using a customized cache implementation rather than
            # lru_cache.
            SIMDScheduling.candidate_tilings.cache_clear()

    def apply_new_loop_order(self, new_order: Sequence[int]) -> None:
        self._body = self._body.reorder_iter_loops(
            new_order,
        )
        self._sizes = self._body.sizes

        self.refresh_dependencies(normalize=False, need_clear_tiling_cache=True)

    def swap_pw_red_dimension(self) -> None:
        num_rdims = self._body.get_original_num_rdims()
        num_pwdims = len(self._body.iter_vars) - num_rdims
        pwdims = tuple(range(num_pwdims))
        rdims = tuple(range(num_pwdims, num_pwdims + num_rdims))

        self.apply_new_loop_order(rdims + pwdims)
        assert len(self.group[1]) == 2
        self.group = self.group[0], (self.group[1][1], self.group[1][0])

    def extract_pw_from_reduction(self) -> BaseSchedulerNode:
        self._body = self._body.extract_pw_from_reduction()
        return self

    def cancel_reduction_split(self) -> None:
        if not MixOrderReduction.is_split_reduction(self):
            return
        assert isinstance(self.node, ir.ComputedBuffer)
        with self.node.with_original_inner_fn():
            self._compute_attrs()

    def expand_dimension_for_pointwise_node(
        self, dimension: int, new_range: int
    ) -> None:
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer))

        self._body = self._body.expand_dimension_for_pointwise_node(
            dimension, new_range
        )
        self._sizes = self._body.sizes

        device = self.node.get_device_or_error()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        # Need normalize the prefix name to facilitate finding common dependencies
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=True)

    def merge_loops(self) -> None:
        self._body = self._body.merge_loops()
        self._sizes = self._body.sizes

        # merge_loops is called after loop reordering.
        # We still need retain fake dependencies since codegen the
        # estimated amount of memory access rely on them.
        #
        # Merge loops does not affect the tiling decision. So we
        # don't need clear the tiling cache.
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=False)

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> bool:
        new_order = None
        self_sizes = self._sizes[0]
        if len(self_sizes) == self_dep.num_vars == other_dep.num_vars:
            new_order = self_dep.decide_loop_order_to_match(other_dep)

        if new_order:
            # pyrefly: ignore [bad-assignment]
            metrics.num_loop_reordering += 1
            loop_ordering_log.debug(
                "Reorder loops for %s with order %s", self.get_name(), new_order
            )
            self.apply_new_loop_order(new_order)
            return True
        else:
            loop_ordering_log.debug(
                "Don't reordering %s because we can not decide the suitable loop order",
                self.get_name(),
            )
            return False

    def debug_str_extra(self) -> str:
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        for dep in self.read_writes.reads_and_writes():
            if not isinstance(dep, WeakDep):
                buf_name = dep.name
                buf = V.graph.get_buffer(buf_name)
                if not isinstance(buf, ir.TorchBindObject):
                    lines.append(f"{buf_name}_layout = {pformat(buf.layout)}")
        if isinstance(self._body, LoopBody):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))

        assert self.node is not None
        lines.extend(self._debug_str_for_device())

        return "\n".join(lines)

    def get_ranges(self) -> Sequence[Sequence[sympy.Expr]]:
        return self._sizes

    def is_reduction(self) -> bool:
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)), (
            f"{type(self.node)=}"
        )

        # self._body containing partial accumulate means the reduction is
        # converted to a pointwise node.  Need this extra check since
        # we change self._body but didn't change self.node (IRNode)
        # when converting a reduction to a pointwise
        return bool(self.node.get_reduction_type()) and (
            self._body is None or not self._body.has_partial_accumulate
        )

    def is_native_matmul(self) -> bool:
        assert isinstance(self.node, ir.ComputedBuffer), f"{type(self.node)=}"
        return self.node.get_reduction_type() == "dot"

    def is_split_scan(self) -> bool:
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)), (
            f"{type(self.node)=}"
        )
        return isinstance(self.node, ir.ComputedBuffer) and isinstance(
            self.node.data, ir.SplitScan
        )

    def is_template(self) -> bool:
        return isinstance(self.node, ir.TemplateBuffer)

    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        return self.node if isinstance(self.node, ir.TemplateBuffer) else None

    def run(self, *index_vars: Sequence[sympy.Expr]) -> None:
        self.decide_inplace_update()
        self.mark_run()
        self.codegen(index_vars)

    def ranges_from_index_vars(
        self, index_vars: Sequence[Sequence[sympy.Expr]]
    ) -> dict[sympy.Expr, sympy.Expr]:
        sizes = self._sizes
        assert sum(map(len, sizes)) == sum(map(len, index_vars))
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars: Sequence[Sequence[sympy.Expr]]) -> None:
        """
        Generate code for this node using the provided index variables.

        This method sets up the appropriate context for code generation, including
        simplifying indexing expressions based on the variable ranges, and then
        calls the node's body function with the index variables.

        Args:
            index_vars: A sequence of sequences of sympy expressions representing
                        the index variables for each dimension of the computation.
        """
        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with (
                V.set_ops_handler(SimplifyIndexing(V.get_ops_handler(), var_ranges)),
                V.kernel.set_current_node(self),
            ):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_or_reduction_read_writes(
        self, pointwise: bool = True
    ) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in either the pointwise or the reduction axes.
        """
        keep_sizes, ignore_sizes = self._sizes if pointwise else reversed(self._sizes)
        return dependencies.extract_read_writes(
            self._body, keep_sizes, hidden_args=[[sympy.S.Zero] * len(ignore_sizes)]
        )

    @cache_on_self
    def pointwise_read_writes(self) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in the non-reduction axes.
        """
        return self.pointwise_or_reduction_read_writes(pointwise=True)

    @cache_on_self
    def reduction_read_writes(self) -> dependencies.ReadWrites:
        """
        Get the memory dependencies in the reduction axes.
        """
        return self.pointwise_or_reduction_read_writes(pointwise=False)

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        if self.is_template():
            return False
        if any(out.get_aliases() for out in self.get_outputs()):
            return False
        if len(self.read_writes.writes) == 1 and isinstance(
            read_dep, dependencies.MemoryDep
        ):
            write_dep = next(iter(self.read_writes.writes))
            assert isinstance(write_dep, dependencies.MemoryDep), f"{type(write_dep)=}"
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    @cache_on_self
    def _get_atomic_add_buffers(self) -> OrderedSet[str]:
        buffers_store_as_atomic_add: OrderedSet[str] = OrderedSet()
        if isinstance(self._body, LoopBody):
            for node in self._body.get_nodes():
                if (
                    node.op == "call_method"
                    and node.target == "store"
                    and (
                        ("mode" in node.kwargs and node.kwargs["mode"] == "atomic_add")
                        or (len(node.args) == 5 and node.args[4] == "atomic_add")
                    )
                ):
                    buffers_store_as_atomic_add.add(
                        node.kwargs["name"]
                        if "name" in node.kwargs
                        else (node.args[1] if len(node.args) >= 2 else "")
                    )
        return buffers_store_as_atomic_add

    @cache_on_self
    def has_side_effects(self) -> bool:
        # self._body is None sometimes that's why this check was added
        if self._body is not None and self._body.has_op("device_assert_async"):
            return True
        return super().has_side_effects()


def refresh_group_node_dependencies(
    group_snode: Union[FusedSchedulerNode, GroupedSchedulerNode],
) -> None:
    snodes = group_snode.snodes
    group_snode.set_read_writes(
        dependencies.ReadWrites.merge_list([x.read_writes for x in snodes])
    )

    group_snode.unmet_dependencies = (
        OrderedSet(
            dep
            for dep in OrderedSet.union(*[x.unmet_dependencies for x in snodes])
            if dep.name not in group_snode.get_buffer_names()
        )
        - group_snode.read_writes.writes
    )


def init_group_node(
    group_snode: Union[FusedSchedulerNode, GroupedSchedulerNode],
    scheduler: Scheduler,
    snodes: list[BaseSchedulerNode],
) -> None:
    assert isinstance(group_snode, (FusedSchedulerNode, GroupedSchedulerNode))
    group_snode.snodes = snodes
    group_snode.scheduler = scheduler
    group_snode.node = None
    group_snode.ancestors = OrderedSet.union(
        *[x.ancestors for x in snodes if x.ancestors is not None]
    )

    refresh_group_node_dependencies(group_snode)

    group_snode.min_order = min(x.min_order for x in group_snode.snodes)
    group_snode.max_order = max(x.max_order for x in group_snode.snodes)
    group_snode.outputs_by_name = {
        buf.get_name(): buf for buf in group_snode.get_outputs()
    }


class FusedSchedulerNode(BaseSchedulerNode):
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be fused together. The way it does this is by maintaining
    its unmet dependencies as the union of its constituent nodes.
    """

    snodes: list[BaseSchedulerNode]

    @classmethod
    def fuse(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> FusedSchedulerNode:
        assert node1.scheduler is node2.scheduler
        assert isinstance(node1, (SchedulerNode, FusedSchedulerNode))
        if node1.is_template() and isinstance(node2, ExternKernelSchedulerNode):
            # Fuse multi outputs template and its outputs
            #   * Node1 has memorydep of MultiOutput in reads
            #   * Node2 has StarDep of MultiOutput in writes
            # Rewrite the Node2' StarDep to MemoryDep, because calculate score_fusion_memory
            # of the template node and its epilogue requires the same type of dependencies
            assert isinstance(node2.node, MultiOutput)
            assert len(node2.read_writes.writes) == 1
            assert isinstance(next(iter(node2.read_writes.writes)), StarDep)
            name = next(iter(node2.read_writes.writes)).name
            template_nodes = [node for node in node1.get_nodes() if node.is_template()]
            assert len(template_nodes) == 1
            template_node = template_nodes[0]
            assert len(template_node.read_writes.writes) == 1
            write = next(iter(template_node.read_writes.writes))
            assert isinstance(write, MemoryDep)
            node2.read_writes.writes = OrderedSet(
                [
                    MemoryDep(
                        name, write.index, write.var_names, write.size, write.mode
                    ),
                ]
            )
        else:
            assert isinstance(node2, (SchedulerNode, FusedSchedulerNode))
        nodes = list(itertools.chain(node1.get_nodes(), node2.get_nodes()))
        return cls(node1.scheduler, nodes)

    def extract_pw_from_reduction(self) -> BaseSchedulerNode:
        for subnode in self.snodes:
            assert isinstance(subnode, SchedulerNode)
            assert subnode.is_reduction()
            subnode.extract_pw_from_reduction()
        return self

    def swap_pw_red_dimension(self) -> None:
        for subnode in self.snodes:
            assert isinstance(subnode, SchedulerNode)
            subnode.swap_pw_red_dimension()

    @cache_on_self
    def estimate_flops(self) -> int | None:
        # don't increment counters in fused methods so we don't double count
        fps = list(
            filter(
                None,
                (
                    node.estimate_flops()
                    for node in self.get_nodes()
                    if node.is_template() or node.is_extern()
                ),
            )
        )
        if len(fps) == 0:
            return None
        ret = sum(fps)
        return ret

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> bool:
        """
        Return true if a loop reordering is performed.
        """
        if self.is_template():
            # We can not really reorder loops for a triton template
            return False
        self_sizes = None
        for snode in self.snodes:
            assert isinstance(snode, SchedulerNode)
            if self_sizes is not None and tuple(self_sizes) != tuple(snode._sizes[0]):
                loop_ordering_log.debug(
                    "Can not reorder fused node due to different sizes"
                )
                return False
            self_sizes = snode._sizes[0]
        new_order = None

        assert self_sizes is not None
        if len(self_sizes) == self_dep.num_vars == other_dep.num_vars:
            new_order = self_dep.decide_loop_order_to_match(other_dep)

        if not new_order:
            loop_ordering_log.debug(
                "Dont reordering fused node %s because we can not decide the suitable loop order",
                self.get_name(),
            )
            return False
        # pyrefly: ignore [bad-assignment]
        metrics.num_loop_reordering += 1
        loop_ordering_log.debug(
            "Reorder loops for fused node %s with order %s", self.get_name(), new_order
        )
        for snode in self.snodes:
            assert isinstance(snode, SchedulerNode)
            snode.apply_new_loop_order(new_order)

        refresh_group_node_dependencies(self)
        return True

    def __init__(self, scheduler: Scheduler, snodes: list[BaseSchedulerNode]) -> None:
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)
        self.users: list[NodeUser] = []
        self.group = max(snodes, key=lambda x: int(x.is_reduction())).group

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(*[x.get_buffer_names() for x in self.snodes])

    def get_outputs(self) -> list[SchedulerBuffer]:
        result: list[SchedulerBuffer] = []
        for node in self.snodes:
            result.extend(node.get_outputs())
        return result

    def debug_str_extra(self) -> str:
        lines = [
            f"{self.get_name()}.snodes[{i}] =\n{node.debug_str()}"
            for i, node in enumerate(self.snodes)
        ]
        node = self.snodes[0].node
        if node is not None:
            lines.extend(self._debug_str_for_device())

        return textwrap.indent("\n".join(lines).rstrip(), "    ")

    def debug_str_short(self) -> str:
        snodes_str = [node.debug_str_short() for node in self.snodes]
        return f"{self}, snodes: {snodes_str}"

    def set_last_usage(
        self, future_used_buffers: OrderedSet[str], mutation_real_name: dict[str, str]
    ) -> None:
        # Set self.last_usage using the global information
        # This will be used for inter-kernel optimisations
        super().set_last_usage(future_used_buffers, mutation_real_name)
        # Set self.last_usage on the snodes
        # This will be used for optimisations within the kernel
        future_used_buffers: OrderedSet[str] = OrderedSet()
        for node in reversed(self.snodes):
            node.set_last_usage(future_used_buffers, mutation_real_name)
            future_used_buffers.update(node.last_usage)

    @cache_on_self
    def used_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(*[x.used_buffer_names() for x in self.snodes])

    @cache_on_self
    def used_or_aliased_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(
            *[x.used_or_aliased_buffer_names() for x in self.snodes]
        )

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        return self.snodes

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nodes={self.get_name()})"

    @cache_on_self
    def is_reduction(self) -> bool:
        return any(x.is_reduction() for x in self.snodes)

    @cache_on_self
    def is_native_matmul(self) -> bool:
        return any(x.is_native_matmul() for x in self.snodes)

    @cache_on_self
    def is_split_scan(self) -> bool:
        return any(x.is_split_scan() for x in self.snodes)

    @cache_on_self
    def is_template(self) -> bool:
        return any(x.is_template() for x in self.snodes)

    @cache_on_self
    def get_template_node(self) -> Optional[ir.TemplateBuffer]:
        for node in self.snodes:
            if node.is_template():
                return node.get_template_node()
        return None

    def get_device(self) -> torch.device:
        return self.group[0]

    @cache_on_self
    def has_aliasing_or_mutation(self) -> bool:
        return any(x.has_aliasing_or_mutation() for x in self.snodes)

    # None of these need to be implemented, as a FusedSchedulerNode is just an
    # abstraction for scheduling purposes
    def update_mutated_names(self, renames: dict[str, str]) -> None:
        raise NotImplementedError

    def add_fake_dep(self, name: Dep) -> None:
        raise NotImplementedError

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        raise NotImplementedError

    def debug_str(self) -> str:
        """Longer form printout for trace logs"""
        name = self.get_name()
        node_typestr = ",".join(type(n).__name__ for n in self.snodes)
        buf = IndentedBuffer()
        buf.splice(
            f"""\
{name}: {type(self).__name__}({node_typestr})
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

    @cache_on_self
    def has_side_effects(self) -> bool:
        if self.snodes is not None:
            return any(node.has_side_effects() for node in self.snodes)
        return super().has_side_effects()


class FusedMixOrderReductions(FusedSchedulerNode):
    def __init__(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> None:
        self.node1 = node1
        self.node2 = node2
        super().__init__(
            node1.scheduler, list(node1.get_nodes()) + list(node2.get_nodes())
        )


class ForeachKernelSchedulerNode(FusedSchedulerNode):
    """
    This is a schedular node that consists of a set of scheduler nodes that
    has no data dependencies among them and can be executed in parallel.
    """

    def get_consumer_subnode_for(
        self, producer: BaseSchedulerNode
    ) -> Optional[BaseSchedulerNode]:
        for buf in producer.get_outputs():
            if buf.get_name() in self.read_to_node:
                return self.read_to_node[buf.get_name()]

        return None

    def get_producer_subnode_for(
        self, consumer: BaseSchedulerNode
    ) -> Optional[BaseSchedulerNode]:
        producers = OrderedSet[BaseSchedulerNode]()
        for rd in consumer.read_writes.reads:
            if rd.name not in self.scheduler.name_to_buf:
                continue

            node_name = self.scheduler.name_to_buf[rd.name].defining_op_name()
            if node_name in self.name_to_node:
                producers.add(self.name_to_node[node_name])

        # Don't permit fusion if there are multiple subnodes
        # that this consumer reads from
        if len(producers) == 1:
            return next(iter(producers))
        else:
            return None

    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        why = WhyNoFuse(producer, consumer)
        if producer.is_foreach() and consumer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            foreach_match = len(producer.snodes) == len(consumer.snodes)
            if not foreach_match:
                why("foreach do not have same length")
            return foreach_match and all(
                producer.scheduler.can_fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            )
        elif consumer.is_foreach():
            if producer.is_reduction():
                why(
                    "candidate producer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            if consumer_subnode is not None:
                return consumer.scheduler.can_fuse(producer, consumer_subnode)

            why("candidate producer is not dep of any foreach consumer")
            return False

        elif producer.is_foreach():
            if consumer.is_reduction():
                why(
                    "candidate consumer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            producer_subnode = producer.get_producer_subnode_for(consumer)
            if producer_subnode is not None:
                return producer.scheduler.can_fuse(producer_subnode, consumer)

            why("candidate consumer has no dep in any foreach producer")
            return False

        raise AssertionError(
            "At least one node passed to ForeachKernelSchedulerNode.can_fuse should be a foreach node"
        )

    @classmethod
    def fuse(
        cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode
    ) -> ForeachKernelSchedulerNode:
        assert producer.is_foreach() or consumer.is_foreach()
        if producer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            use_custom_partition_algo = producer.use_custom_partition_algo
            enable_autotune = producer.enable_autotune
        else:
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            use_custom_partition_algo = consumer.use_custom_partition_algo
            enable_autotune = consumer.enable_autotune
        prev_node_1 = None
        prev_node_2 = None
        fused_nodes: list[BaseSchedulerNode]
        if producer.is_foreach() and consumer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            fused_nodes = [
                FusedSchedulerNode.fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            ]
        elif producer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            producer_subnode = producer.get_producer_subnode_for(consumer)
            fused_nodes = []
            prev_node_1 = producer
            prev_node_2 = None
            for node in producer.snodes:
                if node is producer_subnode:
                    new_node = FusedSchedulerNode.fuse(node, consumer)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)

        elif consumer.is_foreach():
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            fused_nodes = []
            prev_node_1 = consumer
            prev_node_2 = None

            for node in consumer.snodes:
                if node is consumer_subnode:
                    new_node = FusedSchedulerNode.fuse(producer, node)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)
        else:
            raise AssertionError(
                "At least one node passed to ForeachKernelSchedulerNode.fuse should be a foreach node"
            )

        return cls(
            producer.scheduler,
            fused_nodes,
            use_custom_partition_algo=use_custom_partition_algo,
            prev_node_1=prev_node_1,
            prev_node_2=prev_node_2,
            enable_autotune=enable_autotune,
        )

    def __init__(
        self,
        scheduler: Scheduler,
        snodes: list[BaseSchedulerNode],
        use_custom_partition_algo: bool,
        prev_node_1: Optional[BaseSchedulerNode] = None,
        prev_node_2: Optional[BaseSchedulerNode] = None,
        enable_autotune: bool = False,
    ) -> None:
        self.read_to_node = {}
        self.name_to_node = {}

        if prev_node_1 is None or prev_node_2 is None:
            super().__init__(scheduler, snodes)

            for node in snodes:
                for read in node.read_writes.reads:
                    self.read_to_node[read.name] = node

                for name in node.get_operation_names():
                    self.name_to_node[name] = node
        else:
            self.scheduler = scheduler
            self.snodes = snodes
            self.node = None
            self.users: list[NodeUser] = []

            self.set_read_writes(
                dependencies.ReadWrites.merge_list(
                    [prev_node_1.read_writes, prev_node_2.read_writes]
                )
            )

            self.unmet_dependencies = (
                OrderedSet(
                    dep
                    for dep in OrderedSet.union(
                        prev_node_1.unmet_dependencies, prev_node_2.unmet_dependencies
                    )
                    if dep.name not in self.get_buffer_names()
                )
                - self.read_writes.writes
            )

            self.min_order = min([prev_node_1.min_order, prev_node_2.min_order])
            self.max_order = max([prev_node_1.max_order, prev_node_2.max_order])

            if prev_node_1.is_foreach():
                assert isinstance(prev_node_1, ForeachKernelSchedulerNode)
                foreach_node, other_node = prev_node_1, prev_node_2
            else:
                assert isinstance(prev_node_2, ForeachKernelSchedulerNode)
                foreach_node, other_node = prev_node_2, prev_node_1

            self.ancestors = foreach_node.ancestors
            self.ancestors.update(other_node.ancestors)

            self.name_to_node = foreach_node.name_to_node
            for name in other_node.get_operation_names():
                self.name_to_node[name] = other_node

            self.outputs_by_name: dict[str, SchedulerBuffer] = {
                k: v for snode in self.snodes for k, v in snode.outputs_by_name.items()
            }

        self.use_custom_partition_algo = use_custom_partition_algo
        device = snodes[0].get_device()
        assert device
        self.group = (device, ((sympy.Expr("combo_kernel"),),))
        self.origins = OrderedSet[torch.fx.Node]()
        self.enable_autotune = enable_autotune

    @classmethod
    def combinable_nodes(
        cls, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
        extern = [x for x in nodes if isinstance(x, ExternKernelSchedulerNode)]
        if extern:
            log.debug(
                "ComboKernels: %d external nodes are filtered %s",
                len(extern),
                [node.node.get_origins() for node in extern if node.node is not None],
            )
        filtered_nodes = [
            x
            for x in nodes
            if not isinstance(x, (NopKernelSchedulerNode, ExternKernelSchedulerNode))
        ]
        foreach_nodes = [
            x for x in filtered_nodes if isinstance(x, ForeachKernelSchedulerNode)
        ]
        if foreach_nodes:
            log.debug("ComboKernels: %d foreach nodes are filtered", len(foreach_nodes))
        filtered_nodes = [
            x for x in filtered_nodes if not isinstance(x, ForeachKernelSchedulerNode)
        ]
        template_nodes = [x for x in filtered_nodes if x.is_template()]
        if template_nodes:
            log.debug(
                "ComboKernels: %d template nodes are filtered: %s",
                len(template_nodes),
                template_nodes,
            )
        filtered_nodes = [x for x in filtered_nodes if x not in template_nodes]
        return filtered_nodes

    @staticmethod
    def _default_group_nodes_for_combo_kernels(
        scheduler: Scheduler,
    ) -> list[list[BaseSchedulerNode]]:
        """
        Returns a list of lists of nodes that are to be grouped together.
        """
        sorted_nodes = scheduler._topological_sort_nodes()
        grouped_nodes = []
        max_num_nodes = 8
        for nodes in sorted_nodes:
            grouped_nodes.extend(
                [
                    nodes[i : i + max_num_nodes]
                    for i in range(0, len(nodes), max_num_nodes)
                ]
            )

        return grouped_nodes

    group_algorithm_for_combo_kernels: Callable[
        [Scheduler], list[list[BaseSchedulerNode]]
    ] = _default_group_nodes_for_combo_kernels

    @staticmethod
    def set_group_algorithm_for_combo_kernels(
        custom_group_algorithm: Callable[[Scheduler], list[list[BaseSchedulerNode]]],
    ) -> None:
        ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels = (
            custom_group_algorithm
        )

    @staticmethod
    def group_nodes_for_combo_kernels(
        scheduler: Scheduler,
    ) -> list[list[BaseSchedulerNode]]:
        return ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels(scheduler)

    def mark_run(self) -> None:
        raise NotImplementedError

    def codegen(self) -> None:
        raise NotImplementedError

    def is_foreach(self) -> bool:
        return True

    def get_subkernel_nodes(self) -> list[BaseSchedulerNode]:
        """Returns a list of nodes which comprise the combo kernel.
        These nodes may be vertically fused."""
        return list(self.snodes)

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        """Returns all nodes contained in this kernel, unpacking fused nodes
        into their constituent scheduler nodes."""
        return list(itertools.chain.from_iterable(x.get_nodes() for x in self.snodes))

    def get_first_name(self) -> str:
        return self.snodes[0].get_first_name()

    def prune_redundant_deps(
        self, name_to_fused_node: dict[str, BaseSchedulerNode]
    ) -> None:
        _prune_redundant_deps(self, name_to_fused_node, self.scheduler.name_to_buf)

        for node in self.snodes:
            node.prune_redundant_deps(name_to_fused_node)


class GroupedSchedulerNode(BaseSchedulerNode):
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be *grouped* together (it does not allow another node to be scheduled
    in between its constituent nodes, nor does it allow another node to fuse into any of its constituent nodes).
    The way it does this is by maintaining its unmet dependencies as the union of its constituent nodes.
    Fusion will still happen among the nodes within each GroupedSchedulerNode.
    At codegen time, this scheduler node will be unpacked and codegen is called on each constituent node.
    """

    snodes: list[BaseSchedulerNode]

    @classmethod
    def create(cls, snodes: list[BaseSchedulerNode]) -> GroupedSchedulerNode:
        scheduler = snodes[0].scheduler
        assert all(node.scheduler is scheduler for node in snodes)
        grouped_snode = cls(scheduler, snodes)
        for snode in snodes:
            scheduler.name_to_fused_node[snode.get_name()] = grouped_snode
        scheduler.name_to_fused_node[grouped_snode.get_name()] = grouped_snode
        return grouped_snode

    def __init__(
        self,
        scheduler: Scheduler,
        snodes: list[BaseSchedulerNode],
        temp_grouping: bool = False,
    ) -> None:
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)
        # This flag is introduced for "temporary" grouping during some passes,
        # Where nodes are grouped and moved together.
        # After the pass those nodes are flattened.
        # Reusing calculation of grouped unmed_dependencies etc.
        # No fusion logic in this case.
        self.temp_grouping = temp_grouping

    def unpack(self) -> list[BaseSchedulerNode]:
        """
        Do fusion among nodes within this GroupedSchedulerNode,
        and then unpack this GroupedSchedulerNode into regular nodes.
        """
        if self.temp_grouping:
            return self.snodes

        for snode in self.snodes:
            self.scheduler.name_to_fused_node[snode.get_name()] = snode
        del self.scheduler.name_to_fused_node[self.get_name()]
        return self.scheduler.fuse_nodes(self.snodes)

    def add_fake_dep(self, fake_dep: Dep) -> None:
        self.set_read_writes(self.read_writes.with_read(fake_dep))
        self.unmet_dependencies.add(fake_dep)

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(*[x.get_buffer_names() for x in self.snodes])

    def get_outputs(self) -> list[SchedulerBuffer]:
        result: list[SchedulerBuffer] = []
        for node in self.snodes:
            result.extend(node.get_outputs())
        return result

    @cache_on_self
    def estimate_flops(self) -> int | None:
        # don't increment counters in fused methods so we don't double count
        fps = list(
            filter(
                None,
                (
                    node.estimate_flops()
                    for node in self.get_nodes()
                    if node.is_template() or node.is_extern()
                ),
            )
        )
        if len(fps) == 0:
            return None
        ret = sum(fps)
        return ret

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        return self.snodes

    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        # GroupedSchedulerNode cannot be fused with another node
        return False


def pick_loop_order(
    stride_lengths: list[list[int]],
    sizes: Sequence[sympy.Expr],
    priority_idx: Sequence[int] = (),
) -> list[int]:
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

    @functools.cmp_to_key
    def index_cmp(a: int, b: int) -> int:
        if sizes[a] == 1 or sizes[b] == 1:
            # 1-sizes don't matter, just move them to the end
            return cmp(sizes[a] == 1, sizes[b] == 1)

        # Take abs, otherwise flipped dimensions are treated as smaller
        # strides than contiguous dims
        stride_len_a = [abs(sl[a]) for sl in stride_lengths]
        stride_len_b = [abs(sl[b]) for sl in stride_lengths]

        # equivalent to
        # np.logical_or(stride_lengths[:, b] == 0, stride_lengths[:, a] < stride_lengths[:, b]).all()
        a_first = sum(
            sl_b == 0 or sl_a < sl_b for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        b_first = sum(
            sl_a == 0 or sl_b < sl_a for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        if a_first > b_first:
            return -1
        if b_first > a_first:
            return 1

        # otherwise contiguous
        return cmp(b, a)

    order = list(reversed(range(len(stride_lengths[0]))))
    if len(priority_idx) > 0:
        # if we have priority node, only use that node's order
        stride_lengths = [stride_lengths[pi] for pi in priority_idx]
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order


def _replace_operation_buffer(
    orig_node: ir.MultiTemplateBuffer, new_node: ir.OperationBuffer
) -> None:
    replaced_buf_name = new_node.get_name()
    orig_buf_name = orig_node.get_name()
    assert isinstance(orig_buf_name, str) and isinstance(replaced_buf_name, str)

    replaced_op_name = new_node.get_operation_name()
    orig_op_name = orig_node.get_operation_name()
    assert isinstance(orig_op_name, str) and isinstance(replaced_op_name, str)

    del V.graph.name_to_buffer[replaced_buf_name]
    new_node.name = orig_buf_name

    del V.graph.name_to_op[replaced_op_name]
    new_node.operation_name = orig_op_name

    orig = V.graph.buffers.index(orig_node)
    V.graph.buffers.remove(new_node)
    V.graph.buffers[orig] = new_node
    V.graph.name_to_buffer[orig_buf_name] = new_node

    orig = V.graph.operations.index(orig_node)
    V.graph.operations.remove(new_node)
    V.graph.operations[orig] = new_node
    V.graph.name_to_op[orig_op_name] = new_node


@dataclasses.dataclass
class NodeUser:
    node: Union[BaseSchedulerNode, OutputNode]
    can_inplace: bool = False

    # A weak user must be scheduled after a given node, but doesn't actually
    # use the result
    is_weak: bool = False

    def __hash__(self) -> int:
        return hash((self.node.get_name(), self.can_inplace, self.is_weak))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, NodeUser)
            and self.get_name() == other.get_name()
            and self.can_inplace == other.can_inplace
            and self.is_weak == other.is_weak
        )

    def get_name(self) -> str:
        return self.node.get_name()

    def merge(self, other: NodeUser) -> NodeUser:
        assert self.node is other.node
        return NodeUser(
            self.node,
            self.can_inplace and other.can_inplace,
            self.is_weak and other.is_weak,
        )


_post_grad_graph_counter = itertools.count()


def used_non_deterministic_runtime_estimations() -> bool:
    return config.runtime_estimations_mms_benchmark


class Scheduler:
    """
    A Scheduler is a graph of BaseSchedulerNodes. It is responsible for
    optimizations such as fusion, reorder, and graph partition.
    """

    def __init__(self, nodes: list[ir.Operation]) -> None:
        with dynamo_timed("Scheduler.__init__"):
            self._init(nodes)

    def _init(self, nodes: list[ir.Operation]) -> None:
        super().__init__()
        V.graph.scheduler = self
        self.backends: dict[torch.device, BaseScheduling] = {}
        self.post_grad_graph_id = next(_post_grad_graph_counter)
        self._graph_partition_counter = itertools.count()

        self.completed_operations: OrderedSet[str] = OrderedSet()
        self.available_buffer_names = OrderedSet(
            [
                *V.graph.graph_inputs.keys(),
                *V.graph.constants.keys(),
                *V.graph.torchbind_constants.keys(),
            ]
        )
        self.nodes = [self.create_scheduler_node(n) for n in nodes]
        self.current_node: Optional[BaseSchedulerNode] = None
        self.update_zero_dim_cpu_tensor()
        # some new constants could have been created above
        self.available_buffer_names.update(V.graph.constants.keys())
        for node in self.nodes:
            node.prune_deps()

        # See [Note: Graph Partition Device Contexts]
        self.default_device_context: Optional[torch.device] = None

        self.name_to_donated_buffer: dict[str, SchedulerDonatedBuffer] = (
            self.get_donated_buffers()
        )
        self.name_to_node: dict[str, BaseSchedulerNode] = {
            n.get_name(): n for n in self.nodes
        }
        self.name_to_buf: dict[str, SchedulerBuffer] = {
            buf.get_name(): buf for node in self.nodes for buf in node.get_outputs()
        }
        self.name_to_fused_node: dict[str, BaseSchedulerNode] = self.name_to_node.copy()

        # mutation_real_name: Maps back to the original name for codegen
        # Example:
        # If you mutate buf0 inside of buf1's kernel, then:
        # mutation_real_name = {"buf0" : "buf1"}
        # all subsequent uses of buf0 become buf1's usage in dependency graph
        self.mutation_real_name: dict[str, str] = {}

        # We handle mutation by renaming modified versions of the same
        # buffer in the dependency graph to prevent cycles.
        # mutation_renames: tracks the current name for a given buffer
        #                   (changed once per mutation)
        # Example:
        # If you mutate buf0 inside of buf1's kernel, then:
        # mutation_renames = {"buf1" : "buf0"}
        # in codegen we only use buf0, never buf1
        self.mutation_renames: dict[str, str] = {}

        # Must run first to correctly set dependencies, before all other passes that rely on
        # reading from .read_writes.reads or .unmet_dependencies
        self.nodes = comms.decide_global_ordering_of_comms(
            self.nodes,
            self.name_to_buf,
            self.name_to_fused_node,
        )

        self.compute_dependencies()
        self.nodes = self.topological_sort_schedule(self.nodes)
        self.dead_node_elimination()
        self.name_to_fused_node = {n.get_name(): n for n in self.nodes}
        self.compute_ancestors()

        # pyrefly: ignore [bad-assignment]
        metrics.ir_nodes_pre_fusion += len(self.nodes)
        from torch._inductor.debug import log_ir_post_fusion, log_ir_pre_fusion

        log_ir_pre_fusion(self.nodes)
        self.num_orig_nodes = len(self.nodes)
        self.create_foreach_nodes()
        self.nodes = self.topological_sort_schedule(self.nodes)
        self.logged_slow_fusion = OrderedSet[tuple[str, str]]()
        if config._pre_fusion_custom_pass is not None:
            self.nodes = config._pre_fusion_custom_pass(self.nodes)

        if config.distributed_max_autotune_gemm:
            from . import distributed_autotune

            distributed_autotune.schedule(self)
            self.compute_ancestors()

        self.nodes = self.fuse_nodes(self.nodes)
        if config._post_fusion_custom_pass is not None:
            self.nodes = config._post_fusion_custom_pass(self.nodes)

        self.merge_loops()
        self.finalize_multi_template_buffers()
        if config.combo_kernels:
            with dynamo_timed(
                "Scheduler.create_combo_kernel_nodes",
                log_pt2_compile_event=True,
                log_waitcounter=True,
            ):
                self.create_combo_kernel_nodes(num_ck_nodes=None)

        # Peak memory pass and overlap pass must run last, otherwise
        # other reordering passes could undo their effects.
        if config.reorder_for_peak_memory:
            from .memo

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 17 class(es): MixOrderReduction, class, class, BaseSchedulerNode, WhyNoFuse, OutputNode, ExternKernelSchedulerNode, NopKernelSchedulerNode, SchedulerNode, FusedSchedulerNode, FusedMixOrderReductions, ForeachKernelSchedulerNode, GroupedSchedulerNode, class, Scheduler, DedupList, BaseScheduling

### Functions
This file defines 321 function(s): is_split_reduction, get_numel_rnumel, has_mix_reduction_orders, _is_full_access, get_common_read, has_common_read, can_fuse, are_mix_order_reductions, is_contiguous_load, defining_op_name, __hash__, debug_str, get_name, allocate, can_free, set_users, get_aliases, get_mutations, get_device, __init__, _init_from_node, __repr__, debug_str, debug_str_extra, _debug_str_for_device, debug_str_short, log_details, reorder_loops_by_dep_pair, update_mutated_names, add_fake_dep


## Key Components

The file contains 19862 words across 6364 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 248172 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
