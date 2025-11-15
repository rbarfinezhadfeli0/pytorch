# Documentation: `docs/torch/_higher_order_ops/triton_kernel_wrap.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_higher_order_ops/triton_kernel_wrap.py_docs.md`
- **Size**: 54,346 bytes (53.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_higher_order_ops/triton_kernel_wrap.py`

## File Metadata

- **Path**: `torch/_higher_order_ops/triton_kernel_wrap.py`
- **Size**: 85,685 bytes (83.68 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import logging
import operator
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional, TYPE_CHECKING, Union
from typing_extensions import Never

import sympy

import torch.fx as fx
import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import redirect_to_mode
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import guard_scalar
from torch.types import IntLikeType
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


if TYPE_CHECKING:
    from triton._C.libtriton.ir import (
        module as TritonIRModule,
        operation as TritonIROperation,
    )

    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.constant import ConstantVariable
    from torch._dynamo.variables.functions import TritonKernelVariable
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI
    from torch.fx.proxy import Proxy
    from torch.utils._triton import has_triton

    TritonMetaParamsType = dict[str, int]
    TritonGridTupleType = tuple[Union[int, sympy.Expr, SymInt], ...]
    TritonGridCallableType = Callable[[TritonMetaParamsType], tuple[int, ...]]
    TritonGridType = Union[TritonGridTupleType, TritonGridCallableType]

    if has_triton():
        from triton.runtime.autotuner import Autotuner, Config as TritonConfig
        from triton.runtime.jit import JITFunction
    else:

        class Autotuner:  # type: ignore[no-redef]
            pass

        class JITFunction:  # type: ignore[no-redef]
            pass

    TritonKernelType = Union[Autotuner, JITFunction]
    # mypy specifically complains that TritonAutotunerType is not a valid type if Autotuner is not inside of a Union.
    TritonAutotunerType = Union[Autotuner]

log = logging.getLogger("torch._dynamo")

# e.g. for a host-side Triton TMA API call ``create_2d_tma_descriptor(ptr, 50, 60, 32, 15, 4)``,
# the metadata will look like ``("experimental", ([50, 60], [32, 15], 4))``
TMAExperimentalMetadata = tuple[
    str,  # type of TMA (should be "experimental")
    tuple[
        list[IntLikeType],  # dims
        list[IntLikeType],  # block_dims
        IntLikeType,  # element_size
    ],
]

# e.g. for host-side Triton TMA API call ``TensorDescriptor.from_tensor(ptr, [32, 64])``
# the metadata will look like ``("stable", ([32, 64],))``
TMAStableMetadata = tuple[
    str,  # type of TMA ("experimental" or "stable")
    tuple[list[IntLikeType],],  # block_shape
]


def create_tma_experimental_metadata(
    dims: list[IntLikeType],
    block_dims: list[IntLikeType],
    element_size: IntLikeType,
) -> TMAExperimentalMetadata:
    return ("experimental", (dims, block_dims, element_size))


def maybe_unpack_tma_experimental_metadata(
    tma_meta: Union[TMAExperimentalMetadata, TMAStableMetadata],
) -> Optional[tuple[list[IntLikeType], list[IntLikeType], IntLikeType]]:
    if not tma_meta or len(tma_meta) != 2:
        return None
    if tma_meta[0] == "experimental":
        return tma_meta[1]  # type: ignore[return-value]
    return None


def create_tma_stable_metadata(
    block_shape: list[IntLikeType],
) -> TMAStableMetadata:
    return ("stable", (block_shape,))


def maybe_unpack_tma_stable_metadata(
    tma_meta: Union[TMAExperimentalMetadata, TMAStableMetadata],
) -> Optional[tuple[list[IntLikeType]]]:
    if not tma_meta or len(tma_meta) != 2:
        return None
    if tma_meta[0] == "stable":
        return tma_meta[1]  # type: ignore[return-value]
    return None


# TMADescriptorMetadata maps kernel parameter names to the metadata that allows
# reconstructing TMA descriptors from the underlying tensors (passed as kernel
# arguments in the fx graph, instead of the TMA descriptors).
#
# Since there are two TMA APIs (the old "experimental" API and the new "stable" API),
# each entry in the dict is a tuple that starts with a string, either "experimental"
# or "stable". The second entry in the tuple is another tuple, with data that depends
# on the API type (see TMAExperimentalMetadata and TMAStableMetadata above).
#
# These are stored as raw tuples (instead of classes) for ease of serialization.
TMADescriptorMetadata = dict[
    str,  # kernel parameter name
    Union[TMAExperimentalMetadata, TMAStableMetadata],
]


###############################################################################
# Kernel Side Table


# We cannot put Triton Kernels into the FX graph as the graph nodes
# do not support arbitrary functions.
# Use a side table.
# We use two dicts so that fetching both the kernel and id are O(1)
class KernelSideTable:
    id_to_kernel: dict[int, "TritonKernelType"] = {}
    kernel_to_id: dict["TritonKernelType", int] = {}
    constant_args: dict[int, dict[str, Any]] = {}
    lock = threading.Lock()

    # Returns index on the table
    def add_kernel(self, kernel: "TritonKernelType") -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]

            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    # Returns the triton kernel at the given index
    def get_kernel(self, idx: int) -> "TritonKernelType":
        # No need to lock here as fetching from dict is atomic
        assert idx in self.id_to_kernel
        return self.id_to_kernel[idx]

    # Not every constant arg can be added to the graph. Use this side table
    # for constant args.
    def add_constant_args(self, args: dict[str, Any]) -> int:
        with self.lock:
            idx = len(self.constant_args)
            self.constant_args[idx] = args
            return idx

    # Returns the constant args
    def get_constant_args(self, idx: int) -> dict[str, Any]:
        # No need to lock here as fetching from dict is atomic
        assert idx in self.constant_args
        return self.constant_args[idx]

    # Resets the table (only meant to be used in unit tests)
    # This is only safe assuming single threaded execution
    def reset_table(self) -> None:
        self.id_to_kernel = {}
        self.kernel_to_id = {}
        self.constant_args = {}


kernel_side_table = KernelSideTable()


###############################################################################
# Mutation Tracker


@dataclasses.dataclass(frozen=True)
class Param:
    idx: int


@dataclasses.dataclass(frozen=True)
class Intermediate:
    idx: int

    def fake(self) -> bool:
        return self.idx < 0


@dataclasses.dataclass(frozen=True)
class Op:
    name: str
    fn_call_name: Optional[str]
    args: list[Union[Param, Intermediate]]
    ret: Intermediate = dataclasses.field(repr=False)
    # used for scf.yield: see [Note: scf.yield fix-up]
    sub_idx: Optional[int] = None
    # used for tt.elementwise_inline_asm
    # `is_pure = True` assumes the asm block has no side-effects
    is_pure: bool = False

    def __post_init__(self) -> None:
        if self.name == "tt.call":
            assert self.fn_call_name is not None
        else:
            assert self.fn_call_name is None


def generate_ttir(
    kernel: "TritonKernelType",
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> tuple["TritonIRModule", list[str]]:
    """
    Uses Triton's internal code generation to create TTIR
    """
    import sympy
    import triton
    import triton.runtime.jit
    from triton.compiler.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    from torch._inductor.utils import (
        get_triton_attrs_descriptor_version,
        triton_version_uses_attrs_dict,
        TritonAttrsDescriptorVersion,
    )
    from torch.utils._triton import has_triton_tensor_descriptor_host_tma

    triton_version = get_triton_attrs_descriptor_version()

    import torch._inductor.ir
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(kernel, Autotuner):
        if len(kernel.configs) > 0:
            # If we are autotuning, then it doesn't matter which version gets
            # picked for tracing purposes, so lets pick the first one
            kwargs = {**kwargs, **kernel.configs[0].kwargs}
        kernel = kernel.fn

    assert isinstance(kernel, JITFunction)

    # pyrefly: ignore  # missing-attribute
    context = triton._C.libtriton.ir.context()
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.compiler.make_backend(target)
    options = backend.parse_options({})

    # ignore backend-specific kwargs same way as in the native Triton code
    # https://github.com/triton-lang/triton/blob/a6bb57d6285e723c58e87dd7cba263db6efff789/python/triton/runtime/jit.py#L594-L596
    # why this is important for user-defined Triton kernels on AMD: https://github.com/pytorch/pytorch/issues/140800
    for name in list(kwargs):
        if name not in kernel.arg_names and name in options.__dict__:
            kwargs.pop(name)

    if len(kwargs) != len(kernel.arg_names):
        raise ValueError(
            "Incorrect number of arguments passed to kernel: "
            f"passed {list(kwargs.keys())}, expected {kernel.arg_names}."
        )

    # Replace all SymExprs with a regular value for TTIR generation
    # Replace all FakeTensor/TensorBox with real tensors
    # These replacements are needed for triton's type, key and config functions
    ordered_args: dict[str, Any] = {}
    for name in kernel.arg_names:
        a = kwargs[name]
        if isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool, sympy.Expr)):
            ordered_args[name] = 2
        elif (
            stable_meta := maybe_unpack_tma_stable_metadata(
                # pyrefly: ignore [bad-argument-type]
                tma_descriptor_metadata.get(name, None)
            )
        ) is not None:
            from triton.tools.tensor_descriptor import TensorDescriptor

            block_shape = stable_meta[0]
            with torch._C._DisableTorchDispatch():
                # need 16-byte aligned strides
                elements_per_dim = max(1, 16 // a.dtype.itemsize)
                base_tensor = torch.empty(
                    [elements_per_dim] * len(block_shape), dtype=a.dtype
                )
            # pyrefly: ignore  # bad-argument-type
            ordered_args[name] = TensorDescriptor.from_tensor(base_tensor, block_shape)
        elif isinstance(a, (FakeTensor, torch._inductor.ir.TensorBox)):
            with torch._C._DisableTorchDispatch():
                ordered_args[name] = torch.empty(2, dtype=a.dtype)
        else:
            ordered_args[name] = a

    def is_stable_tensor_descriptor_arg(arg: Any) -> bool:
        if has_triton_tensor_descriptor_host_tma():
            from triton.tools.tensor_descriptor import TensorDescriptor

            if isinstance(arg, TensorDescriptor):
                return True
        return False

    def is_tensor_like_arg(arg: Any) -> bool:
        if isinstance(arg, Tensor) or is_stable_tensor_descriptor_arg(arg):
            return True
        return False

    # Note: one would expect that each input to the triton kernel maps to
    # one input parameter in the TTIR. This is _not_ true for TMA descriptors:
    # one TMA descriptor gets converted into:
    #   * one TMA descriptor input
    #   * N strides, for a rank-N tensor
    #   * N sizes, for a rank-N tensor
    # To account for this, we inject some fake arg names as placeholders for
    # the stride and size parameters.
    def get_tensor_names(name: str, arg: Any) -> list[str]:
        if isinstance(arg, Tensor):
            return [name]
        if is_stable_tensor_descriptor_arg(arg):
            stable_meta = maybe_unpack_tma_stable_metadata(
                tma_descriptor_metadata[name]
            )
            assert stable_meta is not None
            block_shape = stable_meta[0]
            tensor_rank = len(block_shape)
            names = [name]
            names.extend(name + f" STRIDE PLACEHOLDER {i}" for i in range(tensor_rank))
            names.extend(name + f" SIZE PLACEHOLDER {i}" for i in range(tensor_rank))
            return names
        return []

    ordered_tensor_names = list(
        itertools.chain.from_iterable(
            get_tensor_names(name, arg) for name, arg in ordered_args.items()
        )
    )

    def _get_specialization(args):  # type: ignore[no-untyped-def]
        # Support multiple triton versions.
        # This code basically copies JITFunction.run() logic to get the attrs to construct an ASTSource.
        if triton_version == TritonAttrsDescriptorVersion.V1_COMPILER:
            return kernel._get_config(*args)
        elif triton_version in {
            TritonAttrsDescriptorVersion.V2_BACKENDS,
            TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
        }:
            from triton.backends.compiler import AttrsDescriptor  # noqa: F401

            target = triton.runtime.driver.active.get_current_target()
            backend_ = triton.compiler.compiler.make_backend(target)
            # pyrefly: ignore  # missing-attribute
            return backend_.get_attrs_descriptor(args, kernel.params)
        else:
            assert (
                get_triton_attrs_descriptor_version()
                == TritonAttrsDescriptorVersion.V4_DICT
            )
            # specialize_impl switched to create_specialize_impl in https://github.com/triton-lang/triton/pull/6099
            if hasattr(triton.runtime.jit, "create_specialize_impl"):
                try:
                    # Latest versions of Triton take specialize_extra as an arg to create_specialize_impl
                    specialize_impl = triton.runtime.jit.create_specialize_impl(
                        specialize_extra=backend.get_arg_specialization
                    )
                except TypeError:  # Unknown arg `specialize_extra`
                    # Older versions of Triton take specialize_extra as an arg to specialize_impl
                    specialize_impl = functools.partial(
                        # pyrefly: ignore  # missing-argument
                        triton.runtime.jit.create_specialize_impl(),
                        specialize_extra=backend.get_arg_specialization,
                    )
            # create_specialize_impl is removed in https://github.com/triton-lang/triton/pull/7771
            # switch to native_specialize_impl instead
            elif hasattr(triton.runtime.jit, "native_specialize_impl"):
                from triton.backends import BaseBackend
                from triton.runtime.jit import native_specialize_impl

                def _native_specialize_impl(
                    arg: Any,
                    is_const: bool = False,
                    specialize_value: bool = True,
                    align: bool = True,
                ) -> Callable:
                    return native_specialize_impl(
                        BaseBackend, arg, is_const, specialize_value, align
                    )

                specialize_impl = _native_specialize_impl
            else:
                from triton.runtime.jit import specialize_impl as specialize_impl_orig

                specialize_impl = functools.partial(
                    specialize_impl_orig,
                    specialize_extra=backend.get_arg_specialization,
                )

            from triton._utils import find_paths_if, get_iterable_path

            # logic is copied from: binder = create_function_from_signature(self.signature, self.params, backend)
            attrvals = []
            for arg, kp in zip(args, kernel.params):
                if kp.is_constexpr:
                    attrvals.append(arg)
                else:
                    spec = specialize_impl(
                        arg,
                        is_const=kp.is_const,
                        specialize_value=not kp.do_not_specialize,
                        align=not kp.do_not_specialize_on_alignment,
                    )
                    # pyrefly: ignore [unsupported-operation]
                    attrvals.append(spec[1])

            attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
            attrs = {
                k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs
            }
            return attrs

    specialization = _get_specialization(ordered_args.values())
    constants = {
        name: arg for name, arg in ordered_args.items() if not is_tensor_like_arg(arg)
    }

    if (mangle_type := getattr(triton.runtime.jit, "mangle_type", None)) is not None:

        def get_signature_value(idx: int, arg: Any) -> str:
            if kernel.params[idx].is_constexpr:
                return "constexpr"
            # pyrefly: ignore [not-callable]
            return mangle_type(arg)

    else:

        def get_signature_value(idx: int, arg: Any) -> str:
            return kernel._type_of(kernel.key_of(arg))

    if triton_version_uses_attrs_dict():
        # In newer versions of Triton, the signature includes constexpr args
        signature = {
            name: get_signature_value(i, arg)
            for i, (name, arg) in enumerate(ordered_args.items())
        }
    else:
        # In older versions of Triton, the signature does not include constexpr args
        constexprs = [p.num for p in kernel.params if p.is_constexpr]
        signature = {
            name: get_signature_value(i, arg)
            for i, (name, arg) in enumerate(ordered_args.items())
            if i not in constexprs
        }

    # pyrefly: ignore  # missing-attribute
    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(kernel, signature, constants, specialization)

    # Triton changes ASTSource.make_ir to take 3/4 arguments. Handle
    # backward compatibility here.
    make_ir_sig_params = len(inspect.signature(src.make_ir).parameters)
    get_codegen_implementation_sig_params = len(
        # pyrefly: ignore  # missing-attribute
        inspect.signature(backend.get_codegen_implementation).parameters
    )
    if make_ir_sig_params == 2:
        # pyrefly: ignore  # missing-argument
        ttir_module = src.make_ir(options, context)
    elif make_ir_sig_params == 3:
        # pyrefly: ignore  # missing-attribute
        codegen_fns = backend.get_codegen_implementation()
        # pyrefly: ignore  # missing-argument
        ttir_module = src.make_ir(options, codegen_fns, context)
    elif make_ir_sig_params == 4:
        codegen_args = [options] if get_codegen_implementation_sig_params == 1 else []
        # pyrefly: ignore  # missing-attribute
        codegen_fns = backend.get_codegen_implementation(*codegen_args)
        module_map = backend.get_module_map()
        # pyrefly: ignore[missing-argument,bad-argument-type]
        ttir_module = src.make_ir(options, codegen_fns, module_map, context)
    else:
        codegen_args = [options] if get_codegen_implementation_sig_params == 1 else []
        # pyrefly: ignore  # missing-attribute
        codegen_fns = backend.get_codegen_implementation(*codegen_args)
        module_map = backend.get_module_map()
        # pyrefly: ignore  # bad-argument-count
        ttir_module = src.make_ir(target, options, codegen_fns, module_map, context)
    if not ttir_module.verify():
        raise RuntimeError("Verification for TTIR module has failed")

    return ttir_module, ordered_tensor_names


def ttir_to_functions(
    ttir_module: "TritonIRModule",
) -> dict[str, dict[Intermediate, list[Op]]]:
    """
    Walk the `ttir_module` bottom up to mine the `functions` from
    the structured MLIR entities representing the Triton kernel
    (mlir::Operation, mlir::Block, mlir::Region).
    """
    functions: dict[str, dict[Intermediate, list[Op]]] = {}

    # block id --> op result (Intermediate) --> one or more ops
    op_stack: dict[int, dict[Intermediate, list[Op]]] = defaultdict(
        lambda: defaultdict(list)
    )
    region_id_to_block_ids: dict[int, list[int]] = defaultdict(list)
    block_id_to_block_arg_ids: dict[int, list[int]] = {}
    replacements: dict[int, Union[Intermediate, Param]] = {}
    reindex_map: dict[int, int] = {}
    next_fake_intermediate = 0

    def reindex(idx: int) -> int:
        if idx not in reindex_map:
            reindex_map[idx] = len(reindex_map)
        return reindex_map[idx]

    def mlir_to_functions(op: "TritonIROperation") -> None:
        name: str = op.get_name()
        if name == "builtin.module":
            # this wraps all tt.func ops
            return

        operand_ids: list[int] = [
            reindex(op.get_operand(i).id()) for i in range(op.get_num_operands())
        ]
        result_ids: list[int] = [
            reindex(op.get_result(i).id()) for i in range(op.get_num_results())
        ]

        child_block_ids: list[int] = []
        for i in [op.get_region(i).id() for i in range(op.get_num_regions())]:
            # as the walk is bottom-up, the region_id_to_block_ids[i]
            # must be populated by the time we process the enclosing op
            child_block_ids.extend(region_id_to_block_ids[i])

        parent_block_id = -1
        parent_block = op.get_block()
        if parent_block is not None:
            parent_block_id = parent_block.id()
            if parent_block_id not in block_id_to_block_arg_ids:
                block_id_to_block_arg_ids[parent_block_id] = []
                for i in range(parent_block.get_num_arguments()):
                    block_id_to_block_arg_ids[parent_block_id].append(
                        reindex(parent_block.get_argument(i).id()),
                    )
                # the region info is collected via ops' parent blocks to be
                # used later when the region's encloding op is traversed
                parent_region = parent_block.get_parent()
                if parent_region is not None:
                    region_id_to_block_ids[parent_region.id()].append(parent_block_id)

        nonlocal next_fake_intermediate

        if name == "tt.func":
            # for function ops: gather and inline
            # the ops from all child blocks
            fn_ops = defaultdict(list)
            for child_block_id in child_block_ids:
                for result, block_fn_ops in op_stack.pop(child_block_id).items():
                    for block_fn_op in block_fn_ops:
                        fn_ops[result].append(block_fn_op)

            # replace the corresponding Intermediates in the
            # child op args with the function args (Params)
            for i, idx in enumerate(block_id_to_block_arg_ids[child_block_ids[0]]):
                replacements[idx] = Param(i)

            for fn_op_list in fn_ops.values():
                for fn_op in fn_op_list:
                    for i in range(len(fn_op.args)):
                        arg = fn_op.args[i]
                        seen = set()  # to break cycles
                        # there can be transitive replacements, but likely
                        # no cycles (we keep the `seen` set just in case)
                        while (
                            isinstance(arg, Intermediate)
                            and arg.idx in replacements
                            and arg.idx not in seen
                        ):
                            seen.add(arg.idx)
                            arg = fn_op.args[i] = replacements[arg.idx]

            # next function capture starts
            # with empty replacements
            replacements.clear()

            fn_name = op.get_str_attr("sym_name")
            functions[fn_name] = fn_ops
        elif child_block_ids:
            if name in {"scf.if", "scf.for", "scf.while", "tt.reduce", "tt.scan"}:
                # for blocked ops: inline the enclosed ops into
                # the parent block + rewire the last op in each
                # child block to return the block result
                return_ops = []
                for block_id in child_block_ids:
                    if name == "scf.for":
                        # example:
                        # %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> (i32) ...
                        # block args: 2 (%iv, %arg)
                        # op operands: 4 (%lb, %ub, %step, %init)
                        # `%arg` is mapping to `%init`
                        for i, idx in enumerate(block_id_to_block_arg_ids[block_id]):
                            if i == 0:
                                next_fake_intermediate -= 1
                                replacements[idx] = Intermediate(next_fake_intermediate)
                            else:
                                replacements[idx] = Intermediate(operand_ids[i + 2])
                    elif name == "scf.while":
                        # example:
                        # %3:3 = scf.while (%arg2 = %1, %arg3 = %2, %arg4 = %c0_i32_8) ...
                        # block args: 3 (%arg2, %arg3, %arg4)
                        # op operands: 3 (%1, %2, %c0_i32_8)
                        # `%arg2` is mapping to `%1`, `%arg3` is mapping to `%2`, ...
                        for i, idx in enumerate(block_id_to_block_arg_ids[block_id]):
                            replacements[idx] = Intermediate(operand_ids[i])
                    elif name == "scf.if":
                        # the scf block args are ignored by the pass. but, as they
                        # may be used as operands of the ops inside the block
                        # (and nested blocks inlined in the current block by now),
                        # they are replaced by new fake Intermediates to avoid "this
                        # operand is not returned by any other op in the fn" error
                        # in the downstream analysis
                        for idx in block_id_to_block_arg_ids[block_id]:
                            next_fake_intermediate -= 1
                            replacements[idx] = Intermediate(next_fake_intermediate)
                    else:
                        assert name in ("tt.reduce", "tt.scan")
                        # wire the block arguments to the op arguments
                        num_operands = len(operand_ids)
                        block_arg_ids = block_id_to_block_arg_ids[block_id]
                        assert len(block_arg_ids) == 2 * num_operands, (
                            f"{name} is expected to have twice as "
                            "many block arguments as op arguments: "
                            f"{operand_ids=}, {block_arg_ids=}."
                        )
                        for i, idx in enumerate(block_arg_ids):
                            # for a tt.reduce/tt.scan op with N arguments, the block
                            # arguments comprise N reduced values followed by
                            # N current values corresponding to the N op args
                            replacements[idx] = Intermediate(
                                operand_ids[i % num_operands]
                            )

                    if block_id in op_stack:
                        block_ops = op_stack.pop(block_id)
                        if not block_ops:
                            continue
                        last_ret, last_ops = block_ops.popitem()
                        if all(
                            op.name
                            in ("scf.yield", "tt.reduce.return", "tt.scan.return")
                            for op in last_ops
                        ):
                            # if last_ops are all return ops, treat them separately
                            return_ops.extend(last_ops)
                        else:
                            # otherwise, return last_ops to the block
                            block_ops[last_ret] = last_ops
                        for op_result, child_ops in block_ops.items():
                            op_stack[parent_block_id][op_result].extend(child_ops)

                scf_results = [Intermediate(idx) for idx in result_ids]

                if return_ops and all(
                    (op.name == "scf.yield" and len(result_ids) == len(op.args))
                    for op in return_ops
                ):
                    # [Note: scf.yield fix-up]
                    #
                    # TL;DR: if our scf.yield takes N args, then we'll create N scf.yield ops to handle each of the
                    # args.
                    #
                    #      **Context**:
                    # During mutation analysis, the analysis pass will identify mutating ops (e.g. tt.store)
                    # and then DFS upwards towards the parameters of the function. Specifically, the analysis pass
                    # looks at the mutated arg in tt.store; then looks for its source ops; and then recurses on the
                    # arguments to each of the source ops.
                    #
                    # In the case of scf.if/scf.for, we may have multiple return ops, each passed as an arg
                    # to scf.yield:
                    #
                    # %18:2 = scf.if %... -> (!tt.ptr<f32>, !tt.ptr<f32>) {
                    #   ...
                    #   scf.yield %1, %2
                    # } else {
                    #   scf.yield %3, %4
                    # }
                    #
                    # And for each of the returns of the scf.if, we'd naively assign the source op of each of the
                    # return values to be the scf.yields. But the scf.yields take _all_ the returns as arguments.
                    # Therefore, if _any_ of the return values of the scf.if are mutated, then the analysis pass
                    # would mark _all_ of the yield args as mutated.
                    #
                    #      **Solution**:
                    # For the purposes of this analysis pass, we create N yield ops - one for each
                    # return-val/yield-arg. In the example above, we'll have two scf.yield's for each branch of the
                    # scf.if.

                    for return_op in return_ops:
                        for i, (scf_result, yield_arg) in enumerate(
                            zip(scf_results, return_op.args)
                        ):
                            sub_yield_op = Op(
                                return_op.name,
                                return_op.fn_call_name,
                                [yield_arg],
                                return_op.ret,
                                sub_idx=i,
                            )
                            op_stack[parent_block_id][scf_result].append(sub_yield_op)

                else:
                    for scf_result in scf_results:
                        for return_op in return_ops:
                            op_stack[parent_block_id][scf_result].append(return_op)
            else:
                raise RuntimeError(
                    f"Unknown blocked function: {name}. Can't capture the TTIR."
                )
        else:
            callee = None
            if name == "tt.call":
                callee = op.get_flat_symbol_ref_attr("callee")
            args: list[Union[Param, Intermediate]] = [
                Intermediate(operand) for operand in operand_ids
            ]
            block_ops = op_stack[parent_block_id]

            is_pure = False
            # Handle the case for tt.elementwise_inline_asm to set `is_pure` for mutation analysis
            if name == "tt.elementwise_inline_asm":
                is_pure = op.get_bool_attr("pure")

            if result_ids:
                for result_id in result_ids:
                    res = Intermediate(result_id)
                    block_ops[res].append(Op(name, callee, args, res, is_pure=is_pure))
            else:
                next_fake_intermediate -= 1
                fake_res = Intermediate(next_fake_intermediate)
                block_ops[fake_res].append(
                    Op(name, callee, args, fake_res, is_pure=is_pure)
                )

    ttir_module.walk(mlir_to_functions)

    return functions


class MemoizeWithCycleCheck:
    fn: Callable[..., Any]
    cache: dict[tuple[Any], Any]

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.reset()

    def __call__(
        self,
        functions: dict[str, dict[Intermediate, list[Op]]],
        fn_name: str,
        *args: Any,
    ) -> list[bool]:
        key: tuple[Any, ...] = (fn_name, *args)
        if key not in self.cache:
            self.cache[key] = None
            self.cache[key] = self.fn(functions, fn_name, *args)
        if self.cache[key] is None:
            raise RuntimeError("Recursion is not supported")
        return self.cache[key]

    def reset(self) -> None:
        self.cache = {}


@MemoizeWithCycleCheck
def get_tma_stores(
    functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str
) -> set[Union[Intermediate, Param]]:
    """
    Identifies all intermediates and parameters that are written to by a
    `tt.experimental_descriptor_store`. It tracks only the specific values
    written to via experimental_descriptor_store and the input values to
    `tt.reinterpret_tensor_descriptor` used to construct the direct inputs
    to tt.experimental_descriptor_store - not any recursive values
    used to construct those values.

    For example: for
      tt.reinterpret_tensor_descriptor(Intermediate(idx=0), ...)
      Intermediate(idx=1) = tt.experimental_descriptor_store(Intermediate(idx=0), ...)
    this function will return [Intermediate(idx=0), Intermediate(idx=1)],

    However
      Intermediate(idx=4) = arith.addptr(Intermediate(idx=2), Intermediate(idx=3))
      Intermediate(idx=5) = tt.experimental_descriptor_store(Intermediate(idx=4), ...)
      tt.experimental_descriptor_store(Intermediate(idx=5), ...)
    this function will mark only idx=4 and idx=5 (but not idx=2 or idx=3)

    If an intermediate/parameter is passed into a function and is written to
    via experimental_descriptor_store within that function, the argument to the
    function will also be marked.
    """

    result: set[Union[Intermediate, Param]] = set()

    ops = functions[fn_name]
    for op_list in ops.values():
        for op in op_list:
            if op.name == "tt.call":
                assert op.fn_call_name in functions
                # pyrefly: ignore [bad-argument-type]
                tma_stores = get_tma_stores(functions, op.fn_call_name)
                for i, inp in enumerate(op.args):
                    if Param(idx=i) in tma_stores:
                        result.add(inp)
            elif op.name == "tt.experimental_descriptor_store":
                assert len(op.args) >= 1
                result.add(op.args[0])
            elif op.name == "tt.descriptor_store":
                assert len(op.args) >= 1
                result.add(op.args[0])

    for val in list(result):
        if val in ops:
            if not isinstance(val, Intermediate):
                continue
            for op in ops[val]:
                if op.name == "tt.reinterpret_tensor_descriptor":
                    assert len(op.args) >= 1
                    result.add(op.args[0])

    return result


@MemoizeWithCycleCheck
def analyze_kernel_mutations(
    functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str, num_args: int
) -> list[bool]:
    """
    Analyzes the graph to detect all sinks from a predefined list of sinks
    by using triton's MemWrite trait list. NOTE: What if triton exposed this?
    From each sink, it traverses the CFG backwards to identify all the input
    pointers that are mutated.
    """
    # Name of mutation op to mutated parameter indices
    # List from Triton Github include/triton/Dialect/Triton/IR/TritonOps.td
    # All the OPs that have MemWrite trait.
    # What if Triton exposed this?
    MUTATION_OPS = {
        "tt.store": [0],
        "tt.atomic_cas": [0],
        "tt.atomic_rmw": [0],
        "tt.experimental_descriptor_store": [0],
        "tt.experimental_tensormap_create": [0],
        "tt.descriptor_store": [0],
    }
    # Ops that we want to bail out on
    UNKNOWN_OPS = {"tt.elementwise_inline_asm"}

    stack: list[Union[Param, Intermediate]] = []
    visited = set()
    ops = functions[fn_name]
    tma_stores = get_tma_stores(functions, fn_name)

    for op_list in ops.values():
        for op in op_list:
            # If we encounter an operation with effects that cannot be reliably analyzed
            # (e.g. `tt.elementwise_inline_asm`), we assume it does not mutate any input parameters.
            if op.name in UNKNOWN_OPS:
                if op.name == "tt.elementwise_inline_asm" and op.is_pure:
                    continue
                raise RuntimeError(
                    f"ttir analysis hit an op we do not know how to analyze: {op.name}"
                )

            if op.name == "tt.experimental_tensormap_create":
                # Note: this is how we implement experimental_descriptor_store mutation analysis.
                # for on-device TMA.
                # experimental_tensormap_store(a, b, ...) stores b to the location specified
                # by descriptor in the memory of a.
                # To track this, we first find all the intermediates/params to which we store via
                # experimental_tensormap_store (get_tma_stores, called above). Then, during this
                # analysis we wait to find the corresponding experimental_tensormap_create (if it
                # exists), at which point we will mark the global_ptr as mutated (as done below).
                assert len(op.args) >= 2
                if op.args[0] in tma_stores:
                    stack.append(op.args[1])

            if op.name == "tt.call":
                assert op.fn_call_name in functions
                mutations = analyze_kernel_mutations(
                    functions,
                    # pyrefly: ignore [bad-argument-type]
                    op.fn_call_name,
                    len(op.args),
                )
                stack.extend(arg for arg, mutated in zip(op.args, mutations) if mutated)
            else:
                stack.extend(op.args[idx] for idx in MUTATION_OPS.get(op.name, []))

    # The following is an iterative DFS algorithm
    mutated = [False] * num_args
    while stack:
        arg = stack.pop()
        if arg in visited:
            continue

        visited.add(arg)

        if isinstance(arg, Param):
            if arg.idx >= num_args:
                # This is an argument defined in the kernel, not passed in
                continue
            mutated[arg.idx] = True
        elif isinstance(arg, Intermediate) and not arg.fake():
            for op in ops[arg]:
                # Skip arguments to load
                if op.name != "tt.load":
                    stack.extend(op.args)
    return mutated


def identify_mutated_tensors(
    kernel: "TritonKernelType",
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> list[str]:
    """
    Given a triton kernel and the arguments for this kernel, this function
    1) Retrieves the TTIR converted version of the kernel from Triton's API.
    2) Parses the TTIR and creates a control flow graph
    3) Analyzes the graph to detect all input tensor mutations
    """

    ttir_module = None
    functions = None
    try:
        ttir_module, ordered_tensor_names = generate_ttir(
            kernel, kwargs, tma_descriptor_metadata
        )

        # extract functions from TTIR using MLIR bindings exposed by Triton code
        functions = ttir_to_functions(ttir_module)

        assert functions is not None
        kernel_name = next(iter(functions.keys()))
        # Triton codegen modifies the name
        # pyrefly: ignore [missing-attribute]
        assert kernel.fn.__name__ in kernel_name
        # Reset the cache between top level invocations
        # The cache for analyze kernel mutations is mainly used for cycle
        # detection, so each top level invocation needs a clean cache
        analyze_kernel_mutations.reset()
        get_tma_stores.reset()
        mutations = analyze_kernel_mutations(
            functions, kernel_name, len(ordered_tensor_names)
        )

        return [
            ordered_tensor_names[i] for i, mutated in enumerate(mutations) if mutated
        ]
    except Exception:
        log.warning(
            "Encountered an exception in identify_mutated_tensors, assuming every input is mutated",
            exc_info=True,
        )
        if ttir_module is not None:
            log.debug("TTIR:\n%s", str(ttir_module))
        if functions is not None:
            log.debug("functions:")
            for name, fn in functions.items():
                log.debug("===\t%s\t===", name)
                for ret, ops in fn.items():
                    log.debug("%s\t=>\t%s", ret, ops)
        return [key for key, value in kwargs.items() if isinstance(value, Tensor)]


###############################################################################
# Triton Kernel Wrappers


# Used for wrapping a Triton Kernel
class TritonKernelWrapperMutation(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("triton_kernel_wrapper_mutation", cacheable=True)

    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        grid: list["TritonGridType"],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: dict[str, Any],
    ) -> Any:
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grid,
            tma_descriptor_metadata=tma_descriptor_metadata,
            kwargs=kwargs,
        )


triton_kernel_wrapper_mutation = TritonKernelWrapperMutation()


# Used for wrapping a Triton Kernel in a functional manner
class TritonKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("triton_kernel_wrapper_functional", cacheable=True)

    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        grid: list["TritonGridType"],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: dict[str, Any],
        tensors_to_clone: list[str],
    ) -> dict[str, Any]:
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            grid=grid,
            tma_descriptor_metadata=tma_descriptor_metadata,
            kwargs=kwargs,
            tensors_to_clone=tensors_to_clone,
        )


triton_kernel_wrapper_functional = TritonKernelWrapperFunctional()


@triton_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_mutation_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    from torch._inductor.codegen.wrapper import user_defined_kernel_grid_fn_code

    kernel = kernel_side_table.get_kernel(kernel_idx)
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)

    if len(grid) == 1:
        grid_fn = grid[0]
    else:
        fn_name, code = user_defined_kernel_grid_fn_code(
            # pyrefly: ignore [missing-attribute]
            kernel.fn.__name__,
            # pyrefly: ignore [missing-attribute]
            kernel.configs,
            grid,
        )
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        grid_fn = namespace[fn_name]

    if tma_descriptor_metadata:
        # as we need to launch the kernel here, we "unwrap" the
        # tma_descriptor_metadata, create the TMA descriptors
        # from it, and replace the tensors in the kwargs by the
        # corresponding TMA descriptors before launching
        kwargs = kwargs.copy()
        for k, v in tma_descriptor_metadata.items():
            tensor = kwargs[k]
            if (exp_meta := maybe_unpack_tma_experimental_metadata(v)) is not None:
                from triton.tools.experimental_descriptor import (  # noqa: F401
                    create_1d_tma_descriptor,
                    create_2d_tma_descriptor,
                )

                dims, block_dims, element_size = exp_meta
                create_tma_descriptor = (
                    create_1d_tma_descriptor
                    if len(dims) == 1
                    else create_2d_tma_descriptor
                )
                kwargs[k] = create_tma_descriptor(
                    tensor.data_ptr(),
                    *dims,
                    *block_dims,
                    element_size,
                )
            else:
                stable_meta = maybe_unpack_tma_stable_metadata(v)
                assert stable_meta is not None
                from triton.tools.tensor_descriptor import TensorDescriptor

                block_shape = stable_meta[0]
                # pyrefly: ignore  # bad-argument-type
                kwargs[k] = TensorDescriptor.from_tensor(tensor, block_shape)

    # move as many positional arguments from dicts to args as we
    # can to circumvent the bug with the kwargs and pre_/post_hook:
    # https://github.com/triton-lang/triton/issues/5082
    # TODO: remove this when the Triton issue above is fixed
    args = []
    # copy kwargs and constant_args here to
    # avoid mutating the original inputs
    kwargs = kwargs.copy()
    constant_args = constant_args.copy()
    # pyrefly: ignore [missing-attribute]
    for name in kernel.arg_names:
        if name in kwargs:
            args.append(kwargs.pop(name))
        elif name in constant_args:
            args.append(constant_args.pop(name))
        else:
            break

    # pyrefly: ignore [index-error]
    kernel[grid_fn](*args, **kwargs, **constant_args)


@triton_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def triton_kernel_wrapper_mutation_fake_tensor_mode(
    mode: FakeTensorMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    with mode:
        return None


@triton_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
def _(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    return None


def trace_triton_kernel_wrapper(
    proxy_mode: ProxyTorchDispatchMode,
    func_overload: Callable[..., Any],
    node_args: dict[str, Any],
) -> Optional[dict[str, Any]]:
    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)

    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy,  # type: ignore[union-attr]
        node_args,
    )
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )

    ret = track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
    return ret


@triton_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    trace_triton_kernel_wrapper(
        mode,
        triton_kernel_wrapper_mutation,
        {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": tma_descriptor_metadata,
            "kwargs": kwargs,
        },
    )

    return None


def get_mutated_tensors(
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> list[str]:
    kernel = kernel_side_table.get_kernel(kernel_idx)
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)
    return identify_mutated_tensors(
        kernel, {**kwargs, **constant_args}, tma_descriptor_metadata
    )


@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
) -> None:
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    # TODO(oulgen): Preexisting bug, if two kernel inputs are views of each
    # other, and one gets mutated in kernel, and later another gets mutated,
    # they are no longer equal. Fix this by graph breaking on this condition
    # earlier in dynamo.
    tensors_to_clone = get_mutated_tensors(
    
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_higher_order_ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_higher_order_ops`):

- [`schema.py_docs.md_docs.md`](./schema.py_docs.md_docs.md)
- [`run_const_graph.py_docs.md_docs.md`](./run_const_graph.py_docs.md_docs.md)
- [`effects.py_kw.md_docs.md`](./effects.py_kw.md_docs.md)
- [`partitioner.py_docs.md_docs.md`](./partitioner.py_docs.md_docs.md)
- [`strict_mode.py_docs.md_docs.md`](./strict_mode.py_docs.md_docs.md)
- [`out_dtype.py_kw.md_docs.md`](./out_dtype.py_kw.md_docs.md)
- [`wrap.py_docs.md_docs.md`](./wrap.py_docs.md_docs.md)
- [`while_loop.py_kw.md_docs.md`](./while_loop.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`invoke_subgraph.py_docs.md_docs.md`](./invoke_subgraph.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `triton_kernel_wrap.py_docs.md_docs.md`
- **Keyword Index**: `triton_kernel_wrap.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
