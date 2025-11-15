# Documentation: `torch/_dynamo/variables/torch.py`

## File Metadata

- **Path**: `torch/_dynamo/variables/torch.py`
- **Size**: 93,156 bytes (90.97 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

"""
This module implements variable tracking for torch functions and operations during Dynamo tracing.

It provides classes to handle different types of torch operations:

TorchInGraphFunctionVariable: Handles torch.* functions that should be captured in the FX graph.
Provides special handling for constant folding, tensor methods, and torch function overrides.
Manages complex cases like out= variants and parameter construction.

TorchCtxManagerClassVariable: Handles torch context managers like torch.no_grad(), autocast, etc.
Provides implementations for entering/exiting these contexts during tracing.

DispatchKeySetVariable: Represents torch.DispatchKeySet for managing dispatch keys and
device-specific operations during tracing.

The module includes special handling for:
- Constant folding of pure functions
- Tensor method calls
- torch.nn.Parameter construction
- __torch_function__ overrides
- Context manager state tracking
- Device and dtype management

This is a core part of Dynamo's tracing system, translating torch operations into
traceable graph nodes while preserving correct semantics and handling edge cases.
"""

import functools
import inspect
import logging
import math
import re
from collections.abc import Callable, Sequence
from typing import Any, Optional, TYPE_CHECKING

import torch._C
import torch._refs
import torch.fx
import torch.nn
from torch._guards import TracingContext
from torch._logging import warning_once
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type

from .. import config, graph_break_hints, polyfills, variables
from ..codegen import PyCodegen
from ..create_parameter_op import (
    can_convert_to_tracable_parameter,
    new_parameter_placeholder,
    tracable_create_parameter,
)
from ..device_interface import get_registered_device_interfaces
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    CallFunctionNoArgsSource,
    SyntheticLocalSource,
    TorchSource,
)
from ..utils import (
    check_unspec_or_constant_args,
    guard_if_dyn,
    has_torch_function,
    hashable,
    is_wrapper_or_member_descriptor,
    product,
    proxy_args_kwargs,
    unwrap_if_wrapper,
)
from .base import raise_type_error_exc, typestr, VariableTracker
from .ctx_manager import (
    AutocastModeVariable,
    ProfilerContextVariable,
    TorchFunctionDisableVariable,
)
from .dicts import ConstDictVariable
from .distributed import DistributedVariable, ProcessGroupVariable
from .functions import bind_args_cached, NestedUserFunctionVariable
from .lists import ListVariable, TupleVariable
from .torch_function import (
    can_dispatch_torch_function,
    dispatch_torch_function,
    TensorWithTFOverrideVariable,
    TorchFunctionModeStackVariable,
)


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


log = logging.getLogger(__name__)

supported_ctx_manager_classes = dict.fromkeys(
    [
        torch.profiler.profiler.profile,
        torch.autograd.forward_ad._set_fwd_grad_enabled,
        torch.autograd.forward_ad.dual_level,
        torch.autograd.profiler.profile,
        torch.autograd.profiler.record_function,
        torch._C.DisableTorchFunctionSubclass,
        torch._C.DisableTorchFunction,
        torch._functorch.vmap.vmap_increment_nesting,
        torch._functorch.eager_transforms.grad_increment_nesting,
        torch._functorch.eager_transforms.jvp_increment_nesting,
        torch._functorch.eager_transforms.enable_inplace_requires_grad,
        torch.amp.autocast_mode.autocast,
        torch.autograd.grad_mode.enable_grad,
        torch.autograd.grad_mode.inference_mode,
        torch.autograd.grad_mode.no_grad,
        torch.autograd.grad_mode.set_grad_enabled,
        torch.autograd.graph.disable_saved_tensors_hooks,
        torch.cpu.amp.autocast_mode.autocast,
        torch.cuda.amp.autocast_mode.autocast,
        torch.fx.traceback.annotate,
        torch.fx.traceback.annotate.__wrapped__,  # type: ignore[attr-defined]
        # We'll let Dynamo inline into the contextlib part of these context
        # manager instances, all the way till it invokes the wrapped function
        # itself (at which point we wrap it back to special context manager
        # VTs).
        #
        # This allows us to support calling functions decorated with these
        # context managers, without much extra effort or code dup.
        torch.nn.attention.sdpa_kernel.__wrapped__,  # type: ignore[attr-defined]
    ]
)


REWRITE_OPS_TO_TENSOR_SIZE_METHOD = dict.fromkeys(
    [
        torch._shape_as_tensor,
    ]
)

constant_fold_functions_need_guards = [
    torch.accelerator.current_device_index,
    torch.accelerator.current_accelerator,
    torch.cuda.current_device,
    torch.cuda.is_initialized,
    torch.xpu.current_device,
    torch.xpu.is_initialized,
]

constant_fold_functions = [
    torch._assert,
    torch._utils._get_device_index,
    torch._C._get_cublas_allow_tf32,
    torch._C._is_any_autocast_enabled,
    torch.accelerator.is_available,
    torch.cuda.get_device_properties,
    torch.cuda.is_available,
    torch.distributed.is_available,
    torch.get_autocast_dtype,
    torch.get_autocast_gpu_dtype,
    torch.get_default_dtype,
    torch.is_autocast_cache_enabled,
    torch.is_autocast_cpu_enabled,
    torch.is_autocast_enabled,
    torch.is_complex,
    torch.is_floating_point,
    torch.nn.functional._Reduction.get_enum,  # type: ignore[attr-defined]
    torch.promote_types,
    torch._C._get_privateuse1_backend_name,
    torch.autograd._is_checkpoint_valid,
    torch.xpu.get_device_properties,
    torch.xpu.is_available,
] + constant_fold_functions_need_guards
if torch.distributed.is_available():
    constant_fold_functions.extend(
        [
            torch.distributed.is_initialized,
            torch.distributed.get_rank,
            torch.distributed.get_world_size,
        ]
    )
# Convert to dict for O(1) access times
constant_fold_functions_need_guards = dict.fromkeys(constant_fold_functions_need_guards)
constant_fold_functions = dict.fromkeys(constant_fold_functions)


@functools.cache
def tracing_state_functions() -> dict[Callable[[], Any], Optional[bool]]:
    # Defined as a function to avoid circular import like torch.onnx
    return {
        torch.jit.is_scripting: False,
        torch.jit.is_tracing: False,
        torch._C._get_tracing_state: None,
        torch.fx._symbolic_trace.is_fx_tracing: False,
        torch.fx._symbolic_trace.is_fx_symbolic_tracing: False,
        torch.onnx.is_in_onnx_export: False,
        torch._dynamo.external_utils.is_compiling: True,
        torch._utils.is_compiling: True,
        torch.compiler.is_compiling: True,
        torch.compiler.is_dynamo_compiling: True,
        torch.compiler.is_exporting: True,
        # Look into https://github.com/pytorch/pytorch/pull/164721 why this is
        # turned to True for Dynamo.
        torch.nn.modules.activation._is_make_fx_tracing: True,
    }


bin_ops = dict.fromkeys(["add", "sub", "mul", "div", "sqrt"])

dispatch_key_set_functions = {
    torch._C._dispatch_keys,
    torch._C._dispatch_tls_local_include_set,
    torch._C._dispatch_tls_local_exclude_set,
}


@functools.cache
def get_overridable_functions():
    from itertools import chain

    from torch.overrides import get_overridable_functions as get_overridable_functions_

    funcs = set(chain.from_iterable(get_overridable_functions_().values()))
    more: set[Callable[..., Any]] = {
        torch.ones,
        torch.ones_like,
        torch.zeros,
        torch.zeros_like,
        torch.empty,
        torch.full,
    }
    funcs.update(more)
    return funcs


class BaseTorchVariable(VariableTracker):
    """common base for all torch.* functions, classes, modules and other things"""

    @classmethod
    def create_with_source(cls, value, source):
        if inspect.isclass(value):
            install_guard(source.make_guard(GuardBuilder.CLASS_MATCH))
        elif inspect.ismodule(value):
            install_guard(source.make_guard(GuardBuilder.MODULE_MATCH))
        elif inspect.isfunction(value):
            install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        elif inspect.isbuiltin(value) or isinstance(
            value, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
        ):
            install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        elif is_wrapper_or_member_descriptor(value) or isinstance(
            value, torch._dynamo.compiled_autograd.Op
        ):
            # Dont need to guard on wrappers
            pass
        else:
            install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(value, source=source)

    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    def reconstruct(self, codegen: "PyCodegen"):
        try:
            name = f"{self.value.__module__}.{self.value.__name__}"
        except Exception:
            name = f"torch_obj_{id(self.value)}"
        unique_var_name = "__" + re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        codegen.extend_output(
            codegen.setup_globally_cached(unique_var_name, self.value)
        )

    def as_proxy(self):
        return self.value

    def as_python_constant(self):
        return self.value

    def call_obj_hasattr(self, tx: "InstructionTranslator", name):
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)

    def can_constant_fold_through(self):
        if self.value in constant_fold_functions:
            return True

        if (
            self.value is torch.autograd._profiler_enabled
            and config.constant_fold_autograd_profiler_enabled
        ):
            # The relevant flag is enabled only for export. One might wonder
            # why?
            #
            # Actually we would like to not graph break even in the case of
            # Dynamo. But there is a weird-unsolved bug with Kineto + Dynamo
            # when there are distributed jobs that lead to NCCL timeouts. This
            # bug is a rare edege case, but we have not been able to root cause
            # it yet. See https://www.internalfb.com/sevmanager/view/560336 for
            # more details.
            #
            # So is this safe for export? Yes, for export, we do not anticipate
            # JIT tracing in distributed job training, and the weird edge-case
            # interaction with Kineto is not a valid usecase. So, this is ok.
            return True

        return getattr(self.value, "__module__", None) == "math"


class TorchCtxManagerClassVariable(BaseTorchVariable):
    """Points to a context manager class in torch.* that dynamo has implementations"""

    def __repr__(self) -> str:
        return f"TorchCtxManagerClassVariable({self.value})"

    @staticmethod
    def is_matching_cls(value):
        # Unwrap if it's a functools.lru_cache wrapper
        value = unwrap_if_wrapper(value)
        # We can't do isinstance(value, type) check because some ctx managers
        # are implemented as a function decorated by contextlib.contextmanager,
        # E.g., torch._functorch.vmap.vmap_increment_nesting.
        return (
            # Context manager type or function with @contextmanager is callable
            callable(value)
            and (
                hashable(value)  # accesses value.__hash__()
                and value in supported_ctx_manager_classes
            )
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import (
            DisabledSavedTensorsHooksVariable,
            DualLevelContextManager,
            FSDPParamGroupUseTrainingStateVariable,
            FxTracebackAnnotateVariable,
            GradIncrementNestingCtxManagerVariable,
            GradInplaceRequiresGradCtxManagerVariable,
            GradModeVariable,
            InferenceModeVariable,
            JvpIncrementNestingCtxManagerVariable,
            SDPAKernelVariable,
            SetFwdGradEnabledContextManager,
            StreamVariable,
            VmapIncrementNestingCtxManagerVariable,
        )

        if self.value is torch.no_grad:
            if len(args) == 1 and isinstance(
                args[0], variables.functions.BaseUserFunctionVariable
            ):
                ctx = GradModeVariable.create(tx, False)
                return ctx.call_function(tx, args, kwargs)
            else:
                return GradModeVariable.create(tx, False)
        elif self.value is torch.enable_grad:
            if len(args) == 1 and isinstance(
                args[0], variables.functions.BaseUserFunctionVariable
            ):
                ctx = GradModeVariable.create(tx, True)
                return ctx.call_function(tx, args, kwargs)
            return GradModeVariable.create(tx, True)
        elif self.value is torch.set_grad_enabled and len(args) == 1:
            return GradModeVariable.create(
                tx, args[0].as_python_constant(), initialized=True
            )
        elif self.value is torch.inference_mode:
            assert len(args) <= 1 and len(kwargs) == 0
            inf_mode = args[0].as_python_constant() if len(args) == 1 else True
            return InferenceModeVariable.create(tx, inf_mode)
        elif self.value in (
            torch.fx.traceback.annotate,
            torch.fx.traceback.annotate.__wrapped__,  # type: ignore[attr-defined]
        ):
            assert len(args) <= 1 and len(kwargs) == 0
            return FxTracebackAnnotateVariable(
                args[0].as_python_constant(), source=self.source
            )
        elif inspect.isclass(self.value) and issubclass(self.value, torch.Stream):
            from torch._dynamo.variables.builder import wrap_fx_proxy_cls

            return wrap_fx_proxy_cls(
                StreamVariable,
                tx,
                tx.output.create_proxy(
                    "call_function",
                    self.value,
                    (),
                    {},
                ),
            )
        elif self.value in (
            torch.amp.autocast_mode.autocast,
            torch.cuda.amp.autocast,
            torch.cpu.amp.autocast,
        ):
            # pyrefly: ignore [bad-argument-type]
            return AutocastModeVariable.create(self.value, args, kwargs)
        elif self.value in (
            # NOTE any class added here must align with the semantic
            # requirements of `ProfilerContextVariable`.
            torch.profiler.profile,
            torch.profiler.record_function,
            torch.autograd.profiler.profile,
            torch.autograd.profiler.record_function,
        ):
            warning_once(log, "Profiler function %s will be ignored", self.value)
            return ProfilerContextVariable()
        elif (
            self.value is torch._C.DisableTorchFunctionSubclass
            or self.value is torch._C.DisableTorchFunction
        ):
            assert not (args or kwargs)
            return TorchFunctionDisableVariable.create(
                tx, only_subclass=self.value is torch._C.DisableTorchFunctionSubclass
            )
        elif self.value is torch._functorch.vmap.vmap_increment_nesting:
            assert len(args) == 2
            return VmapIncrementNestingCtxManagerVariable.create(
                tx,
                args,
            )
        elif self.value is torch._functorch.eager_transforms.jvp_increment_nesting:
            assert len(args) == 0
            return JvpIncrementNestingCtxManagerVariable.create(tx)
        elif self.value is torch.autograd.forward_ad._set_fwd_grad_enabled:
            assert len(args) == 1
            return SetFwdGradEnabledContextManager.create(
                tx,
                [guard_if_dyn(x) for x in args],
            )
        elif self.value is torch.autograd.forward_ad.dual_level:
            assert len(args) == 0
            return DualLevelContextManager.create(tx)
        elif self.value is torch._functorch.eager_transforms.grad_increment_nesting:
            assert len(args) == 0
            return GradIncrementNestingCtxManagerVariable.create(tx)
        elif (
            self.value is torch._functorch.eager_transforms.enable_inplace_requires_grad
        ):
            assert len(args) == 1
            return GradInplaceRequiresGradCtxManagerVariable.create(
                tx,
                [guard_if_dyn(x) for x in args],
            )
        elif self.value is torch.autograd.graph.disable_saved_tensors_hooks:
            assert len(args) == 1
            return DisabledSavedTensorsHooksVariable.create(
                tx, args[0].as_python_constant()
            )
        elif (
            _fsdp_param_group is not None
            and self.value is _fsdp_param_group.FSDPParamGroup.use_training_state
        ):
            assert len(args) == 2
            return FSDPParamGroupUseTrainingStateVariable.create(
                tx, args[0], args[1].as_python_constant()
            )
        elif self.value is torch.nn.attention.sdpa_kernel.__wrapped__:  # type: ignore[attr-defined]
            name_to_arg_map = bind_args_cached(
                # pyrefly: ignore[bad-argument-type]
                self.value,
                tx,
                self.source,
                args,
                kwargs,
            )
            backends = name_to_arg_map["backends"].as_python_constant()
            set_priority = name_to_arg_map["set_priority"].as_python_constant()
            return SDPAKernelVariable.create(tx, backends, set_priority)

        return super().call_function(tx, args, kwargs)


class TorchInGraphFunctionVariable(BaseTorchVariable):
    """Points to a torch function/method that should be put in FX graph"""

    def __init__(self, value, nonstrict_traceable=None, **kwargs) -> None:
        super().__init__(value, **kwargs)
        from ..trace_rules import is_nonstrict_trace_callable

        if nonstrict_traceable is None:
            nonstrict_traceable = is_nonstrict_trace_callable(value)
        self.nonstrict_traceable = nonstrict_traceable

    def __repr__(self) -> str:
        return f"TorchInGraphFunctionVariable({self.value}, nonstrict_traceable={self.nonstrict_traceable})"

    def get_function(self):
        return self.value

    @staticmethod
    @functools.cache
    def _get_handlers():
        """Build a dict from function -> method to handle it so that we are O(1)
        in terms of the number of function with special handling."""
        handlers = {}

        def register(*fns):
            def _register(handler):
                for fn in fns:
                    assert fn not in handlers, fn
                    handlers[fn] = handler
                return handler

            assert callable(fns[0])
            return _register

        from torch.backends.cuda import SDPAParams

        from . import (
            ConstantVariable,
            DeterministicAlgorithmsVariable,
            GradModeVariable,
            StreamContextVariable,
            SymNodeVariable,
            TensorVariable,
            UserDefinedObjectVariable,
        )
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        @register(*tracing_state_functions())
        def handle_tracing_state_functions(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            assert not args and not kwargs
            # See: https://github.com/pytorch/pytorch/issues/110765
            if self.value in (
                torch._utils.is_compiling,
                torch._dynamo.external_utils.is_compiling,
                torch.compiler.is_compiling,
                torch.compiler.is_dynamo_compiling,
                torch.compiler.is_exporting,
            ):
                tx.mark_inconsistent_side_effects()
            return ConstantVariable.create(tracing_state_functions()[self.value])

        @register(*dispatch_key_set_functions)
        def handle_dispatch_key_set_functions(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            assert not kwargs
            if self.value is torch._C._dispatch_keys:
                assert len(args) == 1
                assert isinstance(args[0], variables.TensorVariable)
                example_value = args[0].proxy.node.meta["example_value"]
                dks = self.value(example_value)
                # Remove Python and PythonTLSSnapshot from the dispatch key set,
                # as they originate from FakeTensor propagation.
                # This should only be done if the example_value is a FakeTensor.
                # However, if tensor subclasses are present,
                # it is reasonable for Python to remain in the dispatch key set.
                if isinstance(example_value, torch._subclasses.FakeTensor):
                    dks = (
                        dks
                        - torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
                        - torch._C.DispatchKeySet(
                            torch._C.DispatchKey.PythonTLSSnapshot
                        )
                    )
                return DispatchKeySetVariable.create(dks)
            else:
                assert not args
                return DispatchKeySetVariable.create(self.value())

        @register(torch.overrides.get_default_nowrap_functions.__wrapped__)
        def handle_get_default_nowrap_functions(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            # [Note: __torch_function__] we return empty here because we restrict
            # the set of functions that we trace __torch_function__ on to
            # functions outside of the actual set. Implementing this properly will require implementing
            # some variable types to track and compare tensor getset descriptors
            return VariableTracker.build(
                tx, torch.overrides.get_default_nowrap_functions()
            )

        @register(torch.ops.inductor.accumulate_grad_.default)
        def handle_accumulate_grad_(self, tx: "InstructionTranslator", *args, **kwargs):
            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.accumulate_grad), args, kwargs
            )

        @register(math.radians)
        def handle_radians(self, tx: "InstructionTranslator", *args, **kwargs):
            if not check_unspec_or_constant_args(args, kwargs):
                # Use polyfill to convert math.radians(x) into math.pi * x / 180.0
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.radians), args, kwargs
                )

        if hasattr(math, "fma"):  # Python 3.13+

            @register(math.fma)
            def handle_fma(self, tx: "InstructionTranslator", *args, **kwargs):
                if len(args) != 3 or kwargs:
                    return None

                if all(isinstance(arg, variables.TensorVariable) for arg in args):
                    x, y, z = args
                    addcmul_fn = TorchInGraphFunctionVariable(torch.addcmul)
                    return addcmul_fn.call_function(tx, [z, x, y], {})

                # Use math.fma if constants
                return None

        @register(torch.is_inference_mode_enabled)
        def handle_is_inference_mode_enabled(self, tx: "InstructionTranslator"):
            unimplemented(
                gb_type="Encountered torch.is_inference_mode_enabled during tracing",
                context="",
                explanation="torch.is_inference_mode_enabled() is not supported",
                hints=[
                    *graph_break_hints.FUNDAMENTAL,
                    *graph_break_hints.INFERENCE_MODE,
                ],
            )

        @register(torch.is_tensor, torch.overrides.is_tensor_like)
        def handle_is_tensor(self, tx: "InstructionTranslator", arg):
            if isinstance(arg, TensorVariable) or (
                self.value is torch.overrides.is_tensor_like
                and isinstance(arg, UserDefinedObjectVariable)
                and hasattr(arg.value, "__torch_function__")
            ):
                return ConstantVariable.create(True)
            else:
                return ConstantVariable.create(False)

        @register(
            torch.is_floating_point,
            torch.is_complex,
        )
        def handle_is_floating_point(self, tx: "InstructionTranslator", input):
            input_arg = input
            if isinstance(input_arg, TensorVariable) and input_arg.dtype is not None:
                if self.value is torch.is_floating_point:
                    return ConstantVariable.create(input_arg.dtype.is_floating_point)
                elif self.value is torch.is_complex:
                    return ConstantVariable.create(input_arg.dtype.is_complex)
                else:
                    raise AssertionError(f"calling {self.value}")

        @register(torch.numel)
        def handle_numel(self, tx: "InstructionTranslator", input):
            if isinstance(input, TensorVariable) and input.valid_size():
                return ConstantVariable.create(product(input.size))
            elif isinstance(input, TensorVariable):
                # Workaround dynamic shapes issue
                return input.call_method(tx, "numel", [], {})

        @register(torch.compile)
        def handle_torch_compile(self, tx: "InstructionTranslator", *args, **kwargs):
            if len(args) == 1:
                # torch.compile is a no-op in dynamo
                return args[0]

            unimplemented(
                gb_type="torch.compile call with > 1 args",
                context=f"args={args}, kwargs={kwargs}",
                explanation="Attempted to call `torch.compile` with > 1 args. Dynamo does not support this.",
                hints=[
                    "Remove the torch.compile call or its additional args.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        @register(*REWRITE_OPS_TO_TENSOR_SIZE_METHOD)
        def handle_tensor_size_rewrites(self, tx: "InstructionTranslator", input):
            assert isinstance(input, TensorVariable)
            return input.call_method(tx, "size", [], {})

        @register(
            torch.nn.modules.utils._single,
            torch.nn.modules.utils._pair,
            torch.nn.modules.utils._triple,
            torch.nn.modules.utils._quadruple,
            torch.nn.modules.utils._ntuple,
        )
        def handle_ntuple(self, tx: "InstructionTranslator", *args, **kwargs):
            return self._call_ntuple(tx, args, kwargs)

        @register(torch.is_grad_enabled)
        def handle_is_grad_enabled(self, tx):
            install_guard(GradModeVariable._guards_singleton)
            return ConstantVariable.create(torch.is_grad_enabled())

        @register(torch.use_deterministic_algorithms)
        def handle_use_deterministic_algorithms(
            self, tx: "InstructionTranslator", mode, warn_only=False
        ):
            # pyrefly: ignore [missing-attribute]
            if warn_only and warn_only.as_python_constant():
                unimplemented(
                    gb_type="Attempted to use torch.use_deterministic_algorithms(warn_only=True)",
                    context=f"mode={mode}, warn_only={warn_only}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Remove param warn_only in function call torch.use_deterministic_algorithms.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            return DeterministicAlgorithmsVariable.create(tx, mode.as_python_constant())

        @register(torch.are_deterministic_algorithms_enabled)
        def handle_are_deterministic_algorithms_enabled(self, tx):
            install_guard(DeterministicAlgorithmsVariable._guards_singleton)
            return ConstantVariable.create(torch.are_deterministic_algorithms_enabled())

        @register(torch._C._is_torch_function_enabled)
        def handle_is_torch_function_enabled(self, tx):
            install_guard(TorchFunctionDisableVariable._guards_singleton)
            # see comment on SymbolicTorchFunctionState class as to why
            # this is not a bug
            return ConstantVariable.create(
                tx.symbolic_torch_function_state.torch_function_subclass_enabled
            )

        @register(torch._C._is_torch_function_all_disabled)
        def handle_is_torch_function_all_disabled(self, tx):
            install_guard(TorchFunctionDisableVariable._guards_singleton)
            return ConstantVariable.create(
                not tx.symbolic_torch_function_state.torch_function_mode_enabled
            )

        @register(
            torch.overrides.has_torch_function,
            torch.overrides.has_torch_function_variadic,
            torch.overrides.has_torch_function_unary,
        )
        def handle_has_torch_function(self, tx: "InstructionTranslator", *args):
            elems = (
                args[0].unpack_var_sequence(tx)
                if len(args) == 1 and isinstance(args[0], TupleVariable)
                else args
            )
            return ConstantVariable.create(
                any(has_torch_function(x) for x in elems),
            )

        @register(
            *dict.fromkeys(  # remove duplicates
                device_interface.stream
                for _, device_interface in get_registered_device_interfaces()
            )
        )
        def handle_device_interface_stream(self, tx: "InstructionTranslator", stream):
            return StreamContextVariable.create(tx, stream)

        @register(torch.from_numpy)
        def handle_from_numpy(self, tx: "InstructionTranslator", *args):
            if not config.trace_numpy:
                unimplemented(
                    gb_type="call `torch.from_numpy` with `torch._dynamo.config.trace_numpy=False`",
                    context=f"trace_numpy={config.trace_numpy}",
                    explanation=(
                        "Attempted to call `torch.from_numpy` with config "
                        "`torch._dynamo.config.trace_numpy` set to `False`."
                    ),
                    hints=[
                        "Change `torch._dynamo.config.trace_numpy` to `True`.",
                    ],
                )
            if not np:
                unimplemented(
                    gb_type="`torch.from_numpy` with NumPy unavailable",
                    context="",
                    explanation="Attempted to call `torch.numpy` but NumPy could not be imported.",
                    hints=[
                        "Check NumPy version and installation in your environment.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            return wrap_fx_proxy_cls(
                target_cls=TensorVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch.as_tensor,
                    *proxy_args_kwargs(args, {}),
                ),
                example_value=None,
            )

        @register(torch.jit.annotate)
        def handle_jit_annotate(self, tx: "InstructionTranslator", the_type, the_value):
            return the_value

        @register(torch.backends.cudnn.is_acceptable)
        def handle_cudnn_is_acceptable(
            self, tx: "InstructionTranslator", tensor, *extra
        ):
            # is_acceptable(tensor) returns true if
            #   (a) tensor dtype/device are supported by cudnn
            #   (b) cudnn is available
            #   (c) some initialization has completed
            # technically, it depends on some global state from (c) (torch.backends.cudnn.__cudnn_version)
            assert not extra, "Expect 1 input to cudnn.is_acceptable"
            assert isinstance(tensor, TensorVariable), (
                "Expect input to cudnn.is_acceptable to be a tensor"
            )
            tensor_inp = torch.tensor(0, dtype=tensor.dtype, device=tensor.device)
            return ConstantVariable.create(
                torch.backends.cudnn.is_acceptable(tensor_inp)
            )

        @register(torch.utils.hooks.BackwardHook)
        def handle_backward_hook(self, tx: "InstructionTranslator", *args, **kwargs):
            return variables.BackwardHookVariable.create(tx, *args, **kwargs)

        @register(torch.nn.Parameter)
        def handle_parameter(self, tx: "InstructionTranslator", *args, **kwargs):
            return self.call_nn_parameter(tx, *args, **kwargs)

        @register(torch.ops.aten.sym_size, torch.ops.aten.sym_size.int)
        def handle_sym_size(self_, tx, self, dim=None):
            # we see this when retracing already traced code
            if dim is not None:
                return self.call_method(tx, "size", [dim], {})

        @register(torch.ops.aten.sym_stride, torch.ops.aten.sym_stride.int)
        def handle_sym_stride(self_, tx, self, dim=None):
            if dim is not None:
                return self.call_method(tx, "stride", [dim], {})

        @register(torch.addcdiv)
        def handle_addcdiv(self, tx: "InstructionTranslator", *args, **kwargs):
            if len(args) == 3 and "value" in kwargs and len(kwargs) == 1:
                # decompose addcdiv into constituent ops, prevents a graph break due to converting
                # value to a scalar
                result = TorchInGraphFunctionVariable(torch.div).call_function(
                    tx, [*args[1:]], {}
                )
                result = TorchInGraphFunctionVariable(torch.mul).call_function(
                    tx, [result, kwargs["value"]], {}
                )
                return TorchInGraphFunctionVariable(torch.add).call_function(
                    tx, [args[0], result], {}
                )

        @register(torch.full)
        def handle_full(self, tx, size, fill_value, **kwargs):
            if isinstance(fill_value, TensorVariable):
                # Decompose: create empty tensor and fill it
                # This avoids the scalar extraction at compile time
                empty_result = TorchInGraphFunctionVariable(torch.empty).call_function(
                    tx, [size], kwargs
                )
                # Call fill_ method on the empty tensor
                return empty_result.call_method(tx, "fill_", [fill_value], {})

        @register(torch._foreach_lerp_)
        def handle_inplace_foreach_lerp_scalar(
            _, tx: "InstructionTranslator", *args, **kwargs
        ):
            if len(args) == 3 and not isinstance(args[2], ListVariable) and not kwargs:
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.foreach_lerp_inplace),
                    args,
                    kwargs,
                )

        @register(torch._foreach_pow)
        def handle_foreach_pow_scalar(_, tx: "InstructionTranslator", *args, **kwargs):
            # In eager it's more performant to call item() from within the C op implementation
            # in compile, it's more performant to not graph break.
            if len(args) == 2 and isinstance(args[0], TensorVariable) and not kwargs:
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.foreach_pow_scalar),
                    args,
                    kwargs,
                )

        @register(torch._assert)
        def handle_assert(self, tx: "InstructionTranslator", condition, message):
            if (condition.is_python_constant() and condition.as_python_constant()) or (
                isinstance(condition, variables.SymNodeVariable)
                and condition.evaluate_expr()
            ):
                return ConstantVariable(None)

        @register(SDPAParams)
        def handle_sdpa_params(self, tx: "InstructionTranslator", *args, **kwargs):
            return wrap_fx_proxy(
                tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch._C._SDPAParams,
                    *proxy_args_kwargs(args, kwargs),
                ),
                param_vars=args,
            )

        if DistributedVariable.is_available():
            from torch.distributed.distributed_c10d import (
                _get_group_size_by_name,
                _get_group_tag,
                _rank_not_in_group,
                _resolve_group_name_by_ranks_and_tag,
                get_process_group_ranks,
            )
            from torch.distributed.tensor import DTensor

            @register(
                _get_group_size_by_name,
                _get_group_tag,
                _rank_not_in_group,
                get_process_group_ranks,
                _resolve_group_name_by_ranks_and_tag,
            )
            def handle_constant_processgroup_functions(
                self, tx: "InstructionTranslator", *args
            ):
                # because the input is a "ProcessGroupVariable", we'll be guarding on its
                # ID_MATCH based on how it was constructed.

                # We desugar it at trace-time into ranks by directly calling util
                # bake the result into the trace
                if len(args) == 1:
                    # group or group name
                    assert isinstance(args[0], (ProcessGroupVariable, ConstantVariable))
                elif len(args) == 2:
                    # ranks + tag
                    assert isinstance(args[0], ListVariable) and isinstance(
                        args[1], ConstantVariable
                    )
                else:
                    raise AssertionError(
                        f"Invalid group value ({args}) for constant pg "
                        f"function {self.value}"
                    )
                args_as_value = [arg.as_python_constant() for arg in args]
                invocation_result = self.value(*args_as_value)

                # Note - while we *could* cook up sources around invocations, like a FunctionSource
                # the space of invoking functions in the middle of the guard chain is very iffy. As such,
                # guard propagation via options is the best we can do.
                return VariableTracker.build(tx, invocation_result)

            @register(DTensor.from_local)
            def handle_from_local(self, tx: "InstructionTranslator", *args, **kwargs):
                # rewrite non-primitive args/kwargs to be included in the on-the-fly prim function
                # and rewrite args to have only proxyable args, then insert call_function
                args_as_value = [x.as_python_constant() for x in args[1:]]
                kwargs_as_value = {
                    k: v.as_python_constant()
                    for k, v in kwargs.items()
                    if k not in ["shape", "stride"]
                }
                kwargs_to_be_proxied = {
                    k: kwargs[k] for k in ["shape", "stride"] if k in kwargs
                }

                def fn_with_prim_types(x, shape=None, stride=None):
                    return self.value(
                        x, *args_as_value, **kwargs_as_value, shape=shape, stride=stride
                    )

                # attach the same function name for better debugging
                fn_with_prim_types.__name__ = "prim " + self.value.__name__

                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        fn_with_prim_types,
                        *proxy_args_kwargs(
                            [args[0]],
                            kwargs_to_be_proxied,
                        ),
                    ),
                )

        @register(torch.nested.nested_tensor)
        def handle_nested_tensor(
            self,
            tx: "InstructionTranslator",
            tensor_list=None,
            *args,
            layout=None,
            **kwargs,
        ):
            from .lists import BaseListVariable

            if layout and layout.as_python_constant() == torch.strided:
                unimplemented(
                    gb_type="Attempted to use strided NestedTensor",
                    context=f"layout={layout}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change layout=torch.jagged.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            if not isinstance(tensor_list, BaseListVariable):
                unimplemented(
                    gb_type="Attempted to use `nested_tensor` with non-list input",
                    context=f"tensor_list={tensor_list}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change `nested_tensor` with list input.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )

        @register(torch.nn.functional.one_hot)
        def handle_one_hot(self, tx: "InstructionTranslator", *args, **kwargs):
            if len(args) + len(kwargs) == 1 or (
                len(args) == 2
                and args[1].is_python_constant()
                and args[1].as_python_constant() == -1
            ):
                unimplemented(
                    gb_type="Attempted to use `torch.nn.functional.one_hot` with data-dependent output shape",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Explicitly set the `num_classes` param of the function call "
                        "`torch.nn.functional.one_hot` to something other than -1.",
                    ],
                )

        @register(torch.fx.experimental.symbolic_shapes.guard_size_oblivious)
        def handle_guard_size_oblivious(self, tx: "InstructionTranslator", expr):
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return variables.ConstantVariable.create(
                    torch.fx.experimental.symbolic_shapes.guard_size_oblivious(
                        expr.sym_num
                    )
                )
            elif isinstance(expr, ConstantVariable):
                return expr

        @register(torch.fx.experimental.symbolic_shapes.guard_or_true)
        def handle_guard_or_true(self, tx: "InstructionTranslator", expr):
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return variables.ConstantVariable.create(
                    torch.fx.experimental.symbolic_shapes.guard_or_true(expr.sym_num)
                )
            elif isinstance(expr, ConstantVariable):
                return expr

        @register(torch.fx.experimental.symbolic_shapes.guard_or_false)
        def handle_guard_or_false(self, tx: "InstructionTranslator", expr):
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return variables.ConstantVariable.create(
                    torch.fx.experimental.symbolic_shapes.guard_or_false(expr.sym_num)
                )
            elif isinstance(expr, ConstantVariable):
                return expr

        @register(torch.fx.experimental.symbolic_shapes.statically_known_false)
        def handle_statically_known_false(self, tx: "InstructionTranslator", expr):
            if isinstance(expr, SymNodeVariable):
                return variables.ConstantVariable.create(
                    torch.fx.experimental.symbolic_shapes.statically_known_false(
                        expr.sym_num
                    )
                )
            elif isinstance(expr, ConstantVariable):
                return expr

        @register(torch.fx.experimental.symbolic_shapes.guard_scalar)
        def guard_scalar(self, tx: "InstructionTranslator", expr):
            if isinstance(expr, SymNodeVariable):
                val = expr.sym_num
            elif isinstance(expr, ConstantVariable):
                val = expr.value
            else:
                unimplemented(
                    gb_type="torch.fx.experimental.symbolic_shapes.guard_scalar branch not supported",
                    context=f"expr: {expr}",
                    explanation="Expected `expr` to be a symbolic variable or constant.",
                    hints=[],
                )
            return variables.ConstantVariable.create(
                # pyrefly: ignore [bad-argument-type, unbound-name]
                torch.fx.experimental.symbolic_shapes.guard_scalar(val)
            )

        @register(torch.fx.experimental.symbolic_shapes.statically_known_true)
        def handle_statically_known_true(self, tx: "InstructionTranslator", expr):
            if isinstance(expr, SymNodeVariable):
                return variables.ConstantVariable.create(
                    torch.fx.experimental.symbolic_shapes.statically_known_true(
                        expr.sym_num
                    )
                )
            elif isinstance(expr, ConstantVariable):
                return expr

        @register(torch.fx.experimental.symbolic_shapes.sym_and)
        def handle_sym_and(self, tx: "InstructionTranslator", *terms):
            if all(isinstance(x, SymNodeVariable) for x in terms):
                return SymNodeVariable.create(
                    tx,
                    torch.fx.experimental.symbolic_shapes.sym_and(
                        *(x.as_proxy() for x in terms)
                    ),
                    sym_num=None,
                )

        @register(torch.fx.experimental.symbolic_shapes.sym_or)
        def handle_sym_or(self, tx: "InstructionTranslator", *terms):
            if all(isinstance(x, SymNodeVariable) for x in terms):
                return SymNodeVariable.create(
                    tx,
                    torch.fx.experimental.symbolic_shapes.sym_or(
                        *(x.as_proxy() for x in terms)
                    ),
                    sym_num=None,
                )

        @register(torch.fx.experimental.symbolic_shapes.has_static_value)
        def handle_has_static_value(self, tx: "InstructionTranslator", expr):
            if isinstance(expr, SymNodeVariable):
                val = expr.sym_num
            elif isinstance(expr, ConstantVariable):
                val = expr.value
            else:
                return

            return variables.ConstantVariable.create(
                # pyrefly: ignore [bad-argument-type]
                torch.fx.experimental.symbolic_shapes.has_static_value(val)
            )

        @register(torch._C._autograd._unsafe_set_version_counter)
        def handle_unsafe_set_version_counter(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            from ..tensor_version_op import _unsafe_set_version_counter

            return TorchInGraphFunctionVariable(
                _unsafe_set_version_counter
            ).call_function(tx, [*args], kwargs)

        @register(torch._C._functorch.peek_interpreter_stack)
        def handle_functorch_peek_interpreter_stack(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            # Wrap C++ interpreter (torch._C._functorch.CInterpreter) as UserDefinedObjectVariable,
            # but Python interpreter (torch._functorch.pyfunctorch.FuncTorchInterpreter) as FuncTorchInterpreterVariable.
            return UserDefinedObjectVariable(
                torch._C._functorch.peek_interpreter_stack()
            )

        @register(torch._functorch.pyfunctorch.coerce_cinterpreter)
        def handle_functorch_pyfunctorch_coerce_cinterpreter(
            self, tx: "InstructionTranslator", *args, **kwargs
        ):
            cinterpreter = args[0].value
            return FuncTorchInterpreterVariable(
                torch._functorch.pyfunctorch.coerce_cinterpreter(cinterpreter)
            )

        @register(torch.tensor)
        def handle_torch_tensor(self, tx: "InstructionTranslator", *args, **kwargs):
            def check_any_unspec(x):
                # NB: This includes UnspecializedPythonVariable
                if isinstance(x, (TensorVariable, SymNodeVariable)):
                    return True
                elif isinstance(x, (ListVariable, TupleVariable)):
                    return any(check_any_unspec(y) for y in x.items)
                # TODO: there maybe other recursive structures you need to
                # check
                else:
                    return False

            data_arg = None
         
```



## High-Level Overview

"""This module implements variable tracking for torch functions and operations during Dynamo tracing.It provides classes to handle different types of torch operations:TorchInGraphFunctionVariable: Handles torch.* functions that should be captured in the FX graph.Provides special handling for constant folding, tensor methods, and torch function overrides.Manages complex cases like out= variants and parameter construction.TorchCtxManagerClassVariable: Handles torch context managers like torch.no_grad(), autocast, etc.Provides implementations for entering/exiting these contexts during tracing.DispatchKeySetVariable: Represents torch.DispatchKeySet for managing dispatch keys anddevice-specific operations during tracing.The module includes special handling for:- Constant folding of pure functions- Tensor method calls- torch.nn.Parameter construction- __torch_function__ overrides- Context manager state tracking- Device and dtype managementThis is a core part of Dynamo's tracing system, translating torch operations intotraceable graph nodes while preserving correct semantics and handling edge cases.

This Python file contains 9 class(es) and 93 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BaseTorchVariable`, `TorchCtxManagerClassVariable`, `TorchInGraphFunctionVariable`, `DispatchKeySetVariable`, `FuncTorchInterpreterVariable`

**Functions defined**: `tracing_state_functions`, `get_overridable_functions`, `create_with_source`, `__init__`, `reconstruct`, `as_proxy`, `as_python_constant`, `call_obj_hasattr`, `can_constant_fold_through`, `__repr__`, `is_matching_cls`, `call_function`, `__init__`, `__repr__`, `get_function`, `_get_handlers`, `register`, `_register`, `handle_tracing_state_functions`, `handle_dispatch_key_set_functions`

**Key imports**: functools, inspect, logging, math, re, Callable, Sequence, Any, Optional, TYPE_CHECKING, torch._C, torch._refs, torch.fx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/variables`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `inspect`
- `logging`
- `math`
- `re`
- `collections.abc`: Callable, Sequence
- `typing`: Any, Optional, TYPE_CHECKING
- `torch._C`
- `torch._refs`
- `torch.fx`
- `torch.nn`
- `torch._guards`: TracingContext
- `torch._logging`: warning_once
- `torch.utils._python_dispatch`: is_traceable_wrapper_subclass_type
- `..`: config, graph_break_hints, polyfills, variables
- `..codegen`: PyCodegen
- `..device_interface`: get_registered_device_interfaces
- `..exc`: raise_observed_exception, unimplemented
- `..guards`: GuardBuilder, install_guard
- `.base`: raise_type_error_exc, typestr, VariableTracker
- `.dicts`: ConstDictVariable
- `.distributed`: DistributedVariable, ProcessGroupVariable
- `.functions`: bind_args_cached, NestedUserFunctionVariable
- `.lists`: ListVariable, TupleVariable
- `numpy as np`
- `torch.distributed.fsdp._fully_shard`: _fsdp_param_group
- `torch._dynamo.symbolic_convert`: InstructionTranslator
- `like torch.onnx`
- `itertools`: chain
- `torch.overrides`: get_overridable_functions as get_overridable_functions_


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/_dynamo/variables`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`streams.py_docs.md`](./streams.py_docs.md)
- [`nn_module.py_docs.md`](./nn_module.py_docs.md)
- [`higher_order_ops.py_docs.md`](./higher_order_ops.py_docs.md)
- [`tensor.py_docs.md`](./tensor.py_docs.md)
- [`constant.py_docs.md`](./constant.py_docs.md)
- [`dicts.py_docs.md`](./dicts.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`lists.py_docs.md`](./lists.py_docs.md)


## Cross-References

- **File Documentation**: `torch.py_docs.md`
- **Keyword Index**: `torch.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
