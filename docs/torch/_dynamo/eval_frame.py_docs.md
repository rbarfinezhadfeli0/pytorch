# Documentation: `torch/_dynamo/eval_frame.py`

## File Metadata

- **Path**: `torch/_dynamo/eval_frame.py`
- **Size**: 97,440 bytes (95.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# mypy: disable-error-code="method-assign"

"""
This module implements the core frame evaluation handler for TorchDynamo's compilation system.
The eval frame handler intercepts Python bytecode execution at runtime to enable dynamic
compilation and optimization of PyTorch code.

Key components defined here:
- Frame evaluation handlers that intercept and analyze Python execution frames
- Guards management for tracking dependencies and invalidating compiled code
- Optimization contexts and decorators (optimize, run_once, disable, etc.)
- Export functionality for saving optimized graphs
- Backend compiler integrations and callback management

Functions in this file are responsible for modifying the eval frame handler at RUNTIME.
Therefore, all functions in this file are hot and performance-critical. Functions that
only execute at compile time should be placed in torch._dynamo.convert_frame.

The eval frame handler is the core mechanism that enables TorchDynamo to dynamically
intercept, analyze and optimize PyTorch code during execution. It works by registering
a custom frame evaluation function that gets called for every Python frame, allowing
us to detect PyTorch operations and trigger compilation as needed.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import inspect
import logging
import os
import sys
import sysconfig
import textwrap
import threading
import traceback
import types
import unittest
import warnings
import weakref
from collections.abc import Sized
from dataclasses import dataclass
from enum import Enum
from os.path import dirname, join
from typing import Any, NamedTuple, Optional, TYPE_CHECKING, Union
from unittest.mock import patch

import sympy

import torch
import torch.fx
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch import _guards

# see discussion at https://github.com/pytorch/pytorch/issues/120699
from torch._C._dynamo.eval_frame import (  # noqa: F401
    reset_code,
    set_code_exec_strategy,
    set_eval_frame,
    set_guard_complete_hook,
    set_guard_error_hook,
    set_skip_guard_eval_unsafe,
    unsupported,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.types import ConvertFrameReturn, FrameAction, FrameExecStrategy
from torch._export.utils import _compiling_state_context
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch._utils_internal import DISABLE_JUSTKNOBS, justknobs_check, log_export_usage
from torch.export.dynamic_shapes import (
    _combine_args,
    _DimHint,
    _DimHintType,
    _IntWrapper,
    _process_dynamic_shapes,
    _RelaxedConstraint,
    Constraint,
)
from torch.fx import GraphModule, traceback as fx_traceback
from torch.fx.experimental._dynamism import (
    clone_and_convert_to_meta,
    track_dynamism_across_examples,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

from . import config, convert_frame, distributed, external_utils, trace_rules, utils
from .backends.registry import CompilerFn, lookup_backend
from .code_context import code_context
from .exc import (
    CondOpArgsMismatchError,
    ShortenTraceback,
    Unsupported,
    UserError,
    UserErrorType,
)
from .hooks import Hooks
from .mutation_guard import install_generation_tagging_init
from .utils import (
    _get_error_on_graph_break,
    _set_error_on_graph_break,
    common_constant_types,
    compile_times,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._dynamo.package import CompilePackage
    from torch._dynamo.repro.after_dynamo import WrapBackendDebug
    from torch._subclasses import fake_tensor
    from torch.fx.node import Argument, Node, Target

    from .types import (
        CacheEntry,
        DynamoCallback,
        DynamoFrameType,
        GuardFail,
        GuardFilterEntry,
    )


log = logging.getLogger(__name__)


always_optimize_code_objects = utils.ExactWeakKeyDictionary()
null_context = contextlib.nullcontext


# See https://github.com/python/typing/pull/240
class Unset(Enum):
    token = 0


cached_backends: dict[int, CompilerFn] = {}

unset = Unset.token


if DISABLE_JUSTKNOBS:
    _maybe_set_eval_frame = set_eval_frame
else:

    def _maybe_set_eval_frame(callback: DynamoCallback) -> DynamoCallback:
        # A wrapper on set_eval_frame that is guarded by a Justknob.
        # Users can disable torchDynamo by setting the JK to False.
        if not justknobs_check("pytorch/compiler:enable_compiler_set_eval_frame"):
            torch._dynamo.utils.warn_once(
                "Dynamo disabled by Justknob: enable_compiler_set_eval_frame, skipping set_eval_frame"
            )
            return callback
        else:
            return set_eval_frame(callback)


@dataclass
class DynamoStance:
    stance: str = "default"
    skip_guard_eval_unsafe: bool = False
    backend: Union[str, Callable[..., Any], None] = None


_stance = DynamoStance()


def _set_stance(stance: DynamoStance) -> DynamoStance:
    global _stance

    from torch._C._dynamo.eval_frame import get_eval_frame_callback

    callback = get_eval_frame_callback()

    if callback is not False and callback is not None:
        raise RuntimeError("attempted to set_stance in a torch.compile region")

    prior = _stance
    _stance = stance
    return prior


_set_stance._dynamo_forbidden = True  # type: ignore[attr-defined]

_EXAMPLE_INPUTS: Optional[dict[str, list[Any]]] = None


def get_example_inputs(key: str) -> list[Any]:
    global _EXAMPLE_INPUTS
    if _EXAMPLE_INPUTS is None:
        _EXAMPLE_INPUTS = {}

    if key not in _EXAMPLE_INPUTS:
        _EXAMPLE_INPUTS[key] = []

    return _EXAMPLE_INPUTS[key]


def _callback_from_stance(callback: DynamoCallback) -> DynamoCallback:
    if _stance.stance == "default":
        # force_backend
        if _stance.backend is not None and callback not in (False, None):
            callback = _create_wrapped_callback(get_compiler_fn(_stance.backend))

        return callback
    elif _stance.stance == "eager_then_compile":
        if callback not in (False, None):
            return _create_delayed_compile_callback(callback, _stance.stance)
        return callback
    elif _stance.stance == "aot_eager_then_compile":
        if callback not in (False, None):
            return _create_delayed_compile_callback(callback, _stance.stance)
        return callback
    elif _stance.stance == "force_eager":
        # disable
        return None
    elif _stance.stance == "eager_on_recompile":
        # run mode
        return False
    elif _stance.stance == "fail_on_recompile":
        if callback in (False, None):
            return callback

        def fail_callback(
            frame: DynamoFrameType, *args: Any, **kwargs: Any
        ) -> ConvertFrameReturn:
            if trace_rules.check(frame.f_code):
                return ConvertFrameReturn()
            if not convert_frame.has_tensor_in_frame(frame):
                return ConvertFrameReturn()

            from torch._C._dynamo.eval_frame import (
                _debug_get_cache_entry_list,
                _debug_get_precompile_entries,
            )
            from torch._dynamo.guards import get_and_maybe_log_recompilation_reasons

            message = (
                "Detected recompile when torch.compile stance is 'fail_on_recompile'. "
                + f"filename: '{frame.f_code.co_filename}', "
                + f"function name: '{frame.f_code.co_name}', "
                + f"line number: {frame.f_lineno}"
            )
            cache_entries = _debug_get_cache_entry_list(frame.f_code)
            if cache_entries:
                reasons = get_and_maybe_log_recompilation_reasons(
                    cache_entries[0], frame, skip_logging=True
                )
                if reasons:
                    failures = textwrap.indent("\n".join(reasons), "- ")
                    guard_failure_details = (
                        f"triggered by the following guard failure(s):\n{failures}"
                    )
                    message += f"\n{textwrap.indent(guard_failure_details, '    ')}"
            precompile_entries = _debug_get_precompile_entries(frame.f_code)
            if len(precompile_entries) > 0:
                message += "\nFailed on the following precompiled guards: "
                for entry in precompile_entries:
                    message += f"\n{entry.guard_manager}{entry.guard_manager.check_verbose(frame.f_locals)}"  # type: ignore[attr-defined]
            raise RuntimeError(message)

        # to prevent cache miss due to different backend
        fail_callback._torchdynamo_orig_backend = callback  # type: ignore[attr-defined]

        return fail_callback
    else:
        raise RuntimeError(f"invalid torch.compile stance '{_stance}'")


def _create_wrapped_callback(
    compiler_fn: CompilerFn,
) -> convert_frame.CatchErrorsWrapper:
    hooks = Hooks()
    return convert_frame.catch_errors_wrapper(
        convert_frame.convert_frame(  # type: ignore[arg-type]
            compiler_fn,
            hooks,
        ),
        hooks,
    )


def _get_or_add_example_inputs(frame: DynamoFrameType) -> list[Any]:
    key = frame.f_code.co_filename + str(frame.f_code.co_firstlineno)
    example_inputs = get_example_inputs(key)

    if len(example_inputs) < 2:
        example_inputs.append(clone_and_convert_to_meta(frame.f_locals))

    return example_inputs


def _create_delayed_compile_callback(
    callback: DynamoCallback, stance: str
) -> Callable[..., Any]:
    def callback_fn(*args: Any, **kwargs: Any) -> convert_frame.ConvertFrameReturn:
        frame = args[0]
        example_inputs = _get_or_add_example_inputs(frame)

        if len(example_inputs) == 1:
            if stance == "eager_then_compile":
                return ConvertFrameReturn(
                    frame_exec_strategy=FrameExecStrategy(
                        FrameAction.DEFAULT, FrameAction.DEFAULT
                    )
                )
            elif stance == "aot_eager_then_compile":
                aot_eager_fn = get_compiler_fn("aot_eager")
                return _create_wrapped_callback(aot_eager_fn)(*args, **kwargs)

        dynamism = track_dynamism_across_examples(example_inputs)
        code_context.get_context(frame.f_code)["dynamism"] = dynamism
        compiler_fn = callback._torchdynamo_orig_backend._torchdynamo_orig_backend  # type: ignore[union-attr]
        return _create_wrapped_callback(compiler_fn)(*args, **kwargs)

    # to prevent cache miss due to different backend
    callback_fn._torchdynamo_orig_backend = callback  # type: ignore[attr-defined]

    return callback_fn


def _is_skip_guard_eval_unsafe_stance() -> bool:
    return _stance.skip_guard_eval_unsafe


def _reset_guarded_backend_cache() -> None:
    global cached_backends
    for backend in cached_backends.values():
        if hasattr(backend, "reset"):
            backend.reset()
    cached_backends.clear()


DONT_WRAP_FILES = {
    # For tracing into fx modules
    inspect.getsourcefile(GraphModule),
    join(dirname(dirname(__file__)), "onnx/_internal/fx/dynamo_graph_extractor.py"),
}


def _debug_get_cache_entry_list(
    code: Union[types.CodeType, Callable[..., Any]],
) -> list[CacheEntry]:
    """
    Given a code object or a callable object, retrieve the cache entries
     stored in this code.
    """
    if callable(code):
        code = code.__code__
    return torch._C._dynamo.eval_frame._debug_get_cache_entry_list(code)


class OptimizedModule(torch.nn.Module):
    """
    Wraps the original nn.Module object and later patches its
    forward method to optimized self.forward method.
    """

    _torchdynamo_orig_callable: Callable[..., Any]
    get_compiler_config: Callable[[], Any]

    _opt_mod_attributes = {
        "_orig_mod",
        "dynamo_ctx",
        "_torchdynamo_orig_callable",
        "get_compiler_config",
        "forward",
        "_forward",
        "__dict__",
        "named_children_walk",
        "_super_module_initialized",
    }

    def __init__(self, mod: torch.nn.Module, dynamo_ctx: _TorchDynamoContext) -> None:
        # NOTE: this must go first, because attribute reads/writes of `self`
        # uses `_orig_mod`, and sometimes users override `Module.__init__` to
        # do attribute reads/writes on `self`.
        #
        # We also can't use regular setattr because `super().__setattr__` will
        # complain for module value before `super().__init__()`
        object.__setattr__(self, "_orig_mod", mod)
        self._super_module_initialized = False
        super().__init__()
        self._super_module_initialized = True

        # Installs the params/buffer
        self._orig_mod = mod  # `super().__setattr__` will register this module
        self.dynamo_ctx = dynamo_ctx
        self._initialize()
        self.training = self._orig_mod.training

    def __len__(self) -> int:
        # Proxy the len call to the original module
        if isinstance(self._orig_mod, Sized):
            return len(self._orig_mod)
        # Mimic python's default behavior for objects without a length
        raise TypeError(f"{type(self._orig_mod).__name__} does not support len()")

    def _initialize(self) -> None:
        # Do this stuff in constructor to lower overhead slightly
        if isinstance(self.dynamo_ctx, DisableContext):
            # No need to check trace rules
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)
        elif config.wrap_top_frame or (
            isinstance(self._orig_mod.forward, types.MethodType)
            and (
                trace_rules.check(self._orig_mod.forward)
                or getattr(self._orig_mod, "_is_fsdp_managed_module", False)
            )
        ):
            # This may be a torch.nn.* instance in trace_rules.py which
            # won't trigger a frame evaluation workaround to add an extra
            # frame we can capture
            self.forward = self.dynamo_ctx(external_utils.wrap_inline(self._orig_mod))
        else:
            # Invoke hooks outside of dynamo then pickup the inner frame
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)

        if hasattr(self._orig_mod, "_initialize_hook"):
            self._forward = self.forward
            self.forward = self._call_lazy_check

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if torch.nn.modules.module._has_any_global_hook():
            warnings.warn(
                "Using `torch.compile(module)` when there are global hooks on "
                "modules (e.g., from `register_module_forward_hook`); this will"
                " cause the hooks to fire an extra time for the "
                "`OptimizedModule` created by `torch.compile(module)`. If this "
                "causes undesired behavior, please try using `module.compile()`"
                ", or use the per-module hooks instead",
                stacklevel=2,
            )
        return super().__call__(*args, **kwargs)

    def _aot_compile(self, inputs: list[torch._dynamo.aot_compile.ModelInput]) -> None:
        """
        Experimental: AOT Compile a set of inputs and use that as the forward function
        """
        model = self._orig_mod
        hooks = self.dynamo_ctx._hooks
        assert hooks is not None
        if not config.enable_aot_compile:
            raise RuntimeError(
                "AOT Compile is not enabled, please set torch._dynamo.config.enable_aot_config=True"
            )
        if not self.dynamo_ctx.fullgraph:
            raise RuntimeError(
                "Graph breaks are not supported with aot compile. Please use torch.compile(fullgraph=True)."
            )

        if not callable(self.dynamo_ctx.callback):
            raise RuntimeError("aot compile requires a callable dynamo callback.")

        backend = innermost_fn(
            self.dynamo_ctx.callback, unaltered_fn_attr="_torchdynamo_orig_backend"
        )
        from torch._dynamo.aot_compile import aot_compile_module

        self.forward = aot_compile_module(model, inputs, hooks, backend)

    def _save_aot_compiled_module(self, path: Optional[str] = None) -> bytes:
        if not config.enable_aot_compile:
            raise RuntimeError(
                "AOT Compile is not enabled, please set torch._dynamo.config.enable_aot_config=True"
            )
        from torch._dynamo.aot_compile import AOTCompiledModel

        assert isinstance(self.forward, AOTCompiledModel)
        result: bytes = self.forward.serialize()
        if path is not None:
            with open(path, "wb") as f:
                f.write(result)
        return result

    def _load_aot_compiled_module(self, data: bytes) -> None:
        if not config.enable_aot_compile:
            raise RuntimeError(
                "AOT Compile is not enabled, please set torch._dynamo.config.enable_aot_config=True"
            )
        from torch._dynamo.aot_compile import AOTCompiledModel

        compiled_forward = AOTCompiledModel.deserialize(self._orig_mod, data)
        assert isinstance(compiled_forward, AOTCompiledModel)
        self.forward = compiled_forward

    def __reduce__(
        self,
    ) -> tuple[type[OptimizedModule], tuple[torch.nn.Module, _TorchDynamoContext]]:
        return (self.__class__, (self._orig_mod, self.dynamo_ctx))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("forward", None)
        state.pop("__call__", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self._initialize()

    @property
    # pyrefly: ignore [bad-override]
    def training(self) -> bool:
        return self._orig_mod.training

    @training.setter
    def training(self, value: bool) -> None:
        # Ignore the `training` mutation in `super().__init__()`, since that's
        # setting the default on `nn.Module`, but we are mirroring the
        # `training` attr in `self._orig_mod`.
        if self._super_module_initialized:
            self._orig_mod.training = value

    def __getattr__(self, name: str) -> Any:
        if name == "_orig_mod":
            return self._modules["_orig_mod"]
        return getattr(self._orig_mod, name)

    def __setattr__(self, name: str, val: Any) -> None:
        # Allow patching over class attributes
        if hasattr(type(self), name):
            return super().__setattr__(name, val)

        if name in OptimizedModule._opt_mod_attributes:
            return super().__setattr__(name, val)
        return setattr(self._orig_mod, name, val)

    def __delattr__(self, name: str) -> None:
        # This mirrors `__setattr__`
        if hasattr(type(self), name):
            return super().__delattr__(name)

        if name in OptimizedModule._opt_mod_attributes:
            return super().__delattr__(name)
        return delattr(self._orig_mod, name)

    def _call_lazy_check(self, *args: Any, **kwargs: Any) -> Any:
        if (
            hasattr(self._orig_mod, "_initialize_hook")
            and hasattr(self._orig_mod, "_infer_parameters")
            and callable(self._orig_mod._infer_parameters)
        ):
            # In the case of a lazy module, we want to run
            # the pre-hooks which initialize it.
            # Afterwards, lazy module deletes its pre-hooks
            # to avoid treating it as lazy on subsequent recompile.
            self._orig_mod._infer_parameters(self._orig_mod, args, kwargs)
        return self._forward(*args, **kwargs)

    def __dir__(self) -> list[str]:
        orig_mod_attrs = self._orig_mod.__dir__()
        return orig_mod_attrs + [
            attr for attr in super().__dir__() if attr not in orig_mod_attrs
        ]


def remove_from_cache(f: Any) -> None:
    """
    Make sure f.__code__ is not cached to force a recompile
    """
    if isinstance(f, types.CodeType):
        reset_code(f)
    elif hasattr(f, "__code__"):
        reset_code(f.__code__)
    elif hasattr(getattr(f, "forward", None), "__code__"):
        reset_code(f.forward.__code__)
    else:
        from . import reset  # type: ignore[attr-defined]

        reset()
        log.warning("could not determine __code__ for %s", f)


def nothing() -> None:
    pass


def always_false() -> bool:
    return False


def innermost_fn(
    fn: Callable[..., Any], unaltered_fn_attr: str = "_torchdynamo_orig_callable"
) -> Callable[..., Any]:
    """
    In case of nesting of _TorchDynamoContext calls, find the innermost
    function. TorchDynamo caches on fn.__code__ object, so its necessary to find
    the innermost function to pass on the optimize, run, disable etc.
    """
    unaltered_fn = fn
    while hasattr(unaltered_fn, unaltered_fn_attr):
        unaltered_fn = getattr(unaltered_fn, unaltered_fn_attr)
        assert callable(unaltered_fn), (
            f"A callable function is expected, but {type(unaltered_fn)} is provided."
        )
    return unaltered_fn


def make_set_enable_dynamic(enable: bool) -> Any:
    assert isinstance(enable, bool)
    if enable:
        # Assume everything is dynamic by default
        return config._make_closure_patcher(assume_static_by_default=False)
    else:
        return config._make_closure_patcher(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        )


# A thread local storage that serves to store information as Dynamo traces
# through a user provided function.
class DynamoTLS(threading.local):
    # Each string is a summary of a frame Dynamo attempted to trace, stored in
    # temporal order.
    traced_frame_infos: list[str] = []


dynamo_tls = DynamoTLS()


def clear_dynamo_tls() -> None:
    dynamo_tls.traced_frame_infos.clear()


@atexit.register
def _log_traced_frames() -> None:
    """
    At program exit, log all of the frames Dynamo has attempted to trace from,
    excluding the continuation frames generated by Dynamo.
    """
    msg = "\n".join(dynamo_tls.traced_frame_infos)
    msg = textwrap.indent(msg, "  * ")
    msg = f"TorchDynamo attempted to trace the following frames: [\n{msg}\n]"
    log.info(msg)


def guard_collectives_hook(guard_eval_result: bool) -> bool:
    import torch.distributed as dist
    from torch._dynamo.utils import dynamo_timed

    # guard_eval_result == True  ==>  cache hit
    if pg := distributed.get_guard_pg():
        with dynamo_timed(
            "guard_collective", log_pt2_compile_event=False, log_waitcounter=True
        ):
            log.debug("guard_collective %s", guard_eval_result)
            # TODO: a bit awkward to time, this isn't inside of the dynamo compile region
            all_results = [None] * pg.size()
            dist.all_gather_object(all_results, guard_eval_result, group=pg)
            # True = everyone hit, OK to run
            # False = someone missed, force recompile everywhere
            res = all(all_results)
            log.debug("guard_collective %s -> %s", guard_eval_result, res)
            return res
    return guard_eval_result


_not_set = object()


class _TorchDynamoContext:
    def __init__(
        self,
        callback: DynamoCallback,
        on_enter: Callable[[], Any] = nothing,
        backend_ctx_ctor: Callable[
            [], contextlib.AbstractContextManager[Any]
        ] = null_context,
        patch_fn: Callable[[], Any] = nothing,
        first_ctx: bool = False,
        *,
        fullgraph: bool = False,
        error_on_graph_break: Optional[bool] = None,
        export: bool = False,
        dynamic: Optional[bool] = None,
        compiler_config: Optional[Any] = None,
        package: Optional[CompilePackage] = None,
        hooks: Optional[Hooks] = None,
    ) -> None:
        super().__init__()
        assert callable(callback) or callback is False or callback is None
        self.callback: DynamoCallback = callback
        self._backend_ctx_ctor = backend_ctx_ctor
        self.prior: Union[Unset, DynamoCallback] = unset
        self.first_ctx = first_ctx
        self.fullgraph = fullgraph
        self.error_on_graph_break = error_on_graph_break
        self.export = export
        self._dynamic = dynamic
        self.compiler_config = compiler_config
        self.cleanup_fns: list[Callable[[], Any]] = []
        self.enter_exit_hooks = []
        self._package = package
        self._hooks = hooks
        patch_fn()

        # Save the backends so that we can reset them during torch._dynamo.reset
        backend = innermost_fn(callback, unaltered_fn_attr="_torchdynamo_orig_backend")  # type: ignore[arg-type]
        cached_backends.setdefault(id(backend), backend)  # type: ignore[arg-type]

        if dynamic is not None:
            self.enter_exit_hooks.append(make_set_enable_dynamic(dynamic))

        if on_enter is not nothing:
            # this case is not common
            def call_on_enter() -> Callable[[], None]:
                on_enter()
                return nothing

            self.enter_exit_hooks.append(call_on_enter)

        if backend_ctx_ctor is not contextlib.nullcontext:
            # this case is not common
            def call_backend_ctx() -> functools.partial[Optional[bool]]:
                ctx = backend_ctx_ctor()
                ctx.__enter__()
                return functools.partial(ctx.__exit__, None, None, None)

            self.enter_exit_hooks.append(call_backend_ctx)

    def __enter__(self) -> None:
        if config.raise_on_ctx_manager_usage:
            raise RuntimeError(
                "torch._dynamo.optimize(...) is used with a context manager. "
                "Please refer to https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html "
                "to use torch._dynamo.optimize(...) as an annotation/decorator. "
            )
        self.prior = set_eval_frame(None)
        self.cleanup_fns = [enter() for enter in self.enter_exit_hooks]
        self.prior_skip_guard_eval_unsafe = set_skip_guard_eval_unsafe(
            _is_skip_guard_eval_unsafe_stance()
        )
        _maybe_set_eval_frame(_callback_from_stance(self.callback))

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> Optional[bool]:
        assert self.prior is not unset
        set_eval_frame(None)
        set_skip_guard_eval_unsafe(self.prior_skip_guard_eval_unsafe)
        for cleanup in self.cleanup_fns:
            cleanup()
        self.cleanup_fns.clear()
        _maybe_set_eval_frame(_callback_from_stance(self.prior))
        self.prior = unset
        return None

    def __call__(self, fn: Any) -> Any:
        # public api for compiler config/options
        def get_compiler_config() -> Any:
            return self.compiler_config

        from .package import DynamoCache

        # If self._package is lazily initialized, we should check the dynamo cache now
        if config.caching_precompile:
            if self._package is not None and not self._package.is_initialized():
                result = DynamoCache.load(fn)
                if result is None:
                    # Create a fresh CompilePackage
                    self._package.initialize(fn, None, ignore_inlined_sources=False)
                else:
                    try:
                        self._package.initialize(
                            fn, result.dynamo, ignore_inlined_sources=False
                        )
                        self._package.install(result.backends)
                    except RuntimeError:
                        log.warning(
                            "Failed to load entry from dynamo cache", exc_info=True
                        )
                        self._package.initialize(fn, None, ignore_inlined_sources=False)

        fn = innermost_fn(fn)

        def aot_compile(example_inputs: tuple[tuple[Any, ...], dict[str, Any]]) -> Any:
            from torch._dynamo.aot_compile import aot_compile_fullgraph

            if not self.fullgraph:
                raise RuntimeError(
                    "Graph breaks are not supported with aot compile. Please use torch.compile(fullgraph=True)."
                )

            if not callable(self.callback):
                raise RuntimeError("aot compile requires a callable dynamo callback.")

            assert self._hooks is not None
            return aot_compile_fullgraph(
                fn,
                example_inputs,
                hooks=self._hooks,
                backend=innermost_fn(
                    self.callback, unaltered_fn_attr="_torchdynamo_orig_backend"
                ),
            )

        # add context containing GraphModule to any GraphModule forward functions
        if isinstance(fn, GraphModule):
            # add context containing GraphModule to any GraphModule forward functions
            code_context.get_context(fn.forward.__code__)["orig_graphmodule"] = (
                weakref.ref(fn)
            )

        # Optimize the forward method of torch.nn.Module object
        if isinstance(fn, torch.nn.Module):
            mod = fn
            new_mod = OptimizedModule(mod, self)
            # Save the function pointer to find the original callable while nesting
            # of decorators.
            new_mod._torchdynamo_orig_callable = mod.forward

            # when compiling torch.nn.Module,
            # provide public api OptimizedModule.get_compiler_config()
            assert not hasattr(new_mod, "get_compiler_config")
            new_mod.get_compiler_config = get_compiler_config

            return new_mod

        if inspect.isclass(fn):
            # User has wrapped the class with compile/disable decorator. Apply
            # disable to init/call method.
            cls_obj = fn
            cls_obj.__call__ = self(cls_obj.__call__)
            if issubclass(cls_obj, torch.nn.Module):
                # NN module variable tracker directly inlines the _call_impl.
                cls_obj._call_impl = self(cls_obj._call_impl)
            return cls_obj

        assert callable(fn), (
            f"A callable function is expected, but {type(fn)} is provided."
        )

        try:
            filename = inspect.getsourcefile(fn)
        except TypeError:
            filename = None
        if config.debug_force_nested_calls:
            fn = external_utils.wrap_inline(fn)
        elif config.wrap_top_frame or (
            (filename is None or trace_rules.check(fn))
            and (
                getattr(fn, "__name__", "")
                not in ["_call_impl", "_wrapped_call_impl", "_lazy_forward"]
            )
            and filename not in DONT_WRAP_FILES
        ):
            # call to a builtin without a frame for us to capture
            fn = external_utils.wrap_inline(fn)

        def do_nothing(*arg: Any, **kwargs: Any) -> None:
            pass

        callback: Callable[..., Any] = do_nothing
        if hasattr(self, "callback"):
            callback = self.callback  # type: ignore[assignment]

        is_jit_tracing = torch._C._is_tracing
        is_fx_symbolic_tracing = torch.fx._symbolic_trace.is_fx_symbolic_tracing

        @functools.wraps(fn)
        def compile_wrapper(*args: Any, **kwargs: Any) -> Any:
            prior = set_eval_frame(None)
            try:
                # We shouldn't compile inside kernel invocation.
                if tracing_context := torch._guards.TracingContext.try_get():
                    if (
                        tracing_context.fake_mode is not None
                        and tracing_context.fake_mode.in_kernel_invocation
                    ):
                        return fn(*args, **kwargs)
                # Skip nested compile - just inline the function
                if is_fx_symbolic_tracing():
                    if config.error_on_nested_fx_trace:
                        raise RuntimeError(
                            "Detected that you are using FX to symbolically trace "
                            "a dynamo-optimized function. This is not supported at the moment."
                        )
                    else:
                        return fn(*args, **kwargs)

                if is_jit_tracing():
                    raise RuntimeError(
                        "Detected that you are using FX to torch.jit.trace "
                        "a dynamo-optimized function. This is not supported at the moment."
                    )

                cleanups = [enter() for enter in self.enter_exit_hooks]
                prior_skip_guard_eval_unsafe = set_skip_guard_eval_unsafe(
                    _is_skip_guard_eval_unsafe_stance()
                )
                prior_error_on_graph_break = None
                if not self.fullgraph and self.error_on_graph_break is not None:
                    prior_error_on_graph_break = _get_error_on_graph_break()
                    _set_error_on_graph_break(self.error_on_graph_break)

                # Ensure that if an assertion occurs after graph pushes
                # something onto the DynamicLayerStack then we pop it off (the
                # constructed graph code isn't guarded with try/finally).
                #
                # This used to be a context but putting a `with` here is a noticeable
                # perf regression (#126293)
                saved_dynamic_layer_stack_depth = (
                    torch._C._functorch.get_dynamic_layer_stack_depth()
                )

                _maybe_set_eval_frame(_callback_from_stance(callback))

                try:
                    return fn(*args, **kwargs)
                except Unsupported as e:
                    if config.verbose:
                        raise
                    # strip internal tracebacks from causes
                    cur_exn: BaseException = e
                    while cur_exn.__cause__ is not None:
                        cur_exn.__cause__.with_traceback(None)
                        cur_exn = cur_exn.__cause__
                    # pyrefly: ignore [invalid-inheritance]
                    raise e.with_traceback(None) from e.__cause__  # User compiler error
                except ShortenTraceback as e:
                    # Failures in the backend likely don't have useful
                    # data in the TorchDynamo frames, so we strip them out.
                    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
                finally:
                    # Restore the dynamic layer stack depth if necessary.
                    set_eval_frame(None)
                    if prior_error_on_graph_break is not None:
                        _set_error_on_graph_break(prior_error_on_graph_break)
                    torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth(
                        saved_dynamic_layer_stack_depth
                    )

                    set_skip_guard_eval_unsafe(prior_skip_guard_eval_unsafe)
                    for cleanup in cleanups:
                        cleanup()
            finally:
                _maybe_set_eval_frame(prior)

        # hooks to properly handle inlining
        if self.error_on_graph_break is not None:
            compile_wrapper._torchdynamo_inline = (  # type: ignore[attr-defined]
                external_utils.wrap_inline_with_error_on_graph_break(
                    fn, self.error_on_graph_break
                )
            )
        else:
            compile_wrapper._torchdynamo_inline = fn  # type: ignore[attr-defined]

        # Save the function pointer to find the original callable while nesting
        # of decorators.
        compile_wrapper._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]

        # when compiling user function instead of nn.Module
        # provide public api _fn.get_compiler_config()
        assert not hasattr(compile_wrapper, "get_compiler_config")
        compile_wrapper.get_compiler_config = get_compiler_config  # type: ignore[attr-defined]
        if torch._dynamo.config.enable_aot_compile:
            compile_wrapper.aot_compile = aot_compile  # type: ignore[attr-defined]

        # If the function is called using torch._dynamo.optimize decorator, we
        # should prevent any type of skipping.
        if callback not in (None, False):
            if not hasattr(fn, "__code__"):
                raise RuntimeError(
                    textwrap.dedent(
                        """

                        torch._dynamo.optimize is called on a non function object.
                        If this is a callable class, please wrap the relevant code into a function and optimize the
                        wrapper function.

                        >> class CallableClass:
                        >>     def __init__(self) -> None:
                        >>         super().__init__()
                        >>         self.relu = torch.nn.ReLU()
                        >>
                        >>     def __call__(self, x):
                        >>         return self.relu(torch.sin(x))
                        >>
                        >>     def print_hello(self):
                        >>         print("Hello world")
                        >>
                        >> mod = CallableClass()

                        If you want to optimize the __call__ function and other code, wrap that up in a function

                        >> def wrapper_fn(x):
                        >>     y = mod(x)
                        >>     return y.sum()

                        and then optimize the wrapper_fn

                        >> opt_wrapper_fn = torch._dynamo.optimize(wrapper_fn)
                        """
                    )
                )
            always_optimize_code_objects[fn.__code__] = True

        return compile_wrapper


class OptimizeContext(_TorchDynamoContext):
    def __init__(
        self,
        callback: DynamoCallback,
        backend_ctx_ctor: Callable[[], contextlib.AbstractContextManager[Any]],
        first_ctx: bool = False,
        *,
        fullgraph: bool = False,
        error_on_graph_break: Optional[bool] = None,
        export: bool = False,
        dynamic: Optional[bool] = None,
        compiler_config: Optional[Any] = None,
        rebuild_ctx: Optional[
            Callable[[], Union[OptimizeContext, _NullDecorator]]
        ] = None,
        package: Optional[CompilePackage] = None,
        hooks: Optional[Hooks] = None,
    ) -> None:
        def on_enter() -> None:
            install_generation_tagging_init()

        super().__init__(
            callback=callback,
            on_enter=on_enter,
            backend_ctx_ctor=backend_ctx_ctor,
            patch_fn=TorchPatcher.patch,
            first_ctx=first_ctx,
            fullgraph=fullgraph,
            error_on_graph_break=error_on_graph_break,
            export=export,
            dynamic=dynamic,
            compiler_config=compiler_config,
            package=package,
            hooks=hooks,
        )

        if config.compiled_autograd:
            _dynamic = self._dynamic
            if _dynamic is None:
                _dynamic = not torch._dynamo.config.assume_static_by_default

            def call_compiled_autograd() -> functools.partial[Optional[bool]]:
                assert rebuild_ctx is not None
                compiler_fn = rebuild_ctx()
                ctx = torch._dynamo.compiled_autograd._enable(
                    compiler_fn,
                    # pyrefly: ignore [bad-argument-type]
                    dynamic=_dynamic,
                    ignore_active_disable_ctx=False,
                )
                ctx.__enter__()
                return functools.partial(ctx.__exit__, None, None, None)

            self.enter_exit_hooks.append(call_compiled_autograd)

    def __reduce__(
        self,
    ) -> tuple[type[OptimizeContext], tuple[Any, ...], dict[str, Any]]:
        return (
            self.__class__,
            (self.callback, self._backend_ctx_ctor, self.first_ctx),
            {
                "export": self.export,
                "dynamic": self._dynamic,
                "compiler_config": self.compiler_config,
            },
        )


class RunOnlyContext(_TorchDynamoContext):
    def __init__(self) -> None:
        # cudagraph trees relies on generation increment
        def on_enter() -> None:
            torch._dynamo.mutation_guard.GenerationTracker.generation += 1

        super().__init__(callback=False, on_enter=on_enter)

    def __reduce__(self) -> tuple[type[RunOnlyContext], tuple[Any, ...]]:
        return (self.__class__, ())


class DisableContext(_TorchDynamoContext):
    def __init__(self, msg: Optional[str] = None, wrapping: bool = True) -> None:
        super().__init__(callback=None)
        self.msg = msg
        self.wrapping = wrapping

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # Earlier this code was in the base class _TorchDynamoContext. But we
        # moved it here to have better code organization. For disable, we just
        # want the callback to be None. We don't have to check trace_rules or
        # create any wrapper.
        fn = innermost_fn(fn)

        if isinstance(fn, torch.nn.Module):
            mod = fn
            new_mod = OptimizedModule(mod, self)
            new_mod._torchdynamo_orig_callable = mod.forward
            return new_mod

        if isinstance(fn, type):
            # User has wrapped the class with compile/disable decorator. Apply
            # disable to init/call method.
            cls_obj = fn
            # Disable on init is useful for reconstruction of bytecodes where we
            # want to prevent Dynamo from tracing into the init function. Check
            # test_reconstruction in test_model_output.py.
            cls_obj.__init__ = self(cls_obj.__init__)  # type: ignore[misc]
            cls_obj.__call__ = self(cls_obj.__call__)
            if issubclass(cls_obj, torch.nn.Module):
                # NN module variable tracker directly inlines the _call_impl. Disable it.
                # pyrefly: ignore [missing-attribute]
                cls_obj._call_impl = self(cls_obj._call_impl)
            return cls_obj

        assert callable(fn), (
            f"A callable function is expected, but {type(fn)} is provided."
        )

        def _fn(*args: Any, **kwargs: Any) -> Any:
            prior = set_eval_frame(None)
            try:
                _maybe_set_eval_frame(_callback_from_stance(self.callback))
                try:
                    if torch.compiler.is_exporting():
                        with fx_traceback.annotate(
                            {
                                "_torchdynamo_disable": True,
                                "_torchdynamo_disable_recursive": True,
                                "_torchdynamo_disable_method": getattr(
                                    fn, "__name__", type(fn).__name__
                                ),
                            }
                        ):
                            return fn(*args, **kwargs)
                    return fn(*args, **kwargs)
                finally:
                    set_eval_frame(None)
            finally:
                _maybe_set_eval_frame(prior)

        # Under some circumstances (e.g. precompile) we can end up calling @disable
        # decorator in generated bytecode and trigger recompile. This is due to the
        # fact that the old callback from torch.compile() is still active and under
        # this circumstance we will trigger a failure with set_stance("fail_on_recompile").
        # Therefore we want to skip calling into any frame in this case.
        if self.wrapping:
            _fn = functools.wraps(fn)(_fn)

        _fn._torchdynamo_disable = True  # type: ignore[attr-defined]
        _fn._torchdynamo_disable_msg = self.msg  # type: ignore[attr-defined]

        # Save the function pointer to find the original callable while nesting
        # of decorators.
        _fn._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]

        _fn._torchdynamo_disable_recursive = True  # type: ignore[attr-defined]

        return _fn

    def __reduce__(self) -> tuple[type[DisableContext], tuple[Any, ...]]:
        return (self.__class__, ())


def _optimize_catch_errors(
    compile_fn: convert_frame.ConvertFrameProtocol,
    hooks: Hooks,
    backend_ctx_ctor: Callable[
        [], contextlib.AbstractContextManager[Any]
    ] = null_context,
    fullgraph: bool = False,
    error_on_graph_break: Optional[bool] = None,
    export: bool = False,
    dynamic: Optional[bool] = None,
    compiler_config: Optional[Any] = None,
    rebuild_ctx: Optional[Callable[[], Union[OptimizeContext, _NullDecorator]]] = None,
    package: Optional[CompilePackage] = None,
) -> OptimizeContext:
    return OptimizeContext(
        convert_frame.catch_errors_wrapper(compile_fn, hooks),
        backend_ctx_ctor=backend_ctx_ctor,
        first_ctx=True,
        fullgraph=fullgraph,
        error_on_graph_break=error_on_graph_break,
        export=export,
        dynamic=dynamic,
        compiler_config=compiler_config,
        rebuild_ctx=rebuild_ctx,
        package=package,
        hooks=hooks,
    )


def get_compiler_fn(
    compiler_fn: Union[str, Callable[..., Any], None],
) -> WrapBackendDebug:
    from .repro.after_dynamo import wrap_backend_debug

    if compiler_fn is None:
        # Special case None to avoid crashing in hasattr
        compiler_str = None
    elif hasattr(compiler_fn, "compiler_name"):
        compiler_str = compiler_fn.compiler_name  # type: ignore[union-attr]
        assert isinstance(compiler_str, str)
    elif isinstance(compiler_fn, str):
        compiler_str = compiler_fn
    else:
        compiler_str = None
    compiler_fn = lookup_backend(compiler_fn)  # type: ignore[arg-type]
    return wrap_backend_debug(compiler_fn, compiler_str)


class _NullDecorator(contextlib.nullcontext):  # type: ignore[type-arg]
    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        assert callable(fn), (
            f"A callable function is expected, but {type(fn)} is provided."
        )
        return fn


# Make dynamo graph to have same input/output spec as user code
def argument_names(
    f_sig: inspect.Signature,
    args: Union[list[Any], tuple[Any, ...]],
    kwargs: dict[str, Any],
) -> list[str]:
    def signature_to_fullargspec(sig: inspect.Signature) -> inspect.FullArgSpec:
        # Get a list of Parameter objects from the Signature object
        params = list(sig.parameters.values())
        # Separate positional arguments, keyword-only arguments and varargs/varkw
        args = [
            p.name for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        kwonlyargs = [
            p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        varargs = next(
            (p.name for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL),
            None,
        )
        varkw = next(
            (p.name for p in params if p.kind == inspect.Parameter.VAR_KEYWORD),
            None,
        )
        # Get default values for positional arguments and keyword-only arguments
        defaults = tuple(
            p.default
            for p in params
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and p.default is not inspect.Parameter.empty
        )
        kwonlydefaults = {
            p.name: p.default
            for p in params
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            and p.default is not inspect.Parameter.empty
        }
        # Get annotations for parameters and return value
        annotations = {}
        if sig.return_annotation:
            annotations = {"return": sig.return_annotation}
        for parameter in params:
            annotations[parameter.name] = parameter.annotation
        # Return a FullArgSpec object with the extracted attributes
        return inspect.FullArgSpec(
            args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations
        )

    fullargspec = signature_to_fullargspec(f_sig)

    # 1. Map `args` 1-to-1 to positional arguments in original signature.
    input_strs = fullargspec.args[: len(args)]

    if len(args) > len(fullargspec.args):
        # 2. If there are more arguments left in `args`, they map to varargs in original
        # signature. Assign names as {varargs}_0, {varargs}_1, ...
        assert fullargspec.varargs is not None, "More arguments than expected"
        input_strs += [
            f"{fullargspec.varargs}_{i}" for i in range(len(args) - len(input_strs))
        ]
    elif len(args) < len(fullargspec.args):
        # 3. If there are fewer arguments in `args` than `fullargspec.args`,
        # it implies these are arguments either with default values, or provided in
        # `kwargs`. The former can be safely ignored. Because Dynamo.export does not
        # export them as part of the function signature. The latter will be handled
        # in the next step.
        for unprovided_arg in fullargspec.args[
            len(args) : -len(fullargspec.defaults or [])
        ]:
            assert unprovided_arg in kwargs, f"Missing argument {unprovided_arg}"

    # 4. Keyword arguments provided in `kwargs`.
    input_strs += list(kwargs.keys())

    # 5. Keyword-only arguments with default values if not provided are not exported
    # as part of the function signature.
    for kwonly_arg in fullargspec.kwonlyargs:
        kwonlydefaults = fullargspec.kwonlydefaults or {}
        assert kwonly_arg in kwa
```



## High-Level Overview

"""This module implements the core frame evaluation handler for TorchDynamo's compilation system.The eval frame handler intercepts Python bytecode execution at runtime to enable dynamiccompilation and optimization of PyTorch code.Key components defined here:- Frame evaluation handlers that intercept and analyze Python execution frames- Guards management for tracking dependencies and invalidating compiled code- Optimization contexts and decorators (optimize, run_once, disable, etc.)- Export functionality for saving optimized graphs- Backend compiler integrations and callback managementFunctions in this file are responsible for modifying the eval frame handler at RUNTIME.Therefore, all functions in this file are hot and performance-critical. Functions thatonly execute at compile time should be placed in torch._dynamo.convert_frame.The eval frame handler is the core mechanism that enables TorchDynamo to dynamicallyintercept, analyze and optimize PyTorch code during execution. It works by registeringa custom frame evaluation function that gets called for every Python frame, allowingus to detect PyTorch operations and trigger compilation as needed.

This Python file contains 19 class(es) and 105 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Unset`, `DynamoStance`, `OptimizedModule`, `DynamoTLS`, `_TorchDynamoContext`, `CallableClass`, `OptimizeContext`, `RunOnlyContext`, `DisableContext`, `_NullDecorator`, `FlattenInputOutputSignature`, `ExportResult`, `TorchPatcher`

**Functions defined**: `_maybe_set_eval_frame`, `_set_stance`, `get_example_inputs`, `_callback_from_stance`, `fail_callback`, `_create_wrapped_callback`, `_get_or_add_example_inputs`, `_create_delayed_compile_callback`, `callback_fn`, `_is_skip_guard_eval_unsafe_stance`, `_reset_guarded_backend_cache`, `_debug_get_cache_entry_list`, `__init__`, `__len__`, `_initialize`, `__call__`, `_aot_compile`, `_save_aot_compiled_module`, `_load_aot_compiled_module`, `__reduce__`

**Key imports**: annotations, atexit, contextlib, functools, inspect, logging, os, sys, sysconfig, textwrap


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `atexit`
- `contextlib`
- `functools`
- `inspect`
- `logging`
- `os`
- `sys`
- `sysconfig`
- `textwrap`
- `threading`
- `traceback`
- `types`
- `unittest`
- `warnings`
- `weakref`
- `collections.abc`: Sized
- `dataclasses`: dataclass
- `enum`: Enum
- `os.path`: dirname, join
- `typing`: Any, NamedTuple, Optional, TYPE_CHECKING, Union
- `unittest.mock`: patch
- `sympy`
- `torch`
- `torch.fx`
- `torch.utils._pytree as pytree`
- `torch.utils.checkpoint`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch._dynamo.types`: ConvertFrameReturn, FrameAction, FrameExecStrategy


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `eval_frame.py_docs.md`
- **Keyword Index**: `eval_frame.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
