# Documentation: `torch/_dynamo/utils.py`

## File Metadata

- **Path**: `torch/_dynamo/utils.py`
- **Size**: 172,445 bytes (168.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Utility functions and classes used throughout the TorchDynamo system.

This module contains a collection of helper utilities used by various parts of Dynamo for:
- Performance metrics collection and reporting
- Compilation timing and debugging
- Graph manipulation and tensor operations
- Runtime guards and checks
- Common data structure operations
- Testing and development tools

This is an internal module that provides shared functionality used across the Dynamo codebase.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import copy
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import importlib
import inspect
import itertools
import json
import linecache
import logging
import math
import operator
import os
import re
import sys
import textwrap
import threading
import time
import traceback
import types
import typing
import uuid
import warnings
import weakref
from collections import Counter, OrderedDict
from contextlib import AbstractContextManager, contextmanager
from dataclasses import is_dataclass
from functools import lru_cache
from types import CodeType, MethodWrapperType
from typing import (
    Any,
    cast,
    ClassVar,
    Generic,
    Literal,
    Optional,
    overload,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec, TypeIs

import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
import torch.utils._pytree as pytree
from torch import fx
from torch._C import (
    _instruction_counter,
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.metrics_context import MetricsContext, RuntimeMetricsContext
from torch._guards import CompileId, Source, TracingContext
from torch._subclasses.meta_utils import is_sparse_compressed
from torch._utils_internal import (
    justknobs_check,
    log_chromium_event_internal,
    log_compilation_event,
    record_chromium_event_internal,
    signpost_event,
)
from torch.fx._utils import _format_graph_code, lazy_format_graph_code
from torch.monitor import _WaitCounter
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._triton import has_triton, has_triton_package
from torch.utils.hooks import RemovableHandle

from .graph_utils import _get_flat_args


if typing.TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Container,
        Generator,
        ItemsView,
        Iterable,
        Iterator,
        KeysView,
        Mapping,
        Sequence,
        ValuesView,
    )

    from torch._dynamo.replay_record import ExecutionRecord
    from torch._dynamo.symbolic_convert import (
        InstructionTranslator,
        InstructionTranslatorBase,
    )
    from torch._dynamo.variables.base import VariableTracker
    from torch._prims_common import DeviceLikeType


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    import torch._logging
    import torch._numpy as tnp
    from torch._guards import detect_fake_mode  # noqa: F401
    from torch._logging import LazyString

    from . import config

    # NOTE: Make sure `NP_SUPPORTED_MODULES` and `NP_TO_TNP_MODULE` are in sync.
    if np:
        NP_SUPPORTED_MODULES: tuple[types.ModuleType, ...] = (
            np,
            np.fft,
            np.linalg,
            np.random,
        )

        NP_TO_TNP_MODULE = {
            np: tnp,
            np.fft: tnp.fft,
            np.linalg: tnp.linalg,
            np.random: tnp.random,
        }
    else:
        NP_SUPPORTED_MODULES = ()

        NP_TO_TNP_MODULE = {}
    from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
except ImportError:
    pass


T = TypeVar("T")
R = TypeVar("R")
_P = ParamSpec("_P")

unpatched_nn_module_getattr = torch.nn.Module.__getattr__
unpatched_nn_module_call = torch.nn.Module.__call__
unpatched_nn_module_call_impl = torch.nn.Module._call_impl

counters: collections.defaultdict[str, Counter[str]] = collections.defaultdict(
    collections.Counter
)
optimus_scuba_log: dict[str, Any] = {}
troubleshooting_url = (
    "https://pytorch.org/docs/main/compile/programming_model.recompilation.html"
)
nnmodule_doc_url = "https://pytorch.org/docs/main/torch.compiler_nn_module.html"
nnmodule_doc_url_msg = f"See {nnmodule_doc_url} for more information and limitations."
log = logging.getLogger(__name__)

# profiling compilation time by function
compilation_time_metrics: dict[str, list[float]] = {}

# This supports calculate_time_spent(), which reports cumulative times
# across the process for any "phase" populated by dynamo_timed. Reset if
# reset_frame_count() is called.
cumulative_time_spent_ns: dict[str, float] = collections.defaultdict(float)

timer_counter = itertools.count()


# Abstraction on top of counters.
class ReInplaceTrigger(enum.Enum):
    AUTO_FUNC_V1 = 1
    AUTO_FUNC_V2 = 2
    TRITON_OPS = 3


class ReinplaceCounters:
    _values: collections.defaultdict[str, int] = collections.defaultdict(int)

    # Track sizes of known not re-inplaced tensors (exclude dynamic shapes).
    @classmethod
    def add_missed_bytes(cls, trigger: ReInplaceTrigger, bytes: int) -> None:
        if bytes != 0:
            cls._values[f"missed_bytes_{trigger.name}"] += bytes

    # Track number of not re-inplaced tensors.
    @classmethod
    def add_missed_opportunities(cls, trigger: ReInplaceTrigger, count: int) -> None:
        if count != 0:
            cls._values[f"missed_tensors_{trigger}"] += count

    @classmethod
    def clear(cls) -> None:
        cls._values.clear()

    @classmethod
    def get_total_missed(cls) -> int:
        sum = 0
        for trigger in ReInplaceTrigger:
            sum += cls._values.get(f"missed_tensors_{trigger}", 0)
        return sum

    @classmethod
    def get_total_missed_bytes(cls) -> int:
        sum = 0
        for trigger in ReInplaceTrigger:
            sum += cls._values.get(f"missed_bytes_{trigger.name}", 0)
        return sum

    @classmethod
    def log(cls) -> None:
        # if not empty log.
        if cls._values:
            signpost_event("inductor", "reinplace_counters", cls._values)


def tabulate(
    rows: Union[list[tuple[str, Any]], list[list[Any]]],
    headers: Union[tuple[str, ...], list[str]],
) -> str:
    try:
        import tabulate

        return tabulate.tabulate(rows, headers=headers)
    except ImportError:
        return "\n".join(
            ", ".join(map(str, row)) for row in itertools.chain([headers], rows)
        )


curr_frame = 0


# Note: Called for you by dynamo - you almost never ever want to invoke this yourself.
def increment_frame() -> None:
    global curr_frame
    curr_frame = curr_frame + 1


# Note: Called for you by dynamo - you almost never ever want to invoke this yourself.
def reset_frame_count() -> None:
    global curr_frame
    cumulative_time_spent_ns.clear()
    compilation_time_metrics.clear()
    curr_frame = 0


_recompile_user_contexts: Optional[list[Callable[[], str]]] = None


def register_hook_for_recompile_user_context(hook: Callable[[], str]) -> None:
    """
    Register a hook to be called when a recompile is triggered. The hook
    should return a string describing user contexts that are not available
    to the compiler, such as the current training epoch. This is useful for
    debugging and data analysis for recompile. For data retention purposes,
    the user context string is capped at 256 characters.
    """
    global _recompile_user_contexts
    if _recompile_user_contexts is None:
        _recompile_user_contexts = []
    _recompile_user_contexts.append(hook)


def get_hook_for_recompile_user_context() -> Optional[list[Callable[[], str]]]:
    return _recompile_user_contexts


op_count = 0


def increment_op_count(cnt: int) -> None:
    global op_count
    op_count += cnt


# Get the total time in seconds for each "phase"
# For example, {'entire_frame_compile':8.574629999999999, 'backend_compile':5.26806}
def calculate_time_spent() -> dict[str, float]:
    total_by_key = {}
    for phase, timing in cumulative_time_spent_ns.items():
        # pyrefly: ignore [unsupported-operation]
        total_by_key[phase] = timing / 1e9

    total_by_key["total_wall_time"] = total_by_key.get(
        "entire_frame_compile", 0
    ) + total_by_key.get("entire_backward_compile", 0)
    # pyrefly: ignore [bad-return]
    return total_by_key


# Print a report of time spent so far
# Ex:
# TIMING:
# entire_frame_compile:8.574629999999999
# backend_compile:5.26806
def print_time_report() -> None:
    total_by_key = calculate_time_spent()

    out = "TIMING:"
    for key, value in total_by_key.items():
        out = f"{out} {key}:{round(value, 5)}"

    print(out)


# Use the following singleton to capture and log CompilationMetrics. Entering the context
# manager allocates a new record to be logged when it exits. (You should not need to use
# this directly unless you introduce a new code path where compilation metrics would be
# gathered). While compiling, use the setters or timer in MetricsContext to update fields
# in the current context. For example:
#
# To set a single field once (use overwrite=True to overwrite):
#   get_metrics_context().set("metric_name", value)
#
# To set multiple fields at once (use overwrite=True to overwrite):
#   get_metrics_context().update({"name1": val1, "name2": val2})
#
# To increment an integer field:
#   get_metrics_context().increment("metric_name", value)
#
# To record execution time, MetricsContext works with dynamo_timed:
#    def foo(...):
#        # Updates the "metric_us" field.
#        with dynamo_timed("metric", dynamo_compile_column_us="metric_us")
#            ...
#
_METRICS_CONTEXT: MetricsContext
_RUNTIME_METRICS_CONTEXT: RuntimeMetricsContext


def get_metrics_context() -> MetricsContext:
    return _METRICS_CONTEXT


def get_runtime_metrics_context() -> RuntimeMetricsContext:
    return _RUNTIME_METRICS_CONTEXT


class CompileEventLogLevel(enum.Enum):
    """
    Enum that loosely corresponds with a "log level" of a given event.

    CHROMIUM_EVENT: Logs only to tlparse.
    COMPILE_EVENT: Logs to tlparse + PT2 Compile Events
    COMPILATION_METRIC: Logs to tlparse, PT2 Compile Events, and dynamo_compile
    """

    CHROMIUM = 1
    PT2_COMPILE = 2
    COMPILATION_METRIC = 3


class CompileEventLogger:
    """
    Helper class for representing adding metadata(i.e. columns) to various compile events.
    Use CompileEventLogger to add event data to:
    - Chromium events
    - PT2 Compile Events
    - CompilationMetrics

    This should be used in conjunction with dynamo_timed() and metrics contexts, which create
    timed spans and events. CompileEventLogger uses three log levels (described in CompileEventLogLevel),
    where each log level logs to all sources below it in the hierarchy.

    Example usages:
    - I want to log to an existing chromium event within dynamo timed:
    with dynamo_timed("my_event"):
        CompileEventLogger.chromium("my_event", foo=bar)

    - I want to log my event to both chromium + pt2_compile_events:
    with dynamo_timed("my_event", log_pt2_compile_event=True):
        CompileEventLogger.pt2_compile("my_event", foo=bar)

    - I want to add information to dynamo events and dynamo_compile
        CompileEventLogger.compilation_metric(foo=bar)
    """

    @staticmethod
    def log_instant_event(
        event_name: str,
        metadata: dict[str, Any],
        time_ns: Optional[int] = None,
        log_level: CompileEventLogLevel = CompileEventLogLevel.CHROMIUM,
    ) -> None:
        if time_ns is None:
            time_ns = time.time_ns()
        chromium_log = get_chromium_event_logger()
        if log_level == CompileEventLogLevel.CHROMIUM:
            log_pt2_compile_event = False
        elif log_level == CompileEventLogLevel.PT2_COMPILE:
            log_pt2_compile_event = True
        else:
            raise RuntimeError(
                "Cannot log instant event at COMPILATION_METRIC level. Please choose one of CHROMIUM_EVENT or COMPILE_EVENT"
            )
        chromium_log.log_instant_event(
            event_name, time_ns, metadata, log_pt2_compile_event
        )

    @staticmethod
    def add_data(
        event_name: str,
        log_level: CompileEventLogLevel,
        overwrite: bool = False,
        **metadata: object,
    ) -> None:
        """
        Centralized API for adding data to various events
        Log an event to a toplevel "dynamo" event or metrics context
        depending on log level.
        """
        chromium_log = get_chromium_event_logger()
        pt2_compile_substack = chromium_log.get_pt2_compile_substack()

        if log_level == CompileEventLogLevel.CHROMIUM:
            chromium_log.add_event_data(event_name, **metadata)
        elif log_level == CompileEventLogLevel.PT2_COMPILE:
            pt2_compile_substack = chromium_log.get_pt2_compile_substack()
            if event_name not in pt2_compile_substack:
                raise RuntimeError(
                    "Error: specified log level PT2_COMPILE, but the event %s"
                    " is not logged to pt2_compile_events. Make sure the event is active and you passed "
                    "log_pt2_compile_event=True to dynamo_timed",
                    event_name,
                )
            chromium_log.add_event_data(event_name, **metadata)
        else:
            assert log_level == CompileEventLogLevel.COMPILATION_METRIC
            top_event = chromium_log.get_outermost_event()

            if event_name != top_event:
                raise RuntimeError(
                    "Log level is COMPILATION_METRIC, but event_name isn't the toplevel event. "
                    "CompilationMetrics must be logged to the toplevel event. Consider using `log_toplevel_event_data` directly."
                )
            metrics_context = get_metrics_context()
            if not metrics_context.in_progress():
                raise RuntimeError(
                    "No metrics context is in progress. Please only call this function within a metrics context."
                )

            # TODO: should we assert that the keys of metadata are in CompilationMetrics?
            metrics_context.update(metadata, overwrite)
            chromium_log.add_event_data(event_name, **metadata)

    @staticmethod
    def add_toplevel(
        log_level: CompileEventLogLevel, overwrite: bool = False, **metadata: object
    ) -> None:
        """
        Syntactic sugar for logging to the toplevel event
        """
        top_event = get_chromium_event_logger().get_outermost_event()
        if top_event is None:
            raise RuntimeError(
                "No toplevel event active. Please only call this function within a dynamo_timed context."
            )
        CompileEventLogger.add_data(top_event, log_level, overwrite, **metadata)

    @staticmethod
    def increment(
        event_name: str, log_level: CompileEventLogLevel, key: str, value: int
    ) -> None:
        """
        Increments an existing field, or adds it
        """
        chromium_log = get_chromium_event_logger()
        if (
            log_level == CompileEventLogLevel.CHROMIUM
            or log_level == CompileEventLogLevel.PT2_COMPILE
        ):
            chromium_log.increment(event_name, key, value)
        else:
            assert log_level == CompileEventLogLevel.COMPILATION_METRIC
            top_event = chromium_log.get_outermost_event()
            if event_name != top_event:
                raise RuntimeError(
                    "Log level is COMPILATION_METRIC, but event_name isn't the toplevel event. "
                    "CompilationMetrics must be logged to the toplevel event. Consider using `increment_toplevel` directly."
                )

            metrics_context = get_metrics_context()
            if not metrics_context.in_progress():
                raise RuntimeError(
                    "No metrics context is in progress. Please only call this function within a metrics context/dynamo_timed."
                )

            metrics_context.increment(key, value)
            chromium_log.increment(event_name, key, value)

    @staticmethod
    def increment_toplevel(
        key: str,
        value: int = 1,
        log_level: CompileEventLogLevel = CompileEventLogLevel.COMPILATION_METRIC,
    ) -> None:
        """
        Increments a value on the toplevel metric. By default, logs to metric.
        """
        chromium_log = get_chromium_event_logger()
        top_event = chromium_log.get_outermost_event()
        if top_event is None:
            raise RuntimeError(
                "No toplevel event active. Please only call this function within a metrics context/dynamo_timed."
            )
        CompileEventLogger.increment(top_event, log_level, key, value)

    @staticmethod
    def add_to_set(
        event_name: str, log_level: CompileEventLogLevel, key: str, value: Any
    ) -> None:
        """
        Add metadata <value> to a set of values with key <key>. Creates a set if it doesn't exist.
        """
        chromium_log = get_chromium_event_logger()
        if (
            log_level == CompileEventLogLevel.CHROMIUM
            or log_level == CompileEventLogLevel.PT2_COMPILE
        ):
            chromium_log.add_to_set(event_name, key, value)
        else:
            assert log_level == CompileEventLogLevel.COMPILATION_METRIC
            top_event = chromium_log.get_outermost_event()
            if event_name != top_event:
                raise RuntimeError(
                    "Log level is COMPILATION_METRIC, but event_name isn't the toplevel event. "
                    "CompilationMetrics must be logged to the toplevel event. Consider using `add_to_set_metric` directly."
                )

            metrics_context = get_metrics_context()
            if not metrics_context.in_progress():
                raise RuntimeError(
                    "No metrics context is in progress. Please only call this function within a metrics context/dynamo_timed."
                )

            metrics_context.add_to_set(key, value)
            chromium_log.add_to_set(event_name, key, value)

    @staticmethod
    def add_to_set_toplevel(
        key: str,
        value: Any,
        log_level: CompileEventLogLevel = CompileEventLogLevel.COMPILATION_METRIC,
    ) -> None:
        """
        Same as add to set, just does it automatically to the toplevel event instead of having to explicitly name it.
        Defaults to COMPILATION_METRIC log level.
        """
        chromium_log = get_chromium_event_logger()
        top_event = chromium_log.get_outermost_event()
        if top_event is None:
            raise RuntimeError(
                "No toplevel event active. Please only call this function within a metrics context/dynamo_timed."
            )
        CompileEventLogger.add_to_set(top_event, log_level, key, value)

    # Helper functions that are syntactic sugar

    @staticmethod
    def chromium(event_name: str, **metadata: object) -> None:
        """
        Add <metadata> to <event_name> in chromium. Each key/value of metadata will appear in the chromium trace.
        <event_name> should be the name of a timed event span passed to `dynamo_timed`.
        """
        CompileEventLogger.add_data(
            event_name, CompileEventLogLevel.CHROMIUM, overwrite=False, **metadata
        )

    @staticmethod
    def pt2_compile(event_name: str, **metadata: object) -> None:
        """
        Add <metadata> to <event_name> in chromium and PT2 Compile Events.
        Each key/value of metadata will appear in the chromium trace. Each kwarg name becomes
        a column in PT2 Compile Events, with the corresponding kwarg value.
        <event_name> should be the name of a timed event span passed to `dynamo_timed`,
        with log_to_pt2_compile_events=True.
        """
        CompileEventLogger.add_data(
            event_name, CompileEventLogLevel.PT2_COMPILE, overwrite=False, **metadata
        )

    @staticmethod
    def compilation_metric(overwrite: bool = False, **metadata: object) -> None:
        """
        Add <metadata> to the CompilationMetrics context. Also logs to PT2 Compile Events
        and chromium.
        Each key/value of metadata will appear in the chromium trace. Each kwarg name becomes
        a column in PT2 Compile Events and Dynamo Compile, with the corresponding kwarg value.
        """
        CompileEventLogger.add_toplevel(
            CompileEventLogLevel.COMPILATION_METRIC, overwrite, **metadata
        )

    @staticmethod
    def instant(
        event_name: str, metadata: dict[str, Any], time_ns: Optional[int] = None
    ) -> None:
        """
        Log an instant event to chromium logs with name <event_name> at time <time_ns>. The `args` field in
        Perfetto will point to metadata. <time_ns> should be a value obtained from time.time_ns().
        """
        CompileEventLogger.log_instant_event(
            event_name, metadata, time_ns, CompileEventLogLevel.CHROMIUM
        )

    @staticmethod
    def try_add_pt2_compile(event_name: str, **metadata: object) -> None:
        """
        Adds to an existing pt2_compile event, but silently returns if the event doesn't exist
        or ChromiumEventLogger is not initialized.
        This function is syntactic sugar for chromium_event_logger().try_add_event_data.
        """
        if not chromium_event_log_active():
            return
        chromium_log = get_chromium_event_logger()
        chromium_log.try_add_event_data(event_name, **metadata)

    @staticmethod
    def try_(method_fn: Callable[_P, Any], *args: _P.args, **kwargs: _P.kwargs) -> None:
        """
        Special function that quietly runs a given method, returning if CHROMIUM_EVENT_LOG is None or metrics context is not set
        """
        if not chromium_event_log_active():
            return
        metrics_context = get_metrics_context()
        if not metrics_context.in_progress():
            return
        method_fn(*args, **kwargs)


_dynamo_timed_tls = threading.local()


@contextmanager
def dynamo_timed(
    key: str,
    # TODO(masneral): Deprecate this param.
    phase_name: Optional[str] = None,
    log_pt2_compile_event: bool = False,
    metadata: Optional[dict[str, object]] = None,
    dynamo_compile_column_us: Optional[str] = None,
    compile_id: Optional[CompileId] = None,
    is_backward: Optional[bool] = None,
    log_waitcounter: bool = False,
    waitcounter_name_override: Optional[str] = None,
) -> Generator[Any, None, None]:
    """
    dynamo_timed is a context manager
    By wrapping a function in dynamo_timed, we can get a few things:

    1) Optionally log timings to pt2_compile_events.
    2) Optionally log timings to CompilationMetrics (dynamo_compile).
    3) Optionally log chromium events.
    4) Optionally increment a WaitCounter.
    5) Store a record in compilation_time_metrics
       For example:

        def _foo(...):
            with dynamo_timed("_foo"):
                ...

        Would show up as an entry in our timing dict:
        OrderedDict([('_foo', [0.083690, 0.23949, 3.1425e-05])])
        This is extremely useful for granular debugging.

    Although it is tempting to use dynamo_timed as a decorator, please do not.
    In its decorator form it makes cProfile traces less useful as dynamo_timed
    suddenly becomes a bottleneck for lots of function calls (as only one parent
    pointer is recorded).

    Params:
    - key: key into compile_time_metrics. If phase_name is not provided, this is
      also the event name used for pt2_compile_events logs and chromium events.
    - phase_name: Optional override for the event name.
    - log_pt2_compile_event: Whether to log a pt2 compile event internally.
    - metadata: Extra metadata to put in pt2_compile_events.
    - dynamo_compile_column_us: If provided, updates the specified CompilationMetrics
      field to be logged to dyname_compile column. We expect all columns to be _us;
      therefore, the field name must end with "_us".
    - compile_id: In the typical case, this parameter should not be needed. Use to
      supply the compile_id for those cases where we want to log a compile_id where
      it's not naturally available, e.g., for runtime autotuning.
    - is_backward: Specify forward/backward directly when not available in a
      CompileContext, e.g., during runtime autotuning.
      that support it.
    - log_waitcounter: If set, we'll log a waitcounter of the form "pytorch.dynamo_timed.{key}"
    """
    if phase_name:
        event_name = phase_name
        fn_name = key
    else:
        event_name = key
        fn_name = None

    if key not in compilation_time_metrics:
        compilation_time_metrics[key] = []

    metrics = compilation_time_metrics[key]
    event_metadata = {}
    if metadata:
        event_metadata.update(metadata)
    if fn_name:
        event_metadata.update({"fn_name": fn_name})
    if is_backward is not None:
        event_metadata.update({"is_backward": is_backward})

    chromium_log: ChromiumEventLogger = get_chromium_event_logger()
    start_ns = time.time_ns()
    chromium_log.log_event_start(
        event_name, start_ns, event_metadata, log_pt2_compile_event, compile_id
    )

    cx_mgrs: list[typing.Any] = [
        torch.profiler.record_function(f"{key} (dynamo_timed)")
    ]
    if log_waitcounter:
        wc_name = waitcounter_name_override if waitcounter_name_override else key
        cx_mgrs.append(_WaitCounter(f"pytorch.wait_counter.{wc_name}").guard())

    is_compile_time = torch._guards.CompileContext.current_compile_id() is not None
    if dynamo_compile_column_us:
        # We're standardizing on microseconds for dynamo_compile timings.
        assert dynamo_compile_column_us.endswith("_us")

        # Track nested dynamo_timed calls that update CompilationMetrics so we can
        # bump a total duration only for the outermost metric.
        if not hasattr(_dynamo_timed_tls, "depth"):
            _dynamo_timed_tls.depth = 0
        _dynamo_timed_tls.depth += 1

        # The corresponding WaitCounters that we bump for all overheads
        if _dynamo_timed_tls.depth == 1:
            cx_mgrs.append(_WaitCounter("pytorch.wait_counter.dynamo_compile").guard())
            if not is_compile_time:
                runtime_wc = "pytorch.wait_counter.compile_runtime_overheads"
                cx_mgrs.append(_WaitCounter(runtime_wc).guard())

    try:
        with contextlib.ExitStack() as stack:
            for cx in cx_mgrs:
                stack.enter_context(cx)
            yield
    finally:
        end_ns = time.time_ns()
        time_spent_ns = end_ns - start_ns
        metrics.append(time_spent_ns / 1e9)
        chromium_log.log_event_end(
            event_name, end_ns, {}, start_ns, log_pt2_compile_event, compile_id
        )
        if dynamo_compile_column_us:
            # TODO: the events that we capture in calculate_time_spent() seem a little
            # arbitrary. Currently, it's only those fields that are present in
            # CompilationMetrics (but note that we accumulate by the associated event
            # name, not the field name in CompilationMetrics). Do we want to keep it
            # this way?
            cumulative_time_spent_ns[event_name] += time_spent_ns

            # Bump the total duration for every outer event.
            _dynamo_timed_tls.depth -= 1
            is_outer_event = _dynamo_timed_tls.depth == 0

            duration_us = time_spent_ns // 1000
            if is_compile_time:
                metrics_context = get_metrics_context()
                if metrics_context.in_progress():
                    metrics_context.increment(dynamo_compile_column_us, duration_us)
                    if is_outer_event:
                        metrics_context.increment("duration_us", duration_us)
            else:
                runtime_context = get_runtime_metrics_context()
                runtime_context.increment(dynamo_compile_column_us, duration_us)
                if is_outer_event:
                    extra = {
                        "compile_id": compile_id,
                        "is_runtime": True,
                        "is_forward": not is_backward,
                    }
                    runtime_context.increment("duration_us", duration_us, extra)


@overload
def compile_times(repr: Literal["str"], aggregate: bool = False) -> str: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def compile_times(
    repr: Literal["csv"], aggregate: bool = False
) -> tuple[list[str], list[object]]: ...


def compile_times(  # type: ignore[misc]
    repr: str = "str", aggregate: bool = False
) -> Union[str, None, tuple[list[str], list[str]]]:
    """
    Get metrics about torchdynamo frontend/backend compilation times.

    Accumulates information from functions tagged with `dynamo_timed`.

    repr='str' returns a printable string for user interaction, and 'csv'
    returns headers, rows which can be logged for output

    aggregate causes values from multiple compilations (e.g. split graphs)
    to be accumulated into one value.  If false, expect more than one value
    per metric.
    """

    def fmt_fn(values: list[float], item_fn: Callable[[float], str] = str) -> str:
        if aggregate:
            return item_fn(sum(values))
        return ", ".join(map(item_fn, values))

    if repr == "str":
        rows = [
            (k, fmt_fn(compilation_time_metrics[k], item_fn=lambda x: f"{x:.4f}"))
            for k in compilation_time_metrics
        ]
        out = "TorchDynamo compilation metrics:\n"
        out += tabulate(rows, headers=("Function", "Runtimes (s)"))
        return out
    elif repr == "csv":
        values = [
            fmt_fn(v, item_fn=lambda x: f"{x:.6f}")
            for v in compilation_time_metrics.values()
        ]
        headers = list(compilation_time_metrics.keys())
        return headers, values
    return None


@atexit.register
def dump_compile_times() -> None:
    log.info(compile_times(repr="str", aggregate=True))


tensortype_to_dtype = {
    torch.FloatTensor: (torch.float32, torch.float),
    torch.DoubleTensor: (torch.float64, torch.double),
    torch.HalfTensor: (torch.float16, torch.half),
    torch.BFloat16Tensor: (torch.bfloat16,),
    torch.ByteTensor: (torch.uint8,),
    torch.CharTensor: (torch.int8,),
    torch.LongTensor: (torch.int64, torch.long),
    torch.IntTensor: (torch.int32, torch.int),
    torch.ShortTensor: (torch.int16, torch.short),
    torch.BoolTensor: (torch.bool,),
}


class DuplicateWarningChecker:
    def __init__(self, maxsize: int = 4096) -> None:
        self.maxsize = maxsize
        self.reset()

    def reset(self) -> None:
        self.set: OrderedDict[Any, Any] = OrderedDict()

    def add(self, key: Union[str, tuple[object, object]]) -> bool:
        if key in self.set:
            self.set.move_to_end(key, last=True)
            if not config.verbose:
                return False
        else:
            self.set[key] = None
            while len(self.set) > self.maxsize:
                self.set.popitem(last=False)
        return True


graph_break_dup_warning_checker = DuplicateWarningChecker()


def setup_compile_debug() -> contextlib.ExitStack:
    compile_debug = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    if compile_debug:
        return add_file_handler()

    return contextlib.ExitStack()


def reset_graph_break_dup_checker() -> None:
    graph_break_dup_warning_checker.reset()


def add_file_handler() -> contextlib.ExitStack:
    log_path = os.path.join(get_debug_dir(), "torchdynamo")
    os.makedirs(log_path, exist_ok=True)

    log_file_handler = logging.FileHandler(os.path.join(log_path, "debug.log"))
    logger = logging.getLogger("torch._dynamo")
    logger.addHandler(log_file_handler)

    exitstack = contextlib.ExitStack()
    exitstack.callback(lambda: logger.removeHandler(log_file_handler))
    return exitstack


def setup_log_file() -> contextlib.ExitStack:
    exitstack = contextlib.ExitStack()
    if config.log_file_name is not None:
        log_file_handler = logging.FileHandler(config.log_file_name)
        for logger in torch._logging._internal.get_loggers():
            logger.addHandler(log_file_handler)
            exitstack.callback(lambda: logger.removeHandler(log_file_handler))
        return exitstack

    return exitstack


def gen_record_file_name(exc: Exception, code: CodeType) -> str:
    return f"{get_debug_dir()}/error_recordings/\
{code.co_name}_{type(exc).__name__}_{code.co_firstlineno}.rec"


def write_record_to_file(filename: str, exec_record: ExecutionRecord) -> None:
    try:
        if os.path.exists(filename):
            log.warning(
                "Unable to write execution record %s; file already exists.", filename
            )
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                exec_record.dump(f)
    except Exception:
        log.exception("Unable to write execution record %s", filename)


def count_calls(g: fx.Graph) -> int:
    c = 0
    for n in g.nodes:
        if "call" in n.op:
            c += 1
    return c


def identity(x: T) -> T:
    return x


def hashable(x: Any) -> bool:
    try:
        hash(x)
        return True
    except TypeError:
        return False
    # cannot hash writable memoryview object
    except ValueError:
        return False


def nothing(*args: Any, **kwargs: Any) -> None:
    pass


class ExactWeakKeyDictionary:
    """Similar to weakref.WeakKeyDictionary, but use `is`/`id` rather than `==` to compare equality"""

    def __init__(self) -> None:
        self.values: dict[int, Any] = {}
        self.refs: dict[int, weakref.ReferenceType[Any]] = {}

    def __getitem__(self, key: Any) -> Any:
        return self.values[id(key)]

    def get(self, key: Any, default: Any = None) -> Any:
        return self.values.get(id(key), default)

    def __contains__(self, key: Any) -> bool:
        return id(key) in self.values

    def __setitem__(self, key: Any, value: Any) -> None:
        idx = id(key)
        if idx not in self.refs:
            self.refs[idx] = weakref.ref(key, lambda ref: self._remove_id(idx))
        self.values[idx] = value

    def _remove_id(self, idx: int) -> None:
        if idx in self.values:
            del self.values[idx]
        if idx in self.refs:
            del self.refs[idx]

    def clear(self) -> None:
        self.refs.clear()
        self.values.clear()


@overload
def istype(obj: object, allowed_types: type[T]) -> TypeIs[T]: ...


@overload
def istype(
    obj: object, allowed_types: tuple[type[list[T]], type[tuple[T, ...]]]
) -> TypeIs[T]: ...


@overload
def istype(obj: object, allowed_types: Iterable[type]) -> bool: ...


def istype(obj: object, allowed_types: Any) -> bool:
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types


if sys.version_info >= (3, 12):
    # Some typing classes moved to C in 3.12,
    # which no longer have the _Final mixin.
    # Check for consistency e.g. here:
    # https://github.com/python/cpython/blob/f2b82b3b3b1f8c7a81e84df35ee921e44517cf32/Lib/typing.py#L32
    _builtin_final_typing_classes = (
        typing.ParamSpecArgs,
        typing.ParamSpecKwargs,
        typing.ParamSpec,
        typing.TypeVar,
        typing.TypeVarTuple,
        typing.TypeAliasType,
    )


def is_typing(value: Any) -> bool:
    # _Final catches most of typing classes:
    #   - Any
    #   - Callable
    #   - Union (Python < 3.14)
    #   ...
    #
    # NB: we intentionally ignore classes that inherit from Generic, since they
    # can be used as both TypingVariable as well as UserDefinedClassVariable.
    if sys.version_info >= (3, 12) and isinstance(value, _builtin_final_typing_classes):
        return True
    return (
        isinstance(value, typing._Final)  # type: ignore[attr-defined]
        or value is typing.Generic
        or value is typing.Union
    )


def is_numpy_int_type(value: Any) -> bool:
    if not np:
        return False

    return istype(
        value,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    )


def is_numpy_float_type(value: Any) -> bool:
    if not np:
        return False

    return istype(
        value,
        (
            np.float16,
            np.float32,
            np.float64,
        ),
    )


@overload
def is_lru_cache_wrapped_function(
    value: Callable[..., T],
) -> TypeGuard[functools._lru_cache_wrapper[T]]: ...


@overload
def is_lru_cache_wrapped_function(
    value: Any,
) -> TypeGuard[functools._lru_cache_wrapper[Any]]: ...


def is_lru_cache_wrapped_function(
    value: Any,
) -> bool:
    return isinstance(value, functools._lru_cache_wrapper) and is_function(
        inspect.getattr_static(value, "__wrapped__")
    )


_FuncTypes: TypeAlias = Union[
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
]


def is_function_or_wrapper(
    value: Any,
) -> TypeIs[Union[_FuncTypes, torch._ops.OpOverloadPacket, torch._ops.OpOverload]]:
    return is_function(value) or isinstance(
        value, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
    )


def is_function(
    value: Any,
) -> TypeIs[_FuncTypes]:
    return isinstance(
        value,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodDescriptorType,
            types.WrapperDescriptorType,
        ),
    )


cmp_name_to_op_mapping = {
    "__eq__": operator.eq,
    "__ne__": operator.ne,
    "__lt__": operator.lt,
    "__le__": operator.le,
    "__gt__": operator.gt,
    "__ge__": operator.ge,
}


cmp_name_to_op_str_mapping = {
    "__eq__": "==",
    "__ne__": "!=",
    "__lt__": "<",
    "__le__": "<=",
    "__gt__": ">",
    "__ge__": ">=",
}


def is_wrapper_or_member_descriptor(
    value: Any,
) -> TypeIs[
    Union[
        types.GetSetDescriptorType,
        types.MethodDescriptorType,
        types.WrapperDescriptorType,
        types.MemberDescriptorType,
        types.MethodWrapperType,
    ]
]:
    return isinstance(
        value,
        (
            # set up by PyGetSetDef
            types.GetSetDescriptorType,
            # set by PyMethodDef, e.g. list.append
            types.MethodDescriptorType,
            # slots - list.__add__
            types.WrapperDescriptorType,
            # set up by PyMemberDef
            types.MemberDescriptorType,
            # wrapper over C functions
            types.MethodWrapperType,
        ),
    )


def unwrap_if_wrapper(fn: Any) -> Any:
    return unwrap_with_attr_name_if_wrapper(fn)[0]


def unwrap_with_attr_name_if_wrapper(fn: Any) -> tuple[Any, Optional[str]]:
    # TODO(anijain2305) - Investigate if we can get rid of this function
    # unpack @torch._dynamo.optimize()(fn) wrapped function
    if is_function(fn) and inspect.getattr_static(fn, "_torchdynamo_inline", False):
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        attr_name = "_torchdynamo_inline"
    else:
        attr_name = None
    return fn, attr_name


def is_numpy_ndarray(value: Any) -> TypeGuard[np.ndarray]:  # type: ignore[type-arg]
    if not np:
        return False

    return istype(value, np.ndarray)


def istensor(obj: Any) -> bool:
    """Check of obj is a tensor"""
    tensor_list: tuple[type, ...] = (
        torch.Tensor,
        torch.nn.Parameter,
        *config.traceable_tensor_subclasses,
    )
    tensor_list = tensor_list + (torch._subclasses.FakeTensor,)
    return istype(obj, tensor_list)


def is_lazy_module(mod: Any) -> bool:
    return isinstance(mod, LazyModuleMixin)


@functools.lru_cache(4096)
def print_once(*args: Any) -> None:
    print(*args)


def make_cell(val: Any = None) -> types.CellType:
    """Some black magic to create a cell object that usually only exists in a closure"""
    x = val

    def f() -> Any:
        return x

    assert f.__closure__ is not None and len(f.__closure__) == 1
    return f.__closure__[0]


def proxy_args_kwargs(args: Any, kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    try:
        proxy_args = tuple(arg.as_proxy() for arg in args)
        proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return proxy_args, proxy_kwargs
    except NotImplementedError as e:
        from .exc import unimplemented
        from .variables.base import typestr

        unimplemented(
            gb_type="Failed to convert args/kwargs to proxy",
            context=f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}",
            explanation="Missing `as_proxy()` implementation for some arg/kwarg.",
            hints=[],
            from_exc=e,
        )


def to_int_ms(v: Optional[float]) -> Optional[int]:
    return None if v is None else int(v * 1000)


# float64 timestamp has a quarter microsecond precision in 2024, so while
# this is suboptimal we shouldn't meaningfully lose precision
def to_int_us(v: Optional[float]) -> Optional[int]:
    return None if v is None else int(v * 1_000_000)


# Version field added to every log. Increment to make it easier to distinguish new
# vs. old entries when you make a substantive change to how the logs are populated.
LOG_FORMAT_VERSION = 3


@dataclasses.dataclass
class CompilationMetrics:
    compile_id: Optional[str] = None
    frame_key: Optional[str] = None
    co_name: Optional[str] = None
    co_filename: Optional[str] = None
    co_firstlineno: Optional[int] = None
    cache_size: Optional[int] = None
    accumulated_cache_size: Optional[int] = None
    guard_count: Optional[int] = None
    shape_env_guard_count: Optional[int] = None
    graph_op_count: Optional[int] = None
    graph_node_count: Optional[int] = None
    graph_input_count: Optional[int] = None
    start_time: Optional[float] = None
    entire_frame_compile_time_s: Optional[float] = None
    backend_compile_time_s: Optional[float] = None
    inductor_compile_time_s: Optional[float] = None
    code_gen_time_s: Optional[float] = None
    fail_type: Optional[str] = None
    fail_reason: Optional[str] = None
    fail_user_frame_filename: Optional[str] = None
    fail_user_frame_lineno: Optional[int] = None
    non_compliant_ops: Optional[set[str]] = None
    compliant_custom_ops: Optional[set[str]] = None
    restart_reasons: Optional[set[str]] = None
    dynamo_time_before_restart_s: Optional[float] = None
    stack_trace: Optional[list[str]] = None
    exception_stack_trace: Optional[list[str]] = None
    graph_node_shapes: Optional[str] = None
    # Sometimes, we will finish analyzing a frame but conclude we don't want
    # to install any guarded code.  True means we actually decided to install
    # a compiled frame
    has_guarded_code: Optional[bool] = None
    remote_cache_time_saved_s: Optional[float] = None
    structured_logging_overhead_s: Optional[float] = None
    config_suppress_errors: Optional[bool] = None
    config_inline_inbuilt_nn_modules: Optional[bool] = None
    specialize_float: Optional[bool] = None
    dynamo_config: Optional[str] = None
    compiler_config: Optional[str] = None
    is_forward: Optional[bool] = None
    num_triton_bundles: Optional[int] = None
    remote_fx_graph_cache_get_time_ms: Optional[int] = None
    remote_fx_graph_cache_put_time_ms: Optional[int] = None
    start_time_us: Optional[int] = None
    duration_us: Optional[int] = None
    dynamo_cumulative_compile_time_us: Optional[int] = None
    aot_autograd_cumulative_compile_time_us: Optional[int] = None
    inductor_cumulative_compile_time_us: Optional[int] = None
    inductor_code_gen_cumulative_compile_time_us: Optional[int] = None
    triton_compile_time_us: Optional[int] = None
    runtime_cudagraphify_time_us: Optional[int] = None
    runtime_triton_autotune_time_us: Optional[int] = None
    dynamo_compile_time_before_restart_us: Optional[int] = None
    distributed_ephemeral_timeout_us: Optional[int] = None
    structured_logging_overhead_us: Optional[int] = None
    remote_fx_graph_cache_get_time_us: Optional[int] = None
    remote_fx_graph_cache_put_time_us: Optional[int] = None
    backward_cumulative_compile_time_us: Optional[int] = None
    end_time_us: Optional[int] = None
    pre_grad_pass_time_us: Optional[int] = None
    post_grad_pass_time_us: Optional[int] = None
    joint_graph_pass_time_us: Optional[int] = None
    log_format_version: int = LOG_FORMAT_VERSION
    inductor_config: Optional[str] = None
    remote_cache_version: Optional[int] = None
    inductor_fx_remote_cache_hit_count: Optional[int] = None
    inductor_fx_remote_cache_miss_count: Optional[int] = None
    inductor_fx_remote_cache_backend_type: Optional[str] = None
    inductor_fx_remote_cache_hit_keys: Optional[str] = None
    inductor_fx_remote_cache_miss_keys: Optional[str] = None
    cuda_version: Optional[str] = None
    triton_version: Optional[str] = None
    feature_usage: Optional[dict[str, bool]] = None
    compile_time_autotune_time_us: Optional[int] = None
    is_runtime: Optional[bool] = False
    gc_time_us: Optional[int] = None
    tensorify_float_attempt: Optional[bool] = None
    tensorify_float_success: Optional[bool] = None
    tensorify_float_failure: Optional[set[str]] = None
    guard_latency_us: Optional[float] = None
    recompile_reason: Optional[str] = None
    num_graph_breaks: Optional[int] = None
    triton_kernel_compile_times_us: Optional[str] = None
    ir_count: Optional[int] = None
    cudagraph_skip_reason: Optional[str] = None
    python_version: Optional[str] = None
    pgo_put_remote_code_state_time_us: Optional[int] = None
    pgo_get_remote_code_state_time_us: Optional[int] = None
    # The number of elements within parameters. This is classically what people
    # think of when they think of parameters in a ML model.
    param_numel: Optional[int] = None
    # The number of elements counted by bytes - i.e. a float32 is 4 bytes
    # per element.
    param_bytes: Optional[int] = None
    # The number of parameters counted by fields. This is mostly a proxy for
    # the number of distinct type of params.
    param_count: Optional[int] = None
    recompile_user_contexts: Optional[set[str]] = None
    inline_inbuilt_nn_modules_candidate: Optional[bool] = False
    pytorch_version: Optional[str] = None
    inductor_provenance: Optional[set[str]] = None

    @classmethod
    def create(cls, metrics: dict[str, Any]) -> CompilationMetrics:
        """
        Factory method to create a CompilationMetrics from a dict of fields.
        Includes the logic to add legacy fields and any pre-processing, e.g.,
        we transform some fields to comma-separated strings for scuba logging.
        """

        def us_to_s(metric: Optional[int]) -> Optional[float]:
            return metric / 1e6 if metric is not None else None

        def us_to_ms(metric: Optional[int]) -> Optional[int]:
            return metric // 1000 if metric is not None else None

        def collection_to_str(metric: Optional[Any]) -> Optional[str]:
            def safe_str(item: Any) -> str:
                try:
                    return str(item)
                except Exception:
                    return "<unknown>"

            if metric is None:
                return None

            if not isinstance(metric, (set, list)):
                return "<unknown>"

            return ",".join(safe_str(item) for item in sorted(metric))

        def collection_to_json_str(metric: Optional[Any]) -> Optional[str]:
            if metric is None:
                return None
            try:
                return json.dumps(list(metric))
            except Exception:
                return "<unknown>"

        # TODO: The following are legacy fields, populated from the fields that replace
        # them. Remove these when we decide we can really deprecate them.
        legacy_metrics = {
            "start_time": us_to_s(metrics.get("start_time_us")),
            "entire_frame_compile_time_s": us_to_s(
                metrics.get("dynamo_cumulative_compile_time_us")
            ),
            "backend_compile_time_s": us_to_s(
                metrics.get("aot_autograd_cumulative_compile_time_us")
            ),
            "inductor_compile_time_s": us_to_s(
                metrics.get("inductor_cumulative_compile_time_us")
            ),
            "code_gen_time_s": us_to_s(
                metrics.get("inductor_code_gen_cumulative_compile_time_us")
            ),
            "remote_cache_time_saved_s": us_to_s(
                metrics.get("distributed_ephemeral_timeout_us")
            ),
            "remote_fx_graph_cache_get_time_ms": us_to_ms(
                metrics.get("remote_fx_graph_cache_get_time_us")
            ),
            "remote_fx_graph_cache_put_time_ms": us_to_ms(
                metrics.get("remote_fx_graph_cache_put_time_us")
            ),
            "structured_logging_overhead_s": us_to_s(
                metrics.get("structured_logging_overhead_us")
            ),
        }

        all_metrics = {**legacy_metrics, **metrics}

        # Processing before logging:
        all_metrics["inductor_fx_remote_cache_hit_keys"] = collection_to_str(
            all_metrics.get("inductor_fx_remote_cache_hit_keys")
        )
        all_metrics["inductor_fx
```



## High-Level Overview

"""Utility functions and classes used throughout the TorchDynamo system.This module contains a collection of helper utilities used by various parts of Dynamo for:- Performance metrics collection and reporting- Compilation timing and debugging- Graph manipulation and tensor operations- Runtime guards and checks- Common data structure operations- Testing and development toolsThis is an internal module that provides shared functionality used across the Dynamo codebase.

This Python file contains 33 class(es) and 294 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ReInplaceTrigger`, `ReinplaceCounters`, `CompileEventLogLevel`, `CompileEventLogger`, `DuplicateWarningChecker`, `ExactWeakKeyDictionary`, `CompilationMetrics`, `TypeSafeSerializer`, `ChromiumEventLogger`, `CleanupHook`, `CleanupManager`, `Marker`, `TensorStaticReason`, `numpy_to_tensor_wrapper`, `numpy_method_wrapper`, `numpy_operator_wrapper`, `_Anchors`, `Invalid`, `GmWrapper`, `Lit`

**Functions defined**: `add_missed_bytes`, `add_missed_opportunities`, `clear`, `get_total_missed`, `get_total_missed_bytes`, `log`, `tabulate`, `increment_frame`, `reset_frame_count`, `register_hook_for_recompile_user_context`, `get_hook_for_recompile_user_context`, `increment_op_count`, `calculate_time_spent`, `print_time_report`, `foo`, `get_metrics_context`, `get_runtime_metrics_context`, `log_instant_event`, `add_data`, `add_toplevel`

**Key imports**: annotations, atexit, collections, contextlib, copy, dataclasses, datetime, dis, enum, functools


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `atexit`
- `collections`
- `contextlib`
- `copy`
- `dataclasses`
- `datetime`
- `dis`
- `enum`
- `functools`
- `gc`
- `importlib`
- `inspect`
- `itertools`
- `json`
- `linecache`
- `logging`
- `math`
- `operator`
- `os`
- `re`
- `sys`
- `textwrap`
- `threading`
- `time`
- `traceback`
- `types`
- `typing`
- `uuid`
- `warnings`


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
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

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
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
