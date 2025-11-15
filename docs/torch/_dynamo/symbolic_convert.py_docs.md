# Documentation: `torch/_dynamo/symbolic_convert.py`

## File Metadata

- **Path**: `torch/_dynamo/symbolic_convert.py`
- **Size**: 214,847 bytes (209.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Core module responsible for converting Python bytecode into TorchDynamo's symbolic execution format.

This module implements the bytecode-level tracing system that allows TorchDynamo to analyze
and transform Python code. It converts Python bytecode instructions into a symbolic format
that tracks the flow of tensors and other values through the program.

Key components:
- InstructionTranslatorBase: Base class for converting bytecode to symbolic execution
- InstructionTranslator: Main translator for function bytecode
- InliningInstructionTranslator: Handles inlining of called functions
- SpeculationLog: Manages state for speculative execution and rollback

The symbolic conversion process handles:
- Control flow (loops, conditionals, etc.)
- Function inlining and call stack management
- Tracking of program values and side effects
- Graph breaks and resumption points
- Exception handling and stack frame management

This is a core part of TorchDynamo's tracing system that enables ahead-of-time
optimization of PyTorch programs.
"""

from __future__ import annotations

import collections
import collections.abc
import contextlib
import copy
import dataclasses
import dis
import functools
import importlib
import inspect
import itertools
import linecache
import logging
import operator
import re
import sys
import threading
import traceback
import types
import weakref
from collections import deque
from traceback import StackSummary
from typing import Any, cast, NoReturn, Optional, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import TypeIs

import torch
import torch._logging
from torch._dynamo.exc import ObservedException, TensorifyScalarRestartAnalysis
from torch._guards import tracing, TracingContext
from torch._logging.structured import dump_file
from torch.fx.experimental.symbolic_shapes import guard_bool
from torch.utils._functools import cache_method

from . import (
    config,
    exc,
    graph_break_hints,
    logging as torchdynamo_logging,
    trace_rules,
    variables,
)
from .bytecode_analysis import (
    get_indexof,
    JUMP_OPNAMES,
    livevars_analysis,
    propagate_line_nums,
)
from .bytecode_transformation import (
    cleaned_instructions,
    create_binary_slice,
    create_call_function,
    create_call_function_ex,
    create_copy,
    create_dup_top,
    create_instruction,
    create_jump_absolute,
    create_rot_n,
    create_swap,
    get_code_keys,
    Instruction,
    is_generator,
    is_jump_absolute,
    unique_id,
)
from .code_context import code_context
from .codegen import PyCodegen
from .exc import (
    ArgsMismatchError,
    BackendCompilerFailed,
    collapse_resume_frames,
    format_graph_break_message,
    format_loop_skip_frame_message,
    format_skip_frame_message,
    get_stack_above_dynamo,
    ResumePrologueTracingError,
    StepUnsupported,
    unimplemented,
    Unsupported,
)
from .funcname_cache import get_funcname
from .guards import GuardBuilder, install_guard
from .output_graph import GraphCompileReason, OutputGraph, StackLocalsMetadata
from .polyfills import impl_CONTAINS_OP_fallback
from .replay_record import DummyModule, ExecutionRecorder
from .resume_execution import (
    ContinueExecutionCache,
    IS_TRACING_RESUME_PROLOGUE_VARNAME,
    ReenterWith,
)
from .source import (
    AttrSource,
    DictGetItemSource,
    GlobalSource,
    GlobalWeakRefSource,
    LocalCellSource,
    LocalSource,
    SkipGuardSource,
    Source,
)
from .trace_rules import is_builtin_constant, is_forbidden
from .utils import (
    _get_error_on_graph_break,
    counters,
    get_fake_value,
    get_instruction_source_311,
    get_metrics_context,
    graph_break_dup_warning_checker,
    istype,
    LazyString,
    proxy_args_kwargs,
)
from .variables.base import typestr, ValueMutationNew, VariableTracker
from .variables.builder import FrameStateSizeEntry, VariableBuilder, wrap_fx_proxy
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable
from .variables.ctx_manager import (
    ContextWrappingVariable,
    GenericContextWrappingVariable,
    WithEnterFunctionVariable,
    WithExitFunctionVariable,
)
from .variables.dicts import ConstDictVariable, SetVariable
from .variables.functions import (
    BaseUserFunctionVariable,
    LocalGeneratorFunctionVariable,
    LocalGeneratorObjectVariable,
    NestedUserFunctionVariable,
    SkipFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .variables.iter import MAX_ITERATOR_LIMIT
from .variables.lazy import LazyVariableTracker
from .variables.lists import (
    BaseListVariable,
    IteratorVariable,
    ListIteratorVariable,
    ListVariable,
    SliceVariable,
    TupleVariable,
)
from .variables.misc import (
    CellVariable,
    ExceptionVariable,
    GetAttrVariable,
    NullVariable,
    PythonModuleVariable,
    UnknownVariable,
)
from .variables.nn_module import NNModuleVariable, UnspecializedNNModuleVariable
from .variables.streams import SymbolicStreamState
from .variables.tensor import supported_comparison_ops, SymNodeVariable, TensorVariable
from .variables.torch_function import (
    SymbolicTorchFunctionState,
    TorchFunctionModeVariable,
)
from .variables.user_defined import (
    RemovableHandleVariable,
    UserDefinedClassVariable,
    UserDefinedExceptionClassVariable,
    UserDefinedExceptionObjectVariable,
    UserDefinedObjectVariable,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch._subclasses.fake_tensor import FakeTensorMode

    from .package import CompilePackage

log = logging.getLogger(__name__)
graph_break_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")
trace_call_log = torch._logging.getArtifactLogger(__name__, "trace_call")
trace_source_log = torch._logging.getArtifactLogger(__name__, "trace_source")
trace_bytecode_log = torch._logging.getArtifactLogger(__name__, "trace_bytecode")
tls = threading.local()
compare_op_handlers: dict[str, Any] = {
    k: BuiltinVariable(v).call_function for k, v in supported_comparison_ops.items()
}
handle_contains = BuiltinVariable(operator.contains).call_function
handle_not = BuiltinVariable(operator.not_).call_function
compare_op_handlers["in"] = lambda tx, args, _: handle_contains(
    tx, [*reversed(args)], {}
)
compare_op_handlers["not in"] = lambda tx, args, _: handle_not(
    tx, [handle_contains(tx, [*reversed(args)], {})], {}
)

PT2_ISSUE_TRACKER_URL = "https://github.com/pytorch/pytorch/issues/new?&labels=oncall%3A+pt2&projects=&template=pt2-bug-report.yml"

ExceptionVals: TypeAlias = Union[
    variables.ExceptionVariable,
    UserDefinedExceptionClassVariable,
    UserDefinedExceptionObjectVariable,
]


@functools.cache
def _import_module(name: str) -> types.ModuleType:
    """
    Import the named module and cache the result. importlib.import_module()
    seems to do some filesystem checking to validate the name so not caching
    this can be slow.
    """
    return importlib.import_module(name)


@dataclasses.dataclass
class SpeculationEntry:
    filename: str
    lineno: int
    instruction_pointer: int
    inst: Instruction  # for debugging only
    _failed: bool = False
    error_on_graph_break: Optional[bool] = None
    reason: Optional[GraphCompileReason] = None

    def fail_and_restart_analysis(self, error_on_graph_break: bool) -> None:
        """
        Start tracing of the current frame over again, and don't take this branch.
        """
        self._failed = True
        self.error_on_graph_break = error_on_graph_break
        if self.reason is not None:
            restart_reason = self.reason.reason
        else:
            restart_reason = "Unknown fail_and_restart_analysis"
        raise exc.SpeculationRestartAnalysis(restart_reason=restart_reason)

    def failed(self, tx: InstructionTranslatorBase) -> bool:
        if self._failed:
            assert self.error_on_graph_break is not None
            tx.error_on_graph_break = self.error_on_graph_break
            return True
        return False


@dataclasses.dataclass
class SpeculationLog:
    """
    SpeculationLog replaces the prior copy_graphstate/restore_graphstate
    checkpointing.  Rather than saving/restoring state, we restart the
    dynamo conversion process over from the beginning -- but when we
    hit the start of the speculation that failed, we instead generate
    a graph break.
    """

    entries: list[SpeculationEntry] = dataclasses.field(default_factory=list)
    index: int = 0

    def restart(self) -> None:
        self.index = 0

    def clear(self) -> None:
        self.entries.clear()
        self.index = 0

    def next(
        self, filename: str, lineno: int, instruction_pointer: int, inst: Instruction
    ) -> SpeculationEntry:
        """
        Lookup or create a SpeculationEntry() that is shared across
        RestartAnalysis calls.  Args are used only for debug checks.
        """
        if len(self.entries) == self.index:
            self.entries.append(
                SpeculationEntry(filename, lineno, instruction_pointer, inst)
            )
        entry = self.entries[self.index]
        prev_entry_msg = ""
        if self.index != 0:
            prev_entry = self.entries[self.index - 1]
            prev_entry_msg = (
                f"Previous instruction: {prev_entry.filename}:{prev_entry.lineno}"
                f"({prev_entry.inst.opname} @ {prev_entry.instruction_pointer})\n"
            )
        if not (
            entry.instruction_pointer == instruction_pointer
            and entry.filename == filename
            and entry.lineno == lineno
        ):
            raise SpeculationLogDivergence(
                f"""
SpeculationLog diverged at index {self.index} (log had {len(self.entries)} entries):
- Expected: {entry.filename}:{entry.lineno} ({entry.inst.opname} at ip={entry.instruction_pointer})
- Actual: {filename}:{lineno} ({inst.opname} at ip={instruction_pointer})
{prev_entry_msg}
There are two usual reasons why this may have occurred:
- When Dynamo analysis restarted, the second run took a different path than
  the first.  If this occurred, the previous instruction is the critical instruction that
  behaved differently.
- Speculation entries are only added under certain conditions (as seen in
  step()), e.g., there must exist operators in the graph; those conditions may
  have changed on restart.

If this divergence was intentional, clear the speculation log before restarting (do NOT
do this for graph breaks, you will infinite loop).

Otherwise, please submit a bug report, ideally including the contents of TORCH_LOGS=+dynamo
"""
            )
        self.index += 1
        return entry


@dataclasses.dataclass
class LocalState:
    automatic_dynamic: dict[str, FrameStateSizeEntry] = dataclasses.field(
        default_factory=dict
    )

    def render(self) -> str:
        return "\n".join(
            f"{k}: {v.render()}" for k, v in self.automatic_dynamic.items()
        )


# Mutable box that is shared across restarts
@dataclasses.dataclass
class DistributedState:
    compile_pg: Any
    local_state: LocalState
    all_states: Optional[list[LocalState]] = None


class TensorifyState:
    # These are the set of string symfloats names (eg. "zf0") that we collect
    # from the tensorify_python_scalars.py joint fx pass to inform us about
    # which float inputs we should specialize when we restart analysis.
    force_specializations: set[str] = set()

    @classmethod
    def specialize(cls, index: str) -> None:
        cls.force_specializations.add(index)

    @classmethod
    def should_specialize(cls, index: str) -> bool:
        return index in cls.force_specializations

    @classmethod
    def clear(cls) -> None:
        cls.force_specializations.clear()

    @classmethod
    def empty(cls) -> bool:
        return len(cls.force_specializations) == 0


@functools.cache
def _step_logger() -> Callable[..., None]:
    return torchdynamo_logging.get_step_logger(log)


@contextlib.contextmanager
def save_and_restart_speculation_log(
    tx: InstructionTranslatorBase,
) -> Generator[None, None, None]:
    # When reconstructing a generator after a graph break, we advance it until
    # it is fully exhausted. This process adds new entries to the speculation
    # log that were not previously observed. Without temporarily clearing the
    # speculation log, this could lead to a divergence error.

    entries = tx.speculation_log.entries
    index = tx.speculation_log.index
    try:
        tx.speculation_log.entries = []
        tx.speculation_log.index = 0
        yield
    finally:
        tx.speculation_log.entries = entries
        tx.speculation_log.index = index


@contextlib.contextmanager
def temporarely_allow_writes_to_output_graph(
    tx: InstructionTranslatorBase,
) -> Generator[None, None, None]:
    try:
        tmp = tx.output.should_exit
        tx.output.should_exit = False
        yield
    finally:
        tx.output.should_exit = tmp


@dataclasses.dataclass
class BlockStackEntry:
    # Current instruction that pushes something to block_stack
    inst: Instruction
    target: Instruction
    stack_index: int
    with_context: Optional[
        Union[ContextWrappingVariable, GenericContextWrappingVariable]
    ] = None

    def can_restore(self) -> bool:
        return self.with_context is not None

    def resume_fn(self) -> ReenterWith:
        assert self.stack_index is not None
        if (
            self.with_context
            and hasattr(self.with_context, "target_values")
            and self.with_context.target_values
        ):
            return ReenterWith(
                self.stack_index - 1, tuple(self.with_context.target_values)
            )
        else:
            return ReenterWith(self.stack_index - 1)

    def exit(
        self, tx: InstructionTranslatorBase, is_graph_break: bool
    ) -> VariableTracker | None:
        assert self.with_context is not None
        if (
            is_graph_break and self.with_context.exit_on_graph_break()
        ) or not is_graph_break:
            return self.with_context.exit(tx)  # type: ignore[arg-type]
        return None


class SpeculationLogDivergence(AssertionError):
    pass


class ReturnValueOp(Exception):
    pass


class YieldValueOp(Exception):
    """
    Signal to the symbolic tracer to stop and return control flow to the
    caller
    """


def stack_op(fn: Callable[..., object]) -> Callable[..., Any]:
    nargs = len(inspect.signature(fn).parameters)
    fn_var = BuiltinVariable(fn)

    @functools.wraps(fn)
    def impl(self: InstructionTranslator, inst: Instruction) -> None:
        self.push(fn_var.call_function(self, self.popn(nargs), {}))

    return impl


def is_stdlib(mod: object) -> bool:
    if not isinstance(mod, types.ModuleType):
        return False
    return mod.__name__.split(".")[0] in sys.stdlib_module_names


@functools.cache
def get_assert_bytecode_sequence(with_msg: bool) -> list[str]:
    if with_msg:

        def fn(x: Any) -> None:
            assert x, "msg"
    else:

        def fn(x: Any) -> None:
            assert x

    insts = [inst.opname for inst in dis.get_instructions(fn)]

    # expect to find POP_JUMP_[FORWARD_]IF_TRUE
    begin_idx = next(i for i, inst in enumerate(insts) if inst.startswith("POP_JUMP"))
    end_idx = insts.index("RAISE_VARARGS")

    return insts[begin_idx + 1 : end_idx + 1]


def _detect_and_normalize_assert_statement(
    self: InstructionTranslatorBase,
    truth_fn: Callable[[object], bool],
    push: bool,
) -> bool:
    # Detect if this jump instruction is assert and normalize the assert
    # by pushing dummy error message when nothing is given.
    #
    # Python 3.9-3.13 assertion is in following format (minus small differences)
    # 18 POP_JUMP_IF_TRUE       28
    # 20 LOAD_ASSERTION_ERROR
    # 22 LOAD_CONST               3 ('Assert message') -> optional instruction
    # 24 CALL_FUNCTION            1                    -> optional instruction
    # 26 RAISE_VARARGS

    if (truth_fn is not operator.truth) or push:
        return False

    assert isinstance(self.instruction_pointer, int)
    current_instruction_pointer = self.instruction_pointer

    for with_msg in (False, True):
        assert_insts = get_assert_bytecode_sequence(with_msg)
        cur_insts = self.instructions[
            current_instruction_pointer : current_instruction_pointer
            + len(assert_insts)
        ]
        cur_insts = [inst.opname for inst in cur_insts]
        if cur_insts == assert_insts:
            if with_msg:
                load_const_idx = assert_insts.index("LOAD_CONST")
                error_msg = self.instructions[
                    current_instruction_pointer + load_const_idx
                ].argval
            else:
                error_msg = "assertion error"
            self.push(ConstantVariable.create(error_msg))
            return True

    return False


explain = False


# [NOTE] graph break handling in symbolic_convert
# There are 4 possible graph break cases that InstructionTranslatorBase handles:
#   1. Regular graph breaks from CALL, BINARY_SUBSCR, etc. (implemented by break_graph_if_unsupported)
#   2. Data-dependent condition graph breaks (implemented by generic_jump)
#   3. STORE_ATTR graph breaks (implemented in InstructionTranslatorBase.STORE_ATTR)
#   4. All other unhandled graph breaks - unsupported step graph breaks (implemented in InstructionTranslatorBase.step)
#
# Graph breaks are handled in the following manner:
#   1. The Unsupported exception is caught. If we cannot compile a partial graph (should_compile_partial_graph() is False),
#      then propagate the exception upward. For unsupported step graph breaks, the condition to abort partial compilation is
#      more restrictive (see InstructionTranslatorBase.step).
#   2. If the Unsupported exception escapes symbolic_convert.py, then we are done.
#      Otherwise, we want to attempt partial compilation.
#      Log the graph break via log_graph_break. If we're handling a data-dependent graph break (type 2.), then we can immediately
#      codegen the compiled graph and resume function and we're done. This is because the jump instruction we graph break on is
#      limited in how it can manipulate Python state (say, in comparison, to CALL, which can modify Python state arbitrarily).
#      Otherwise, we need to restart compilation. We need to restart because by processing the unsupported instruction,
#      we may have modified the VariableTrackers, and we need all of our VariableTrackers to be in the state BEFORE tracing the
#      unsupported instruction.
#   3. During the first compilation, we updated a speculation log, indicating points in the code that we can resume from.
#      On the second compilation, we will stop tracing at the first speculation log that fails. Then we compile the partial
#      graph and resume function.
#
# Logging invariants:
#   1. No logs need to be made if Unsupported escapes symbolic_convert.py. Python's default exception printing will
#      print out all of the necessary information and no partial compilation will be attempted.
#   2. log_graph_break should be called as soon as Unsupported is caught and we determined we want to partial compile.
#      This always happens on the first compilation, NOT the restart handling this graph
#   3. Any compile_subgraph call should be preceded immediately by a log in the form of "... triggered compile".


def generic_jump(
    truth_fn: Callable[[object], bool], push: bool
) -> Callable[[InstructionTranslatorBase, Instruction], None]:
    # graph break message fields for data dependent branching
    _gb_type = "Data-dependent branching"
    _explanation = (
        "Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). "
        "Dynamo does not support tracing dynamic control flow."
    )
    _hints = [
        *graph_break_hints.FUNDAMENTAL,
        "Use `torch.cond` to express dynamic control flow.",
    ]

    def jump_graph_break(
        self: InstructionTranslatorBase,
        inst: Instruction,
        value: VariableTracker,
        extra_msg: str = "",
    ) -> None:
        assert self.should_compile_partial_graph()
        self.log_graph_break(
            self.code_options,
            reason=format_graph_break_message(
                gb_type=_gb_type,
                context=f"attempted to jump with {value}",
                explanation=_explanation,
                hints=_hints,
            ),
        )
        # compile a partial subgraph prefix then jump into user code
        if self.maybe_has_backedge():
            msg = format_loop_skip_frame_message(
                self.f_code,
                "".join(traceback.format_list([self.frame_summary()])),
            )
            log.info(msg)
            raise exc.SkipFrame(msg)

        self.push(value)
        log.debug("generic_jump triggered compile")
        all_stack_locals_metadata = self.output.compile_subgraph(
            self,
            reason=GraphCompileReason(
                f"generic_jump {typestr(value)}{extra_msg}", [self.frame_summary()]
            ),
            stack_pops=1,
        )
        self.pop()

        if_next = self.create_call_resume_at(
            self.next_instruction,
            all_stack_locals_metadata,
        )
        if push:
            self.push(value)
        assert inst.target is not None
        if_jump = self.create_call_resume_at(
            inst.target,
            all_stack_locals_metadata,
        )

        if sys.version_info >= (3, 13):
            # 3.13 requires stack[-1] to be bool type
            self.output.add_output_instructions([create_instruction("TO_BOOL")])

        jump_inst = create_instruction(inst.opname, target=if_jump[0])
        jump_inst.copy_positions(inst)
        self.output.add_output_instructions([jump_inst] + if_next + if_jump)

    def inner(self: InstructionTranslatorBase, inst: Instruction) -> None:
        value: VariableTracker = self.pop()
        if (
            config.rewrite_assert_with_torch_assert
            and _detect_and_normalize_assert_statement(self, truth_fn, push)
        ):
            error_msg: VariableTracker = self.pop()
            # Skip over things like `assert True`
            if value.is_python_constant():
                if bool(value.as_python_constant()):
                    return self.jump(inst)
                elif self.should_compile_partial_graph():
                    jump_graph_break(self, inst, value)
                else:
                    unimplemented(
                        gb_type="Data-dependent assertion failed (cannot compile partial graph)",
                        context=f"value: {value}",
                        explanation="Dynamo has determined when encountering a data-dependent assert failure "
                        "that it should not compile the partial graph.",
                        hints=[
                            *graph_break_hints.FUNDAMENTAL,
                            "Use `torch._assert()` to raise a hard AssertionError when the check fails. "
                            "This error will propagate back the user code "
                            "that called the compiled function (i.e. Dynamo will not trace any exception handling).",
                            "Remove the assert statement.",
                            "Move the assert statement outside of any context managers in order to graph break with "
                            "partial graph compilation (if fullgraph=False).",
                        ],
                    )

            # TODO maybe should respect DtoH sync intention of users later??
            # Manually insert torch._assert_async instead of python assert and jump over
            # assert related instructions as we don't need them anymore.

            # if we see Tensor as assert statement, no need to call scalar_tensor
            if isinstance(value, TensorVariable):
                self.output.create_proxy(
                    "call_function",
                    torch._assert_async,
                    *proxy_args_kwargs((value, error_msg), {}),
                )
                self.jump(inst)
                return

            if isinstance(value, SymNodeVariable):
                # if the assertion is normal shape expression.
                # just install guard and bail out.
                sym_expr = value.sym_num
                if not isinstance(sym_expr, torch.SymBool):
                    sym_expr = sym_expr != 0

                result = torch.fx.experimental.symbolic_shapes.expect_true(sym_expr)
                if not result:
                    unimplemented(
                        gb_type="Assertion failed on symbolic shapes",
                        context=str(sym_expr),
                        explanation="",
                        hints=[*graph_break_hints.USER_ERROR],
                    )
                self.jump(inst)
                return

            scalar_to_tensor_proxy = self.output.create_proxy(
                "call_function", torch.scalar_tensor, *proxy_args_kwargs((value,), {})
            )

            scalar_to_tensor = wrap_fx_proxy(
                self,
                scalar_to_tensor_proxy,
                example_value=get_fake_value(scalar_to_tensor_proxy.node, self),
            )

            self.output.create_proxy(
                "call_function",
                torch._assert_async,
                *proxy_args_kwargs((scalar_to_tensor, error_msg), {}),
            )
            self.jump(inst)
            return

        if value.is_python_constant():
            # ConstDictVariable is optimized to be very lazy about insertion of
            # guards, so we have to manually insert a SEQUENCE_LENGTH guard
            # here.
            if isinstance(value, ConstDictVariable) and value.source:
                install_guard(value.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            if truth_fn(value.as_python_constant()):
                if push:
                    self.push(value)
                self.jump(inst)
        elif (
            isinstance(value, (TensorVariable)) and self.should_compile_partial_graph()
        ):
            jump_graph_break(self, inst, value)
        elif isinstance(value, NNModuleVariable):
            # Equivalent of "self.nn_module is not None"
            mod = self.output.get_submodule(value.module_key)
            if truth_fn(mod):
                if push:
                    self.push(value)
                self.jump(inst)
        elif isinstance(value, UserDefinedObjectVariable):
            try:
                x = value.var_getattr(self, "__bool__")  # type: ignore[arg-type]
            except exc.ObservedAttributeError:
                exc.handle_observed_exception(self)
                # if __bool__ is missing, trying __len__ to infer a truth value.
                try:
                    x = value.var_getattr(self, "__len__")  # type: ignore[arg-type]
                except exc.ObservedAttributeError:
                    exc.handle_observed_exception(self)
                    x = None

            # __bool__ or __len__ is function
            if isinstance(x, UserMethodVariable):
                result = x.call_function(self, [], {})  # type: ignore[arg-type, assignment]
                if isinstance(result, ConstantVariable) and isinstance(
                    result.value, (bool, int)
                ):
                    if truth_fn(result.value):
                        if push:
                            self.push(value)
                        self.jump(inst)
                elif isinstance(result, SymNodeVariable):
                    if result.evaluate_expr():
                        if push:
                            self.push(value)
                        self.jump(inst)
                else:
                    unimplemented(
                        gb_type="Data-dependent branching with non-constant __bool__",
                        context=f"method: {x}, result: {result}",
                        explanation="Attempted to perform data-dependent branching on a user-defined "
                        "object with a __bool__ method that did not return a constant.",
                        hints=[],
                    )
            # __bool__ or __len__ is non-function or not existed in the user defined object
            else:
                if truth_fn(True):
                    if push:
                        self.push(value)
                    self.jump(inst)
        elif not isinstance(value, TensorVariable) and value.has_unpack_var_sequence(
            self
        ):
            if truth_fn(len(value.unpack_var_sequence(self))):
                if push:
                    self.push(value)
                self.jump(inst)
        elif isinstance(value, SymNodeVariable):
            try:
                # if the user is branching on a SymBool, guard on it
                # if the user has code like:
                #    if size:
                #        ...
                # then they are just testing truthiness: guard that the expr != 0
                if isinstance(value.sym_num, torch.SymBool):
                    eval_result = value.evaluate_expr(self.output)
                else:
                    eval_result = guard_bool(value.sym_num != 0)
            except exc.UserError as e:
                if self.should_compile_partial_graph():
                    return jump_graph_break(self, inst, value, extra_msg=f"\n{e}")
                raise
            if truth_fn(eval_result):
                if push:
                    self.push(value)
                self.jump(inst)
        elif isinstance(value, variables.BackwardHookVariable):
            if truth_fn(True):
                if push:
                    self.push(value)
                self.jump(inst)
        else:
            from .source import is_constant_source

            if value.source is not None and is_constant_source(value.source):
                if truth_fn(value.get_real_value()):  # type: ignore[attr-defined]
                    if push:
                        self.push(value)
                    self.jump(inst)
            else:
                unimplemented(
                    gb_type="Data-dependent branching",
                    context=f"attempted to jump with {value}",
                    explanation=_explanation,
                    hints=[
                        *graph_break_hints.FUNDAMENTAL,
                        "Use `torch.cond` to express dynamic control flow.",
                    ],
                )

    return inner


def break_graph_if_unsupported(
    *, push: int
) -> Callable[
    [Callable[..., None]], Callable[[InstructionTranslatorBase, Instruction], None]
]:
    def decorator(
        inner_fn: Callable[..., None],
    ) -> Callable[[InstructionTranslatorBase, Instruction], None]:
        @functools.wraps(inner_fn)
        def wrapper(self: InstructionTranslatorBase, inst: Instruction) -> None:
            speculation = self.speculate()
            if speculation.failed(self):
                assert speculation.reason is not None
                return handle_graph_break(self, inst, speculation.reason)
            try:
                return inner_fn(self, inst)
            except Unsupported as excp:
                if self.active_generic_context_managers:
                    # We don't support graph break under GenericContextWrappingVariable,
                    # If there is, we roll back to the checkpoint and fall back.
                    excp.remove_from_stats()
                    unimplemented(
                        gb_type="Graph break under GenericContextWrappingVariable",
                        context=f"Active generic context managers: {self.active_generic_context_managers}",
                        explanation="Attempted to graph break in an active context manager(s) that doesn't support graph breaking.",
                        hints=[
                            "Move the offending context manager(s) to outside the compiled region.",
                            *graph_break_hints.CAUSED_BY_EARLIER_GRAPH_BREAK,
                        ],
                        from_exc=excp,
                    )

                if isinstance(excp, exc.UncapturedHigherOrderOpError):
                    raise

                if not self.should_compile_partial_graph():
                    raise

                self.log_graph_break(
                    self.code_options,
                    reason=str(excp),
                    user_stack=excp.real_stack,
                )

                if self.maybe_has_backedge():
                    msg = format_loop_skip_frame_message(
                        self.f_code,
                        "".join(traceback.format_list([self.frame_summary()])),
                    )
                    log.info(msg)
                    raise exc.SkipFrame(msg) from excp

                excp.remove_from_stats()
                excp.add_to_stats("graph_break")
                speculation.reason = GraphCompileReason(excp.msg, excp.real_stack)
            speculation.fail_and_restart_analysis(self.error_on_graph_break)

        def handle_graph_break(
            self: InstructionTranslatorBase,
            inst: Instruction,
            reason: GraphCompileReason,
        ) -> None:
            if (
                sys.version_info >= (3, 11)
                and sys.version_info < (3, 12)
                and inst.opname == "CALL"
            ):
                # stack effect for PRECALL + CALL is split between the two instructions
                stack_effect = dis.stack_effect(
                    dis.opmap["PRECALL"], inst.arg
                ) + dis.stack_effect(dis.opmap["CALL"], inst.arg)
            else:
                stack_effect = dis.stack_effect(inst.opcode, inst.arg)

            log.debug("%s triggered compile", inst.opname)
            all_stack_locals_metadata = self.output.compile_subgraph(
                self, reason=reason, stack_pops=push - stack_effect
            )
            cg = PyCodegen(self.output.root_tx)
            cleanup: list[Instruction] = []
            # Reconstruct the context variable CLASS in the block stack
            for b in self.block_stack:
                # Don't exit any modes we have entered,
                # output bytecode will mutate the tf mode stack accordingly
                if isinstance(b.with_context, TorchFunctionModeVariable):
                    cg.extend_output(
                        b.resume_fn().try_except_torch_function_mode(
                            cg.code_options, cleanup
                        )
                    )
                    continue
                assert b.with_context is not None
                assert isinstance(b.with_context, (ContextWrappingVariable))
                b.with_context.reconstruct_type(cg)
                cg.extend_output(b.resume_fn().try_finally(cg.code_options, cleanup))
            self.output.add_output_instructions(cg.get_instructions())
            del cg

            if sys.version_info >= (3, 11) and inst.opname == "CALL":
                kw_names = (
                    self.kw_names.as_python_constant()
                    if self.kw_names is not None
                    else ()
                )
                if len(kw_names) > 0:
                    # KW_NAMES no longer used in 3.13
                    assert sys.version_info < (3, 13)
                    self.output.add_output_instructions(
                        [create_instruction("KW_NAMES", argval=kw_names)]
                    )
                assert inst.arg is not None
                call_insts = create_call_function(inst.arg, False)
                call_insts[-1].copy_positions(inst)
                self.output.add_output_instructions(call_insts)
            else:
                # copy instruction, but without exception table data
                assert inst.target is None
                inst_copy = copy.copy(inst)
                inst_copy.exn_tab_entry = None
                self.output.add_output_instructions([inst_copy])

            self.output.add_output_instructions(cleanup)

            self.popn(push - stack_effect)
            for _ in range(push):
                self.push(UnknownVariable())
            self.output.add_output_instructions(
                self.create_call_resume_at(
                    self.next_instruction,
                    all_stack_locals_metadata,
                )
            )

        return wrapper

    return decorator


class BytecodeDispatchTableMeta(type):
    """Installs a `cls.dispatch_table` on every subclass to speed up calls to self.OPCODE()"""

    def __init__(cls: type, name: str, bases: Any, dct: Any) -> None:
        super().__init__(name, bases, dct)  # type: ignore[misc]

        def _missing(opname: str, *args: Any) -> None:
            unimplemented(
                gb_type="Missing bytecode handler",
                context=f"{opname} with args {args}",
                explanation=f"Dynamo does not know how to handle the bytecode instruction `{opname}`.",
                hints=[
                    f"Do not trace code that produces the `{opname}` bytecode instruction "
                    "(see https://docs.python.org/3/library/dis.html for bytecode semantics).",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        dispatch_table = {
            op: getattr(cls, opname, functools.partial(_missing, opname))
            for opname, op in dis.opmap.items()
        }
        # pyrefly: ignore [missing-attribute]
        cls.dispatch_table = [dispatch_table.get(i) for i in range(2**8)]


@dataclasses.dataclass
class ExceptionStack:
    """
    Exception stack that it is shared among all InstructionTranslator instances
    """

    # Exception handling in CPython is a bit confusing and some of the bytecode
    # have a slightly different behavior than what is documented. While reading
    # the documentation, is important to notice that the terms "current exception"
    # and "stack" sometimes refers to a C variable with the same name and the
    # exception stack, respectively.
    #
    # The lifetime of an exception is (Python 3.11+):
    #  + tx._raise_exception_variable(...) := sets the current_exception variable
    #  + PUSH_EXC_INFO := pushes the current_exception to the *exception stack*
    #  + POP_EXCEPT := pops TOS from the *exception stack*

    _exc_stack: list[ExceptionVals] = dataclasses.field(default_factory=list)
    _current_exception: Optional[ExceptionVals] = dataclasses.field(default=None)

    def clear_current_exception(self) -> None:
        self._current_exception = None

    def set_current_exception(self, val: ExceptionVals) -> None:
        self._set_context_and_break_context_reference_cycle(val)
        self._current_exception = val

    def move_current_exception_to_stack(self) -> None:
        assert self._current_exception is not None
        self.append(self._current_exception)
        self.clear_current_exception()

    def get_current_exception(self) -> ExceptionVals:
        assert self._current_exception is not None
        return self._current_exception

    def _set_context_recursive(
        self, val: ExceptionVals, prev_idx: int
    ) -> ExceptionVals:
        if (ctx := val.__context__) and type(ctx) is not ConstantVariable:  # type: ignore[union-attr]
            return val
        if len(self._exc_stack) + prev_idx > 0:
            prev = self._exc_stack[prev_idx]
            self._set_context_recursive(prev, prev_idx - 1)
            val.set_context(prev)  # type: ignore[union-attr, arg-type]
        return val

    def _break_context_reference_cycle(self, val: ExceptionVals) -> None:
        # See test_exceptions::test_raise_does_not_create_context_chain_cycle
        # Based on https://github.com/python/cpython/blob/e635bf2e49797ecb976ce45a67fce2201a25ca68/Python/errors.c#L207-L228
        # As noted on CPython, this is O(chain length) but the context chains
        # are usually very small
        o = slow_o = val
        slow_update_toggle = False  # floyd's algorithm for detecting cycle
        while True:
            context = o.__context__  # type: ignore[union-attr]
            if type(context) is ConstantVariable:  # context not set
                break

            if context is val:
                o.set_context(ConstantVariable(None))  # type: ignore[union-attr, arg-type]
                break

            o = context  # type: ignore[assignment]
            if o is slow_o:
                # pre-existing cycle - all exceptions on the path were
                # visited and checked
                break

            if slow_update_toggle:
                # visited all exceptions
                slow_o = slow_o.__context__  # type: ignore[union-attr, assignment]
            slow_update_toggle = not slow_update_toggle

    def _set_context_and_break_context_reference_cycle(
        self, val: ExceptionVals
    ) -> None:
        # set Exception.__context__
        self._set_context_recursive(val, len(self._exc_stack) - 1)
        self._break_context_reference_cycle(val)

    def pop(self) -> ExceptionVals:
        return self._exc_stack.pop()

    def append(self, val: ExceptionVals) -> None:
        self._exc_stack.append(val)

    def __len__(self) -> int:
        return len(self._exc_stack)

    def __getitem__(self, index: int) -> ExceptionVals:
        return self._exc_stack[index]

    def __str__(self) -> str:
        return f"{self._exc_stack=} - {self._current_exception=}"

    __repr__ = __str__


class InstructionTranslatorBase(
    metaclass=BytecodeDispatchTableMeta,
):
    output: OutputGraph
    symbolic_locals: dict[str, VariableTracker]
    symbolic_globals: dict[str, VariableTracker]
    symbolic_torch_function_state: SymbolicTorchFunctionState
    symbolic_stream_state: SymbolicStreamState
    post_prune_cell_and_freevars: Optional[dict[str, VariableTracker]]
    stack: list[VariableTracker]
    instruction_pointer: Optional[int]
    current_instruction: Instruction
    block_stack: list[BlockStackEntry]
    lineno: int
    kw_names: Optional[ConstantVariable]
    accept_prefix_inst: bool
    prefix_insts: list[Instruction]
    inline_depth: int
    inconsistent_side_effects: bool
    current_speculation: Optional[SpeculationEntry]
    dispatch_table: list[Any]
    exn_vt_stack: ExceptionStack
    exec_recorder: Optional[ExecutionRecorder]
    strict_checks_fn: Optional[Callable[[VariableTracker], bool]]
    start_point: Optional[int]
    is_leaf_tracer: bool
    parent: Optional[InstructionTranslatorBase]
    debug_locals: list[tuple[VariableTracker, list[VariableTracker]]]
    package: Optional[CompilePackage]
    latest_bytecode_queue: deque[str]
    # Store the latest bytecode before graph_break() call by user

    def mark_inconsistent_side_effects(self) -> None:
        """
        InstructionTranslator has encountered instructions which may cause
        dynamo to see a different version of history from eager
        See: https://github.com/pytorch/pytorch/issues/110765
        """
        self.inconsistent_side_effects = True

    def maybe_has_backedge(self) -> bool:
        # This function employs a heuristic. It does not reliably detect a backedge.
        # The heuristic is straightforward: starting from the current instruction and
        # continuing to the end, if any jump instruction targets an instruction before
        # the current one, there might be a backedge.

        # Python 3.12 introduced changes to bytecode that group common paths in
        # blockstacks (with or try...else) and allow for early returns. Consequently,
        # there can be multiple RETURN_VALUE instructions. Another heuristic is to
        # halt detection upon encountering the first RETURN_VALUE or RETURN_CONST.

        # These heuristics can result in both false positives and negatives, but
        # in either case, the Dynamo code remains valid. For false positives
        # (where an edge is incorrectly marked as a backedge), Dynamo will
        # perform a SkipFrame instead of potentially applying optimizations. For
        # false negatives (where an edge that should be marked as a backedge
        # isn't), multiple graphs may be generated if there's a break in the
        # graph during a for loop. In general, its better to have fewer false
        # negatives so that Dynamo does not skip the whole frame.

        # If any parent tx has a backedge, then return True
        cur_tx: Optional[InstructionTranslatorBase] = self
        while cur_tx is not None:
            cur_offset = cur_tx.current_instruction.offset
            assert cur_tx.instruction_pointer is not None
            for inst in cur_tx.instructions[cur_tx.instruction_pointer :]:
                if inst.opname in ("RETURN_VALUE", "RETURN_CONST"):
                    break
                if inst.opname in JUMP_OPNAMES:
                    jump_offset = inst.argval
                    if jump_offset < cur_offset:
                        return True
            cur_tx = cur_tx.parent
        return False

    def cellvars(self) -> list[str]:
        return self.code_options["co_cellvars"]

    def freevars(self) -> list[str]:
        return self.code_options["co_freevars"]

    def cell_and_freevars(self) -> list[str]:
        if not hasattr(self, "_cell_and_freevars"):
            self._cell_and_freevars = self.cellvars() + self.freevars()
        return self._cell_and_freevars

    def prune_dead_locals(self) -> None:
        # keep cell and freevar references alive
        self.post_prune_cell_and_freevars = {
            k: v
            for k, v in self.symbolic_locals.items()
            if k in self.cell_and_freevars()
        }
        # Only keep the locals that must remain on the stack.
        reads = livevars_analysis(self.instructions, self.current_instruction)
        self.symbolic_locals = {
            k: v for k, v in self.symbolic_locals.items() if k in reads
        }

    def call_function(
        self,
        fn: VariableTracker,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> None:
        assert isinstance(fn, VariableTracker)
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        assert all(
            isinstance(x, VariableTracker)
            for x in itertools.chain(args, kwargs.values())
        )
        inner_fn = None
        if hasattr(fn, "value"):
            inner_fn = fn.value
        if hasattr(fn, "fn"):
            inner_fn = fn.fn
        if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
            raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
        self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]

    def inline_generator_function(
        self, fn: VariableTracker, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> Any:
        """
        Redirect the call to the generator "call_function"
        """
        if not isinstance(fn, LocalGeneratorFunctionVariable):
            fn = LocalGeneratorFunctionVariable(fn)  # type: ignore[arg-type]
        return fn.call_function(self, args, kwargs)  # type: ignore[arg-type]

    def inline_user_function_return(
        self, fn: VariableTracker, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> Any:
        """
        A call to some user defined function by inlining it.
        """
        self.is_leaf_tracer = False
        if config.enable_faithful_generator_behavior and is_generator(fn.get_code()):  # type: ignore[attr-defined]
            return self.inline_generator_function(fn, args, kwargs)
        else:
            return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

    def get_line_of_code_header(self, lineno: Optional[int] = None) -> str:
        if lineno is None:
            lineno = self.lineno
        inline_depth_str = (
            f" (inline depth: {self.inline_depth})" if self.inline_depth > 0 else ""
        )
        funcname = get_funcname(self.f_code.co_filename, lineno)
        funcname_str = "" if funcname is None else f" ({funcname})"
        return f"{self.f_code.co_filename}:{lineno} in {self.f_code.co_name}{funcname_str}{inline_depth_str}"

    def get_log_starts_line_log_str(self) -> str:
        log_str = f"TRACE starts_line {self.get_line_of_code_header()}\n"
        line = linecache.getline(self.f_code.co_filename, self.lineno).rstrip()
        log_str += f"    {line}"
        return log_str

    def starts_line(self, lineno: int) -> None:
        if self.lineno == lineno:
            return
        self.lineno = lineno
        TracingContext.set_current_loc(
            self.f_code.co_filename, lineno, self.f_code.co_name
        )

        if self.is_trace_source_log_enabled:
            trace_source_log.debug("%s", LazyString(self.get_log_starts_line_log_str))

    def step(self) -> bool:
        """Process exactly one instruction, return False we should exit"""
        self.error_on_graph_break = _get_error_on_graph_break()

        ip = self.instruction_pointer
        if ip is None:
            return False
        self.current_instruction = inst = self.instructions[ip]
        self.instruction_pointer = ip + 1

        if inst.starts_line:
            self.starts_line(inst.starts_line)

        if (
            not self.stack
            and self.should_compile_partial_graph()
            and self.is_non_empty_graph()
        ):
            self.current_speculation = self.speculate()
            if self.current_speculation.failed(self):
                self.step_graph_break(inst)
                return False

        if self.is_trace_byt
```



## High-Level Overview

"""Core module responsible for converting Python bytecode into TorchDynamo's symbolic execution format.This module implements the bytecode-level tracing system that allows TorchDynamo to analyzeand transform Python code. It converts Python bytecode instructions into a symbolic formatthat tracks the flow of tensors and other values through the program.Key components:- InstructionTranslatorBase: Base class for converting bytecode to symbolic execution- InstructionTranslator: Main translator for function bytecode- InliningInstructionTranslator: Handles inlining of called functions- SpeculationLog: Manages state for speculative execution and rollbackThe symbolic conversion process handles:- Control flow (loops, conditionals, etc.)- Function inlining and call stack management- Tracking of program values and side effects- Graph breaks and resumption points- Exception handling and stack frame managementThis is a core part of TorchDynamo's tracing system that enables ahead-of-timeoptimization of PyTorch programs.

This Python file contains 19 class(es) and 248 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SpeculationEntry`, `SpeculationLog`, `LocalState`, `DistributedState`, `TensorifyState`, `BlockStackEntry`, `SpeculationLogDivergence`, `ReturnValueOp`, `YieldValueOp`, `BytecodeDispatchTableMeta`, `ExceptionStack`, `InstructionTranslatorBase`, `InstructionTranslator`, `InliningInstructionTranslator`, `InliningGeneratorInstructionTranslator`

**Functions defined**: `_import_module`, `fail_and_restart_analysis`, `failed`, `restart`, `clear`, `next`, `render`, `specialize`, `should_specialize`, `clear`, `empty`, `_step_logger`, `save_and_restart_speculation_log`, `temporarely_allow_writes_to_output_graph`, `can_restore`, `resume_fn`, `exit`, `stack_op`, `impl`, `is_stdlib`

**Key imports**: annotations, collections, collections.abc, contextlib, copy, dataclasses, dis, functools, importlib, inspect


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `collections`
- `collections.abc`
- `contextlib`
- `copy`
- `dataclasses`
- `dis`
- `functools`
- `importlib`
- `inspect`
- `itertools`
- `linecache`
- `logging`
- `operator`
- `re`
- `sys`
- `threading`
- `traceback`
- `types`
- `weakref`
- `typing`: Any, cast, NoReturn, Optional, TYPE_CHECKING, TypeAlias, Union
- `typing_extensions`: TypeIs
- `torch`
- `torch._logging`
- `torch._dynamo.exc`: ObservedException, TensorifyScalarRestartAnalysis
- `torch._guards`: tracing, TracingContext
- `torch._logging.structured`: dump_file
- `torch.fx.experimental.symbolic_shapes`: guard_bool


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

- **File Documentation**: `symbolic_convert.py_docs.md`
- **Keyword Index**: `symbolic_convert.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
