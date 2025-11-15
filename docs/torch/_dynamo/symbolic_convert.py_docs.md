# Documentation: symbolic_convert.py

## File Metadata
- **Path**: `torch/_dynamo/symbolic_convert.py`
- **Size**: 214847 bytes
- **Lines**: 5263
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

        if self.is_trace_bytecode_log_enabled:
            trace_bytecode_log.debug(
                "TRACE %s %s %s", inst.opname, inst.argval, repr(self.stack)
            )

        # Store the latest 20 bytecode execution for the process,
        # Used repr for byte processing and limiting the length to 2048
        if config.verbose:
            try:
                stack_repr = repr(self.stack)
            except ValueError:
                # Handle large integers that exceed sys.int_info.str_digits_check_threshold
                stack_repr = "<self.stack repr truncated due to large integer>"
            self.latest_bytecode_queue.append(
                f"TRACE {inst.opname} {repr(inst.argval)} {stack_repr}"
            )

        self.update_block_stack(inst)

        try:
            self.dispatch_table[inst.opcode](self, inst)
            return not self.output.should_exit
        except TensorifyScalarRestartAnalysis:
            raise
        except exc.ObservedException as e:
            self.exception_handler(e)
            return True
        except (ReturnValueOp, YieldValueOp):
            return False
        except (Unsupported, StepUnsupported) as e:
            # More restrictive condition than should_compile_partial_graph:
            # if this condition is true, then we SHOULD NOT attempt to find
            # a previous checkpoint to resume from and try to resume - we should
            # immediately error out.
            # The condition is more restrictive because, it may be possible to resume significantly earlier
            # in the code (the most recent speculation point). This happens, for example, in the case
            # of a graph break in a try block.
            if (
                self.one_graph
                or self.error_on_graph_break
                or self.is_tracing_resume_prologue
            ):
                if isinstance(e, StepUnsupported):
                    unimplemented(
                        gb_type="cannot resume from torch._dynamo.step_unsupported()",
                        context="",
                        explanation="traced torch._dynamo.step_unsupported(), but Dynamo is instructed "
                        "to error on graph break. This graph break is used for debugging only.",
                        hints=[
                            "Remove the torch._dynamo.step_unsupported() call.",
                            "Make sure fullgraph=False and error_on_graph_break=False.",
                            *graph_break_hints.DYNAMO_BUG,
                        ],
                    )
                raise
            if self.current_speculation is None:
                log.debug("empty checkpoint - cannot resume from graph break")
                if isinstance(e, StepUnsupported):
                    unimplemented(
                        gb_type="torch._dynamo.step_unsupported() with empty checkpoint",
                        context="",
                        explanation="traced torch._dynamo.step_unsupported(), but there is no checkpoint "
                        "to step_graph_break from. This graph break is used for debugging only.",
                        hints=[
                            "Remove the torch._dynamo.step_unsupported() call.",
                            "Include at least one checkpoint: (1) include at least 2 ops and (2) make sure there is some "
                            "line of code that is not in a try/with block, and has an empty Python stack.",
                            *graph_break_hints.DYNAMO_BUG,
                        ],
                    )
                raise
            reason = (
                "Encountered graph break that we cannot resume from. "
                "Compiling up to the previous resumable state, "
                "then skipping the rest of the function. "
                f"Graph break encountered:\n{str(e)}"
            )
            self.log_graph_break(
                self.code_options,
                reason=reason,
                user_stack=e.real_stack,
            )

        self.current_speculation.fail_and_restart_analysis(self.error_on_graph_break)
        return False

    if sys.version_info >= (3, 11):

        def update_block_stack(self, inst: Instruction) -> None:
            # 3.11+ no longer uses a block stack, but we still keep track of one
            # so that we know which contexts are currently active.
            # For our purposes, all exception table entries with the same target
            # are considered to be part of the same "block".
            # NOTE: we only keep track of with blocks that are not contained in try blocks.
            # This is because we will not create continuation functions on graph breaks in try blocks,
            # but we may for with blocks. We do not push blocks here since
            # with blocks are pushed when handling BEFORE_WITH.
            entry = inst.exn_tab_entry
            if entry:
                # Detect when we have exited the top with block.
                # The with blocks on the block stack are not enclosed in try
                # blocks, so a with block's cleanup code should be in the
                # previous with block (if any).
                if (
                    len(self.block_stack) >= 2
                    and entry.target is not self.block_stack[-1].target
                    and entry.target is self.block_stack[-2].target
                ):
                    # exit the current block
                    self.block_stack.pop()
            else:
                # no longer in any block
                # It is possible for NOPs to be between two instructions
                # in the same block, but the NOPs are not covered by an
                # exception table entry. In this case, assume that we
                # are still in the same block.
                # In 3.12+, JUMP_BACKWARD might also not be covered by
                # an exception table entry, so we also assume that we
                # are still in the same block. It is probably safe to do
                # this in 3.11, even though we haven't encountered this case before.
                # In 3.14+, NOT_TAKEN might also not be covered by an exn table entry.
                if self.block_stack and inst.opname not in (
                    "NOP",
                    "JUMP_BACKWARD",
                    "NOT_TAKEN",
                ):
                    # If we really escape from a block and the current
                    # instruction is not in another block, then there
                    # should be no other nested blocks that we are in.
                    assert len(self.block_stack) == 1
                    self.block_stack.pop()

    else:

        def update_block_stack(self, inst: Instruction) -> None:
            pass

    @property
    def next_instruction(self) -> Instruction:
        assert self.instruction_pointer is not None
        return self.instructions[self.instruction_pointer]

    def step_graph_break(self, continue_inst: Instruction) -> None:
        # generate code from checkpoint
        assert not self.output.output_instructions
        assert self.current_speculation is not None
        # NOTE: adding an assert here since it seems like the only place
        # where we call step_graph_break right now is when the stack is empty,
        # so let's enforce that for now.
        assert not self.stack
        # NOTE: if we support non-empty self.stack in the future, the `stack_pops` argument
        # below should be set to the stack length to ensure that the stack is codegen'd
        # for the rest of the function.
        log.debug("step triggered compile")
        all_stack_locals_metadata = self.output.compile_subgraph(
            self,
            partial_convert=True,
            reason=GraphCompileReason("step_unsupported", [self.frame_summary()]),
        )
        # current frame state
        # cells,
        # [
        #   frame N locals,
        #   frame N-1 stack + locals,
        #   ...,
        #   frame 1 stack + locals,
        # ],
        if self.parent:
            from .eval_frame import skip_code

            # nested graph break
            assert config.nested_graph_breaks
            cg = PyCodegen(self.output.root_tx)

            # codegen cells and frame values only for frame N
            cg.extend_output(
                [
                    *create_copy(2),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    create_instruction("BUILD_LIST", arg=1),
                    *create_copy(2),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    create_instruction("BUILD_LIST", arg=1),
                ]
            )
            # No need to fix stack, since stack is assumed to be empty here.
            # Do NOT handle_inactive_ctx because we will be skipping this resume code.
            leaf_resume_code, leaf_resume_name = self.create_resume(
                0, continue_inst, all_stack_locals_metadata[0], [], cg, True, False
            )
            skip_code(leaf_resume_code)

            # current frame state
            # cells,
            # [
            #   frame N locals,
            #   frame N-1 stack + locals,
            #   ...,
            #   frame 1 stack + locals,
            # ], [frame N cells], [frame N locals],
            self.codegen_call_resume([leaf_resume_code], [leaf_resume_name], cg)

            # current frame state
            # cells,
            # [
            #   frame N locals,
            #   frame N-1 stack + locals,
            #   ...,
            #   frame 1 stack + locals,
            # ], leaf_resume result

            # pop frame N cells and locals
            cg.extend_output(
                [
                    *create_copy(2),
                    cg.create_load_const(0),
                    create_instruction("DELETE_SUBSCR"),
                    *create_copy(3),
                    cg.create_load_const(0),
                    create_instruction("DELETE_SUBSCR"),
                ]
            )

            # add the leaf_resume result to frame N-1 stack
            num_stack = all_stack_locals_metadata[1].num_stack
            cg.extend_output(
                [
                    create_instruction("BUILD_LIST", arg=1),
                    *create_copy(2),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    *create_binary_slice(num_stack, num_stack, True),
                ]
            )
            self.parent.push(UnknownVariable())
            all_stack_locals_metadata[1].num_stack += 1

            # current frame state
            # cells, frame_values
            # extract frame N-1 stack to stack
            cg.extend_output(
                [
                    create_dup_top(),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    *create_binary_slice(0, num_stack + 1),
                ]
            )

            # current frame state
            # cells, frame_values, frame N-1 stack + leaf_resume result
            # remove frame N-1 stack from frame_values
            cg.extend_output(
                # frame_values[0] = frame_values[0][num_stack + 1:]
                [
                    *create_copy(2),
                    cg.create_load_const(0),
                    cg.create_binary_subscr(),
                    create_dup_top(),
                    *create_binary_slice(num_stack + 1, None),
                    *create_swap(2),
                    cg.create_load_const(0),
                    create_instruction("STORE_SUBSCR"),
                ]
            )

            # current frame state
            # cells, frame_values, frame N-1 stack + leaf_resume result
            # unpack the stack (need to unpack twice since UNPACK_SEQUENCE unpacks in reverse order)
            cg.extend_output(
                [
                    create_instruction("UNPACK_SEQUENCE", arg=num_stack + 1),
                    create_instruction("BUILD_LIST", arg=num_stack + 1),
                    create_instruction("UNPACK_SEQUENCE", arg=num_stack + 1),
                ]
            )

            # call the remaining resume functions
            # current frame state
            # [frame N-1 cells, ..., frame 1 cells],
            # [
            #   frame N-1 locals,
            #   frame N-2 stack + locals,
            #   ...,
            #   frame 1 stack + locals,
            # ], *(frame N-1 stack), leaf_resume result
            self.output.add_output_instructions(
                cg.get_instructions()
                + self.parent.create_call_resume_at(
                    self.parent.next_instruction, all_stack_locals_metadata[1:]
                )
            )
        else:
            # pop cells
            self.output.add_output_instructions(
                [
                    *create_swap(2),
                    create_instruction("POP_TOP"),
                ]
            )
            # load locals from frame values
            cg = PyCodegen(self.output.root_tx)
            self.output.add_output_instructions(
                [
                    cg.create_load_const(-1),
                    cg.create_binary_subscr(),
                ]
            )
            for local, idx in all_stack_locals_metadata[-1].locals_names.items():
                self.output.add_output_instructions(
                    [
                        create_dup_top(),
                        cg.create_load_const(idx),
                        cg.create_binary_subscr(),
                        cg.create_store(local),
                    ]
                )
            self.output.add_output_instructions(
                [
                    create_instruction("POP_TOP"),
                    create_jump_absolute(continue_inst),
                    *self.instructions,
                ]
            )

    def run_ctx_mgr(self) -> Any:
        # NB: Don't push the top level frame summary; set_current_loc will
        # take care of it.  However, DO make sure we attach real_stack to
        # exceptions
        return TracingContext.current_frame(None)

    def run(self) -> None:
        with self.run_ctx_mgr():
            dump_file(self.f_code.co_filename)
            try:
                self.output.push_tx(self)
                self.start_point = self.instruction_pointer
                try:
                    while self.step():
                        pass
                except Exception as e:
                    if self.is_tracing_resume_prologue:
                        raise ResumePrologueTracingError(
                            "Error while tracing through a Dynamo-generated resume function prologue. "
                            "Errors are not allowed when tracing resume function prologues.\n"
                            f"{type(e).__qualname__}: {str(e)}"
                        ).with_traceback(e.__traceback__) from None
                    raise
            except TensorifyScalarRestartAnalysis:
                raise
            except BackendCompilerFailed:
                raise
            except RuntimeError as e:
                if hasattr(e, "msg") and "Data-dependent" in e.msg:
                    readable_graph = torch.fx.GraphModule(
                        self.output.nn_modules, self.output.graph
                    ).print_readable(
                        print_output=False, include_stride=True, include_device=True
                    )
                    e.partial_fx_graph = readable_graph  # type: ignore[attr-defined]
                    raise

                raise
            except Exception as e:
                if self.exec_recorder:
                    e.exec_record = self.exec_recorder.get_record()  # type: ignore[attr-defined]

                raise
            finally:
                self.output.pop_tx()
                # Cleanup the outputGraph to delete the held tensors. We perform the
                # cleanup only for InstructionTranslator and not
                # InliningInstructionTranslator. The InliningInstructionTranslator
                # mutates the output object and is restored to original state if
                # there was an exception.
                if isinstance(self, InstructionTranslator):
                    self.output.cleanup()

                    # Note that this call maybe redundant if compile_subgraph is
                    # called. This is ok, because calling exit stack close()
                    # twice is not an issue (second stop is a no op).
                    self.output.mark_bytecode_tracing_stop()

    def push(self, val: Optional[VariableTracker]) -> None:
        assert val is None or isinstance(val, VariableTracker), (
            f"push expects VariableTracker, got {typestr(val)}"
        )
        self.stack.append(val)  # type: ignore[arg-type]

    def push_many(self, vals: list[VariableTracker]) -> None:
        for val in vals:
            self.push(val)

    def pop(self) -> VariableTracker:
        return self.stack.pop()

    def popn(self, n: int) -> list[VariableTracker]:
        return [*reversed([self.pop() for _ in range(n)])]

    def LOAD_FAST(self, inst: Instruction) -> None:
        name = inst.argval
        if self.exec_recorder and name in self.f_locals:
            self.exec_recorder.add_local_var(name, self.f_locals[name])

        try:
            self.push(self.symbolic_locals[name].unwrap())
        except KeyError:
            if name.startswith("."):
                try:
                    # This happens in dict/list comprehensions
                    new_name = name.replace(".", "implicit")
                    self.push(self.symbolic_locals[new_name])
                except KeyError:
                    unimplemented(
                        gb_type="Attempted to read undefined local variable (implicit)",
                        context=f"LOAD_FAST {name}",
                        explanation=f"Could not find an implicit local variable with name `{name}`",
                        hints=[
                            "This happens in dict/list comprehensions",
                            *graph_break_hints.USER_ERROR,
                        ],
                    )
            else:
                unimplemented(
                    gb_type="Attempted to read undefined local variable",
                    context=f"LOAD_FAST {name}",
                    explanation=f"Could not find a local variable with name `{name}`",
                    hints=[*graph_break_hints.USER_ERROR],
                )

        # for continuation functions
        if name.startswith("__stack"):
            self.symbolic_locals.pop(name)

    def LOAD_DEREF(self, inst: Instruction) -> None:
        assert inst.argval in self.cell_and_freevars()
        cell = self.symbolic_locals[inst.argval]
        contents_var = self.output.side_effects.load_cell(cell)
        self.push(contents_var)

        if self.exec_recorder and inst.argval in self.f_locals:
            self.exec_recorder.add_local_var(inst.argval, self.f_locals[inst.argval])

    def STORE_FAST(self, inst: Instruction) -> None:
        name = inst.argval
        loaded_vt = self.pop()
        loaded_vt.set_name_hint(name)
        self.symbolic_locals[name] = loaded_vt
        if name == IS_TRACING_RESUME_PROLOGUE_VARNAME:
            val = loaded_vt.as_python_constant()
            assert type(val) is bool
            self.is_tracing_resume_prologue = val

    def DELETE_FAST(self, inst: Instruction) -> None:
        del self.symbolic_locals[inst.argval]

    def STORE_DEREF(self, inst: Instruction) -> None:  # type: ignore[override]
        assert inst.argval in self.cell_and_freevars()
        cell = self.symbolic_locals[inst.argval]
        val = self.pop()
        self.output.side_effects.store_cell(cell, val)

        assert isinstance(cell, CellVariable)  # tame mypy
        if cell.local_name is not None:
            val.set_name_hint(cell.local_name)  # type: ignore[attr-defined]

    LOAD_CLOSURE = LOAD_FAST

    def _load_const(self, inst: Instruction) -> VariableTracker:
        i = inst.arg
        if i is None:
            return ConstantVariable.create(value=inst.argval)  # type: ignore[return-value]
        val = self._constants_cache[i]
        if not val:
            self._constants_cache[i] = ConstantVariable.create(value=inst.argval)  # type: ignore[call-overload]
            val = self._constants_cache[i]
        assert val is not None
        return val

    def LOAD_CONST(self, inst: Instruction) -> None:
        self.push(self._load_const(inst))

    def _load_global(self, inst: Instruction) -> None:
        name = inst.argval

        if self.exec_recorder:
            if name in self.f_globals:
                self.exec_recorder.add_global_var(name, self.f_globals[name])
            else:
                assert name in self.f_builtins
                self.exec_recorder.builtins[name] = self.f_builtins[name]

        if name not in self.f_globals:
            return self.load_builtin(inst)

        if name in self.symbolic_globals:
            variable = self.output.side_effects[self.symbolic_globals[name]]
            self.push(self.output.side_effects.load_global(variable, name))
            return

        value = self.f_globals[name]
        self.push(VariableTracker.build(self, value, GlobalSource(name)))

    @functools.cached_property
    def nn_modules_globals_vt(self) -> VariableTracker:
        module_name = "torch.nn.modules.module"
        module_source = self.import_source(module_name)
        fglobals_value = _import_module(module_name)
        return VariableTracker.build(self, fglobals_value, module_source)

    def LOAD_GLOBAL(self, inst: Instruction) -> None:
        assert inst.arg is not None
        if sys.version_info >= (3, 11) and sys.version_info < (3, 13) and inst.arg % 2:
            self.PUSH_NULL(inst)
        self._load_global(inst)
        if sys.version_info >= (3, 13) and inst.arg % 2:
            self.PUSH_NULL(inst)

    def STORE_GLOBAL(self, inst: Instruction) -> None:
        value = self.pop()
        name = inst.argval
        source = GlobalSource(name)
        if name not in self.symbolic_globals:
            self.symbolic_globals[name] = object()  # type: ignore[assignment]  # sentinel object
        variable = self.output.side_effects.track_global_existing(
            source, self.symbolic_globals[name]
        )
        if isinstance(value, RemovableHandleVariable):
            unimplemented(
                gb_type="Storing Tensor hook handle in globals",
                context=name,
                explanation="This is not supported.",
                hints=[],
            )
        self.output.side_effects.store_global(variable, name, value)

    # Cache note: This cache only exists for the duration of this
    # InstructionTranslator - so it should be safe to do.
    @cache_method
    def import_source(self, module_name: str) -> GlobalSource:
        """Create an alias to a module for use in guards"""
        if "torch_package" in module_name:
            value = torch.package.package_importer._package_imported_modules[
                module_name
            ]
            alias = (
                module_name.replace(">", "_").replace("<", "_").replace(".", "_dot_")
            )
        else:
            value = _import_module(module_name)
            alias = f"__import_{module_name.replace('.', '_dot_')}"

        if self.package is not None:
            self.package.add_import_source(alias, module_name)
        self.output.import_sources[alias] = module_name
        f_globals = self.output.global_scope
        assert alias not in f_globals or f_globals[alias] is value
        f_globals[alias] = value
        self.output.update_co_names(alias)
        return GlobalSource(alias)

    def resolve_name(self, name: str, package: str, level: int) -> str:
        """
        Copied from the Cpython implementation of __import__
        Resolve a relative module name to an absolute one.
        https://github.com/python/cpython/blob/5a094f0255eea1db58fb2cf14c200971e64ec36e/Lib/importlib/_bootstrap.py#L902
        """
        bits = package.rsplit(".", level - 1)
        if len(bits) < level:
            raise ImportError("attempted relative import beyond top-level package")
        base = bits[0]
        return f"{base}.{name}" if name else base

    def calc_package(self) -> str:
        """
        Copied from the Cpython implementation of __import__
        https://github.com/python/cpython/blob/5a094f0255eea1db58fb2cf14c200971e64ec36e/Lib/importlib/_bootstrap.py#L1090
        """
        package = self.f_globals.get("__package__")
        spec = self.f_globals.get("__spec__")
        if package is not None:
            if spec is not None and package != spec.parent:
                log.warning(
                    "__package__ != __spec__.parent (%r != %r)",
                    package,
                    spec.parent,
                    stacklevel=3,
                )
            return package
        elif spec is not None:
            return spec.parent
        else:
            log.warning(
                "can't resolve package from __spec__ or __package__, "
                "falling back on __name__ and __path__",
                stacklevel=3,
            )
            package = self.f_globals["__name__"]
            if "__path__" not in self.f_globals:
                package = package.rpartition(".")[0]
        return package

    def IMPORT_NAME(self, inst: Instruction) -> None:
        level, fromlist = self.popn(2)
        level = level.as_python_constant()
        fromlist = fromlist.as_python_constant()
        module_name = inst.argval

        # Are we replaying? if so, load recorded module
        recorded_name = (
            f"{ExecutionRecorder.LOCAL_MOD_PREFIX}_{level}_{fromlist}_{module_name}"
        )
        if recorded_name in self.f_globals:
            value = self.f_globals[recorded_name]
            source = GlobalSource(recorded_name)
        else:
            try:
                value = __import__(
                    module_name,
                    fromlist=fromlist,
                    level=level,
                    globals=self.f_globals,
                )
            except ImportError:
                unimplemented(
                    gb_type="Import failure",
                    context=f"module_name: {module_name}, fromlist: {fromlist}, level={level}",
                    explanation="Failure when attempting to import.",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            if level != 0:
                pkg = self.calc_package()
                module_name = self.resolve_name(module_name, pkg, level)

            # For __import__, when the name variable is of the form package.module,
            # normally, the top-level package (the name up till the first dot) is
            # returned, not the module named by module_name. However, when a
            # non-empty fromlist argument is given, the module named by name is
            # returned. Therefore, we set the source correctly here.
            if not fromlist:
                top_level_module_name = module_name.partition(".")[0]
                source = self.import_source(top_level_module_name)
            else:
                source = self.import_source(module_name)

        if self.exec_recorder:
            # pyrefly: ignore [unbound-name]
            self.exec_recorder.add_local_mod(recorded_name, value)

        # pyrefly: ignore [unbound-name]
        if istype(value, (types.ModuleType, DummyModule)):
            # pyrefly: ignore [unbound-name]
            self.push(PythonModuleVariable(value, source=source))
        else:
            unimplemented(
                gb_type="Bad import result",
                # pyrefly: ignore [unbound-name]
                context=typestr(value),
                explanation="Import result is not a Python module.",
                hints=[],
            )

    # fb internal 3.12 opcode
    EAGER_IMPORT_NAME = IMPORT_NAME

    def IMPORT_FROM(self, inst: Instruction) -> None:
        self.DUP_TOP(inst)
        self._load_attr(inst.argval)

    # Cache note: This cache only exists for the duration of this
    # InstructionTranslator - so it should be safe to do.
    @cache_method
    def load_builtin_from_argval(self, argval: Any) -> VariableTracker:
        if argval not in self.f_builtins:
            unimplemented(
                gb_type="failed to find name in frame builtins",
                context="",
                explanation=f"Failed to find name `{argval}` in frame's builtins.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        val = self.f_builtins[argval]

        if callable(val):
            builtins_source = GlobalSource(
                self.output.name_of_builtins_dict_key_in_fglobals
            )
            var_source = DictGetItemSource(builtins_source, argval)
            return VariableTracker.build(self, val, var_source)
        else:
            assert is_builtin_constant(val)
            return ConstantVariable.create(value=val)

    def load_builtin(self, inst: Instruction) -> None:
        self.push(self.load_builtin_from_argval(inst.argval))

    def jump(self, inst: Instruction) -> None:
        assert self.instruction_pointer is not None
        assert self.start_point is not None
        assert inst.target is not None
        get_metrics_context().increment(
            "ir_count", self.instruction_pointer - self.start_point
        )
        self.instruction_pointer = self.indexof[inst.target]
        self.start_point = self.instruction_pointer

    JUMP_FORWARD = jump
    JUMP_ABSOLUTE = jump

    POP_JUMP_IF_FALSE = generic_jump(operator.not_, False)
    POP_JUMP_IF_TRUE = generic_jump(operator.truth, False)
    JUMP_IF_FALSE_OR_POP = generic_jump(operator.not_, True)
    JUMP_IF_TRUE_OR_POP = generic_jump(operator.truth, True)

    def SETUP_LOOP(self, inst: Instruction) -> None:
        # only exists in python<=3.7
        assert inst.target is not None
        self.block_stack.append(BlockStackEntry(inst, inst.target, len(self.stack)))

    def SETUP_EXCEPT(self, inst: Instruction) -> None:
        # only exists in python<=3.7
        assert inst.target is not None
        self.block_stack.append(BlockStackEntry(inst, inst.target, len(self.stack)))

    def POP_BLOCK(self, inst: Instruction) -> None:
        self.block_stack.pop()

    def SETUP_WITH(self, inst: Instruction) -> None:
        self.setup_or_before_with(inst)

    def SETUP_FINALLY(self, inst: Instruction) -> None:
        assert inst.target is not None
        self.block_stack.append(BlockStackEntry(inst, inst.target, len(self.stack)))

    def BEGIN_FINALLY(self, inst: Instruction) -> None:
        self.push(None)

    def WITH_CLEANUP_START(self, inst: Instruction) -> None:
        exit, exc = self.popn(2)
        assert exc is None
        self.push(exc)
        # pyrefly: ignore [bad-argument-type]
        self.push(exit.call_function(self, [ConstantVariable.create(None)] * 3, {}))

    def WITH_CLEANUP_FINISH(self, inst: Instruction) -> None:
        self.popn(2)
        self.push(None)

    def FOR_ITER(self, inst: Instruction) -> None:
        it = self.pop().realize()
        try:
            val = it.next_variable(self)
            self.push(it)
            self.push(val)
        except (StopIteration, exc.ObservedUserStopIteration) as e:
            if isinstance(e, exc.ObservedUserStopIteration):
                exc.handle_observed_exception(self)

            # leave iterator upon exhaustion in 3.12
            if sys.version_info >= (3, 12):
                # CPython 3.12 actually jumps to the instruction after the END_FOR
                # and performs the action of END_FOR as part of FOR_ITER. We jump
                # to the END_FOR and run it, so we need to make sure 2 values are
                # on the stack for it to pop.
                self.push(it)
                self.push(ConstantVariable.create(None))
            self.jump(inst)

    def _create_exception_type(self, val: VariableTracker) -> VariableTracker:
        if isinstance(
            val, (variables.BuiltinVariable, UserDefinedExceptionClassVariable)
        ):
            # Create the instance of the exception type
            # https://github.com/python/cpython/blob/3.11/Python/ceval.c#L6547-L6549
            val = val.call_function(self, [], {})  # type: ignore[arg-type]
        return val

    def _raise_exception_variable(self, val: VariableTracker) -> NoReturn:
        # User can raise exception in 2 ways
        #   1) raise exception type - raise NotImplementedError
        #   2) raise exception instance - raise NotImplementedError("foo")

        # 1) when user raises exception type
        val = self._create_exception_type(val)

        # Handle https://peps.python.org/pep-0479/
        # CPython 3.12+ has a specific bytecode instruction (CALL_INTRINSIC_1 3) for this
        if (
            is_generator(self.f_code)
            and isinstance(val, variables.ExceptionVariable)
            and val.exc_type is StopIteration
        ):
            val = variables.BuiltinVariable(RuntimeError).call_function(self, [], {})  # type: ignore[arg-type]

        # Save the exception in a global data structure
        self.exn_vt_stack.set_current_exception(val)  # type: ignore[arg-type]

        # 2) when user raises exception instance
        if self._isinstance_exception(val):
            observed_exception_type = exc.get_dynamo_observed_exception(val.exc_type)  # type: ignore[attr-defined, union-attr]
            raise observed_exception_type(f"raised exception {val}")
        unimplemented(
            gb_type="Failed to raise exception",
            context=str(exc),
            explanation="Attempted to raise a non-Exception type/value.",
            hints=[*graph_break_hints.USER_ERROR],
        )

    def RAISE_VARARGS(self, inst: Instruction) -> None:
        if inst.arg == 0:
            if not len(self.exn_vt_stack):
                msg = ConstantVariable("No active exception to reraise")
                exc.raise_observed_exception(RuntimeError, self, args=[msg])

            # re-raise the previous exception. Here CPython refers to the exception
            # on top of the exception stack
            assert len(self.exn_vt_stack)
            val = self.exn_vt_stack[-1]
            assert self._isinstance_exception(val), val
            self._raise_exception_variable(val)
        elif inst.arg == 1:
            # raise TOS
            val = self.stack[-1]  # type: ignore[assignment]
            self._raise_exception_variable(val)
        else:
            # raise .. from ...
            from_vt = self.pop()
            val = self.pop()  # type: ignore[assignment]
            try:
                self._raise_exception_variable(val)
            finally:
                # Update __cause__/__suppress_context__ in the raised exception
                curr_exc = self.exn_vt_stack.get_current_exception()
                cause = self._create_exception_type(from_vt)
                curr_exc.call_setattr(self, ConstantVariable("__cause__"), cause)  # type: ignore[arg-type, union-attr, assignment]

    def CLEANUP_THROW(self, inst: Instruction) -> None:
        # https://github.com/python/cpython/pull/96010
        tos = self.stack[-1]
        assert isinstance(tos, ExceptionVariable)
        if tos.exc_type is StopIteration:
            unimplemented(
                gb_type="CLEANUP_THROW with StopIteration",
                context="",
                explanation="Received StopIteration when handling generator.throw/close. This is not supported.",
                hints=[],
            )
        else:
            self.RERAISE(inst)

    def RERAISE(self, inst: Instruction) -> None:
        # https://docs.python.org/3/library/dis.html#opcode-RERAISE
        #   Re-raises the exception currently on top of the stack. If oparg is
        #   non-zero, pops an additional value from the stack which is used to
        #   set f_lasti of the current frame.

        if sys.version_info >= (3, 11):
            # RERAISE is currently supported in a narrow case of `raise ... from None`
            val = self.pop()
            if inst.argval:
                # RERAISE 1
                _ = self.pop()
                self._raise_exception_variable(val)
            else:
                # RERAISE 0
                self.push(val)
                self._raise_exception_variable(val)
        else:
            _exc = self.pop()
            val = self.pop()
            _tb = self.pop()
            self._raise_exception_variable(val)

    def _isinstance_exception(self, val: VariableTracker) -> TypeIs[ExceptionVals]:
        return isinstance(
            val,
            (
                variables.ExceptionVariable,
                UserDefinedExceptionClassVariable,
                UserDefinedExceptionObjectVariable,
            ),
        )

    def WITH_EXCEPT_START(self, inst: Instruction) -> None:
        args: list[VariableTracker] = []
        if sys.version_info >= (3, 11):
            fn_loc = 4 if sys.version_info < (3, 14) else 5
            # At the top of the stack are 4 values:
            #    - TOP = exc_info()
            #    - SECOND = previous exception
            #    - THIRD: lasti of exception in exc_info()
            #    - FOURTH: the context.__exit__ bound method
            #    We call FOURTH(type(TOP), TOP, GetTraceback(TOP)).
            #    Then we push the __exit__ return value.
            # In Python 3.14+, there is a NULL placed between the context.__exit__ bound method and the lasti,
            # that is, fn is now the 5th from TOS.
            assert len(self.stack) >= fn_loc
            fn = self.stack[-fn_loc]
            val = self.stack[-1]
            assert self._isinstance_exception(val)
            typ = BuiltinVariable(val.exc_type)  # type: ignore[attr-defined, union-attr]
            tb = ConstantVariable(None)
            if sys.version_info >= (3, 14):
                if not isinstance(self.stack[-4], NullVariable):
                    args.append(self.stack[-4])
        else:
            assert len(self.stack) >= 7
            fn = self.stack[-7]
            val = self.stack[-2]
            assert self._isinstance_exception(val)
            typ = BuiltinVariable(val.exc_type)  # type: ignore[attr-defined]
            tb = ConstantVariable(None)

        args += [typ, val, tb]
        self.call_function(fn, args, {})

    def exception_handler(self, raised_exception: ObservedException) -> None:
        observed_exn_gb_explanation = (
            "Dynamo found no exception handler at the top-level compiled function "
            "when encountering an exception. Exception will propagate outside the compiled region."
        )

        def bubble_exception_to_interpreter() -> None:
            # Bubble the exception to the interpreter
            curr_exc = self.exn_vt_stack.get_current_exception()
            dynamo_exc = exc.get_dynamo_observed_exception(curr_exc.python_type())
            assert isinstance(raised_exception, dynamo_exc)  # sanity check
            unimplemented(
                gb_type="Observed exception",
                context=f"raised exception {curr_exc.python_type_name()}({curr_exc.args})",  # type: ignore[union-attr]
                explanation=observed_exn_gb_explanation,
                hints=[
                    *graph_break_hints.USER_ERROR,
                    *graph_break_hints.SUPPORTABLE,
                ],
                from_exc=raised_exception,
            )

        if sys.version_info >= (3, 11):
            exn_tab_entry = self.current_instruction.exn_tab_entry
            if exn_tab_entry:
                # Implementation is based on https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt

                # 1) pop values from the stack until it matches the stack depth
                # for the handler
                while len(self.stack) > exn_tab_entry.depth:
                    self.pop()

                # 2) if 'lasti' is true, then push the offset that the exception was raised at
                if exn_tab_entry.lasti:
                    self.push(
                        variables.ConstantVariable(self.current_instruction.offset)
                    )

                # 3) push the exception to the stack
                self.push(self.exn_vt_stack.get_current_exception())

                # 4) jump to the handler
                self.jump(exn_tab_entry)  # type: ignore[arg-type]
            else:
                # No handler found. Bubble the exception to the parent
                # instruction translator. We use special exception for this.
                self.stack.clear()
                if type(self) is InstructionTranslator:
                    bubble_exception_to_interpreter()
                raise raised_exception
        else:
            if len(self.block_stack):
                # base implementation - https://github.com/python/cpython/blob/3.10/Python/ceval.c#L4455

                block_stack_entry = self.block_stack.pop()

                while block_stack_entry.inst.opname == "EXCEPT_HANDLER":
                    # TODO(anijain2305) - This is not tested .. unable to create a testcase
                    # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1456
                    self.popn(3)
                    self.exn_vt_stack.pop()
                    if len(self.block_stack) == 0:
                        # No handler found in this frame. Bubble the exception to the parent
                        # instruction translator.
                        self.stack.clear()
                        if type(self) is InstructionTranslator:
                            unimplemented(
                                gb_type="Observed exception (EXCEPT_HANDLER)",
                                context=str(raised_exception),
                                explanation=observed_exn_gb_explanation
                                + " This graph break is unexpected.",
                                hints=[*graph_break_hints.DYNAMO_BUG],
                            )

                        raise raised_exception
                    block_stack_entry = self.block_stack.pop()

                exception_var = self.exn_vt_stack.get_current_exception()
                self.exn_vt_stack.move_current_exception_to_stack()

                # 1) pop values from the stack until it matches the stack depth
                # for the handler
                while len(self.stack) > block_stack_entry.stack_index:
                    self.pop()

                # Push a dummy block stack entry of EXCEPT_HANDLER
                # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1456
                except_handler_inst = Instruction(1e6, "EXCEPT_HANDLER", None, 0)
                self.block_stack.append(
                    BlockStackEntry(except_handler_inst, None, len(self.stack))
                )

                # Push old exception
                if len(self.exn_vt_stack) >= 2:
                    old_exception = self.exn_vt_stack[-2]

                    # Push the old exception on to stack - tb, value, type
                    # Traceback is currently mapped to UnknownVariable
                    self.push(variables.UnknownVariable())
                    self.push(old_exception)
                    self.push(variables.BuiltinVariable(old_exception.exc_type))
                else:
                    # Push empty exception tb, value, type
                    self.push(variables.ConstantVariable(None))
                    self.push(variables.ConstantVariable(None))
                    self.push(variables.ConstantVariable(None))

                # Push new exception - tb, val, type
                # Traceback is currently mapped to UnknownVariable
                self.push(variables.UnknownVariable())
                self.push(exception_var)
                self.push(variables.BuiltinVariable(exception_var.exc_type))

                # Jump to target
                self.jump(block_stack_entry)
            else:
                # No handler found. Bubble the exception to the parent
                # instruction translator. We use special exception for this.
                self.stack.clear()
                if type(self) is InstructionTranslator:
                    bubble_exception_to_interpreter()
                raise raised_exception

    def PUSH_EXC_INFO(self, inst: Instruction) -> None:
        # https://docs.python.org/3/library/dis.html#opcode-PUSH_EXC_INFO
        #   Pops a value from the stack. Pushes the current exception to the top
        #   of the stack. Pushes the value originally popped back to the stack.
        #
        # The behavior of this opcode in CPython is a bit different than what it
        # is described. It pops a value from the stack, pushes the top of the
        # exception stack to the interpreter stack and moves the
        # "current exception" to the exception stack.
        #
        # As an example, suppose the stack is in the following state:
        #   + stack = [..., ConstantVariable(1), ConstantVariable(2)]
        #   + current_exception = TypeError
        #   + exception_stack = [ValueError]
        #
        # After PUSH_EXC_INFO is executed
        #   + stack = [..., ConstantVariable(1), ValueError, ConstantVariable(2)]
        #   + current_exception = None
        #   + exception_stack = [ValueError, TypeError]

        val = self.pop()
        if len(self.exn_vt_stack) == 0:
            prev_exc: VariableTracker = ConstantVariable(None)
        else:
            prev_exc = self.exn_vt_stack[-1]
        self.push(prev_exc)
        self.push(val)
        self.exn_vt_stack.move_current_exception_to_stack()

    def POP_EXCEPT(self, inst: Instruction) -> None:
        if sys.version_info >= (3, 11):
            _ = self.pop()
            # This exception is handled and therefore we can clear the error indicator
            assert len(self.exn_vt_stack)
            self.exn_vt_stack.pop()
        else:
            assert len(self.block_stack) > 0
            if self.block_stack[-1].inst.opname != "EXCEPT_HANDLER":
                raise AssertionError(
                    "Bug in Dynamo tracing of exception handling."
                    "Top of the block stack is not EXCEPT_HANDLER."
                )
            self.block_stack.pop()

            self.popn(3)

            # This exception is handled and therefore we can clear the error indicator
            assert len(self.exn_vt_stack)
            self.exn_vt_stack.pop()

    def check_if_exc_matches(self) -> bool:
        assert len(self.stack) >= 2
        expected_exc_types = self.pop()
        if sys.version_info >= (3, 11):
            # CHECK_EXC_MATCH (which is used from 3.11 onwards) does not pop.
            # This is the description from the disassembly doc
            #
            # Performs exception matching for ``except``. Tests whether the ``STACK[-2]``
            # is an exception matching ``STACK[-1]``. Pops ``STACK[-1]`` and pushes the boolean
            # result of the test.
            exc_instance = self.stack[-1]
        else:
            # This is used prior to 3.11 via opcode JUMP_IF_NOT_EXC_MATCH
            # There is no documentation but here is the code pointer that does 2 pops
            # https://github.com/python/cpython/blob/3.10/Python/ceval.c#L3650-L3665
            exc_instance = self.stack.pop()

        # Users can check exception in 3 ways
        # 1) except NotImplementedError --> BuiltinVariable
        # 2) except CustomException --> UserDefinedExceptionClassVariable
        # 3) except (NotImplementedError, AttributeError) -> TupleVariable

        if not isinstance(
            expected_exc_types,
            (
                BuiltinVariable,
                TupleVariable,
                UserDefinedExceptionClassVariable,
                UserDefinedExceptionObjectVariable,
            ),
        ):
            unimplemented(
                gb_type="Exception with bad expected type",
                context=str(expected_exc_types),
                explanation=f"`except ...` has unsupported type {expected_exc_types}.",
                hints=[*graph_break_hints.USER_ERROR],
            )

        if sys.version_info >= (3, 11):
            if not self._isinstance_exception(exc_instance):
                unimplemented(
                    gb_type="Caught non-Exception value",
                    context=str(exc_instance),
                    explanation=f"Except expects to receive an object of Exception type but received {exc_instance}.",
                    hints=[*graph_break_hints.USER_ERROR],
                )

        if isinstance(expected_exc_types, TupleVariable):
            expected_types = expected_exc_types.items
        else:
            expected_types = [
                expected_exc_types,
            ]

        for expected_type in expected_types:
            if not isinstance(
                expected_type,
                (
                    BuiltinVariable,
                    UserDefinedExceptionObjectVariable,
                    UserDefinedExceptionClassVariable,
                ),
            ):
                unimplemented(
                    gb_type="Exception with non-type expectation",
                    context=str(expected_type),
                    explanation=f"`except ...` expects a non-type: {exp

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 14 class(es): class, class, class, class, TensorifyState, class, SpeculationLogDivergence, ReturnValueOp, YieldValueOp, BytecodeDispatchTableMeta, class, InstructionTranslator, InliningInstructionTranslator, InliningGeneratorInstructionTranslator

### Functions
This file defines 248 function(s): _import_module, fail_and_restart_analysis, failed, restart, clear, next, render, specialize, should_specialize, clear, empty, _step_logger, save_and_restart_speculation_log, temporarely_allow_writes_to_output_graph, can_restore, resume_fn, exit, stack_op, impl, is_stdlib, get_assert_bytecode_sequence, fn, fn, _detect_and_normalize_assert_statement, generic_jump, jump_graph_break, inner, break_graph_if_unsupported, decorator, wrapper


## Key Components

The file contains 17552 words across 5263 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 214847 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
