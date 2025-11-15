# Documentation: `torch/_dynamo/guards.py`

## File Metadata

- **Path**: `torch/_dynamo/guards.py`
- **Size**: 186,430 bytes (182.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Core guard system for Dynamo that detects when compiled code needs to be recompiled due to
changes in program state. Guards are conditions that must remain true for previously-compiled
code to be valid for reuse.

This module provides the infrastructure for creating, managing and checking guards, including:
- Guard creation and composition
- Guard state management and invalidation
- Guard checking and failure handling
- Utilities for guard optimization and debugging
- Integration with Dynamo's compilation caching

The guard system is critical for Dynamo's ability to efficiently reuse compiled code while
maintaining correctness by detecting when recompilation is necessary due to changes in
program state, tensor properties, or control flow.
"""

from __future__ import annotations

import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import io
import logging
import math
import pickle
import sys
import textwrap
import traceback
import types
import warnings
import weakref
from contextlib import contextmanager
from copy import deepcopy
from inspect import currentframe
from typing import Any, NoReturn, Optional, TYPE_CHECKING, Union


try:
    from typing import LiteralString
except ImportError:
    from typing_extensions import LiteralString

from typing_extensions import TypeAliasType, TypeVar
from weakref import ReferenceType

import torch
import torch.overrides
import torch.utils._device
from torch._C._dynamo.eval_frame import code_framelocals_names
from torch._C._dynamo.guards import (
    check_obj_id,
    check_type_id,
    ClosureGuardAccessor,
    CodeGuardAccessor,
    dict_version,
    DictGetItemGuardAccessor,
    DictGuardManager,
    FuncDefaultsGuardAccessor,
    FuncKwDefaultsGuardAccessor,
    GetAttrGuardAccessor,
    GetGenericDictGuardAccessor,
    GuardAccessor,
    GuardDebugInfo,
    GuardManager,
    install_no_tensor_aliasing_guard,
    install_object_aliasing_guard,
    install_storage_overlapping_guard,
    install_symbolic_shape_guard,
    LeafGuard,
    profile_guard_manager,
    RelationalGuard,
    RootGuardManager,
    TupleGetItemGuardAccessor,
    TypeDictGuardAccessor,
    TypeGuardAccessor,
    TypeMROGuardAccessor,
)
from torch._dynamo.source import (
    get_global_source_name,
    get_local_source_name,
    IndexedSource,
    is_from_flatten_script_object_source,
    is_from_local_source,
    is_from_optimizer_source,
    is_from_skip_guard_source,
    is_from_unspecialized_builtin_nn_module_source,
    TensorProperty,
    TensorPropertySource,
)
from torch._dynamo.utils import CompileEventLogger, get_metrics_context
from torch._guards import (
    CompileContext,
    CompileId,
    DuplicateInputs,
    Guard,
    GuardBuilderBase,
    GuardEnvExpr,
    GuardSource,
    Source,
    StorageOverlap,
)
from torch._inductor.utils import IndentedBuffer
from torch._logging import structured
from torch._utils_internal import justknobs_check
from torch.fx.experimental.symbolic_shapes import (
    _CppShapeGuardsHelper,
    _ShapeGuardsHelper,
    EqualityConstraint,
    is_symbolic,
    SYMPY_INTERP,
)
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef

from . import config, convert_frame, exc
from .eval_frame import set_guard_error_hook
from .source import (
    AttrProxySource,
    AttrSource,
    CallFunctionNoArgsSource,
    CallMethodItemSource,
    ChainedSource,
    ClosureSource,
    CodeSource,
    ConstantSource,
    ConstDictKeySource,
    CurrentStreamSource,
    DataclassFieldsSource,
    DefaultsSource,
    DictGetItemSource,
    DictSubclassGetItemSource,
    DynamicScalarSource,
    FlattenScriptObjectSource,
    FloatTensorSource,
    FSDPNNModuleSource,
    GenericAttrSource,
    GetItemSource,
    GlobalSource,
    GlobalStateSource,
    GlobalWeakRefSource,
    GradSource,
    ListGetItemSource,
    LocalSource,
    NamedTupleFieldsSource,
    NNModuleSource,
    NonSerializableSetGetItemSource,
    NumpyTensorSource,
    OptimizerSource,
    ScriptObjectQualifiedNameSource,
    ShapeEnvSource,
    SubclassAttrListSource,
    TorchFunctionModeStackSource,
    TorchSource,
    TupleIteratorGetItemSource,
    TypeDictSource,
    TypeMROSource,
    TypeSource,
    UnspecializedBuiltinNNModuleSource,
    UnspecializedNNModuleSource,
    UnspecializedParamBufferSource,
    WeakRefCallSource,
)
from .types import (  # noqa: F401
    CacheEntry,
    DynamoFrameType,
    ExtraState,
    GuardedCode,
    GuardFail,
    GuardFilterEntry,
    GuardFn,
)
from .utils import (
    builtin_dict_keys,
    common_constant_types,
    dataclass_fields,
    dict_keys,
    get_current_stream,
    get_custom_getattr,
    get_torch_function_mode_stack,
    get_torch_function_mode_stack_at,
    guard_failures,
    istype,
    key_is_id,
    key_to_id,
    normalize_range_iter,
    orig_code_map,
    tensor_always_has_static_shape,
    tuple_iterator_getitem,
    tuple_iterator_len,
    unpatched_nn_module_getattr,
    verify_guard_fn_signature,
)


if TYPE_CHECKING:
    from collections.abc import Callable


guard_manager_testing_hook_fn: Optional[Callable[[Any, Any, Any], Any]] = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from collections.abc import Generator, KeysView, Sequence

    from sympy import Symbol

    from torch._C import DispatchKeySet
    from torch._dynamo.output_graph import OutputGraphCommon, OutputGraphGuardsState

T = TypeVar("T")
log = logging.getLogger(__name__)
guards_log = torch._logging.getArtifactLogger(__name__, "guards")
recompiles_log = torch._logging.getArtifactLogger(__name__, "recompiles")
recompiles_verbose_log = torch._logging.getArtifactLogger(
    __name__, "recompiles_verbose"
)
verbose_guards_log = torch._logging.getArtifactLogger(__name__, "verbose_guards")


dunder_attrs_assumed_constants = (
    "__defaults__",
    "__kwdefaults__",
    "__code__",
    "__closure__",
    "__annotations__",
    "__func__",
    "__mro__",
)


def get_framelocals_idx(code: types.CodeType, var_name: str) -> int:
    # Refer to index in the frame's localsplus directly.
    # NOTE: name order for a code object doesn't change.
    # NOTE: we need to find the LAST matching index because <= 3.10 contains
    # duplicate names in the case of cells: a name can be both local and cell
    # and will take up 2 slots of the frame's localsplus. The correct behavior
    # is to refer to the cell, which has a higher index.
    framelocals_names_reversed = code_framelocals_names_reversed_cached(code)
    framelocals_idx = (
        len(framelocals_names_reversed) - framelocals_names_reversed.index(var_name) - 1
    )
    return framelocals_idx


class IndentedBufferWithPrefix(IndentedBuffer):
    def prefix(self) -> str:
        return "| " * (self._indent * self.tabwidth)

    def writeline(self, line: str, skip_prefix: bool = False) -> None:  # type: ignore[override]
        if skip_prefix:
            super().writeline(line)
        else:
            super().writeline("+- " + line)


class GuardManagerWrapper:
    """
    A helper class that contains the root guard manager. An instance of this
    class is stored in the Dynamo cache entry, so that the cache entry can
    access the RootGuardManager stored in the "root" attribute and directly call
    the check_nopybind from C++.
    """

    def __init__(self, root: Optional[RootGuardManager] = None) -> None:
        if root is None:
            self.root = RootGuardManager()
        else:
            self.root = root

        self.diff_guard_root: Optional[RootGuardManager] = None
        self.closure_vars: Optional[dict[str, Any]] = None
        self.args: Optional[list[str]] = None
        self.code_parts: list[str] = []
        self.verbose_code_parts: Optional[list[str]] = None
        self.global_scope: Optional[dict[str, Any]] = None
        self.guard_fail_fn: Optional[Callable[[GuardFail], None]] = None
        self.cache_entry: Optional[CacheEntry] = None
        self.extra_state: Optional[ExtraState] = None
        self.id_matched_objs: dict[str, ReferenceType[object]] = {}
        self.no_tensor_aliasing_sources: list[str] = []

        self.printed_relational_guards: set[RelationalGuard] = set()

        self.diff_guard_sources: OrderedSet[str] = OrderedSet()

    @contextmanager
    def _preserve_printed_relational_guards(self) -> Generator[None, None, None]:
        self.printed_relational_guards = set()
        try:
            yield
        finally:
            self.printed_relational_guards = set()

    # TODO: clarify what fn and attributes guard manager has to get the right things here
    def collect_diff_guard_sources(self) -> OrderedSet[str]:
        # At the time of finalize, we have only marked guard managers with
        # TENSOR_MATCH guards as diff guard managers. So, we do a tree traversal
        # and collect all the nodes in the tree (branches) that lead to tensor
        # guards.

        # After a recompilation, some of guard managers will have a fail_count >
        # 0, so we collect them as well. Later on, we accumulate the diff guard
        # sources for all the guard managers.

        def visit_dict_manager(node: DictGuardManager) -> bool:
            is_diff_guard_node = (
                node.get_source() in self.diff_guard_sources or node.fail_count() > 0
            )
            for _idx, (key_mgr, val_mgr) in sorted(
                node.get_key_value_managers().items()
            ):
                is_diff_guard_node |= visit(key_mgr) | visit(val_mgr)

            if is_diff_guard_node:
                self.diff_guard_sources.add(node.get_source())

            return is_diff_guard_node

        def visit_manager(node: GuardManager) -> bool:
            assert not isinstance(node, DictGuardManager)

            is_diff_guard_node = (
                node.get_source() in self.diff_guard_sources or node.fail_count() > 0
            )
            for child_mgr in node.get_child_managers():
                is_diff_guard_node |= visit(child_mgr)

            if is_diff_guard_node:
                self.diff_guard_sources.add(node.get_source())

            return is_diff_guard_node

        def visit(node: GuardManager) -> bool:
            if node is None:
                return False
            if isinstance(node, DictGuardManager):
                return visit_dict_manager(node)
            return visit_manager(node)

        visit(self.root)

        return self.diff_guard_sources

    def finalize(self) -> None:
        if config.use_recursive_dict_tags_for_guards and justknobs_check(
            "pytorch/compiler:use_recursive_dict_tags_for_guards"
        ):
            self.find_tag_safe_roots()
        self.prepare_diff_guard_manager()

    def prepare_diff_guard_manager(self) -> None:
        self.collect_diff_guard_sources()
        self.populate_diff_guard_manager()

    def find_tag_safe_roots(self) -> None:
        """
        Identify ``tag safe nodes`` and ``tag safe roots`` within a guard tree.

        -----------------------------------------------------------------------
        tag safe node
        -----------------------------------------------------------------------
        A *tag safe node* is a ``GuardManager`` whose guarded value satisfies one
        of the following conditions:

        1. Immutable value - The value is intrinsically immutable according to
        ``is_immutable_object``. Tensors are considered immutable. To ensure
        that symbolic guards run, we also check that the GuardManager has no
        accessors.

        2. Nested tag safe dictionary - The value is a ``dict`` whose keys and
        values are all tag safe nodes  (checked recursively).  Such dictionaries
        allow entire nested structures to be skipped once their identity tag
        matches.

        3. Pure ``nn.Module`` - The value is an ``nn.Module`` whose sole
        accessor is ``GetGenericDictGuardAccessor``—i.e., it only exposes its
        ``__dict__`` and nothing else that could mutate between runs.

        For every tag safe node, verifying the identity/tag of just the top-level
        dictionary is enough to guarantee the entire subtree is unchanged, enabling
        a *fast-path* guard check.

        -----------------------------------------------------------------------
        tag safe root
        -----------------------------------------------------------------------
        A ``tag safe root`` is a tag safe node whose parent is not tag safe.
        These boundary nodes mark the points where guard evaluation can safely
        prune traversal: if a tag-safe root's dictionary tag matches, the entire
        subtree beneath it is skipped.

        One strong requirement for tag safe root is for the guarded object to
        support weakref. Refer to more details in the Recursive dict tag
        matching note. In short, we need to save the weakref of the object on
        first invocation, and check if it is still valid in later iterations, to
        apply recursive dict tag optimizations. `dict` objects do NOT support
        weakref. Therefore, as of now, we only mark nn module related guard
        managers as tag safe roots.

        Algorithm
        ---------
        The search runs in post-order traversal

        1. Visit leaves and classify them as tag safe or not.
        2. Propagate tag-safety upward: a parent dictionary becomes tag safe only if
        all of its children are already tag-safe.
        3. Propagate tag-safe-rootness upward: if the whole subtree is tag safe,
        the current node becomes the new tag safe root, otherwise propagate the
        subtree tag safe roots.
        4. Collect every tag safe node and, by inspecting parent tags, label the
        subset that are tag safe roots.
        """

        def check_tag_safety(
            node: GuardManager, accepted_accessors: tuple[type[GuardAccessor], ...]
        ) -> bool:
            accessors = node.get_accessors()
            child_mgrs = node.get_child_managers()
            return all(
                isinstance(accessor, accepted_accessors) and mgr.is_tag_safe()
                for accessor, mgr in zip(accessors, child_mgrs)
            )

        def visit_dict_manager(node: DictGuardManager) -> list[GuardManager]:
            # Just recurse through the key and value dict managers and check if
            # all of them are tag safe nodes.
            assert issubclass(node.get_type_of_guarded_value(), dict)

            tag_safe_roots = []
            is_subtree_tag_safe = True

            # Recurse to get the tag safe roots from subtree.
            for _idx, (key_mgr, val_mgr) in sorted(
                node.get_key_value_managers().items()
            ):
                if key_mgr is not None:
                    visit(key_mgr)
                if val_mgr is not None:
                    tag_safe_roots.extend(visit(val_mgr))

            for key_mgr, val_mgr in node.get_key_value_managers().values():
                if key_mgr:
                    is_subtree_tag_safe &= key_mgr.is_tag_safe()

                if val_mgr:
                    is_subtree_tag_safe &= val_mgr.is_tag_safe()

            if is_subtree_tag_safe:
                node.mark_tag_safe()
            return tag_safe_roots

        def visit_manager(node: GuardManager) -> list[GuardManager]:
            assert not isinstance(node, DictGuardManager)

            # Collect the subtree tag safe roots
            tag_safe_roots = []
            for child_mgr in node.get_child_managers():
                tag_safe_roots.extend(visit(child_mgr))

            if node.is_guarded_value_immutable():
                # If the node guards a tensor, mark it tag safe only if there
                # are no accessors. Presence of accessors means presence of
                # symbolic shape guards.
                if issubclass(node.get_type_of_guarded_value(), torch.Tensor):
                    if node.has_no_accessors() and not node.has_object_aliasing_guard():
                        node.mark_tag_safe()
                else:
                    node.mark_tag_safe()
            elif issubclass(node.get_type_of_guarded_value(), dict):
                accessors = node.get_accessors()
                child_mgrs = node.get_child_managers()
                is_subtree_tag_safe = all(
                    isinstance(accessor, DictGetItemGuardAccessor) and mgr.is_tag_safe()
                    for accessor, mgr in zip(accessors, child_mgrs)
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()
            elif issubclass(node.get_type_of_guarded_value(), torch.nn.Module):
                is_subtree_tag_safe = check_tag_safety(
                    node, (GetGenericDictGuardAccessor, TypeGuardAccessor)
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()
                    # Return the current node as tag safe root, discarding the
                    # subtree tag safe roots.
                    return [
                        node,
                    ]
            elif (
                node.get_type_of_guarded_value()
                in (
                    types.FunctionType,
                    types.MethodType,
                    staticmethod,
                    classmethod,
                )
                and config.assume_dunder_attributes_remain_unchanged
            ):
                # Assumption: callers will not reassignthe attributes
                #   func.__code__, func.__closure__, func.__defaults__, or func.__kwdefaults__.
                # Mutating the objects those attributes point to is fine;
                # rebinding the attribute itself is not.
                # Example ─ allowed:   foo.__defaults__[0].bar = 99
                #          forbidden: foo.__defaults__ = (3, 4)
                is_subtree_tag_safe = check_tag_safety(
                    node,
                    (
                        CodeGuardAccessor,
                        ClosureGuardAccessor,
                        FuncDefaultsGuardAccessor,
                        FuncKwDefaultsGuardAccessor,
                        GetAttrGuardAccessor,
                    ),
                )

                for accessor in node.get_accessors():
                    if isinstance(accessor, GetAttrGuardAccessor):
                        is_subtree_tag_safe &= (
                            accessor.get_attr_name() in dunder_attrs_assumed_constants
                        )

                if is_subtree_tag_safe:
                    node.mark_tag_safe()
            elif issubclass(node.get_type_of_guarded_value(), types.CellType):
                is_subtree_tag_safe = check_tag_safety(node, (GetAttrGuardAccessor,))

                is_subtree_tag_safe &= all(
                    isinstance(accessor, GetAttrGuardAccessor)
                    and accessor.get_attr_name() == "cell_contents"
                    for accessor in node.get_accessors()
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()
            elif (
                issubclass(node.get_type_of_guarded_value(), tuple)
                and node.get_source().endswith(dunder_attrs_assumed_constants)
                and config.assume_dunder_attributes_remain_unchanged
            ):
                # We trust tuples obtained from a function's __closure__ or
                # __defaults__. Any *other* tuple-valued attribute can be
                # silently replaced—for example:
                #
                #     foo.bar = (1, 2)      # original
                #     foo.bar = (3, 4)      # rebinding that our dict-tag optimisation won't see
                #
                # Therefore only tuples from __closure__ / __defaults__ participate in the
                # recursive-dict-tag optimization; all others are ignored.
                is_subtree_tag_safe = check_tag_safety(
                    node, (TupleGetItemGuardAccessor,)
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()
            elif issubclass(node.get_type_of_guarded_value(), type):
                is_subtree_tag_safe = check_tag_safety(
                    node, (TypeDictGuardAccessor, TypeMROGuardAccessor)
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()

            return tag_safe_roots

        def visit(node: GuardManager) -> list[GuardManager]:
            if node is None:
                return []
            if isinstance(node, DictGuardManager):
                return visit_dict_manager(node)
            return visit_manager(node)

        tag_safe_roots = visit(self.root)
        for node in tag_safe_roots:
            if issubclass(node.get_type_of_guarded_value(), torch.nn.Module):
                node.mark_tag_safe_root()

    def populate_diff_guard_manager(self) -> None:
        self.diff_guard_root = self.clone_with_chosen_sources(self.diff_guard_sources)

        # Ensure that that C++ side points to the updated diff guard manager.
        # When a new GuardManagerWrapper is created, it does not have a
        # cache_entry attribute, so it relies on the CacheEntry constructor to
        # set the diff_guard_root in C++.  But once it is saved in the Dynamo
        # cache, C++ side adds a cache_entry attribute. On recompiles, this
        # cache_entry is visible, so we update the C++ side to point to the
        # update guard manager.
        if self.cache_entry:
            self.cache_entry.update_diff_guard_root_manager()

    def clone_with_chosen_sources(
        self, chosen_sources: OrderedSet[str]
    ) -> RootGuardManager:
        def filter_fn(node_mgr: GuardManager) -> bool:
            return node_mgr.get_source() in chosen_sources

        return self.root.clone_manager(filter_fn)

    def get_guard_lines(self, guard: LeafGuard) -> list[str]:
        guard_name = guard.__class__.__name__
        parts = guard.verbose_code_parts()
        parts = [guard_name + ": " + part for part in parts]
        return parts

    def get_manager_line(
        self, guard_manager: GuardManager, accessor_str: Optional[str] = None
    ) -> str:
        source = guard_manager.get_source()
        t = guard_manager.__class__.__name__
        s = t + ": source=" + source
        if accessor_str:
            s += ", " + accessor_str
        s += f", type={guard_manager.get_type_of_guarded_value()}"
        s += f", tag_safe=({guard_manager.is_tag_safe()}, {guard_manager.is_tag_safe_root()})"
        return s

    def construct_dict_manager_string(
        self, mgr: DictGuardManager, body: IndentedBufferWithPrefix
    ) -> None:
        for idx, (key_mgr, val_mgr) in sorted(mgr.get_key_value_managers().items()):
            body.writeline(f"KeyValueManager pair at index={idx}")
            with body.indent():
                if key_mgr:
                    body.writeline(f"KeyManager: {self.get_manager_line(key_mgr)}")
                    self.construct_manager_string(key_mgr, body)

                if val_mgr:
                    body.writeline(f"ValueManager: {self.get_manager_line(val_mgr)}")
                    self.construct_manager_string(val_mgr, body)

    def construct_manager_string(
        self, mgr: GuardManager, body: IndentedBufferWithPrefix
    ) -> None:
        with body.indent():
            for guard in mgr.get_leaf_guards():
                if isinstance(guard, RelationalGuard):
                    if guard not in self.printed_relational_guards:
                        self.printed_relational_guards.add(guard)
                        # pyrefly: ignore [bad-argument-type]
                        body.writelines(self.get_guard_lines(guard))
                    else:
                        body.writelines(
                            [
                                guard.__class__.__name__,
                            ]
                        )
                else:
                    body.writelines(self.get_guard_lines(guard))

            # This works for both DictGuardManager and SubclassedDictGuardManager
            if isinstance(mgr, DictGuardManager):
                self.construct_dict_manager_string(mgr, body)

            # General case of GuardManager/RootGuardManager
            for accessor, child_mgr in zip(
                mgr.get_accessors(), mgr.get_child_managers()
            ):
                body.writeline(
                    self.get_manager_line(child_mgr, f"accessed_by={accessor.repr()}")
                )
                self.construct_manager_string(child_mgr, body)

    def __str__(self) -> str:
        with self._preserve_printed_relational_guards():
            body = IndentedBufferWithPrefix()
            body.tabwidth = 1
            body.writeline("", skip_prefix=True)
            body.writeline("TREE_GUARD_MANAGER:", skip_prefix=True)
            body.writeline("RootGuardManager")
            self.construct_manager_string(self.root, body)
            if hasattr(self.root, "get_epilogue_lambda_guards"):
                for guard in self.root.get_epilogue_lambda_guards():
                    body.writelines(self.get_guard_lines(guard))
            return body.getvalue()

    def check(self, x: Any) -> bool:
        # Only needed for debugging purposes.
        return self.root.check(x)

    def check_verbose(self, x: Any) -> GuardDebugInfo:
        # Only needed for debugging purposes.
        return self.root.check_verbose(x)

    def populate_code_parts_for_debugging(self) -> None:
        # This should be called when the guard manager is fully populated
        relational_guards_seen = set()

        def get_code_parts(leaf_guard: LeafGuard) -> list[str]:
            code_parts = []
            for verbose_code_part in leaf_guard.verbose_code_parts():
                code_part = verbose_code_part.split("#")[0].rstrip()
                code_parts.append(code_part)
            return code_parts

        def visit(mgr: GuardManager) -> None:
            nonlocal relational_guards_seen
            for guard in mgr.get_leaf_guards():
                if isinstance(guard, RelationalGuard):
                    if guard not in relational_guards_seen:
                        # pyrefly: ignore [bad-argument-type]
                        self.code_parts.extend(get_code_parts(guard))
                        relational_guards_seen.add(guard)
                else:
                    self.code_parts.extend(get_code_parts(guard))

            for child_mgr in mgr.get_child_managers():
                visit(child_mgr)

        visit(self.root)


def from_numpy(a: Any) -> torch.Tensor:
    # If not numpy array, piggy back on e.g. tensor guards to check type
    # Re-enable torch function since we disable it on leaf guards
    # we need it to properly construct the tensor if a default device is set
    with torch.overrides._enable_torch_function():
        # pyrefly: ignore [missing-attribute]
        return torch.as_tensor(a) if isinstance(a, (np.generic, np.ndarray)) else a


# For user stack printing
@functools.cache
def uninteresting_files() -> set[str]:
    import torch._dynamo.external_utils
    import torch._dynamo.polyfills

    mods = [torch._dynamo.external_utils, torch._dynamo.polyfills]

    from torch._dynamo.polyfills.loader import POLYFILLED_MODULES

    # pyrefly: ignore [bad-argument-type]
    mods.extend(POLYFILLED_MODULES)

    return {inspect.getfile(m) for m in mods}


_CLOSURE_VARS: Optional[dict[str, object]] = None


def _get_closure_vars() -> dict[str, object]:
    global _CLOSURE_VARS
    if _CLOSURE_VARS is None:
        _CLOSURE_VARS = {
            "___check_type_id": check_type_id,
            "___check_obj_id": check_obj_id,
            "___odict_getitem": collections.OrderedDict.__getitem__,
            "___key_to_id": key_to_id,
            "___dict_version": dict_version,
            "___dict_contains": lambda a, b: dict.__contains__(b, a),
            "___tuple_iterator_len": tuple_iterator_len,
            "___normalize_range_iter": normalize_range_iter,
            "___tuple_iterator_getitem": tuple_iterator_getitem,
            "___dataclass_fields": dataclass_fields,
            "___namedtuple_fields": lambda x: x._fields,
            "___get_torch_function_mode_stack_at": get_torch_function_mode_stack_at,
            "___get_current_stream": get_current_stream,
            "__math_isnan": math.isnan,
            "__numpy_isnan": None if np is None else np.isnan,
            "inf": float("inf"),
            "__load_module": importlib.import_module,
            "utils_device": torch.utils._device,
            "device": torch.device,
            "___from_numpy": from_numpy,
            "___as_tensor": torch._as_tensor_fullprec,
            "torch": torch,
            "inspect": inspect,
        }
    return _CLOSURE_VARS


def _ast_unparse(node: ast.AST) -> str:
    return ast.unparse(node).replace("\n", "")


strip_function_call = torch._C._dynamo.strip_function_call


def get_verbose_code_part(code_part: str, guard: Optional[Guard]) -> str:
    extra = ""
    if guard is not None:
        if guard.user_stack:
            for fs in reversed(guard.user_stack):
                if fs.filename not in uninteresting_files():
                    extra = f"  # {format_frame(fs, line=True)}"
                    if len(extra) > 1024:
                        # For fx graphs, the line can be very long in case of
                        # torch.stack ops, where many inputs are set to None
                        # after the operation.  This increases the size of the
                        # guards log file.  In such cases, do not print the line
                        # contents.
                        extra = f"  # {format_frame(fs)}"
                    break
        elif guard.stack:
            summary = guard.stack.summary()
            if len(summary) > 0:
                extra = f"  # {format_frame(summary[-1])}"
            else:
                extra = "  # <unknown>"
    return f"{code_part:<60}{extra}"


def get_verbose_code_parts(
    code_parts: Union[str, list[str]],
    guard: Optional[Guard],
    recompile_hint: Optional[str] = None,
) -> list[str]:
    if not isinstance(code_parts, list):
        code_parts = [code_parts]

    verbose_code_parts = [
        get_verbose_code_part(code_part, guard) for code_part in code_parts
    ]
    if recompile_hint:
        verbose_code_parts = [
            f"{part} (HINT: {recompile_hint})" for part in verbose_code_parts
        ]

    return verbose_code_parts


def convert_int_to_concrete_values(dim: Any) -> Optional[int]:
    if dim is None:
        return None
    if not is_symbolic(dim):
        return dim
    else:
        assert isinstance(dim, torch.SymInt)
        return dim.node.maybe_as_int()


def convert_to_concrete_values(size_or_stride: list[Any]) -> list[Optional[int]]:
    return [convert_int_to_concrete_values(dim) for dim in size_or_stride]


def get_tensor_guard_code_part(
    value: torch.Tensor,
    name: str,
    sizes: list[Optional[int]],
    strides: list[Optional[int]],
    pytype: type,
    dispatch_keys: DispatchKeySet,
) -> str:
    dispatch_key = (
        dispatch_keys | torch._C._dispatch_tls_local_include_set()
    ) - torch._C._dispatch_tls_local_exclude_set()
    dtype = value.dtype
    device_index = value.device.index
    requires_grad = value.requires_grad
    guard_str = (
        f"check_tensor({name}, {pytype.__qualname__}, {dispatch_key}, {dtype}, "
        f"device={device_index}, requires_grad={requires_grad}, size={sizes}, stride={strides})"
    )
    return guard_str


def get_key_index(dct: dict[Any, Any], key: Any) -> int:
    # Ensure that we call dict.keys and not value.keys (which can call
    # overridden keys method). In the C++ guards, we relied on PyDict_Next
    # to traverse the dictionary, which uses the internal data structure and
    # does not call the overridden keys method.
    return list(builtin_dict_keys(dct)).index(key)


def get_key_index_source(source: Any, index: Any) -> str:
    return f"list(dict.keys({source}))[{index}]"


def raise_local_type_error(obj: Any) -> NoReturn:
    raise TypeError(
        f"Type {type(obj)} for object {obj} cannot be saved "
        + "into torch.compile() package since it's defined in local scope. "
        + "Please define the class at global scope (top level of a module)."
    )


def should_optimize_getattr_on_nn_module(value: Any) -> bool:
    # If inline_inbuilt_nn_modules flag is True, Dynamo has already traced
    # through the __getattr__, and therefore it is always safe to optimize
    # getattr on nn modules.
    return isinstance(value, torch.nn.Module) and (
        config.inline_inbuilt_nn_modules
        or get_custom_getattr(value) is unpatched_nn_module_getattr
    )


@dataclasses.dataclass(frozen=True)
class NNModuleAttrAccessorInfo:
    # Represents where is the attr name is present in the nn module attribute
    # access

    # Tells that the attribute can be accessed via __dict__
    present_in_generic_dict: bool = False

    # Either the actual name or _parameters/_buffers/_modules
    l1_key: Optional[str] = None

    # Actual parameter/buffer/submodule name
    l2_key: Optional[str] = None


def getitem_on_dict_manager(
    source: Union[DictGetItemSource, DictSubclassGetItemSource],
    base_guard_manager: DictGuardManager,
    base_example_value: Any,
    example_value: Any,
    guard_manager_enum: GuardManagerType,
) -> GuardManager:
    base_source_name = source.base.name()
    if isinstance(source.index, ConstDictKeySource):
        index = source.index.index
    else:
        assert isinstance(base_example_value, dict)
        index = get_key_index(base_example_value, source.index)

    key_source = get_key_index_source(base_source_name, index)

    # Ensure that we call dict.keys and not value.keys (which can call
    # overridden keys method). In the C++ guards, we relied on PyDict_Next
    # to traverse the dictionary, which uses the internal data structure and
    # does not call the overridden keys method.
    key_example_value = list(builtin_dict_keys(base_example_value))[index]
    if isinstance(key_example_value, (int, str)):
        value_source = f"{base_source_name}[{key_example_value!r}]"
    else:
        value_source = f"{base_source_name}[{key_source}]"
    if not isinstance(source.index, ConstDictKeySource):
        # We have to insert a key manager guard here
        # TODO - source debug string is probably wrong here.
        base_guard_manager.get_key_manager(
            index=index,
            source=key_source,
            example_value=source.index,
            guard_manager_enum=GuardManagerType.GUARD_MANAGER,
        ).add_equals_match_guard(
            source.index, [f"{key_source} == {key_example_value!r}"]
        )

    return base_guard_manager.get_value_manager(
        index=index,
        source=value_source,
        example_value=example_value,
        guard_manager_enum=guard_manager_enum,
    )


def match_on_id_for_tensor(guard: Guard) -> bool:
    source = guard.originating_source
    # For numpy tensors, always use TENSOR_MATCH because __from_numpy leads
    # to a new tensor every time and therefore id differs.
    if isinstance(source, NumpyTensorSource):
        return False

    if guard.is_specialized_nn_module():
        return True

    return source.is_dict_key() and not isinstance(source, GradSource)


# The ready to eval generated code (possibly multiple parts) for a guard, plus
# the original guard object that created it for provenance
@dataclasses.dataclass
class GuardCodeList:
    code_list: list[str]
    guard: Guard


class GuardManagerType(enum.Enum):
    GUARD_MANAGER = 1
    DICT_GUARD_MANAGER = 2


@functools.cache
def code_framelocals_names_reversed_cached(code: types.CodeType) -> list[str]:
    return list(reversed(code_framelocals_names(code)))


class GuardBuilder(GuardBuilderBase):
    def __init__(
        self,
        f_code: types.CodeType,
        id_ref: Callable[[object, str], int],
        source_ref: Callable[[Source], str],
        lookup_weakrefs: Callable[[object], Optional[weakref.ref[object]]],
        local_scope: dict[str, object],
        global_scope: dict[str, object],
        guard_manager: GuardManagerWrapper,
        check_fn_manager: CheckFunctionManager,
        save_guards: bool = False,
        runtime_global_scope: Optional[dict[str, object]] = None,
        source_get_cache: Optional[dict[str, Any]] = None,
    ) -> None:
        self.f_code = f_code
        self.id_ref = id_ref
        self.source_ref = source_ref
        self.lookup_weakrefs = lookup_weakrefs
        self.scope: dict[str, dict[str, object]] = {"L": local_scope, "G": global_scope}
        self.runtime_global_scope = runtime_global_scope or global_scope
        self.source_get_cache = source_get_cache or {}
        self.scope["__builtins__"] = builtins.__dict__.copy()
        for (
            name,
            package_module,
        ) in torch.package.package_importer._package_imported_modules.items():
            name = name.replace(">", "_").replace("<", "_").replace(".", "_dot_")
            # Write the package module into the scope so that we can import it
            self.scope["__builtins__"][name] = package_module
            # Write the demangled name to the scope so that we can use it
            self.scope[name] = package_module
        self.guard_manager = guard_manager

        self.argnames: list[str] = []
        # Code is python expression strings generated for each guard
        self.code: list[GuardCodeList] = []
        # shape_env_code is only used by builder and is used for
        # shape env code.  This exists only because we need to make sure
        # shape env guards get run after tensor match guards (since the
        # tensor match guards make sure we actually have tensors)
        self.shape_env_code: list[GuardCodeList] = []

        # Collect the guard managers and debug info to insert no tensor aliasing
        # guards.
        self.no_tensor_aliasing_names: list[str] = []
        self.no_tensor_aliasing_guard_managers: list[GuardManager] = []

        self.check_fn_manager: CheckFunctionManager = check_fn_manager

        self.guard_tree_values: dict[int, Any] = {}
        self.save_guards = save_guards

        # Collect the ids of dicts which need key order guarding. source_name is
        # not sufficient because for nn modules, we can have different sources
        # to access the same object - self._module["param"] is same as
        # self.param.
        self.key_order_guarded_dict_ids = set()
        assert self.check_fn_manager.output_graph is not None
        for source in self.check_fn_manager.output_graph.guard_on_key_order:
            dict_obj = self.get(source.name())
            if self.save_guards:
                self.source_get_cache[source.name()] = dict_obj
            self.key_order_guarded_dict_ids.add(id(dict_obj))

        # Keep track of weak references of objects with ID_MATCH guard. This
        # info is stored alongside optimized_code and guard_manager and is used to
        # limit the number of cache entries with same ID_MATCH'd object.
        self.id_matched_objs: dict[str, ReferenceType[object]] = {}

        # Save the guard managers to avoid repeatedly traversing sources.
        self._cached_guard_managers: dict[str, GuardManager] = {}
        self._cached_duplicate_input_guards: set[tuple[str, str]] = set()
        self.object_aliasing_guard_codes: list[tuple[str, str]] = []
        self.guard_nn_modules = config.guard_nn_modules and justknobs_check(
            "pytorch/compiler:guard_nn_modules"
        )
        self.already_added_code_parts: OrderedSet[str] = OrderedSet()

    def guard_on_dict_keys_and_ignore_order(
        self, example_value: dict[Any, Any], guard: Guard
    ) -> None:
        dict_mgr = self.get_guard_manager(guard)
        if isinstance(dict_mgr, DictGuardManager):
            raise NotImplementedError(
                "Not expecting a DictGuardManager. Seems like Dynamo incorrectly "
                f"added the dict to tx.output.guard_on_key_order for {guard.name}"
            )

        # Iterate over the dicts and install a dict_getitem_manager.
        dict_source = guard.originating_source.name()

        # Ensure that we call dict.keys and not value.keys (which can call
        # overridden keys method). In the C++ guards, we relied on PyDict_Next
        # to traverse the dictionary, which uses the internal data structure and
        # does not call the overridden keys method.
        for key in builtin_dict_keys(example_value):
            value = example_value[key]
            value_source = DictGetItemSource(guard.originating_source, index=key)
            guard_manager_enum = self.get_guard_manager_type(
                value_source, example_value
            )
            dict_mgr.dict_getitem_manager(
                key=key,
                source=f"{dict_source}[{key!r}]",
                example_value=value,
                guard_manager_enum=guard_manager_enum,
            )

    def guard_on_dict_keys_and_order(self, value: dict[Any, Any], guard: Guard) -> None:
        # Add key managers for the DictGuardManager. Then add either an
        # ID_MATCH or EQUALS_MATCH guard on the key.
        dict_mgr = self.get_guard_manager(guard)
        if not isinstance(dict_mgr, DictGuardManager):
            raise NotImplementedError(
                "Expecting a DictGuardManager. Seems like Dynamo forgot "
                f"to set the right guard manager enum for {guard.name}"
            )
        assert isinstance(dict_mgr, DictGuardManager)

        # Ensure that we call dict.keys and not value.keys (which can call
        # overridden keys method). In the C++ guards, we relied on PyDict_Next
        # to traverse the dictionary, which uses the internal data structure and
        # does not call the overridden keys method.
        for idx, key in enumerate(builtin_dict_keys(value)):
            key_source = get_key_index_source(guard.name, idx)
            key_manager = dict_mgr.get_key_manager(
                index=idx,
                source=key_source,
                example_value=key,
                guard_manager_enum=GuardManagerType.GUARD_MANAGER,
            )
            if key_is_id(key):
                # Install ID_MATCH guard
                id_val = self.id_ref(key, key_source)
                key_manager.add_id_match_guard(
                    id_val,
                    get_verbose_code_parts(
                        f"__check_obj_id({key_source}, {id_val})", guard
                    ),
                )
            else:
                # Install EQUALS_MATCH guard
                key_manager.add_equals_match_guard(
                    key, get_verbose_code_parts(f"{key_source} == {key!r}", guard)
                )

    @staticmethod
    def _get_generic_dict_manager_example_value(example_value: Any) -> Optional[Any]:
        # due to a bug in 3.13.0 (introduced by https://github.com/python/cpython/pull/116115,
        # reported in https://github.com/python/cpython/issues/125608,
        # fixed by https://github.com/python/cpython/pull/125611), we cannot take
        # advantage of __dict__ versions to speed up guard checks.
        if (
            config.issue_3_13_0_warning
            and sys.version_info >= (3, 13)
            and sys.version_info < (3, 13, 1)
        ):
            warnings.warn(
                "Guards may run slower on Python 3.13.0. Consider upgrading to Python 3.13.1+.",
                RuntimeWarning,
            )
            return None
        return example_value

    def getattr_on_nn_module(
        self,
        source: AttrSource,
        base_guard_manager: GuardManager,
        base_example_value: Any,
        example_value: Any,
        base_source_name: str,
        source_name: str,
        guard_manager_enum: GuardManagerType,
    ) -> GuardManager:
        """
        This tries to avoid calling the expensive nn module custom getattr method by
        checking if the attribute is accessible via __dict__. For attributes that
        are not accessible via __dict__ (like descriptors), we fallback to
        PyObject_GetAttr.

        There are two cases that we optimize for
        1) attributes present directly in __dict__, e.g training.
        2) parameters/buffers/modules - they can be accessed via _parameters,
        _buffers, _modules keys in __dict__. For example, mod.linear can be
        accessed as mod.__dict__["_parameters"]["linear"]

        The most common and expensive case for nn module guards is of type
        mod.submod1.submod2.submod3.training. We avoid the python getattr of nn
        modules by going through the __dict__.
        """

        def getitem_on_dict_mgr(
            mgr: GuardManager,
            key: Any,
            source_name: str,
            base_example_value: Any,
            example_value: Any,
            guard_manager_enum: GuardManagerType,
        ) -> GuardManager:
            if isinstance(mgr, DictGuardManager):
                # Case where the user code relies on key order, e.g.,
                # named_parameters
                index = get_key_index(base_example_value, key)

                # Install the key manager and add equals match guard
                key_source = f"list(dict.keys({source_name}))[{index!r}]"
                mgr.get_key_manager(
                    index=index,
                    source=key_source,
                    example_value=key,
                    guard_manager_enum=GuardManagerType.GUARD_MANAGER,
                ).add_equals_match_guard(key, [f"{key_source} == {key!r}"])

                # Install the value manager
                return mgr.get_value_manager(
                    index=index,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )
            else:
                return mgr.dict_getitem_manager(
                    key=key,
                    source=source_name,
                    example_value=example_value,
                    guard_manager_enum=guard_manager_enum,
                )

        attr_name = source.member
        mod_dict = base_example_value.__dict__

        all_class_attribute_names: set[str] = set()
        for x in inspect.getmro(base_example_value.__class__):
            all_class_attribute_names.update(x.__dict__.keys())

        accessor_info = NNModuleAttrAccessorInfo(False, None, None)

        if attr_name in mod_dict:
            accessor_info = NNModuleAttrAccessorInfo(True, attr_name, None)
        elif "_parameters" in mod_dict and attr_name in mod_dict["_parameters"]:
            accessor_info = NNModuleAttrAccessorInfo(True, "_parameters", attr_name)
        elif "_buffers" in mod_dict and attr_name in mod_dict["_buffers"]:
            accessor_info = NNModuleAttrAccessorInfo(True, "_buffers", attr_name)
        elif (
            attr_name not in all_class_attribute_names
            and "_modules" in mod_dict
            and attr_name in mod_dict["_modules"]
        ):
            # Check test_attr_precedence test - instance attributes always take precedence unless its an nn.Module.
            accessor_info = NNModuleAttrAccessorInfo(True, "_modules", attr_name)

        if not accessor_info.present_in_generic_dict:
            # The attribute can be accessed by __getattribute__ call, so rely on
            # PyObject_GetAttr
            return base_guard_manager.getattr_manager(
                attr=source.member,
                source=source_name,
                example_value=example_value,
                guard_manager_enum=guard_manager_enum,
            )
        else:
            assert accessor_info.l1_key
            l1_key = accessor_info.l1_key
            l2_key = accessor_info.l2_key

            # Set source strings for debug info
            mod_dict_source = f"{base_source_name}.__dict__"
            l1_source_name = l2_source_name = None
            l1_value = l2_value = None
            l1_guard_manager_enum = l2_guard_manager_enum = None
            if l2_key:
                l1_source = AttrSource(source.base, l1_key)
                l1_source_name = l1_source.name()
                l1_value = mod_dict[l1_key]
                # do not guard on key order for _parameters etc unless the user code
                # actually needs the key order (e.g. calling named_parameters)
                l1_guard_manager_enum = self.get_guard_manager_type(l1_source, l1_value)

                l2_source_name = source_name
                l2_value = example_value
                l2_guard_manager_enum = self.get_guard_manager_type(
                    source, example_value
                )
            else:
                l1_source_name = source_name
                l1_value = example_value
                l1_guard_manager_enum = self.get_guard_manager_type(
                    source, example_value
                )

            # Get __dict__ accessor. No need to guard on dict key order, so use base
            # Guard Manager
            mod_generic_dict_manager = base_guard_manager.get_generic_dict_manager(
                source=mod_dict_source,
                example_value=self._get_generic_dict_manager_example_value(mod_dict),
                guard_manager_enum=GuardManagerType.GUARD_MANAGER,
            )

            l1_
```



## High-Level Overview

"""Core guard system for Dynamo that detects when compiled code needs to be recompiled due tochanges in program state. Guards are conditions that must remain true for previously-compiledcode to be valid for reuse.This module provides the infrastructure for creating, managing and checking guards, including:- Guard creation and composition- Guard state management and invalidation- Guard checking and failure handling- Utilities for guard optimization and debugging- Integration with Dynamo's compilation cachingThe guard system is critical for Dynamo's ability to efficiently reuse compiled code whilemaintaining correctness by detecting when recompilation is necessary due to changes inprogram state, tensor properties, or control flow.

This Python file contains 24 class(es) and 179 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `IndentedBufferWithPrefix`, `GuardManagerWrapper`, `NNModuleAttrAccessorInfo`, `GuardCodeList`, `GuardManagerType`, `GuardBuilder`, `PyExprCSEPass`, `Config`, `ExprCounter`, `Replacer`, `DeletedGuardManagerWrapper`, `ShapeCodeParts`, `GuardsState`, `_Missing`, `GuardsStatePickler`, `CheckFunctionManager`

**Functions defined**: `get_framelocals_idx`, `prefix`, `writeline`, `__init__`, `_preserve_printed_relational_guards`, `collect_diff_guard_sources`, `visit_dict_manager`, `visit_manager`, `visit`, `finalize`, `prepare_diff_guard_manager`, `find_tag_safe_roots`, `check_tag_safety`, `visit_dict_manager`, `visit_manager`, `visit`, `populate_diff_guard_manager`, `clone_with_chosen_sources`, `filter_fn`, `get_guard_lines`

**Key imports**: annotations, ast, builtins, collections, dataclasses, enum, functools, importlib, inspect, io


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `ast`
- `builtins`
- `collections`
- `dataclasses`
- `enum`
- `functools`
- `importlib`
- `inspect`
- `io`
- `logging`
- `math`
- `pickle`
- `sys`
- `textwrap`
- `traceback`
- `types`
- `warnings`
- `weakref`
- `contextlib`: contextmanager
- `copy`: deepcopy
- `typing`: Any, NoReturn, Optional, TYPE_CHECKING, Union
- `typing_extensions`: LiteralString
- `torch`
- `torch.overrides`
- `torch.utils._device`


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
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `guards.py_docs.md`
- **Keyword Index**: `guards.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
