# Documentation: `docs/torch/_export/pass_base.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_export/pass_base.py_docs.md`
- **Size**: 22,017 bytes (21.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_export/pass_base.py`

## File Metadata

- **Path**: `torch/_export/pass_base.py`
- **Size**: 18,513 bytes (18.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import operator
import traceback
import typing
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional, Union

import torch
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._higher_order_ops.map import _unstack_pytree
from torch._subclasses import FakeTensor, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    PropagateUnbackedSymInts,
)
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree


__all__ = ["_ExportPassBaseDeprecatedDoNotUse"]


Argument = Any
Value = Any
Fn = Callable[..., Any]
PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


_TORCH_SYM_OPS: set[Callable] = {
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_not,
    torch.sym_sqrt,
}


class ExportPassBaseError(RuntimeError):
    pass


class _ExportPassBaseDeprecatedDoNotUse(PassBase):
    """
    Interpreter-based pass class to help users maintain the IR spec while writing
    transformations.
    """

    @staticmethod
    def _create_dummy_node_metadata():
        return NodeMetadata({"stack_trace": "".join(traceback.format_stack(limit=1))})

    class ExportTracer(PythonKeyTracer):
        def __init__(
            self, callback: "_ExportPassBaseDeprecatedDoNotUse", codegen: CodeGen
        ) -> None:
            super().__init__()
            self.callback = callback
            self.root = torch.nn.Module()
            self.graph = torch.fx.Graph()
            self.graph.set_codegen(codegen)
            self.tensor_attrs: dict[str, torch.Tensor] = {}  # type: ignore[assignment]
            self.fake_tensor_mode: Optional[FakeTensorMode] = None
            self.submodules: dict[torch.nn.Module, str] = {}

        def trace(self) -> None:  # type: ignore[override]
            raise ExportPassBaseError("ExportTracer doesn't support trace().")

        def create_arg(self, a: Argument) -> torch.fx.Node:
            if isinstance(a, torch.nn.Module):
                if a not in self.submodules:
                    name_submodule = f"submodule_{len(self.submodules)}"
                    self.root.add_module(name_submodule, a)
                    self.submodules[a] = name_submodule
            elif isinstance(a, FakeTensor):
                if not hasattr(a, "constant") or a.constant is None:
                    raise ExportPassBaseError(f"Cannot add {a} to graph.")
                a = a.constant
            node = super().create_arg(a)
            if (
                isinstance(a, torch.Tensor)
                and isinstance(node, torch.fx.Node)
                and node.op == "get_attr"
            ):
                self.set_metadata(node, a)
                self.callback.on_attr(ProxyValue(a, node))
            return node

        def set_metadata(
            self,
            node: torch.fx.Node,
            value: Argument,
        ) -> None:
            # propagate the fake tensor or sym nodes
            def make_val(
                x: Argument,
            ) -> Union[
                FakeTensor,
                torch.SymInt,
                torch.SymFloat,
                torch.SymBool,
                int,
                float,
                bool,
                str,
                None,
            ]:
                if isinstance(x, FakeTensor):
                    return x
                elif isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        assert self.fake_tensor_mode is not None
                        # TODO we should allocate static shapes
                        # for param/buffer values
                        if isinstance(x, torch.nn.Parameter):
                            fake_tensor = self.fake_tensor_mode.from_tensor(
                                x, static_shapes=True
                            )
                        else:
                            fake_tensor = self.fake_tensor_mode.from_tensor(x)
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        print(
                            "Fakeifying a Tensor subclass is not supported \
                            right now. Instead a TensorMetadata is used."
                        )
                        fake_tensor = None
                    return fake_tensor
                elif isinstance(
                    x,
                    (
                        torch.SymInt,
                        torch.SymFloat,
                        torch.SymBool,
                        int,
                        float,
                        bool,
                        str,
                    ),
                ):
                    return x
                else:
                    return None

            node.meta["val"] = pytree.tree_map(make_val, value)

            # Set the tensor_metadata for values that do not have a corresponding FakeTensor
            def make_tensor_meta(x: Argument) -> Optional[TensorMetadata]:
                if not isinstance(x, FakeTensor) and isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        assert self.fake_tensor_mode is not None
                        _ = self.fake_tensor_mode.from_tensor(x)
                        tensor_meta = None
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        tensor_meta = _extract_tensor_metadata(x)
                    return tensor_meta
                else:
                    return None

            node.meta["tensor_meta"] = pytree.tree_map(make_tensor_meta, value)

    class ExportInterpreter(fx.Interpreter):
        def __init__(
            self, callback: "_ExportPassBaseDeprecatedDoNotUse", gm: fx.GraphModule
        ) -> None:
            super().__init__(gm)
            self.callback = callback
            self.node: torch.fx.Node = next(iter(gm.graph.nodes))

        # pyrefly: ignore [bad-override]
        def placeholder(
            self,
            target: str,  # type: ignore[override]
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            arg = super().placeholder(target, args, kwargs)
            return self.callback.placeholder(target, arg, NodeMetadata(self.node.meta))

        def output(
            self,
            target: torch.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            return self.callback.output(args[0], NodeMetadata(self.node.meta)).data  # type: ignore[return-value]

        def call_function(
            self,
            target: torch.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            meta = NodeMetadata(self.node.meta)

            if target is operator.getitem:
                value, key = args
                return self.callback.call_getitem(value, key, meta)
            elif getattr(target, "__module__", None) in {
                "_operator",
                "builtins",
                "math",
            }:
                assert callable(target)
                return self.callback.call_sym(target, args, meta)
            elif target in _TORCH_SYM_OPS:
                assert callable(target)
                return self.callback.call_sym(target, args, meta)
            elif isinstance(
                target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
            ):
                return self.callback.call_operator(
                    target,
                    args,
                    kwargs,
                    meta,
                )
            elif target is torch.ops.higher_order.cond:
                pred, true_fn, false_fn, inputs = args
                return self.callback.call_cond(pred, true_fn, false_fn, inputs, meta)
            elif target is torch.ops.higher_order.map_impl:
                f, mapped_args, operands = args  # type: ignore[assignment]
                return self.callback.call_map(f, mapped_args, operands, meta)
            # For other unregistered HigherOrderOps, just interpret them blindly
            elif isinstance(target, torch._ops.HigherOrderOperator):
                return self.callback._fx(
                    "call_function",
                    target,
                    args,
                    kwargs,
                    meta,
                )
            else:
                raise ExportPassBaseError(f"Unsupported target type: {target}")

        def get_attr(  # type: ignore[override]
            self,
            target: str,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> Argument:
            return super().get_attr(target, args, kwargs)

        def call_module(
            self,
            target: torch.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> None:
            raise ExportPassBaseError("call_module is not supported.")

        def call_method(  # type: ignore[override]
            self,
            target: str,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> None:
            raise ExportPassBaseError("call_method is not supported.")

        def run_node(self, n: torch.fx.Node) -> Argument:
            self.node = n
            self.callback.node_debug_str = n.format_node()
            return super().run_node(n)

    def __init__(self) -> None:
        self.interpreter = PropagateUnbackedSymInts(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        self.tracer = self.ExportTracer(self, CodeGen())
        self.fake_tensor_mode: Optional[FakeTensorMode] = None
        self._initialized = True
        self.node_debug_str: typing.Optional[str] = None

    def _fx(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        res_data = getattr(self.interpreter, kind)(target, args_data, kwargs_data)
        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue, lambda x: x.proxy, (args, kwargs)
        )

        name = None
        if isinstance(target, torch._ops.OpOverload):
            name = self.tracer.graph._target_to_str(target.overloadpacket.__name__)

        res_proxy = self.tracer.create_proxy(
            kind, target, args_proxy, kwargs_proxy, name=name
        )
        res_proxy.node.meta.update(meta.data)
        if self.fake_tensor_mode and (shape_env := self.fake_tensor_mode.shape_env):
            if symbol_to_path := compute_unbacked_bindings(shape_env, res_data):
                res_proxy.node.meta["unbacked_bindings"] = symbol_to_path
        self.tracer.set_metadata(res_proxy.node, res_data)
        return ProxyValue(res_data, res_proxy)

    def inputs(self, graph_module: torch.fx.GraphModule) -> list[Argument]:
        # TODO(angelayi): Update this with what we decide to do for metadata in
        # the exported graph module
        if (args := graph_module.meta.get("args", None)) is not None:
            return list(args)

        def extract_input(node: torch.fx.Node) -> Optional[FakeTensor]:
            if "val" in node.meta:
                fake = node.meta["val"]
                if hasattr(fake, "constant") and fake.constant is not None:
                    return fake.constant
                return fake
            elif tensor_meta := node.meta.get("tensor_meta"):
                assert self.fake_tensor_mode is not None
                return FakeTensor(
                    self.fake_tensor_mode,
                    torch.empty(
                        tensor_meta.shape,
                        dtype=tensor_meta.dtype,
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                        memory_format=tensor_meta.memory_format,
                    ),
                    torch.device("cpu"),
                )
            elif len(node.users) == 0:
                return None
            raise ExportPassBaseError(
                f"Cannot construct an input for graph module: {graph_module}.",
            )

        return [
            extract_input(node)
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
        ]

    def on_attr(self, attr: ProxyValue) -> None:
        pass

    def placeholder(self, name: str, arg: Argument, meta: NodeMetadata) -> ProxyValue:
        arg_proxy = self.tracer.create_proxy("placeholder", name, (), {})
        arg_proxy.node.meta = meta.data
        self.tracer.set_metadata(arg_proxy.node, arg)
        return ProxyValue(arg, arg_proxy)

    def call_operator(
        self,
        op,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", op, args, kwargs, meta)

    def call_sym(
        self,
        target: Fn,
        args: tuple[Argument, ...],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", target, args, {}, meta)

    def call_cond(
        self,
        pred: ProxyValue,
        true_fn: torch.fx.GraphModule,
        false_fn: torch.fx.GraphModule,
        inputs: list[Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        true_branch = self.call_submodule(true_fn, tuple(inputs))
        false_branch = self.call_submodule(false_fn, tuple(inputs))
        assert true_branch is not None
        assert false_branch is not None
        return self._fx(
            "call_function",
            torch.ops.higher_order.cond,
            (pred, true_branch.graph_module, false_branch.graph_module, list(inputs)),
            {},
            meta,
        )

    def call_map(
        self,
        f: torch.fx.GraphModule,
        mapped_args: list[ProxyValue],
        operands: list[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        xs = _unstack_pytree([arg.data for arg in mapped_args])[0]
        f_branch = self.call_submodule(f, tuple(xs + [arg.data for arg in operands]))
        assert f_branch is not None
        return self._fx(
            "call_function",
            torch.ops.higher_order.map_impl,
            (f_branch.graph_module, mapped_args, operands),
            {},
            meta,
        )

    def call_getitem(
        self, value: ProxyValue, key: int, meta: NodeMetadata
    ) -> ProxyValue:
        return self._fx("call_function", operator.getitem, (value, key), {}, meta)

    def output(self, results: list[Argument], meta: NodeMetadata) -> ProxyValue:
        return self._fx("output", "output", (results,), {}, meta)

    def call_submodule(
        self, graph_module: fx.GraphModule, inputs: tuple[Argument, ...]
    ) -> PassResult:
        prev_tracer, self.tracer = (
            self.tracer,
            self.ExportTracer(self, graph_module.graph._codegen),
        )
        self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode
        interpreter = self.ExportInterpreter(self, graph_module)
        # pyrefly: ignore [bad-assignment]
        prev_interpreter, self.interpreter = (
            self.interpreter,
            torch.fx.Interpreter(  # type: ignore[assignment]
                torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
            ),
        )
        inputs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, inputs)
        with fx_traceback.preserve_node_meta():
            interpreter.run(*inputs_data)

        new_graph_module = torch.fx.GraphModule(self.tracer.root, self.tracer.graph)

        self.tracer = prev_tracer
        self.interpreter = prev_interpreter
        return PassResult(
            new_graph_module,
            True,
        )

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        if not getattr(self, "_initialized", False):
            raise ExportPassBaseError(
                "ExportPass is not initialized with __init__().",
            )

        inputs = self.inputs(graph_module)

        fake_tensor_mode = None
        for i in inputs:
            if isinstance(i, FakeTensor):
                assert fake_tensor_mode is None or fake_tensor_mode is i.fake_mode, (
                    "Multiple fake tensor mode detected."
                )
                fake_tensor_mode = i.fake_mode
        if fake_tensor_mode is None:
            self.tracer.fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_tensor_mode = nullcontext()  # type: ignore[assignment]
            dispatcher_mode = nullcontext()  # type: ignore[assignment]
        else:
            fake_tensor_mode.allow_non_fake_inputs = True
            self.tracer.fake_tensor_mode = fake_tensor_mode
            dispatcher_mode = enable_python_dispatcher()  # type: ignore[assignment]
        self.fake_tensor_mode = self.tracer.fake_tensor_mode

        with fake_tensor_mode, dispatcher_mode:  # type: ignore[assignment, union-attr]
            result = self.call_submodule(graph_module, tuple(inputs))

        return result

```



## High-Level Overview


This Python file contains 8 class(es) and 29 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ExportPassBaseError`, `_ExportPassBaseDeprecatedDoNotUse`, `ExportTracer`, `ExportInterpreter`

**Functions defined**: `_create_dummy_node_metadata`, `__init__`, `trace`, `create_arg`, `set_metadata`, `make_val`, `make_tensor_meta`, `__init__`, `placeholder`, `output`, `call_function`, `get_attr`, `call_module`, `call_method`, `run_node`, `__init__`, `_fx`, `inputs`, `extract_input`, `on_attr`

**Key imports**: operator, traceback, typing, Callable, nullcontext, Any, Optional, Union, torch, fx, enable_python_dispatcher, NodeMetadata


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `traceback`
- `typing`
- `collections.abc`: Callable
- `contextlib`: nullcontext
- `torch`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch._export.pass_infra.node_metadata`: NodeMetadata
- `torch._export.pass_infra.proxy_value`: ProxyValue
- `torch._higher_order_ops.map`: _unstack_pytree
- `torch._subclasses`: FakeTensor, UnsupportedFakeTensorException
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `torch.fx`: traceback as fx_traceback
- `torch.fx.experimental.proxy_tensor`: PythonKeyTracer
- `torch.fx.graph`: CodeGen
- `torch.fx.passes.infra.pass_base`: PassBase, PassResult
- `torch.fx.passes.shape_prop`: _extract_tensor_metadata, TensorMetadata
- `torch.utils`: _pytree as pytree


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/_export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`error.py_docs.md`](./error.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`tools.py_docs.md`](./tools.py_docs.md)
- [`non_strict_utils.py_docs.md`](./non_strict_utils.py_docs.md)
- [`converter.py_docs.md`](./converter.py_docs.md)
- [`wrappers.py_docs.md`](./wrappers.py_docs.md)
- [`verifier.py_docs.md`](./verifier.py_docs.md)


## Cross-References

- **File Documentation**: `pass_base.py_docs.md`
- **Keyword Index**: `pass_base.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/_export`):

- [`error.py_kw.md_docs.md`](./error.py_kw.md_docs.md)
- [`converter.py_kw.md_docs.md`](./converter.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`pass_base.py_kw.md_docs.md`](./pass_base.py_kw.md_docs.md)
- [`wrappers.py_docs.md_docs.md`](./wrappers.py_docs.md_docs.md)
- [`converter.py_docs.md_docs.md`](./converter.py_docs.md_docs.md)
- [`config.py_kw.md_docs.md`](./config.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`verifier.py_kw.md_docs.md`](./verifier.py_kw.md_docs.md)
- [`verifier.py_docs.md_docs.md`](./verifier.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `pass_base.py_docs.md_docs.md`
- **Keyword Index**: `pass_base.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
