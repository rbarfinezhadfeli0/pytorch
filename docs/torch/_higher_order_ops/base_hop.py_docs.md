# Documentation: `torch/_higher_order_ops/base_hop.py`

## File Metadata

- **Path**: `torch/_higher_order_ops/base_hop.py`
- **Size**: 10,686 bytes (10.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import abc

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.auto_functionalize import FunctionalCallableWithEpilogue
from torch._higher_order_ops.utils import (
    check_input_alias_and_mutation_return_outputs,
    HopInstance,
    materialize_as_graph,
    reenter_make_fx,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class BaseHOP(HigherOrderOperator, abc.ABC):
    """
    This is the "Base" HOP implementation for a HOP that looks like:

        call_subgraph_hop(subgraph, *operands, **kwargs)

    That is:
    1) the HOP stays alive until Inductor
    2) the HOP's semantics are subgraph(*operands)
    3) kwargs may be some config options but aren't passed directly to the subgraph.

    To use this, please subclass this class and override methods as necessary:
    ```
    class InvokeQuant(BaseHOP):
        def __init__(self):
            return super().__init__("invoke_quant")


    invoke_quant = InvokeQuant()


    def g(x):
        return x.sin().cos()


    @torch.compile(backend="aot_eager")
    def f(x):
        return invoke_quant(g, x, scheme="nf4")
    ```

    NOTE: don't subclass BaseHOP out of tree! That is not allowed. All
    usages must be in tree.
    """

    def __init__(self, hop_name) -> None:
        super().__init__(hop_name)

        # Set up the registrations
        # If you want to override any of these, override them in your subclass.
        self.py_autograd_impl(self._call_Autograd)
        self.py_functionalize_impl(self._call_Functionalize)
        self.py_impl(ProxyTorchDispatchMode)(self._call_ProxyTorchDispatchMode)
        self.py_impl(FakeTensorMode)(self._call_FakeTensorMode)
        self.py_impl(DispatchKey.CompositeExplicitAutograd)(
            self._call_CompositeExplicitAutograd
        )

    def __call__(self, subgraph, *operands, **kwargs):
        if not isinstance(
            subgraph,
            (
                torch.fx.GraphModule,
                FunctionWithNoFreeVars,
                FunctionalCallableWithEpilogue,
            ),
        ):
            raise RuntimeError(
                f"{self._name}: when calling this API without torch.compile, "
                f"we require that the subgraph be a torch.fx.GraphModule (or "
                f"a function we know doesn't have free variables)."
            )
        return super().__call__(subgraph, *operands, **kwargs)

    def _call_Autograd(self, subgraph, *operands, **kwargs):
        if isinstance(subgraph, torch.fx.GraphModule):
            pass

        # We assume the subgraph doesn't mutate inputs and there is no aliasing.
        # In the PT2 stack, this is Dynamo's responsibility to figure out.
        return BaseHOPFunction.apply(self, subgraph, kwargs, *operands)

    def _call_CompositeExplicitAutograd(self, subgraph, *operands, **kwargs):
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        mode = _get_current_dispatch_mode()
        assert mode is None, "Mode should never be enabled for CPU/CUDA key"
        return subgraph(*operands)

    def _call_ProxyTorchDispatchMode(self, proxy_mode, subgraph, *operands, **kwargs):
        traced_graph = reenter_make_fx(subgraph)(*operands)
        assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
        qualname = proxy_mode.tracer.get_fresh_qualname("subgraph")
        proxy_mode.tracer.root.register_module(qualname, traced_graph)

        node_args = (traced_graph, *operands)
        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)  # type: ignore[attr-defined]
        proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, kwargs)  # type: ignore[attr-defined]
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", self, proxy_args, proxy_kwargs
        )

        out = self(subgraph, *operands, **kwargs)
        return track_tensor_tree(
            out,
            out_proxy,
            constant=None,
            tracer=proxy_mode.tracer,  # type: ignore[arg-type]
        )

    def _call_FakeTensorMode(self, mode, subgraph, *operands, **kwargs):
        # TODO: this should probably route through FakeTensorMode to reuse caching
        with mode:
            return subgraph(*operands)

    # NOTE [Support input mutation of hops]
    # To support input mutation, hop's subgraph must be functionalized because many inductor passes are
    #   applied to subgraph recursively and only work on functional graph. However, we could inline an
    #   epilogue graph (i.e. the copy_) into the subgraph because this is how input mutation
    #   is implemented in the top-level graph when no hop is presented. All passes must have been and will be
    #   aware of the epilogue graph.
    #
    # Since we've supported input mutation for custom op with auto_functionalized, we share the infra for hops
    # The plan is:
    #   1. In hop's Functionalization key, it calls do_auto_functionalize_v2 if subgraph mutates input
    #   2. In do_auto_functionalize_v2:
    #       a. we functionalize the callables in hop's argument. This is to make the subgraphs functional so we
    #          could recursively run passes on them. Also the epilogue graph is inlined at the end.
    #       b. we call auto_functionalized_v2 and pass in an additional schema in order to properly invoke
    #          the hop with normalized kwargs.
    #   3. In inductor, we decompose the auto_functionalized hop by callilng into the dense implementation, which
    #      copies the mutated inputs to the hop if necessary and call the hop.
    # After these steps, the rest of the inductor stack knows how to fuse the copy_ in subgraph with other ops.
    def _call_Functionalize(self, ctx, subgraph, *operands, **kwargs):
        from torch._higher_order_ops.auto_functionalize import (
            can_auto_functionalize,
            do_auto_functionalize_v2,
        )

        # invoke_quant has non-proxable argument of type InvokeQuant that
        # we cannot generate schema for.
        if self is not torch.ops.higher_order.invoke_quant_packed:
            hop_instance = HopInstance.create(self, subgraph, *operands, **kwargs)
            if can_auto_functionalize(hop_instance):
                return do_auto_functionalize_v2(
                    ctx.mode, hop_instance, (subgraph, *operands), kwargs
                )

        unwrapped_operands = ctx.unwrap_tensors(operands)
        with ctx.redispatch_to_next():
            # We assume the subgraph doesn't mutate inputs and there is no aliasing.
            # In the PT2 stack, this is Dynamo's responsibility to figure out.
            functionalized_subgraph = FunctionWithNoFreeVars(
                ctx.functionalize(subgraph)
            )
            out = self(functionalized_subgraph, *unwrapped_operands, **kwargs)
        return ctx.wrap_tensors(out)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, subgraph, *operands, **kwargs):
        from .schema import HopSchemaGenerator

        subgraph = materialize_as_graph(subgraph, operands)
        (
            inp_inp_alias,
            inp_out_alias,
            out_out_alias,
            mutated_inp_idx,
            output,
        ) = check_input_alias_and_mutation_return_outputs(subgraph)

        if not (
            len(inp_inp_alias) == 0
            and len(inp_out_alias) == 0
            and len(out_out_alias) == 0
        ):
            # TODO: turn this into an error.
            # test_foreach_map_backward_binary_foreach_map_addrecip_op fails the alias test.
            import warnings

            warnings.warn(
                "Aliasing is not supported for HOP subgraph.\n"
                f"{subgraph.print_readable(print_output=False)}\n"
                f"Alias info: inp-inp alias: {inp_inp_alias}, inp-out alias: {inp_out_alias}, out-out alias{out_out_alias}"
                f"This may lead to silent incorrectness.",
                stacklevel=2,
            )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("subgraph", subgraph)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"arg{idx}", arg, is_mutated=idx in mutated_inp_idx)

        for name, arg in kwargs.items():
            schema_gen.add_arg(name, arg, default_value=arg, kw_only=True)

        for out in output:
            schema_gen.add_output(out)

        return schema_gen.gen_schema()


class BaseHOPFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, hop, subgraph, kwargs, *operands):
        ctx.hop = hop
        ctx.operands = operands
        ctx.subgraph = subgraph
        ctx.kwargs = kwargs

        with torch._C._AutoDispatchBelowAutograd():
            return hop(subgraph, *operands, **kwargs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        subgraph = ctx.subgraph
        operands = ctx.operands
        kwargs = ctx.kwargs

        # TODO: Something special needs to happen with min cut partitioner
        with (
            suspend_functionalization(),
            disable_functional_mode(),
            torch.enable_grad(),
        ):
            with disable_proxy_modes_tracing():
                from .invoke_subgraph import create_fw_bw_graph
                from .utils import _from_fun

                fw_inputs = pytree.tree_map(_from_fun, operands)
                (
                    _,
                    joint_graph,
                    _,
                ) = create_fw_bw_graph(subgraph, fw_inputs, grad_outputs)

        # The joint graph returns (*grad_inputs, *fwd_outputs).
        # We only need the grad_inputs.
        def bwd_fn(*args):
            operands = args[: -len(grad_outputs)]
            grad_outs = args[-len(grad_outputs) :]
            result = joint_graph(*operands, *grad_outs)
            grad_inputs = result[: -len(grad_outputs)]
            return grad_inputs

        return (
            None,
            None,
            None,
            *ctx.hop(
                FunctionWithNoFreeVars(bwd_fn), *operands, *grad_outputs, **kwargs
            ),
        )


class FunctionWithNoFreeVars:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

```



## High-Level Overview

"""    This is the "Base" HOP implementation for a HOP that looks like:        call_subgraph_hop(subgraph, *operands, **kwargs)    That is:    1) the HOP stays alive until Inductor    2) the HOP's semantics are subgraph(*operands)    3) kwargs may be some config options but aren't passed directly to the subgraph.    To use this, please subclass this class and override methods as necessary:    ```    class InvokeQuant(BaseHOP):        def __init__(self):            return super().__init__("invoke_quant")    invoke_quant = InvokeQuant()    def g(x):        return x.sin().cos()

This Python file contains 7 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BaseHOP`, `InvokeQuant`, `BaseHOPFunction`, `FunctionWithNoFreeVars`

**Functions defined**: `__init__`, `g`, `f`, `__init__`, `__call__`, `_call_Autograd`, `_call_CompositeExplicitAutograd`, `_call_ProxyTorchDispatchMode`, `_call_FakeTensorMode`, `_call_Functionalize`, `gen_schema`, `forward`, `backward`, `bwd_fn`, `__init__`, `__call__`

**Key imports**: abc, torch, torch.utils._pytree as pytree, DispatchKey, suspend_functionalization, FunctionalCallableWithEpilogue, HigherOrderOperator, FakeTensorMode, disable_functional_mode, _get_current_dispatch_mode


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `torch`
- `torch.utils._pytree as pytree`
- `torch._C`: DispatchKey
- `torch._dispatch.python`: suspend_functionalization
- `torch._higher_order_ops.auto_functionalize`: FunctionalCallableWithEpilogue
- `torch._ops`: HigherOrderOperator
- `torch._subclasses`: FakeTensorMode
- `torch._subclasses.functional_tensor`: disable_functional_mode
- `torch.utils._python_dispatch`: _get_current_dispatch_mode
- `.schema`: HopSchemaGenerator
- `warnings`
- `.invoke_subgraph`: create_fw_bw_graph
- `.utils`: _from_fun


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/_higher_order_ops`):

- [`associative_scan.py_docs.md`](./associative_scan.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`effects.py_docs.md`](./effects.py_docs.md)
- [`foreach_map.py_docs.md`](./foreach_map.py_docs.md)
- [`strict_mode.py_docs.md`](./strict_mode.py_docs.md)
- [`torchbind.py_docs.md`](./torchbind.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`run_const_graph.py_docs.md`](./run_const_graph.py_docs.md)
- [`_invoke_quant.py_docs.md`](./_invoke_quant.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `base_hop.py_docs.md`
- **Keyword Index**: `base_hop.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
