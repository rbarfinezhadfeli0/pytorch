# Documentation: `torch/fx/experimental/schema_type_annotation.py`

## File Metadata

- **Path**: `torch/fx/experimental/schema_type_annotation.py`
- **Size**: 5,379 bytes (5.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import inspect
from typing import Any, Optional

import torch
import torch.fx
from torch._jit_internal import boolean_dispatched
from torch.fx import Transformer
from torch.fx.node import Argument, Target
from torch.fx.operator_schemas import _torchscript_type_to_python_type


class AnnotateTypesWithSchema(Transformer):
    """
    Use Python function signatures to annotate types for `Nodes` within an FX graph.
    This pulls out Python function signatures for:

        1. Standard `torch.nn` Module calls
        2. `torch.nn.functional` calls
        3. Attribute fetches via `get_attr`

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = AnnotateTypesWithSchema(traced).transform()

    """

    def __init__(
        self,
        module: torch.nn.Module,
        annotate_functionals: bool = True,
        annotate_modules: bool = True,
        annotate_get_attrs: bool = True,
    ):
        super().__init__(module)
        self.annotate_functionals = annotate_functionals
        self.annotate_modules = annotate_modules
        self.annotate_get_attrs = annotate_get_attrs

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ):
        python_ret_type = None
        if self.annotate_functionals and target.__module__ == "torch.nn.functional":
            target_for_analysis = target
            if target in boolean_dispatched:
                # HACK: `boolean_dispatch` as used in `torch.nn.functional` makes it so that we have
                # a 2-way dispatch based on a boolean value. Here we check that the `true` and `false`
                # branches of the dispatch have exactly the same signature. If they do, use the `true`
                # branch signature for analysis. Otherwise, leave this un-normalized
                assert not isinstance(target, str)
                dispatched = boolean_dispatched[target]
                if_true, if_false = dispatched["if_true"], dispatched["if_false"]
                # TODO: can we emit the union of these? What are the implications on TorchScript
                # compilation?
                if (
                    inspect.signature(if_true).return_annotation
                    != inspect.signature(if_false).return_annotation
                ):
                    return super().call_function(target, args, kwargs)
                target_for_analysis = if_true

            python_ret_type = self._extract_python_return_type(target_for_analysis)

        return_proxy = super().call_function(target, args, kwargs)
        return_proxy.node.type = (
            return_proxy.node.type if return_proxy.node.type else python_ret_type
        )
        return return_proxy

    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ):
        python_ret_type = None
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        if self.annotate_modules and hasattr(submod.__class__, "__name__"):
            classname = submod.__class__.__name__
            if getattr(torch.nn, classname, None) == submod.__class__:
                python_ret_type = self._extract_python_return_type(submod.forward)
        return_proxy = super().call_module(target, args, kwargs)
        return_proxy.node.type = (
            return_proxy.node.type if return_proxy.node.type else python_ret_type
        )
        return return_proxy

    def get_attr(
        self,
        target: torch.fx.node.Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Any],
    ):
        attr_proxy = super().get_attr(target, args, kwargs)

        if self.annotate_get_attrs:
            module_itr = self.module
            assert isinstance(target, str)
            atoms = target.split(".")
            for i, atom in enumerate(atoms):
                if not hasattr(module_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonextent target {'.'.join(atoms[:i])}!"
                    )
                module_itr = getattr(module_itr, atom)

            maybe_inferred_ts_type = torch._C._jit_try_infer_type(module_itr)
            if maybe_inferred_ts_type.success():
                python_type = _torchscript_type_to_python_type(
                    maybe_inferred_ts_type.type()
                )
                attr_proxy.node.type = (
                    python_type if not attr_proxy.node.type else attr_proxy.node.type
                )

        return attr_proxy

    def _extract_python_return_type(self, target: Target) -> Optional[Any]:
        """
        Given a Python call target, try to extract the Python return annotation
        if it is available, otherwise return None

        Args:

            target (Callable): Python callable to get return annotation for

        Returns:

            Optional[Any]: Return annotation from the `target`, or None if it was
                not available.
        """
        assert callable(target)
        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            return None

        return (
            sig.return_annotation
            if sig.return_annotation is not inspect.Signature.empty
            else None
        )

```



## High-Level Overview

"""    Use Python function signatures to annotate types for `Nodes` within an FX graph.    This pulls out Python function signatures for:        1. Standard `torch.nn` Module calls        2. `torch.nn.functional` calls        3. Attribute fetches via `get_attr`    Example usage:        m = torchvision.models.resnet18()        traced = torch.fx.symbolic_trace(m)        traced = AnnotateTypesWithSchema(traced).transform()

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AnnotateTypesWithSchema`

**Functions defined**: `__init__`, `call_function`, `call_module`, `get_attr`, `_extract_python_return_type`

**Key imports**: inspect, Any, Optional, torch, torch.fx, boolean_dispatched, Transformer, Argument, Target, _torchscript_type_to_python_type


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `inspect`
- `typing`: Any, Optional
- `torch`
- `torch.fx`
- `torch._jit_internal`: boolean_dispatched
- `torch.fx.node`: Argument, Target
- `torch.fx.operator_schemas`: _torchscript_type_to_python_type


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/fx/experimental`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_gradual_typechecker.py_docs.md`](./graph_gradual_typechecker.py_docs.md)
- [`validator.py_docs.md`](./validator.py_docs.md)
- [`accelerator_partitioner.py_docs.md`](./accelerator_partitioner.py_docs.md)
- [`unify_refinements.py_docs.md`](./unify_refinements.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`const_fold.py_docs.md`](./const_fold.py_docs.md)
- [`merge_matmul.py_docs.md`](./merge_matmul.py_docs.md)
- [`rewriter.py_docs.md`](./rewriter.py_docs.md)
- [`partitioner_utils.py_docs.md`](./partitioner_utils.py_docs.md)


## Cross-References

- **File Documentation**: `schema_type_annotation.py_docs.md`
- **Keyword Index**: `schema_type_annotation.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
