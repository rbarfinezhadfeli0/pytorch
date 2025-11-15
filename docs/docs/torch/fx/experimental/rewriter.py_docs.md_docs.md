# Documentation: `docs/torch/fx/experimental/rewriter.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/experimental/rewriter.py_docs.md`
- **Size**: 8,869 bytes (8.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/experimental/rewriter.py`

## File Metadata

- **Path**: `torch/fx/experimental/rewriter.py`
- **Size**: 5,495 bytes (5.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import ast
import copy
import functools
import inspect
import textwrap
from collections.abc import Callable
from types import FunctionType
from typing import Any, cast, Optional, Union

import torch
from torch._sources import normalize_source_lines
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph import Graph


class AST_Rewriter(ast.NodeTransformer):
    """
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """

    # This function checks for new keys added in the globals dict. TorchDynamo
    # can insert new keys in the global dict and upset the check. Therefore, put
    # a disable here. This function is an optimization pass and not really
    # suitable for dynamo tracing anyways.
    @torch._dynamo.disable
    def rewrite(self, fn: FunctionType):
        # Normalize the source lines
        sourcelines, _ = inspect.getsourcelines(fn)
        sourcelines = normalize_source_lines(sourcelines)
        source = "".join(sourcelines)
        normalized_str = textwrap.dedent(source)

        # Rewrite the original AST
        source_ast = ast.parse(normalized_str)
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))

        # Pull out the compiled function from the newly-created Module
        code = compile(dest_ast, "", "exec")
        globals_dict = copy.copy(fn.__globals__)
        keys_before = set(globals_dict.keys())
        exec(code, globals_dict)
        new_keys = list(set(globals_dict.keys()) - keys_before)
        assert len(new_keys) == 1
        fn_compiled = globals_dict[new_keys[0]]

        # return the compiled function with the original globals
        def change_func_globals(f, globals):
            """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
            # __globals__ is a private member of the function class
            # so we have to copy the function, f, all of its member, except f.__globals__
            g = FunctionType(
                f.__code__,
                globals,
                name=f.__name__,
                argdefs=f.__defaults__,
                closure=f.__closure__,
            )
            g = functools.update_wrapper(g, f)
            g.__kwdefaults__ = copy.copy(f.__kwdefaults__)  # type:ignore[attr-defined]
            return g

        # Return the correct FunctionType object
        return change_func_globals(fn_compiled, globals=fn.__globals__)

    def visit_Assert(self, node):
        """
        Swap out the Assert node (Python's `assert`) with a callsite to the
        symbolically-traceable torch._assert function
        """
        # Create the Call node
        n = ast.parse("torch._assert()", mode="eval")
        assert isinstance(n, ast.Expression)
        call_node = n.body
        assert isinstance(call_node, ast.Call)
        msg = node.msg if node.msg else ast.Constant(value="", kind=None)
        call_node.args = [node.test, msg]

        # Ensure that the new node conforms to the Python AST grammar
        expr_wrapper = ast.Expr(value=call_node)

        # Return the new Call node to signify that we want to use it as
        # a replacement for the original _assert node
        return ast.copy_location(expr_wrapper, node)

    def visit_AnnAssign(self, node):
        """
        Swap out Python's AnnAssign with an Assign node where the annotation function is called.
        Example:
             Original:
             y: Tensor_Type(1,2,3, Dyn) = f2(x)
            Output:
             y = annotate(f2(x),Tensor_Type((1,2,3,Dyn)))
        """
        return ast.Assign(
            targets=[node.target],
            value=ast.Call(
                func=ast.Name(id="annotate", ctx=ast.Load()),
                args=[node.value, node.annotation],
                keywords=[],
            ),
        )


class RewritingTracer(Tracer):
    def trace(
        self,
        root: Union[torch.nn.Module, Callable],
        concrete_args: Optional[dict[str, Any]] = None,
    ) -> Graph:
        return super().trace(_rewrite(root), concrete_args)


def _rewrite(fn: Union[torch.nn.Module, Callable]) -> Union[torch.nn.Module, Callable]:
    if isinstance(fn, torch.nn.Module):
        # Rewrite this module's `forward` as well as the `forward`s of
        # all of this module's recursive descendents. Return the new,
        # rewritten module hierarchy.
        def rewrite_module(m: torch.nn.Module):
            class RewrittenModule(torch.nn.Module):
                def __init__(self, orig):
                    super().__init__()
                    for k, v in orig.__dict__.items():
                        if isinstance(v, torch.nn.Module):
                            self.__dict__[k] = copy.copy(rewrite_module(v))
                        else:
                            self.__dict__[k] = copy.copy(v)

            RewrittenModule.forward = AST_Rewriter().rewrite(
                cast(FunctionType, m.forward)
            )
            return RewrittenModule(m)

        return rewrite_module(fn)
    else:
        # Rewrite this single free function
        return AST_Rewriter().rewrite(cast(FunctionType, fn))

```



## High-Level Overview

"""    Take a FunctionType object representing a `forward` method, then    perform an AST rewrite to swap out nodes that are not symbolically    traceable with a callsite to the FX alternative.    To support swapping out an AST node, define a new `visit` method on    that node. For more details, see:    https://docs.python.org/3/library/ast.html#ast.NodeTransformer

This Python file contains 3 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AST_Rewriter`, `RewritingTracer`, `RewrittenModule`

**Functions defined**: `rewrite`, `change_func_globals`, `visit_Assert`, `visit_AnnAssign`, `trace`, `_rewrite`, `rewrite_module`, `__init__`

**Key imports**: ast, copy, functools, inspect, textwrap, Callable, FunctionType, Any, cast, Optional, Union, torch, normalize_source_lines


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `ast`
- `copy`
- `functools`
- `inspect`
- `textwrap`
- `collections.abc`: Callable
- `types`: FunctionType
- `typing`: Any, cast, Optional, Union
- `torch`
- `torch._sources`: normalize_source_lines
- `torch.fx._symbolic_trace`: Tracer
- `torch.fx.graph`: Graph


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/fx/experimental`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_gradual_typechecker.py_docs.md`](./graph_gradual_typechecker.py_docs.md)
- [`validator.py_docs.md`](./validator.py_docs.md)
- [`accelerator_partitioner.py_docs.md`](./accelerator_partitioner.py_docs.md)
- [`unify_refinements.py_docs.md`](./unify_refinements.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`const_fold.py_docs.md`](./const_fold.py_docs.md)
- [`merge_matmul.py_docs.md`](./merge_matmul.py_docs.md)
- [`partitioner_utils.py_docs.md`](./partitioner_utils.py_docs.md)


## Cross-References

- **File Documentation**: `rewriter.py_docs.md`
- **Keyword Index**: `rewriter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/experimental`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/fx/experimental`):

- [`schema_type_annotation.py_kw.md_docs.md`](./schema_type_annotation.py_kw.md_docs.md)
- [`proxy_tensor.py_kw.md_docs.md`](./proxy_tensor.py_kw.md_docs.md)
- [`partitioner_utils.py_docs.md_docs.md`](./partitioner_utils.py_docs.md_docs.md)
- [`recording.py_docs.md_docs.md`](./recording.py_docs.md_docs.md)
- [`validator.py_kw.md_docs.md`](./validator.py_kw.md_docs.md)
- [`recording.py_kw.md_docs.md`](./recording.py_kw.md_docs.md)
- [`accelerator_partitioner.py_kw.md_docs.md`](./accelerator_partitioner.py_kw.md_docs.md)
- [`optimization.py_kw.md_docs.md`](./optimization.py_kw.md_docs.md)
- [`graph_gradual_typechecker.py_docs.md_docs.md`](./graph_gradual_typechecker.py_docs.md_docs.md)
- [`_dynamism.py_kw.md_docs.md`](./_dynamism.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `rewriter.py_docs.md_docs.md`
- **Keyword Index**: `rewriter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
