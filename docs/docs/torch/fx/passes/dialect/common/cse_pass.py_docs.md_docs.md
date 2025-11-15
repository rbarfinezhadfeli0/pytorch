# Documentation: `docs/torch/fx/passes/dialect/common/cse_pass.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/passes/dialect/common/cse_pass.py_docs.md`
- **Size**: 7,461 bytes (7.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/passes/dialect/common/cse_pass.py`

## File Metadata

- **Path**: `torch/fx/passes/dialect/common/cse_pass.py`
- **Size**: 5,248 bytes (5.12 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Any

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._pytree import tree_flatten


aten = torch.ops.aten


# stateful ops are banned from CSE
rand_ops = {
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
}  # noqa: E501,B950

inplace_ops = {
    aten.add_,
    aten.sub_,
    aten.mul_,
    aten.div_,
    aten.pow_,
    aten.lerp_,
    aten.relu_,
    aten.sigmoid_,
    aten.tanh_,
}  # noqa: E501


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def get_CSE_banned_ops():
    return rand_ops.union(inplace_ops)


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
class CSEPass(PassBase):
    def __init__(self, banned_ops=None):
        """
        This version of CSE Pass aims to be dialect agnostic, and it's implemented purely based on the connectivity between fx.Node.

        For functional dialects, user would only need to specify the random ops in ban list.

        Warning: CSE Pass cannot be safely applied on a FX graph in non-functional dialects.
        If your dialect contains stateful operators, please customized the banned_ops.

        """
        if banned_ops is None:
            banned_ops = set()
        self.banned_ops = banned_ops
        super().__init__()

    def call(self, graph_module: GraphModule) -> PassResult:
        """
        Return a new copy of torch.fx.GraphModule with CSE applied to the input graph

        Example usage:

        from torch.fx.experimental.proxy_tensor import make_fx
        def f(a):
            b = a * a
            c = a * a
            return b+c

        p = CSEPass()
        traced_graph = make_fx(f)(torch.tensor(1))
        print(traced_graph)
        result = p(traced_graph)
        print(result.graph_module)
        """

        def get_aten_target(node):
            if hasattr(node.target, "overloadpacket"):
                return node.target.overloadpacket
            return node.target

        modified = False
        new_graph = Graph()
        env: dict[
            Node, Node
        ] = {}  # map from node in the old graph to node in the new graph
        hash_env: dict[
            tuple[torch._ops.OpOverload, int], Node
        ] = {}  # map from hash to a node in the new graph
        token_map: dict[
            tuple[torch._ops.OpOverload, int], dict[str, Any]
        ] = {}  # map from hash to token
        for n in graph_module.graph.nodes:
            # The placeholder, output, and get_attr nodes are copied to the new graph without change
            # do not CSE away random operations
            if (
                n.op == "placeholder"
                or n.op == "output"
                or n.op == "get_attr"
                or get_aten_target(n) in self.banned_ops
            ):
                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
            else:  # n.op == 'call_function', should never see n.op == 'call_module' or 'call_method'
                # substitute args and kwargs members to their mapping in env if exists
                # specs can be used to reconstruct nested list/dictionaries
                def substitute(arg_list):
                    arg_list, spec = tree_flatten(arg_list)
                    for i in range(len(arg_list)):
                        v = arg_list[i]
                        if isinstance(v, Node) and v in env:
                            arg_list[i] = env[v]
                    return tuple(arg_list), spec

                args, args_spec = substitute(n.args)
                kwargs, kwargs_spec = substitute(n.kwargs)

                # each token corresponds to a unique node
                # nodes with the same token can be substituted
                token = {
                    "target": n.target,
                    "args": args,
                    "args_spec": args_spec,
                    "kwargs": kwargs,
                    "kwargs_spec": kwargs_spec,
                }

                # hash substituted args to a number, do not hash specs because specs are not hashable
                hash_arg = hash((args, kwargs))
                hash_val = (n.target, hash_arg)

                # check if a node has a substitute and can be eliminated
                hash_val_in_hash_env = hash_val in hash_env
                if hash_val_in_hash_env and token_map[hash_val] == token:
                    modified = True  # substitution happens and the graph is modified
                    env[n] = hash_env[hash_val]
                    continue

                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
                if not hash_val_in_hash_env:
                    hash_env[hash_val] = new_node
                    token_map[hash_val] = token

        csed_gm = GraphModule(graph_module, new_graph)
        return PassResult(csed_gm, modified)

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CSEPass`

**Functions defined**: `get_CSE_banned_ops`, `__init__`, `call`, `f`, `get_aten_target`, `substitute`

**Key imports**: Any, torch, Graph, GraphModule, Node, PassBase, PassResult, tree_flatten, make_fx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes/dialect/common`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch`
- `torch.fx`: Graph, GraphModule, Node
- `torch.fx.passes.infra.pass_base`: PassBase, PassResult
- `torch.utils._pytree`: tree_flatten
- `torch.fx.experimental.proxy_tensor`: make_fx


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/fx/passes/dialect/common`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `cse_pass.py_docs.md`
- **Keyword Index**: `cse_pass.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/passes/dialect/common`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/passes/dialect/common`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/fx/passes/dialect/common`):

- [`cse_pass.py_kw.md_docs.md`](./cse_pass.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cse_pass.py_docs.md_docs.md`
- **Keyword Index**: `cse_pass.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
