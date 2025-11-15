# Documentation: `tools/code_analyzer/gen_op_registration_allowlist.py`

## File Metadata

- **Path**: `tools/code_analyzer/gen_op_registration_allowlist.py`
- **Size**: 3,225 bytes (3.15 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
"""
This util is invoked from cmake to produce the op registration allowlist param
for `ATen/gen.py` for custom mobile build.
For custom build with dynamic dispatch, it takes the op dependency graph of ATen
and the list of root ops, and outputs all transitive dependencies of the root
ops as the allowlist.
For custom build with static dispatch, the op dependency graph will be omitted,
and it will directly output root ops as the allowlist.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import yaml


DepGraph = dict[str, set[str]]


def canonical_name(opname: str) -> str:
    # Skip the overload name part as it's not supported by code analyzer yet.
    return opname.split(".", 1)[0]


def load_op_dep_graph(fname: str) -> DepGraph:
    with open(fname) as stream:
        result = defaultdict(set)
        for op in yaml.safe_load(stream):
            op_name = canonical_name(op["name"])
            for dep in op.get("depends", []):
                dep_name = canonical_name(dep["name"])
                result[op_name].add(dep_name)
        return dict(result)


def load_root_ops(fname: str) -> list[str]:
    result = []
    with open(fname) as stream:
        for op in yaml.safe_load(stream):
            result.append(canonical_name(op))
    return result


def gen_transitive_closure(
    dep_graph: DepGraph,
    root_ops: list[str],
    train: bool = False,
) -> list[str]:
    result = set(root_ops)
    queue = root_ops.copy()

    # The dependency graph might contain a special entry with key = `__BASE__`
    # and value = (set of `base` ops to always include in custom build).
    queue.append("__BASE__")

    # The dependency graph might contain a special entry with key = `__ROOT__`
    # and value = (set of ops reachable from C++ functions). Insert the special
    # `__ROOT__` key to include ops which can be called from C++ code directly,
    # in addition to ops that are called from TorchScript model.
    # '__ROOT__' is only needed for full-jit. Keep it only for training.
    # TODO: when FL is migrated from full-jit to lite trainer, remove '__ROOT__'
    if train:
        queue.append("__ROOT__")

    while queue:
        cur = queue.pop()
        for dep in dep_graph.get(cur, []):
            if dep not in result:
                result.add(dep)
                queue.append(dep)

    return sorted(result)


def gen_transitive_closure_str(dep_graph: DepGraph, root_ops: list[str]) -> str:
    return " ".join(gen_transitive_closure(dep_graph, root_ops))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Util to produce transitive dependencies for custom build"
    )
    parser.add_argument(
        "--op-dependency",
        help="input yaml file of op dependency graph "
        "- can be omitted for custom build with static dispatch",
    )
    parser.add_argument(
        "--root-ops",
        required=True,
        help="input yaml file of root (directly used) operators",
    )
    args = parser.parse_args()

    deps = load_op_dep_graph(args.op_dependency) if args.op_dependency else {}
    root_ops = load_root_ops(args.root_ops)
    print(gen_transitive_closure_str(deps, root_ops))

```



## High-Level Overview

"""This util is invoked from cmake to produce the op registration allowlist paramfor `ATen/gen.py` for custom mobile build.For custom build with dynamic dispatch, it takes the op dependency graph of ATenand the list of root ops, and outputs all transitive dependencies of the rootops as the allowlist.For custom build with static dispatch, the op dependency graph will be omitted,and it will directly output root ops as the allowlist.

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `canonical_name`, `load_op_dep_graph`, `load_root_ops`, `gen_transitive_closure`, `gen_transitive_closure_str`

**Key imports**: annotations, argparse, defaultdict, yaml


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/code_analyzer`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `collections`: defaultdict
- `yaml`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`tools/code_analyzer`):

- [`gen_operators_yaml.py_docs.md`](./gen_operators_yaml.py_docs.md)
- [`gen_oplist.py_docs.md`](./gen_oplist.py_docs.md)


## Cross-References

- **File Documentation**: `gen_op_registration_allowlist.py_docs.md`
- **Keyword Index**: `gen_op_registration_allowlist.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
