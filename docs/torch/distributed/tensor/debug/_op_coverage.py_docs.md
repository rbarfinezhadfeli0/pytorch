# Documentation: `torch/distributed/tensor/debug/_op_coverage.py`

## File Metadata

- **Path**: `torch/distributed/tensor/debug/_op_coverage.py`
- **Size**: 3,238 bytes (3.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from operator import itemgetter

import torch
import torch.fx
import torch.nn as nn
from functorch.compile import make_boxed_func
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table
from torch.distributed.tensor import DTensor


inductor_decomps = select_decomp_table()

graphs: list[torch.fx.GraphModule] = []


def fwd_bwd_compiler(fx_g, _):
    graphs.append(fx_g)
    return make_boxed_func(fx_g)


def get_inductor_decomp_graphs(model: nn.Module, args, kwargs):
    """
    Obtain forward and backward graphs of a model with inductor decompositions using tracing and aot_module.

    Convenient util to get the fwd and bwd graphs of an arbitrary model
    with inductor decompositions. Note that this would simply do tracing
    with aot_module and don't ensure correctness. This is useful to track
    the ops needed in DTensor.
    """
    compiled_mod = aot_module(
        model, fw_compiler=fwd_bwd_compiler, decompositions=inductor_decomps
    )
    output = compiled_mod(*args, **kwargs)

    if output.ndim != 0:
        # if output is not a scalar tensor, by default sum it in order to
        # run backward
        output = output.sum()

    output.backward()

    # one fwd, one bwd graph
    assert len(graphs) == 2
    return graphs


def print_op_coverage_summary(model: nn.Module, args, kwargs, *, output_csv=False):
    """
    Util to print the operator coverage summary of a certain model with tabulute.

    Must have tabulate module installed.
    """
    # python module required for summary
    import csv

    from tabulate import tabulate

    fwd_graph, bwd_graph = get_inductor_decomp_graphs(model, args, kwargs)

    op_counts = {}

    for node in fwd_graph.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target not in op_counts:
                op_counts[node.target] = 0

            op_counts[node.target] += 1

    for node in bwd_graph.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target not in op_counts:
                op_counts[node.target] = 0

            op_counts[node.target] += 1

    op_infos = []

    for op, count in op_counts.items():
        supported = op in DTensor._op_dispatcher.sharding_propagator.op_to_rules
        op_infos.append([op, str(op._schema), count, supported])

    # sort the op info base on the total count index
    count_idx = 2
    op_infos.sort(key=itemgetter(count_idx), reverse=True)

    headers = ["Operator", "Schema", "Total Count", "Supported"]
    # pyrefly: ignore [bad-argument-type]
    print(tabulate(op_infos, headers=headers))

    if output_csv:
        # Open a CSV file for writing
        with open("op_summary.csv", "w", newline="") as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(headers)
            # Write each table row to the CSV file
            for row in op_infos:
                # pyrefly: ignore [bad-argument-type]
                csv_writer.writerow(row)

```



## High-Level Overview

"""    Obtain forward and backward graphs of a model with inductor decompositions using tracing and aot_module.    Convenient util to get the fwd and bwd graphs of an arbitrary model    with inductor decompositions. Note that this would simply do tracing    with aot_module and don't ensure correctness. This is useful to track    the ops needed in DTensor.

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `fwd_bwd_compiler`, `get_inductor_decomp_graphs`, `print_op_coverage_summary`

**Key imports**: itemgetter, torch, torch.fx, torch.nn as nn, make_boxed_func, aot_module, select_decomp_table, DTensor, csv, tabulate


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor/debug`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`: itemgetter
- `torch`
- `torch.fx`
- `torch.nn as nn`
- `functorch.compile`: make_boxed_func
- `torch._functorch.compilers`: aot_module
- `torch._inductor.decomposition`: select_decomp_table
- `torch.distributed.tensor`: DTensor
- `csv`
- `tabulate`: tabulate


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`torch/distributed/tensor/debug`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_visualize_sharding.py_docs.md`](./_visualize_sharding.py_docs.md)
- [`_comm_mode.py_docs.md`](./_comm_mode.py_docs.md)
- [`comm_mode_broswer_visual.js_docs.md`](./comm_mode_broswer_visual.js_docs.md)


## Cross-References

- **File Documentation**: `_op_coverage.py_docs.md`
- **Keyword Index**: `_op_coverage.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
