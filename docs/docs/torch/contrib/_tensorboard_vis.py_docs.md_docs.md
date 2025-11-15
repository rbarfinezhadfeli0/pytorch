# Documentation: `docs/torch/contrib/_tensorboard_vis.py_docs.md`

## File Metadata

- **Path**: `docs/torch/contrib/_tensorboard_vis.py_docs.md`
- **Size**: 8,712 bytes (8.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/contrib/_tensorboard_vis.py`

## File Metadata

- **Path**: `torch/contrib/_tensorboard_vis.py`
- **Size**: 5,875 bytes (5.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import time
from collections import defaultdict
from functools import partial

import torch


# Unfortunately it doesn't seem as if there was any way to get TensorBoard to do
# anything without having TF installed, and so this file has a hard dependency on it
# as well. It really is a debugging tool, so it doesn't matter.
try:
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.util import event_pb2
    from tensorflow.python.summary.writer.writer import FileWriter
except ImportError:
    raise ImportError(
        "TensorBoard visualization of GraphExecutors requires having "
        "TensorFlow installed"
    ) from None


def dump_tensorboard_summary(graph_executor, logdir):
    with FileWriter(logdir) as w:
        pb_graph = visualize(graph_executor)
        evt = event_pb2.Event(
            wall_time=time.time(), graph_def=pb_graph.SerializeToString()
        )
        w.add_event(evt)


def visualize(graph, name_prefix="", pb_graph=None, executors_it=None):
    """Visualizes an independent graph, or a graph executor."""
    value_map = {}
    pb_graph = pb_graph or graph_pb2.GraphDef()

    if isinstance(graph, torch._C.GraphExecutorState):
        visualize_graph_executor(
            graph, name_prefix, pb_graph, partial(visualize, pb_graph=pb_graph)
        )
        return pb_graph

    # Set up an input node
    pb_graph.node.add(op="input", name=name_prefix + "input")
    for i, value in enumerate(graph.param_node().outputs()):
        value_map[value.unique()] = name_prefix + "input:" + str(i)

    visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it)

    # Gather all outputs
    return_node = pb_graph.node.add(op="output", name=name_prefix + "output")
    for value in graph.return_node().inputs():
        return_node.input.append(value_map[value.unique()])

    return pb_graph


def visualize_graph_executor(state, name_prefix, pb_graph, inline_graph):
    """Append the state of a given GraphExecutor to the graph protobuf.

    Args:
        state (GraphExecutor or GraphExecutorState): GraphExecutor to display.
        name_prefix (str): Name prefix of the containing subgraph.
        pb_graph (GraphDef): graph to append to.
        inline_graph (Callable): a function that handles setting up a value_map,
            so that some graphs in here can be inlined. This is necessary, because
            this will simply be `visualize` for the top-level GraphExecutor,
            or `inline_graph` for all nested ones.

            The signature should look like (Graph, name_prefix) -> ().
            It will be called exactly once.

    The strategy is to embed all different configurations as independent subgraphs,
    while inlining the original graph as the one that actually produces the values.
    """
    if state.autograd_fallback_graph is not None:
        visualize(
            graph=state.autograd_fallback_graph,
            name_prefix=name_prefix + "autograd_fallback/",
            pb_graph=pb_graph,
            executors_it=iter(state.autograd_fallback.executors()),
        )

    for i, (arg_spec, plan) in enumerate(state.execution_plans.items()):
        subgraph_name = name_prefix + f"plan{i}/"

        # Create a disconnected node that will keep information regarding the input
        # types of this trace. This is unfortunately a bit too verbose to be included
        # in the subgraph name.
        input_kinds = pb_graph.node.add(op="INPUT_KIND", name=subgraph_name)
        input_kinds.attr["inputs"].s = repr(arg_spec).encode("ascii")

        visualize(plan.graph, subgraph_name, pb_graph, iter(plan.code.executors()))

        # Show gradient as an independent subgraph of this plan
        if plan.grad_executor is not None:
            grad_subgraph_name = subgraph_name + "grad/"
            visualize(plan.grad_executor, grad_subgraph_name, pb_graph)

    return inline_graph(state.graph, name_prefix + "original/")


def visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it=None):
    """Recursive part of visualize (basically skips setting up the input and output nodes)."""

    def inline_graph(subgraph, name, node):
        rec_value_map = {
            inp.unique(): value_map[val.unique()]
            for inp, val in zip(subgraph.inputs(), node.inputs())
        }
        visualize_rec(
            graph=subgraph, value_map=rec_value_map, name_prefix=name, pb_graph=pb_graph
        )
        for out, val in zip(subgraph.outputs(), node.outputs()):
            value_map[val.unique()] = rec_value_map[out.unique()]

    op_id_counter: defaultdict[str, int] = defaultdict(int)

    def name_for(node):
        kind = node.kind()[node.kind().index("::") + 2 :]
        op_id_counter[kind] += 1
        return kind, name_prefix + kind + "_" + str(op_id_counter[kind])

    def add_fusion_group(node):
        op, name = name_for(node)
        inline_graph(node.g("Subgraph"), name + "/", node)

    def add_graph_executor(node):
        op, name = name_for(node)
        if executors_it is None:
            add_node(node)
        else:
            ge = next(executors_it)
            visualize_graph_executor(
                ge, name + "/", pb_graph, partial(inline_graph, node=node)
            )

    def add_node(node):
        if node.kind() == "prim::FusionGroup":
            return add_fusion_group(node)
        elif node.kind() == "prim::GraphExecutor":
            return add_graph_executor(node)
        op, name = name_for(node)
        pb_node = pb_graph.node.add(op=op, name=name)
        for value in node.inputs():
            pb_node.input.append(value_map[value.unique()])
        # TODO: handle attrs
        for i, value in enumerate(node.outputs()):
            value_map[value.unique()] = name + ":" + str(i)

    for node in graph.nodes():
        add_node(node)

```



## High-Level Overview

"""Visualizes an independent graph, or a graph executor."""    value_map = {}    pb_graph = pb_graph or graph_pb2.GraphDef()    if isinstance(graph, torch._C.GraphExecutorState):        visualize_graph_executor(            graph, name_prefix, pb_graph, partial(visualize, pb_graph=pb_graph)        )        return pb_graph    # Set up an input node    pb_graph.node.add(op="input", name=name_prefix + "input")    for i, value in enumerate(graph.param_node().outputs()):        value_map[value.unique()] = name_prefix + "input:" + str(i)    visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it)    # Gather all outputs

This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `dump_tensorboard_summary`, `visualize`, `visualize_graph_executor`, `visualize_rec`, `inline_graph`, `name_for`, `add_fusion_group`, `add_graph_executor`, `add_node`

**Key imports**: time, defaultdict, partial, torch, graph_pb2, event_pb2, FileWriter


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/contrib`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `time`
- `collections`: defaultdict
- `functools`: partial
- `torch`
- `tensorflow.core.framework`: graph_pb2
- `tensorflow.core.util`: event_pb2
- `tensorflow.python.summary.writer.writer`: FileWriter


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/contrib`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `_tensorboard_vis.py_docs.md`
- **Keyword Index**: `_tensorboard_vis.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/contrib`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/contrib`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/torch/contrib`):

- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`_tensorboard_vis.py_kw.md_docs.md`](./_tensorboard_vis.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_tensorboard_vis.py_docs.md_docs.md`
- **Keyword Index**: `_tensorboard_vis.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
