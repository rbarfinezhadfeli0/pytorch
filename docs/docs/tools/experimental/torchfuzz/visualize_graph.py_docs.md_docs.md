# Documentation: `docs/tools/experimental/torchfuzz/visualize_graph.py_docs.md`

## File Metadata

- **Path**: `docs/tools/experimental/torchfuzz/visualize_graph.py_docs.md`
- **Size**: 9,732 bytes (9.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/experimental/torchfuzz/visualize_graph.py`

## File Metadata

- **Path**: `tools/experimental/torchfuzz/visualize_graph.py`
- **Size**: 7,069 bytes (6.90 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# mypy: ignore-errors

"""
Visualization tools for operation stacks and graphs as DAGs.
"""

import subprocess

from torchfuzz.ops_fuzzer import OperationGraph
from torchfuzz.tensor_fuzzer import TensorSpec


def save_and_render_dot(dot_content: str, filename: str = "operation_stack"):
    """
    Save DOT content to file and render as PNG/PDF.

    Args:
        dot_content: DOT format string
        filename: Base filename (without extension)
    """
    import os

    dot_file = f"{filename}.dot"
    png_file = f"{filename}.png"

    # Get absolute path for clickable link
    abs_png = os.path.abspath(png_file)

    # Save DOT file
    with open(dot_file, "w") as f:
        f.write(dot_content)

    # Render to PNG
    try:
        subprocess.run(["dot", "-Tpng", dot_file, "-o", png_file], check=True)
        print(f"🖼️  View: file://{abs_png}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def operation_graph_to_dot(
    graph: OperationGraph, title: str = "Operation Graph"
) -> str:
    """
    Convert an operation graph to Graphviz DOT format for visualization.

    Args:
        graph: OperationGraph instance
        title: Title for the graph

    Returns:
        DOT format string
    """
    dot_lines = [
        "digraph OperationGraph {",
        f'    label="{title}";',
        "    rankdir=TB;",  # Top to bottom layout
        "    node [shape=box, style=filled, fontsize=10];",
        "    edge [fontsize=8];",
        "",
    ]

    # Add nodes with styling based on operation type
    for node_id, node in graph.nodes.items():
        # Choose color and shape based on operation type
        if node.op_name.startswith("arg_"):
            color = "lightblue"
            shape = "ellipse"
        elif node.op_name == "constant":
            color = "lightgreen"
            shape = "ellipse"
        elif "aten" in node.op_name:
            color = "lightyellow"
            shape = "box"
        else:
            color = "lightgray"
            shape = "box"

        # Create comprehensive label
        if node.op_name.startswith("arg_"):
            label_parts = [node.op_name]
        else:
            label_parts = [node_id, node.op_name, f"depth {node.depth}"]

        if hasattr(node.output_spec, "dtype"):
            dtype_str = str(node.output_spec.dtype).replace("torch.", "")
            label_parts.append(dtype_str)

        # Only add size for TensorSpec, not ScalarSpec
        if isinstance(node.output_spec, TensorSpec) and node.output_spec.size:
            size_str = "x".join(map(str, node.output_spec.size))
            label_parts.append(f"size {size_str}")

        label = "\\n".join(label_parts)

        # Special highlighting for root node
        extra_style = ""
        if node_id == graph.root_node_id:
            extra_style = ", penwidth=3, color=red"

        dot_lines.append(
            f'    {node_id} [label="{label}", fillcolor="{color}", shape="{shape}"{extra_style}];'
        )

    dot_lines.append("")

    # Add edges based on the graph structure
    for node_id, node in graph.nodes.items():
        for i, input_node_id in enumerate(node.input_nodes):
            # Add edge from input node to current node with input position label
            edge_label = f"input_{i}"
            dot_lines.append(
                f'    {input_node_id} -> {node_id} [label="{edge_label}"];'
            )

    dot_lines.extend(
        [
            "",
            "    // Legend",
            "    subgraph cluster_legend {",
            '        label="Legend";',
            "        style=filled;",
            "        fillcolor=white;",
            '        legend_arg [label="arg", fillcolor=lightblue, shape=ellipse];',
            '        legend_const [label="constant", fillcolor=lightgreen, shape=ellipse];',
            '        legend_aten [label="aten ops", fillcolor=lightyellow, shape=box];',
            '        legend_root [label="root", fillcolor=orange, shape=box, penwidth=3, color=red];',
            "    }",
            "}",
        ]
    )

    return "\n".join(dot_lines)


def visualize_operation_graph(
    graph: OperationGraph,
    title: str = "Operation Graph",
    output_folder: str = ".",
):
    """
    Complete visualization pipeline for an operation graph.

    Args:
        graph: OperationGraph instance
        title: Title for the visualization
        output_folder: Folder where to save the visualization files
    """
    # Generate DOT content
    dot_content = operation_graph_to_dot(graph, title)

    # Save and render in the specified folder
    import os

    filename = os.path.join(output_folder, "operation_graph")
    save_and_render_dot(dot_content, filename)


def operation_graph_to_networkx(graph: OperationGraph):
    """
    Convert operation graph to NetworkX graph for Python visualization.
    Requires: pip install networkx matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print(
            "⚠️  NetworkX/Matplotlib not installed. Run: pip install networkx matplotlib"
        )
        return

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for node_id, node in graph.nodes.items():
        label = f"{node_id}\n{node.op_name}\ndepth {node.depth}"
        G.add_node(node_id, label=label, node=node)

    # Add edges based on the graph structure
    for node_id, node in graph.nodes.items():
        for input_node_id in node.input_nodes:
            if input_node_id in graph.nodes:  # Only add edges to nodes in the graph
                G.add_edge(input_node_id, node_id)

    # Plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes with colors based on operation type
    node_colors = []
    for node_id in G.nodes():
        node = graph.nodes[node_id]
        if node.op_name.startswith("arg_"):
            node_colors.append("lightblue")
        elif node.op_name == "constant":
            node_colors.append("lightgreen")
        elif "aten" in node.op_name:
            node_colors.append("lightyellow")
        else:
            node_colors.append("lightgray")

    # Highlight root node
    node_sizes = []
    for node_id in G.nodes():
        if node_id == graph.root_node_id:
            node_sizes.append(2000)  # Larger size for root
        else:
            node_sizes.append(1500)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

    # Draw labels
    labels = {
        node_id: f"{node_id}\n{graph.nodes[node_id].op_name}" for node_id in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("Operation Graph Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("operation_graph_networkx.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ NetworkX graph visualization saved as operation_graph_networkx.png")

```



## High-Level Overview

"""Visualization tools for operation stacks and graphs as DAGs.

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `save_and_render_dot`, `operation_graph_to_dot`, `visualize_operation_graph`, `operation_graph_to_networkx`

**Key imports**: subprocess, OperationGraph, TensorSpec, os, os, matplotlib.pyplot as plt, networkx as nx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/experimental/torchfuzz`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `subprocess`
- `torchfuzz.ops_fuzzer`: OperationGraph
- `torchfuzz.tensor_fuzzer`: TensorSpec
- `os`
- `matplotlib.pyplot as plt`
- `networkx as nx`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/experimental/torchfuzz`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`codegen.py_docs.md`](./codegen.py_docs.md)
- [`tensor_fuzzer.py_docs.md`](./tensor_fuzzer.py_docs.md)
- [`fuzzer.py_docs.md`](./fuzzer.py_docs.md)
- [`checks.py_docs.md`](./checks.py_docs.md)
- [`test_determinism.py_docs.md`](./test_determinism.py_docs.md)
- [`type_promotion.py_docs.md`](./type_promotion.py_docs.md)
- [`ops_fuzzer.py_docs.md`](./ops_fuzzer.py_docs.md)
- [`multi_process_fuzzer.py_docs.md`](./multi_process_fuzzer.py_docs.md)


## Cross-References

- **File Documentation**: `visualize_graph.py_docs.md`
- **Keyword Index**: `visualize_graph.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/experimental/torchfuzz`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/experimental/torchfuzz`, which is part of the **core PyTorch library**.



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

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/experimental/torchfuzz`):

- [`ops_fuzzer.py_docs.md_docs.md`](./ops_fuzzer.py_docs.md_docs.md)
- [`multi_process_fuzzer.py_docs.md_docs.md`](./multi_process_fuzzer.py_docs.md_docs.md)
- [`multi_process_fuzzer.py_kw.md_docs.md`](./multi_process_fuzzer.py_kw.md_docs.md)
- [`checks.py_kw.md_docs.md`](./checks.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`checks.py_docs.md_docs.md`](./checks.py_docs.md_docs.md)
- [`runner.py_docs.md_docs.md`](./runner.py_docs.md_docs.md)
- [`type_promotion.py_docs.md_docs.md`](./type_promotion.py_docs.md_docs.md)
- [`fuzzer.py_kw.md_docs.md`](./fuzzer.py_kw.md_docs.md)
- [`test_determinism.py_kw.md_docs.md`](./test_determinism.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `visualize_graph.py_docs.md_docs.md`
- **Keyword Index**: `visualize_graph.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
