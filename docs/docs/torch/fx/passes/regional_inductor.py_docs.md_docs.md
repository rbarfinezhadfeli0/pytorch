# Documentation: `docs/torch/fx/passes/regional_inductor.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/passes/regional_inductor.py_docs.md`
- **Size**: 9,803 bytes (9.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/passes/regional_inductor.py`

## File Metadata

- **Path**: `torch/fx/passes/regional_inductor.py`
- **Size**: 6,522 bytes (6.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import functools
import logging

import torch
from torch.fx._compatibility import compatibility


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor"]


# standalone_inductor returns a callable class object - this does not sit well
# with Fx graph node op call_function which expects a function. So this is just
# a wrapper function to make Fx graph codegen happy.
def _dummy_wrapper(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


def _partition_by_supported_nodes(gm, supported_ops, prefix):
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
    from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

    partitioner = CapabilityBasedPartitioner(
        gm, supported_ops, allows_single_node_partition=True
    )

    candidate_partitions = partitioner.propose_partitions()
    partitioned_gm = fuse_by_partitions(
        partitioner.graph_module,
        [partition.nodes for partition in candidate_partitions],
        prefix=prefix,
        always_return_tuple=True,
    )

    return partitioned_gm


def _compile_submod(gm, prefix):
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])
                else:
                    raise RuntimeError(
                        f"Partition is bad because non fake tensor value is seen {inp_node}"
                    )

            submod = getattr(gm, node.target)

            # Get inductor configs from annotation
            # TODO we should change partition when there are multiple differently
            # annotated regions.
            inductor_options = {}
            for sub_node in submod.graph.nodes:
                if hasattr(sub_node, "meta") and sub_node.meta.get("custom", None):
                    custom = sub_node.meta["custom"]
                    if isinstance(custom, dict) and "compile_with_inductor" in custom:
                        compile_value = custom["compile_with_inductor"]
                        if (
                            isinstance(compile_value, dict)
                            and "inductor_configs" in compile_value
                        ):
                            inductor_options = compile_value["inductor_configs"]
                            break

            # Log the options being used
            logger.info(
                "Compiling submodule %s with inductor options: %s",
                node.target,
                inductor_options,
            )

            # Apply config patches before compilation
            import torch._inductor.config as inductor_config

            # Validate that all config keys exist
            for key in inductor_options:
                if not hasattr(inductor_config, key):
                    raise ValueError(
                        f"Invalid inductor config key '{key}' in regional_inductor annotation. "
                        f"Available config keys can be found in torch._inductor.config"
                    )

            with inductor_config.patch(inductor_options):
                compiled_fn = torch._inductor.standalone_compile(
                    submod, fake_inputs, dynamic_shapes="from_tracing_context", aot=True
                )
            assert isinstance(compiled_fn, AOTCompiledArtifact)
            # _dummy_wrapper is to make call_function happy
            compiled_submod = _dummy_wrapper(compiled_fn)
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                del gm._modules[node.target]

    gm.recompile()
    return gm


def _needs_inductor_compile(node: torch.fx.Node):
    return (
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and "compile_with_inductor" in node.meta["custom"]
    )


def _compile_fx_annotated_nodes_with_inductor(gm):
    from torch.fx.passes.operator_support import OperatorSupport

    found_marked_node = False
    for node in gm.graph.nodes:
        if _needs_inductor_compile(node):
            found_marked_node = True
            break

    if not found_marked_node:
        logger.info("No inductor marked nodes found")
        return gm

    class InductorMarkedNodes(OperatorSupport):
        def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
            return _needs_inductor_compile(node)

    marked_nodes = InductorMarkedNodes()
    gm = _partition_by_supported_nodes(gm, marked_nodes, "__marked_inductor_submod")
    gm = _compile_submod(gm, "__marked_inductor_submod")
    return gm


def _recursive_compile_fx_annotated_nodes_with_inductor(gm):
    for node in gm.graph.find_nodes(op="get_attr"):
        if _needs_inductor_compile(node):
            # If the get_attr itself is marked for compile, the outer graph will
            # take care of it. If we dont do that, we end up with nested
            # regional inductor compiles that do not work well.
            continue
        submod = getattr(gm, node.target)
        if isinstance(submod, torch.fx.GraphModule):
            _recursive_compile_fx_annotated_nodes_with_inductor(submod)

    return _compile_fx_annotated_nodes_with_inductor(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor(gm, *example_args):
    """
    Scoops out inductor marked regions and compiles them with inductor.

    Inductor options should be provided via the annotation API:
    with fx_traceback.annotate({
        "compile_with_inductor": {
            "inductor_configs": {
                "max_autotune": True,
                "triton.cudagraphs": False
            }
        }
    }):
    """
    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _recursive_compile_fx_annotated_nodes_with_inductor(gm)

```



## High-Level Overview


This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `InductorMarkedNodes`

**Functions defined**: `_dummy_wrapper`, `inner`, `_partition_by_supported_nodes`, `_compile_submod`, `_needs_inductor_compile`, `_compile_fx_annotated_nodes_with_inductor`, `is_node_supported`, `_recursive_compile_fx_annotated_nodes_with_inductor`, `regional_inductor`

**Key imports**: functools, logging, torch, compatibility, CapabilityBasedPartitioner, fuse_by_partitions, AOTCompiledArtifact, torch._inductor.config as inductor_config, OperatorSupport


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `logging`
- `torch`
- `torch.fx._compatibility`: compatibility
- `torch.fx.passes.infra.partitioner`: CapabilityBasedPartitioner
- `torch.fx.passes.utils.fuser_utils`: fuse_by_partitions
- `torch._inductor.standalone_compile`: AOTCompiledArtifact
- `torch._inductor.config as inductor_config`
- `torch.fx.passes.operator_support`: OperatorSupport


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/fx/passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`operator_support.py_docs.md`](./operator_support.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_drawer.py_docs.md`](./graph_drawer.py_docs.md)
- [`shape_prop.py_docs.md`](./shape_prop.py_docs.md)
- [`split_utils.py_docs.md`](./split_utils.py_docs.md)
- [`runtime_assert.py_docs.md`](./runtime_assert.py_docs.md)
- [`splitter_base.py_docs.md`](./splitter_base.py_docs.md)
- [`graph_transform_observer.py_docs.md`](./graph_transform_observer.py_docs.md)
- [`fake_tensor_prop.py_docs.md`](./fake_tensor_prop.py_docs.md)


## Cross-References

- **File Documentation**: `regional_inductor.py_docs.md`
- **Keyword Index**: `regional_inductor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/fx/passes`):

- [`split_utils.py_kw.md_docs.md`](./split_utils.py_kw.md_docs.md)
- [`fake_tensor_prop.py_kw.md_docs.md`](./fake_tensor_prop.py_kw.md_docs.md)
- [`tools_common.py_kw.md_docs.md`](./tools_common.py_kw.md_docs.md)
- [`param_fetch.py_kw.md_docs.md`](./param_fetch.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`graph_manipulation.py_docs.md_docs.md`](./graph_manipulation.py_docs.md_docs.md)
- [`annotate_getitem_nodes.py_docs.md_docs.md`](./annotate_getitem_nodes.py_docs.md_docs.md)
- [`split_module.py_docs.md_docs.md`](./split_module.py_docs.md_docs.md)
- [`pass_manager.py_kw.md_docs.md`](./pass_manager.py_kw.md_docs.md)
- [`tools_common.py_docs.md_docs.md`](./tools_common.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `regional_inductor.py_docs.md_docs.md`
- **Keyword Index**: `regional_inductor.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
