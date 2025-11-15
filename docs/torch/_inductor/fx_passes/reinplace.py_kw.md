# Keyword Index: `torch/_inductor/fx_passes/reinplace.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/reinplace.py](../../../../torch/_inductor/fx_passes/reinplace.py)
- **Documentation**: [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`InplaceableOp`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`class`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`from`**: [reinplace.py_docs.md](./reinplace.py_docs.md)

### Functions

- **`_decompose_scatter_functional`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`_decompose_scatter_functional_helper`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`_decompose_scatter_mutating`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`_generalized_scatter`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`_inplace_generalized_scatter`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`_overlap`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`any_use_of_views_after_node`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`bytes`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`can_fuse`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`can_inplace`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`canonicalize_view_scatter_ops`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`decompose_generalized_scatter`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`graph_call_function`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`handle_view_scatter`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`handle_views`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`is_meta_only_user`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`log_inplace_results`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`reinplace_and_refine_tensors_to_clone`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`reinplace_inplaceable_ops`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`reinplace_inplaceable_ops_core`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`scatter`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`scatter_always_uses_mutation`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`should_reinplace_scatter`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`tensor_with_same_storage_already_reinplaced`**: [reinplace.py_docs.md](./reinplace.py_docs.md)

### Imports

- **`Any`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`Autotuner`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`Callable`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`JITFunction`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`OrderedSet`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`ReinplaceCounters`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`V`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`_is_view_op`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`_pytree`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`collections`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`collections.abc`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`compute_overlapping_tensors`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`config`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`contextlib`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`dataclass`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`dataclasses`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`defaultdict`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`detect_fake_mode`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`enable_python_dispatcher`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`get_mutable_args`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`get_node_storage`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`immutable_dict`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`itertools`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`logging`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`nullcontext`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`operator`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._C._dynamo.guards`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._dispatch.python`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._dynamo.utils`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._guards`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._higher_order_ops.auto_functionalize`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._higher_order_ops.triton_kernel_wrap`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._inductor`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._inductor.fx_utils`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._inductor.lowering`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch._inductor.virtualized`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch.fx.immutable_collections`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch.fx.node`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch.fx.passes.reinplace`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch.utils`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`torch.utils._ordered_set`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`triton.runtime.autotuner`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`triton.runtime.jit`**: [reinplace.py_docs.md](./reinplace.py_docs.md)
- **`typing`**: [reinplace.py_docs.md](./reinplace.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
