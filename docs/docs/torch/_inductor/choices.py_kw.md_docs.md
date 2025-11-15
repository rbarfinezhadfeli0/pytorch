# Documentation: `docs/torch/_inductor/choices.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/choices.py_kw.md`
- **Size**: 6,028 bytes (5.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/choices.py`

## File Information

- **Original File**: [torch/_inductor/choices.py](../../../torch/_inductor/choices.py)
- **Documentation**: [`choices.py_docs.md`](./choices.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`InductorChoices`**: [choices.py_docs.md](./choices.py_docs.md)
- **`MyHeuristics`**: [choices.py_docs.md](./choices.py_docs.md)
- **`Sortable`**: [choices.py_docs.md](./choices.py_docs.md)
- **`contains`**: [choices.py_docs.md](./choices.py_docs.md)

### Functions

- **`__lt__`**: [choices.py_docs.md](./choices.py_docs.md)
- **`_finalize_template_configs`**: [choices.py_docs.md](./choices.py_docs.md)
- **`_need_to_fix_layout`**: [choices.py_docs.md](./choices.py_docs.md)
- **`can_fuse`**: [choices.py_docs.md](./choices.py_docs.md)
- **`can_fuse_horizontal`**: [choices.py_docs.md](./choices.py_docs.md)
- **`can_fuse_vertical`**: [choices.py_docs.md](./choices.py_docs.md)
- **`get_config_heuristics`**: [choices.py_docs.md](./choices.py_docs.md)
- **`get_conv_configs`**: [choices.py_docs.md](./choices.py_docs.md)
- **`get_flex_attention_bwd_configs`**: [choices.py_docs.md](./choices.py_docs.md)
- **`get_flex_attention_fwd_configs`**: [choices.py_docs.md](./choices.py_docs.md)
- **`get_flex_decode_configs`**: [choices.py_docs.md](./choices.py_docs.md)
- **`get_ktc`**: [choices.py_docs.md](./choices.py_docs.md)
- **`get_template_configs`**: [choices.py_docs.md](./choices.py_docs.md)
- **`reduction_split_factor`**: [choices.py_docs.md](./choices.py_docs.md)
- **`score_fusion`**: [choices.py_docs.md](./choices.py_docs.md)
- **`should_use_cooperative_reduction`**: [choices.py_docs.md](./choices.py_docs.md)
- **`should_use_persistent_reduction`**: [choices.py_docs.md](./choices.py_docs.md)
- **`triton_kernel_kwargs`**: [choices.py_docs.md](./choices.py_docs.md)

### Imports

- **`.`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.codecache`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.codegen.common`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.codegen.simd_kernel_features`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.codegen.triton`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.ir`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.kernel_inputs`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.kernel_template_choice`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.metrics`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.runtime.hints`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.scheduler`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.select_algorithm`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.template_heuristics`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.template_heuristics.triton`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.utils`**: [choices.py_docs.md](./choices.py_docs.md)
- **`.virtualized`**: [choices.py_docs.md](./choices.py_docs.md)
- **`Any`**: [choices.py_docs.md](./choices.py_docs.md)
- **`BaseSchedulerNode`**: [choices.py_docs.md](./choices.py_docs.md)
- **`ChoiceCaller`**: [choices.py_docs.md](./choices.py_docs.md)
- **`Config`**: [choices.py_docs.md](./choices.py_docs.md)
- **`DeviceProperties`**: [choices.py_docs.md](./choices.py_docs.md)
- **`ExternKernelChoice`**: [choices.py_docs.md](./choices.py_docs.md)
- **`Generator`**: [choices.py_docs.md](./choices.py_docs.md)
- **`KernelInputs`**: [choices.py_docs.md](./choices.py_docs.md)
- **`KernelTemplate`**: [choices.py_docs.md](./choices.py_docs.md)
- **`KernelTemplateChoice`**: [choices.py_docs.md](./choices.py_docs.md)
- **`MixOrderReduction`**: [choices.py_docs.md](./choices.py_docs.md)
- **`OrderedSet`**: [choices.py_docs.md](./choices.py_docs.md)
- **`SIMDKernelFeatures`**: [choices.py_docs.md](./choices.py_docs.md)
- **`TritonKernel`**: [choices.py_docs.md](./choices.py_docs.md)
- **`V`**: [choices.py_docs.md](./choices.py_docs.md)
- **`__future__`**: [choices.py_docs.md](./choices.py_docs.md)
- **`_use_autotune_backend`**: [choices.py_docs.md](./choices.py_docs.md)
- **`annotations`**: [choices.py_docs.md](./choices.py_docs.md)
- **`bound_sympy`**: [choices.py_docs.md](./choices.py_docs.md)
- **`collections.abc`**: [choices.py_docs.md](./choices.py_docs.md)
- **`config`**: [choices.py_docs.md](./choices.py_docs.md)
- **`functools`**: [choices.py_docs.md](./choices.py_docs.md)
- **`get_metric_table`**: [choices.py_docs.md](./choices.py_docs.md)
- **`get_template_heuristic`**: [choices.py_docs.md](./choices.py_docs.md)
- **`make_ktc_generator`**: [choices.py_docs.md](./choices.py_docs.md)
- **`next_power_of_2`**: [choices.py_docs.md](./choices.py_docs.md)
- **`partial`**: [choices.py_docs.md](./choices.py_docs.md)
- **`sympy`**: [choices.py_docs.md](./choices.py_docs.md)
- **`torch`**: [choices.py_docs.md](./choices.py_docs.md)
- **`torch._inductor.runtime.runtime_utils`**: [choices.py_docs.md](./choices.py_docs.md)
- **`torch._inductor.scheduler`**: [choices.py_docs.md](./choices.py_docs.md)
- **`torch.utils._ordered_set`**: [choices.py_docs.md](./choices.py_docs.md)
- **`torch.utils._sympy.value_ranges`**: [choices.py_docs.md](./choices.py_docs.md)
- **`triton`**: [choices.py_docs.md](./choices.py_docs.md)
- **`typing`**: [choices.py_docs.md](./choices.py_docs.md)
- **`write_text`**: [choices.py_docs.md](./choices.py_docs.md)


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

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `choices.py_kw.md_docs.md`
- **Keyword Index**: `choices.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
