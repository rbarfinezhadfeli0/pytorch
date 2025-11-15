# Documentation: `docs/torch/_inductor/template_heuristics/triton.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/template_heuristics/triton.py_kw.md`
- **Size**: 10,615 bytes (10.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/template_heuristics/triton.py`

## File Information

- **Original File**: [torch/_inductor/template_heuristics/triton.py](../../../../torch/_inductor/template_heuristics/triton.py)
- **Documentation**: [`triton.py_docs.md`](./triton.py_docs.md)
- **Folder**: `torch/_inductor/template_heuristics`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BaseConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`BaseHeuristicSingleton`**: [triton.py_docs.md](./triton.py_docs.md)
- **`BaseScaledMMConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`BlackwellTMATemplateConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CPUAddmmTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CPUConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CPUInt8MMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CPUMMPlusMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CPUMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CPUScaledMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAAddMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAAddmmPersistentTMATemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDABlackwellPersistentTMATemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAInt8MMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAMMAHTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAMMPlusMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAPersistentTMATemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAScaledBlackwellTMATemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAScaledMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAScaledTMAEpilogueScalingTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`CUDAScaledTMAMainLoopScalingTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`INT8MMTemplateConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`MMPlusMMTemplateConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`MMTemplateConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`MTIAAddMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`MTIAConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`MTIAInt8MMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`MTIAMMPlusMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`MTIAMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`MTIAScaledMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ROCmAddMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ROCmConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ROCmInt8MMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ROCmMMAHTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ROCmMMPlusMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ROCmMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ROCmScaledMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ScaledBlackwellTMAConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ScaledMMConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`ScaledTMAConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`TMATemplateConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`TMAWorkspaceMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`XPUAddmmPersistentTMATemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`XPUAddmmTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`XPUConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`XPUInt8MMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`XPUMMPlusMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`XPUMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`XPUPersistentTMATemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`XPUScaledMMTemplateConfigHeuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`class`**: [triton.py_docs.md](./triton.py_docs.md)
- **`for`**: [triton.py_docs.md](./triton.py_docs.md)
- **`that`**: [triton.py_docs.md](./triton.py_docs.md)

### Functions

- **`__call__`**: [triton.py_docs.md](./triton.py_docs.md)
- **`__init__`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_convert_config_to_template_kwargs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_filter_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_finalize_mm_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_get_acc_type`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_get_config_generator`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_get_cpu_exclude_function`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_get_exceeding_shared_memory_checker`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_get_template_configs_impl`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_prune_exceeding_max_shared_mem_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_prune_exhaustive_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_scale_mm_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`_valid`**: [triton.py_docs.md](./triton.py_docs.md)
- **`adjust_kernel_inputs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`are_compatible_scales`**: [triton.py_docs.md](./triton.py_docs.md)
- **`exceeds`**: [triton.py_docs.md](./triton.py_docs.md)
- **`exclude_bmm`**: [triton.py_docs.md](./triton.py_docs.md)
- **`exclude_conv`**: [triton.py_docs.md](./triton.py_docs.md)
- **`exclude_mm`**: [triton.py_docs.md](./triton.py_docs.md)
- **`get_conv_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`get_exhaustive_mm_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`get_extra_kwargs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`get_flex_attn_bwd_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`get_flex_attn_fwd_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`get_flex_decode_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`get_mm_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`preprocess_mm_configs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`should_skip_mi350x_config`**: [triton.py_docs.md](./triton.py_docs.md)
- **`triton_config`**: [triton.py_docs.md](./triton.py_docs.md)

### Imports

- **`..`**: [triton.py_docs.md](./triton.py_docs.md)
- **`..kernel.bmm`**: [triton.py_docs.md](./triton.py_docs.md)
- **`..kernel.mm`**: [triton.py_docs.md](./triton.py_docs.md)
- **`..kernel.mm_common`**: [triton.py_docs.md](./triton.py_docs.md)
- **`..kernel.mm_plus_mm`**: [triton.py_docs.md](./triton.py_docs.md)
- **`..kernel_inputs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`..lowering`**: [triton.py_docs.md](./triton.py_docs.md)
- **`..runtime.runtime_utils`**: [triton.py_docs.md](./triton.py_docs.md)
- **`..utils`**: [triton.py_docs.md](./triton.py_docs.md)
- **`..virtualized`**: [triton.py_docs.md](./triton.py_docs.md)
- **`.gemm`**: [triton.py_docs.md](./triton.py_docs.md)
- **`.registry`**: [triton.py_docs.md](./triton.py_docs.md)
- **`AddMMConfigMixin`**: [triton.py_docs.md](./triton.py_docs.md)
- **`Any`**: [triton.py_docs.md](./triton.py_docs.md)
- **`Callable`**: [triton.py_docs.md](./triton.py_docs.md)
- **`Config`**: [triton.py_docs.md](./triton.py_docs.md)
- **`GemmMaxAutotuneTemplateConfigHeuristics`**: [triton.py_docs.md](./triton.py_docs.md)
- **`KernelInputs`**: [triton.py_docs.md](./triton.py_docs.md)
- **`Lock`**: [triton.py_docs.md](./triton.py_docs.md)
- **`OrderedSet`**: [triton.py_docs.md](./triton.py_docs.md)
- **`V`**: [triton.py_docs.md](./triton.py_docs.md)
- **`__future__`**: [triton.py_docs.md](./triton.py_docs.md)
- **`annotations`**: [triton.py_docs.md](./triton.py_docs.md)
- **`bmm_template`**: [triton.py_docs.md](./triton.py_docs.md)
- **`collections.abc`**: [triton.py_docs.md](./triton.py_docs.md)
- **`config`**: [triton.py_docs.md](./triton.py_docs.md)
- **`dataclasses`**: [triton.py_docs.md](./triton.py_docs.md)
- **`functools`**: [triton.py_docs.md](./triton.py_docs.md)
- **`has_triton_stable_tma_api`**: [triton.py_docs.md](./triton.py_docs.md)
- **`itertools`**: [triton.py_docs.md](./triton.py_docs.md)
- **`lowerings`**: [triton.py_docs.md](./triton.py_docs.md)
- **`math`**: [triton.py_docs.md](./triton.py_docs.md)
- **`mm_plus_mm_template`**: [triton.py_docs.md](./triton.py_docs.md)
- **`next_power_of_2`**: [triton.py_docs.md](./triton.py_docs.md)
- **`os`**: [triton.py_docs.md](./triton.py_docs.md)
- **`partial`**: [triton.py_docs.md](./triton.py_docs.md)
- **`register_template_heuristic`**: [triton.py_docs.md](./triton.py_docs.md)
- **`scale_mm_epilogue`**: [triton.py_docs.md](./triton.py_docs.md)
- **`sympy`**: [triton.py_docs.md](./triton.py_docs.md)
- **`threading`**: [triton.py_docs.md](./triton.py_docs.md)
- **`torch`**: [triton.py_docs.md](./triton.py_docs.md)
- **`torch._inductor.template_heuristics.triton_addmm`**: [triton.py_docs.md](./triton.py_docs.md)
- **`torch.utils._ordered_set`**: [triton.py_docs.md](./triton.py_docs.md)
- **`torch.utils._triton`**: [triton.py_docs.md](./triton.py_docs.md)
- **`triton`**: [triton.py_docs.md](./triton.py_docs.md)
- **`typing`**: [triton.py_docs.md](./triton.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_inductor/template_heuristics`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/template_heuristics`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`docs/torch/_inductor/template_heuristics`):

- [`decompose_k.py_docs.md_docs.md`](./decompose_k.py_docs.md_docs.md)
- [`registry.py_kw.md_docs.md`](./registry.py_kw.md_docs.md)
- [`params.py_docs.md_docs.md`](./params.py_docs.md_docs.md)
- [`aten.py_kw.md_docs.md`](./aten.py_kw.md_docs.md)
- [`decompose_k.py_kw.md_docs.md`](./decompose_k.py_kw.md_docs.md)
- [`base.py_kw.md_docs.md`](./base.py_kw.md_docs.md)
- [`cutedsl.py_docs.md_docs.md`](./cutedsl.py_docs.md_docs.md)
- [`gemm.py_kw.md_docs.md`](./gemm.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `triton.py_kw.md_docs.md`
- **Keyword Index**: `triton.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
