# Documentation: `docs/torch/_inductor/fx_passes/joint_graph.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/fx_passes/joint_graph.py_kw.md`
- **Size**: 6,592 bytes (6.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/fx_passes/joint_graph.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/joint_graph.py](../../../../torch/_inductor/fx_passes/joint_graph.py)
- **Documentation**: [`joint_graph.py_docs.md`](./joint_graph.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`UniformValueConstantFolder`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)

### Functions

- **`__init__`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`_add_peephole_patterns`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`_deduce_value`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`_other_is_broadcasted_in_dim`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`_partial_softmax_pattern`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`_support_dynamic_shape`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`add_node_replacement`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`bmm_to_mm`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`canonicalize_aten_ir_passes`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`canonicalize_quant_mapping`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`constant_fold_uniform_value`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`definitely_equal`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`div_softmax_pattern`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`fake_tensors_eq`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`fix_iota_device`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`insert_placerholder_values`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`insertable_tensor_check`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`is_zero_int`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`joint_graph_passes`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`lazy_init`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`mul_softmax_pattern`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`pointless_convert`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`pointless_permute_pair`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`pointless_view`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`pointless_view_pair`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`remove_no_ops`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`remove_redundant_views`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`repl`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`replace_no_op`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`visit`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)

### Imports

- **`..`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`..pattern_matcher`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`.decompose_mem_bound_mm`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`.fuse_attention`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`.misc_patterns`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`.pad_mm`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`.post_grad`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`.replace_random`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`Any`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`ConstantFolder`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`Counter`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`OrderedSet`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`Sequence`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`StorageWeakRef`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`_SymHashingDict`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`_misc_patterns_init`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`_pad_mm_init`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`_sfdp_init`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`check_device`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`collections`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`collections.abc`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`config`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`counters`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`functools`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`get_gpu_type`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`itertools`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`logging`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`operator`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`remove_noop_ops`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`replace_random_passes`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch._dynamo.utils`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch._guards`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch._inductor.constant_folding`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch._inductor.fx_passes.dedupe_symint_uses`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch._inductor.utils`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch.multiprocessing.reductions`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch.utils._ordered_set`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`torch.utils._pytree`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)
- **`typing`**: [joint_graph.py_docs.md](./joint_graph.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_inductor/fx_passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_inductor/fx_passes`):

- [`dedupe_symint_uses.py_kw.md_docs.md`](./dedupe_symint_uses.py_kw.md_docs.md)
- [`overlap_preserving_bucketer.py_kw.md_docs.md`](./overlap_preserving_bucketer.py_kw.md_docs.md)
- [`pre_grad.py_docs.md_docs.md`](./pre_grad.py_docs.md_docs.md)
- [`b2b_gemm.py_docs.md_docs.md`](./b2b_gemm.py_docs.md_docs.md)
- [`freezing_patterns.py_kw.md_docs.md`](./freezing_patterns.py_kw.md_docs.md)
- [`fsdp.py_docs.md_docs.md`](./fsdp.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`replace_random.py_kw.md_docs.md`](./replace_random.py_kw.md_docs.md)
- [`numeric_utils.py_docs.md_docs.md`](./numeric_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `joint_graph.py_kw.md_docs.md`
- **Keyword Index**: `joint_graph.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
