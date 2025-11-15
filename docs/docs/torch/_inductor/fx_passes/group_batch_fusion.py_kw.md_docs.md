# Documentation: `docs/torch/_inductor/fx_passes/group_batch_fusion.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/fx_passes/group_batch_fusion.py_kw.md`
- **Size**: 8,481 bytes (8.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/fx_passes/group_batch_fusion.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/group_batch_fusion.py](../../../../torch/_inductor/fx_passes/group_batch_fusion.py)
- **Documentation**: [`group_batch_fusion.py_docs.md`](./group_batch_fusion.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BatchAddPostGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchClampPreGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchDetachPreGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchDivPostGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchDropoutPreGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchLayernormFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchLinearLHSFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchMathOpsPreGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchMulPostGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchNanToNumPreGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchPointwiseMathOpsPostGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchPointwiseOpsFusionFactory`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchPointwiseOpsPostGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchPointwiseOpsPreGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchReLuPostGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchReLuPreGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchSigmoidPostGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchSigmoidPreGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchSubPostGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchTanhPostGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`BatchTanhPreGradFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`GroupBatchFusionBase`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`GroupFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`GroupLinearFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`PostGradBatchLinearFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`PreGradBatchLinearFusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`_OrderedSet`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)

### Functions

- **`__contains__`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`__init__`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`__iter__`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`__len__`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`_addmm_node_can_be_fused`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`_getitem_args`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`_is_input_2d`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`_is_mutable_node`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`_mm_node_can_be_fused`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`_pointwise_node_can_be_fused`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`append`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`apply_group_batch_fusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`decompose_stack`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`decorator`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`find_dependent_nodes`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`find_independent_subset_greedy`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`fuse`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`generate_fusion_from_config`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`get_fusion_candidates`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`group_batch_fusion_passes`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`is_linear_node_can_be_fused`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`list_group_batch_fusions`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`match`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`register_fusion`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`update_pointwise_example_value`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`update_stack_example_value`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)

### Imports

- **`..`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`..pattern_matcher`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`..utils`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`Any`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`GraphTransformObserver`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`Iterable`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`OPTIMUS_EXCLUDE_POST_GRAD`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`OrderedDict`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`OrderedSet`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`_LazyGraphModule`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`collections`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`collections.abc`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`config`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`counters`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`logging`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`operator`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`torch`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`torch._dynamo.utils`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`torch._logging`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`torch.fx._lazy_graph_module`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`torch.fx.passes.graph_transform_observer`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`torch.utils._ordered_set`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`trace_structured`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)
- **`typing`**: [group_batch_fusion.py_docs.md](./group_batch_fusion.py_docs.md)


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
- [`joint_graph.py_kw.md_docs.md`](./joint_graph.py_kw.md_docs.md)
- [`numeric_utils.py_docs.md_docs.md`](./numeric_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `group_batch_fusion.py_kw.md_docs.md`
- **Keyword Index**: `group_batch_fusion.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
