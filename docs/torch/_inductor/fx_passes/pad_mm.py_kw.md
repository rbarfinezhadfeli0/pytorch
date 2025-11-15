# Keyword Index: `torch/_inductor/fx_passes/pad_mm.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/pad_mm.py](../../../../torch/_inductor/fx_passes/pad_mm.py)
- **Documentation**: [`pad_mm.py_docs.md`](./pad_mm.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_pad_mm_init`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`_should_pad_bench`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`addmm_pattern`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`addmm_replace`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`bmm_pattern`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`bmm_replace`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`check_device`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`check_dtype`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`decorator`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`fallback`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`feedback_fn`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`fetch_fake_tensors`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`fmt_pad`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`get_alignment_size`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`get_alignment_size_dtype`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`get_cached_base_mm_benchmark_time`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`get_cached_should_pad`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`get_context`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`get_do_bench`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`get_non_view_def`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`get_pad_cache`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`get_padded_length`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`is_mm_compute_bound`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`mm_pattern`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`mm_replace`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`orig_bench_fn`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`pad_addmm`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`pad_bench_fn`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`pad_bmm`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`pad_dim`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`pad_mat1`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`pad_mat2`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`pad_mm`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`realize_symbols`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`realize_tensor`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`run_autoheuristic`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`set_cached_base_mm_benchmark_time`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`set_cached_should_pad`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`should_exclude_padding_time`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`should_pad`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`should_pad_addmm`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`should_pad_bench`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`should_pad_bench_key`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`should_pad_bmm`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`should_pad_common`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`should_pad_mm`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`should_pad_mm_bf16`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`tensor_key`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`unwrap_fake_args`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`valid_shape_and_stride`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`wrapper`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`write_pad`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)

### Imports

- **`...utils._triton`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`..pattern_matcher`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`.joint_graph`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`Any`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`Callable`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`FakeTensor`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`Tensor`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`collections.abc`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`counters`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`functools`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`has_triton`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`itertools`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`no_dispatch`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`operator`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`patterns`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`torch`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`torch._dynamo.utils`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`torch._inductor`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`torch._inductor.autoheuristic.autoheuristic`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`torch._inductor.autoheuristic.autoheuristic_utils`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`torch._inductor.runtime.runtime_utils`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`torch.utils._mode_utils`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`typing`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)
- **`utils`**: [pad_mm.py_docs.md](./pad_mm.py_docs.md)


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
