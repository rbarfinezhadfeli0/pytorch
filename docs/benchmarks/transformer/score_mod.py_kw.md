# Keyword Index: `benchmarks/transformer/score_mod.py`

## File Information

- **Original File**: [benchmarks/transformer/score_mod.py](../../../benchmarks/transformer/score_mod.py)
- **Documentation**: [`score_mod.py_docs.md`](./score_mod.py_docs.md)
- **Folder**: `benchmarks/transformer`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Experiment`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`ExperimentConfig`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`ExperimentResults`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`Times`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`class`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`from`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`instance`**: [score_mod.py_docs.md](./score_mod.py_docs.md)

### Functions

- **`__post_init__`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`_output_json_for_dashboard`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`asdict`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`benchmark_torch_function_in_microseconds`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`calculate_bandwidth`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`calculate_speedup`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`calculate_tflops`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`causal`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`cleanup_memory`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`decoding_w_cached_seq_len`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`decorator`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`flash_attn_with_kvcache_renamed`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`gen_offset`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_FA_callable`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_FD_callable`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_attn_mask_linear_score_mod`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_block_mask`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_eager_sdpa`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_experiment_configs`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_inputs`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_jagged_inputs`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_random_lengths`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_score_mod`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`get_average_speedups`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`get_backend_context`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`get_default_split_k`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`get_kernel_options`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`head_bias`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`main`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`offset`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`offsets_to_lengths`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`print_results`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`query_key_value_clones`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`relative_bias`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`run_single_backend_FA`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`run_single_backend_sdpa`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`run_single_experiment`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`safe_backend`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`score_mod_w_offset`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`wrapper`**: [score_mod.py_docs.md](./score_mod.py_docs.md)

### Imports

- **`Any`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`Callable`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`Literal`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`argparse`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`asdict`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`attn_gym.masks`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`attn_gym.masks.document_mask`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`attn_gym.mods`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`benchmarker`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`collections`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`collections.abc`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`config_utils`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`contextlib`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`csv`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`dataclasses`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`defaultdict`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`flash_attn`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`flash_attn.flash_attn_interface`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`flash_attn_with_kvcache`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`functools`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`gc`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`generate_alibi_bias`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`heads_input_type`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`itertools`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`json`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`length_to_offsets`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`math`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`nullcontext`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`numpy`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`partial`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`platform`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`random`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`sdpa_kernel`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`sys`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`tabulate`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`torch`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`torch._inductor.runtime.benchmarking`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`torch.nn.attention`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`torch.nn.attention.flex_attention`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`torch.nn.functional`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`tqdm`**: [score_mod.py_docs.md](./score_mod.py_docs.md)
- **`typing`**: [score_mod.py_docs.md](./score_mod.py_docs.md)


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
