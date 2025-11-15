# Keyword Index: `benchmarks/dynamo/torchbench.py`

## File Information

- **Original File**: [benchmarks/dynamo/torchbench.py](../../../benchmarks/dynamo/torchbench.py)
- **Documentation**: [`torchbench.py_docs.md`](./torchbench.py_docs.md)
- **Folder**: `benchmarks/dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TorchBenchmarkRunner`**: [torchbench.py_docs.md](./torchbench.py_docs.md)

### Functions

- **`__init__`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`_accuracy`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`_batch_size`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`_config`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`_reassign_parameters`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`_require_larger_multiplier_for_smaller_tensor`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`_skip`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`_tolerance`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`compute_loss`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`disable_cudagraph_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`failing_fx2trt_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`force_amp_for_fp16_bf16_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`force_fp16_for_bf16_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`forward_and_backward_pass`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`forward_pass`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`get_output_amp_train_process_func`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`get_tolerance_and_cosine_flag`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`guard_on_nn_module_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`inline_inbuilt_nn_modules_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`iter_model_names`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`load_model`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`non_deterministic_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`pick_grad`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`setup_torchbench_cwd`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_accuracy_check_as_eager_non_deterministic`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_accuracy_checks_large_models_dashboard`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_models_due_to_control_flow`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_models_due_to_export_not_supported`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_models_for_cpu`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_models_for_cpu_aarch64`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_models_for_cuda`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_models_for_freezing_cpu`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_models_for_freezing_cuda`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_models_for_xpu`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_multiprocess_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`skip_not_suitable_for_training_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`slow_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`state_dict_hook`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`torchbench_main`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`use_larger_multiplier_for_smaller_tensor`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`very_slow_models`**: [torchbench.py_docs.md](./torchbench.py_docs.md)

### Imports

- **`.common`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`BenchmarkRunner`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`_list_canary_model_paths`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`abspath`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`any`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`clone_inputs`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`collect_results`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`collections`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`common`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`gc`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`importlib`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`logging`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`namedtuple`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`os`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`os.path`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`re`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`sys`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`torch`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`torch._dynamo.testing`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`torch._dynamo.utils`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`torch_xla`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`torchbenchmark`**: [torchbench.py_docs.md](./torchbench.py_docs.md)
- **`warnings`**: [torchbench.py_docs.md](./torchbench.py_docs.md)


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
