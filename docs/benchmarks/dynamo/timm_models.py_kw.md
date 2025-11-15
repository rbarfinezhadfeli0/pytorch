# Keyword Index: `benchmarks/dynamo/timm_models.py`

## File Information

- **Original File**: [benchmarks/dynamo/timm_models.py](../../../benchmarks/dynamo/timm_models.py)
- **Documentation**: [`timm_models.py_docs.md`](./timm_models.py_docs.md)
- **Folder**: `benchmarks/dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TimmRunner`**: [timm_models.py_docs.md](./timm_models.py_docs.md)

### Functions

- **`__init__`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`_config`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`_download_model`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`_skip`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`compute_loss`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`force_amp_for_fp16_bf16_models`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`force_fp16_for_bf16_models`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`forward_and_backward_pass`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`forward_pass`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`get_family_name`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`get_output_amp_train_process_func`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`get_tolerance_and_cosine_flag`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`guard_on_nn_module_models`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`inline_inbuilt_nn_modules_models`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`iter_model_names`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`load_model`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`pick_grad`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`pip_install`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`populate_family`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`read_models_from_docs`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`refresh_model_names`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`scaled_compute_loss`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`skip_accuracy_check_as_eager_non_deterministic`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`skip_models`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`skip_models_for_cpu`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`skip_models_for_cpu_aarch64`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`timm_main`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`use_larger_multiplier_for_smaller_tensor`**: [timm_models.py_docs.md](./timm_models.py_docs.md)

### Imports

- **`.common`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`BenchmarkRunner`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`__version__`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`clone_inputs`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`collect_results`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`common`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`config`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`create_model`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`glob`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`importlib`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`list_models`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`logging`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`os`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`re`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`resolve_data_config`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`subprocess`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`sys`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`timm`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`timm.data`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`timm.models`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`torch`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`torch._dynamo.testing`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`torch._dynamo.utils`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`torch._inductor`**: [timm_models.py_docs.md](./timm_models.py_docs.md)
- **`warnings`**: [timm_models.py_docs.md](./timm_models.py_docs.md)


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
