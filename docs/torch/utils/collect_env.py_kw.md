# Keyword Index: `torch/utils/collect_env.py`

## File Information

- **Original File**: [torch/utils/collect_env.py](../../../torch/utils/collect_env.py)
- **Documentation**: [`collect_env.py_docs.md`](./collect_env.py_docs.md)
- **Folder**: `torch/utils`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_detect_linux_pkg_manager`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`check_release_file`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_cachingallocator_config`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_clang_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_cmake_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_conda_packages`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_cpu_info`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_cuda_module_loading_config`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_cudnn_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_env_info`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_gcc_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_gpu_info`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_intel_gpu_detected`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_intel_gpu_driver_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_intel_gpu_onboard`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_libc_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_linux_pkg_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_lsb_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_mac_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_nvidia_driver_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_nvidia_smi`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_os`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_pip_packages`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_platform`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_pretty_env_info`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_python_platform`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_running_cuda_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_version_or_na`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`get_windows_version`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`is_xnnpack_available`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`main`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`maybe_start_on_next_line`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`prepend`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`pretty_str`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`replace_bools`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`replace_if_empty`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`replace_nones`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`run`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`run_and_parse_first_match`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`run_and_read_all`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`run_and_return_first_line`**: [collect_env.py_docs.md](./collect_env.py_docs.md)

### Imports

- **`cast`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`collections`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`datetime`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`json`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`locale`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`machine`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`namedtuple`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`os`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`platform`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`re`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`subprocess`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`sys`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`torch`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`torch.backends.xnnpack`**: [collect_env.py_docs.md](./collect_env.py_docs.md)
- **`typing`**: [collect_env.py_docs.md](./collect_env.py_docs.md)


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
