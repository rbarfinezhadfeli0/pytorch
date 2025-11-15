# Keyword Index: `torch/csrc/inductor/aoti_runtime/model_base.h`

## File Information

- **Original File**: [torch/csrc/inductor/aoti_runtime/model_base.h](../../../../../torch/csrc/inductor/aoti_runtime/model_base.h)
- **Documentation**: [`model_base.h_docs.md`](./model_base.h_docs.md)
- **Folder**: `torch/csrc/inductor/aoti_runtime`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AOTInductorModelBase`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`AOTInductorModelKernelsBase`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`ConstInfo`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`Dl_info`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`ParamInfo`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`for`**: [model_base.h_docs.md](./model_base.h_docs.md)

### Functions

- **`RAII_cpuMalloc`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`RAII_gpuMalloc`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`close`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`compute_constant_blob`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`constant_blob_size`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`constant_data_size`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`constant_dtype`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`constant_from_folded`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`constant_layout`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`constant_ndim`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`constant_offset`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`constant_type`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`dladdr`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`for`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`get_access_mode`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`get_creation_disposition`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`get_device_idx`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`get_device_type`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`if`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`is_finished`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`load_constants`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`munmap`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`num_constants`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`num_folded_constants`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`num_inputs`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`num_outputs`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`opaque_metadata_size`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`open`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`parse_device_str`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`run`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`run_single_threaded`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`update_constants_array`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`update_constants_array_from_map`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`update_constants_from_blob`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`update_constants_map`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`wait_for_completion`**: [model_base.h_docs.md](./model_base.h_docs.md)

### Includes

- **`dlfcn.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`errno.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`fcntl.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`functional`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`io.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`optional`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`regex`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`stdexcept`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`sys/mman.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`sys/stat.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`torch/csrc/inductor/aoti_runtime/constant_type.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`torch/csrc/inductor/aoti_runtime/device_utils.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`torch/csrc/inductor/aoti_runtime/utils.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`torch/csrc/inductor/aoti_runtime/utils_xpu.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`torch/csrc/inductor/aoti_torch/c/shim_mps.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`unistd.h`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`unordered_map`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`utility`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`windows.h`**: [model_base.h_docs.md](./model_base.h_docs.md)

### Namespaces

- **`namespace`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`sycl`**: [model_base.h_docs.md](./model_base.h_docs.md)
- **`torch`**: [model_base.h_docs.md](./model_base.h_docs.md)


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
