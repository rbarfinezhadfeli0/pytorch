# Keyword Index: `torch/_inductor/analysis/profile_analysis.py`

## File Information

- **Original File**: [torch/_inductor/analysis/profile_analysis.py](../../../../torch/_inductor/analysis/profile_analysis.py)
- **Documentation**: [`profile_analysis.py_docs.md`](./profile_analysis.py_docs.md)
- **Folder**: `torch/_inductor/analysis`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Device`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`JsonProfile`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`KernelStats`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`ParseException`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`class`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`for`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`from`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)

### Functions

- **`__init__`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`__repr__`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_augment_trace_helper`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_calculate_flops`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_combine_tables`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_compute_stats`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_create_devices`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_create_extern_mapping`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_create_single_table`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_create_tables`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_default_estimate_gb`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_estimate_gb`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_get_size_from_string`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_parse_kernel_name`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_slow_conv2d_adapter`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`addmm_adapter`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`augment_trace`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`baddbmm_adapter`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`bmm_adapter`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`calculate_flops`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`combine_with`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`conv_adapter`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`conv_out_dim`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`conv_out_dims`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`convert_dtype`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`create_ret`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`decorator`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`default_adapter`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`dump`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`estimate_gb`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`main`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`mm_adapter`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`mm_formula`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`parse_list`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`register_adapter`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`report`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`safe_div_format`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)

### Imports

- **`Any`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`Callable`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`DeviceInfo`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`OrderedSet`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`_pytree`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`argparse`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`collections`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`collections.abc`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`dataclass`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`dataclasses`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`defaultdict`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`flop_registry`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`json`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`logging`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`math`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`os`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`tabulate_2d`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`tempfile`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`torch`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`torch._inductor.analysis.device_info`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`torch._inductor.utils`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`torch.utils`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`torch.utils._ordered_set`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`torch.utils.flop_counter`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)
- **`typing`**: [profile_analysis.py_docs.md](./profile_analysis.py_docs.md)


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
