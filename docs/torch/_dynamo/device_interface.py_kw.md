# Keyword Index: `torch/_dynamo/device_interface.py`

## File Information

- **Original File**: [torch/_dynamo/device_interface.py](../../../torch/_dynamo/device_interface.py)
- **Documentation**: [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CpuInterface`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`CudaInterface`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`DeviceGuard`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`DeviceInterface`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`Event`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`MpsInterface`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`MtiaInterface`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`Stream`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`Worker`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`XpuInterface`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`class`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`defining`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`device`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`from`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`into`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`provides`**: [device_interface.py_docs.md](./device_interface.py_docs.md)

### Functions

- **`__enter__`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`__exit__`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`__init__`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`__new__`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`_set_stream_by_id`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`current_device`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`current_stream`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`device_count`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`elapsed_time`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`exchange_device`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`get_compute_capability`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`get_device_properties`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`get_interface_for_device`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`get_raw_stream`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`get_registered_device_interfaces`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`init_device_reg`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`is_available`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`is_bf16_supported`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`is_dtype_supported`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`is_triton_capable`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`maybe_exchange_device`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`memory_allocated`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`raise_if_triton_unavailable`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`record`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`register_interface_for_device`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`set_device`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`set_stream`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`stream`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`synchronize`**: [device_interface.py_docs.md](./device_interface.py_docs.md)

### Imports

- **`Any`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`Callable`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`GPUTooOldForTriton`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`_cuda_getCurrentRawStream`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`_mtia_getCurrentRawStream`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`_xpu_getCurrentRawStream`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`collections`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`collections.abc`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`dataclass`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`dataclasses`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`inspect`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`multiprocessing`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`namedtuple`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`time`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`torch`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`torch._C`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`torch._inductor.exc`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`triton.backends`**: [device_interface.py_docs.md](./device_interface.py_docs.md)
- **`typing`**: [device_interface.py_docs.md](./device_interface.py_docs.md)


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
