# Keyword Index: `torch/cuda/_sanitizer.py`

## File Information

- **Original File**: [torch/cuda/_sanitizer.py](../../../torch/cuda/_sanitizer.py)
- **Documentation**: [`_sanitizer.py_docs.md`](./_sanitizer.py_docs.md)
- **Folder**: `torch/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AccessType`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`ArgumentHandler`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`CUDASanitizer`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`CUDASanitizerDispatchMode`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`CUDASanitizerErrors`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`EventHandler`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`StreamSynchronizations`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`SynchronizationError`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`UnsynchronizedAccessError`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_TensorsAccessed`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`class`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`for`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`wraps`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)

### Functions

- **`__del__`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`__init__`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`__str__`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`__torch_dispatch__`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_ensure_event_does_not_exist`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_ensure_event_exists`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_ensure_stream_exists`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_argument`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_device_synchronization`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_event_creation`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_event_deletion`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_event_record`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_event_synchronization`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_event_wait`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_kernel_launch`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_memory_allocation`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_memory_deallocation`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_stream_creation`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_handle_stream_synchronization`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_state_wait_for_other`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`add_read`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`all_streams_wait_for_event`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`all_streams_wait_for_stream`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`check_conflict`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`create_event`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`create_stream`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`create_tensor`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`delete_event`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`delete_tensor`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`disable`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`enable`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`enable_cuda_sanitizer`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`ensure_tensor_does_not_exist`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`ensure_tensor_exists`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`format_access`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`get_allocation_stack_trace`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`get_reads`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`get_write`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`is_ordered_after`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`parse_inputs`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`parse_outputs`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`record_state`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`set_write`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`stream_wait_for_event`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`sync_all_streams`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`update_seq_num`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`were_there_reads_since_last_write`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`zip_arguments`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`zip_by_key`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)

### Imports

- **`Any`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`Iterator`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`TorchDispatchMode`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`_pytree`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`collections.abc`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`dataclass`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`dataclasses`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`enum`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`functools`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`inspect`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`io`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`logging`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`re`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`sys`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`textwrap`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`torch`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`torch.cuda._gpu_trace`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`torch.utils`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`torch.utils._python_dispatch`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`traceback`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)
- **`typing`**: [_sanitizer.py_docs.md](./_sanitizer.py_docs.md)


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
