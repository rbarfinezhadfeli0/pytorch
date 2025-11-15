# Keyword Index: `torch/cuda/_memory_viz.py`

## File Information

- **Original File**: [torch/cuda/_memory_viz.py](../../../torch/cuda/_memory_viz.py)
- **Documentation**: [`_memory_viz.py_docs.md`](./_memory_viz.py_docs.md)
- **Folder**: `torch/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Bytes`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)

### Functions

- **`__add__`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`__init__`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`__repr__`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_block_extra`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_block_extra_legacy`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_format_size`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_format_viz`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_frame_filter`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_frame_fmt`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_frames_fmt`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_name`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_output`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_profile_to_snapshot`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_read`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_report_free`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_seg_info`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_seg_key`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_write`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_write_blocks`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`allocate`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`calc_active`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`compare`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`filter_alloc_free_pairs`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`find_segment`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`format`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`format_flamegraph`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`frames_fragment`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`free`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`memory`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`profile_plot`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`segment_plot`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`segments`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`segsum`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`to_device`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`trace`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`trace_plot`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)

### Imports

- **`Action`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`Any`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`_EventType`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`argparse`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`base64`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`functools`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`groupby`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`io`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`itertools`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`json`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`lru_cache`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`operator`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`os`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`os.path`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`pickle`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`subprocess`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`sys`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`tempfile`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`torch`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`torch._C._profiler`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`torch.profiler._memory_profiler`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`typing`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`urllib.request`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)
- **`warnings`**: [_memory_viz.py_docs.md](./_memory_viz.py_docs.md)


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
