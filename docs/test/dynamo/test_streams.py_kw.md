# Keyword Index: `test/dynamo/test_streams.py`

## File Information

- **Original File**: [test/dynamo/test_streams.py](../../../test/dynamo/test_streams.py)
- **Documentation**: [`test_streams.py_docs.md`](./test_streams.py_docs.md)
- **Folder**: `test/dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GraphModule`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`TestStreams`**: [test_streams.py_docs.md](./test_streams.py_docs.md)

### Functions

- **`event_generation_backend`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`fn`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`forward`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`print_graph`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`remove_file_comment`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`setUpClass`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`stream_generation_backend`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`tearDownClass`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_current_stream_api`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_event_tracing`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_event_weakref`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_get_current_stream_return`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_get_current_stream_return_different_device`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_get_current_stream_return_no_index`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_inductor_lowering`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_is_marked_side_effectful`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_local_stream_enter_exit`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_local_stream_nested_enter_exit`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_local_stream_return`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_nested_stream_enter_exit`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_nested_stream_enter_exit_graph_break`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_new_event_api`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_new_stream_api`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_run_opcheck_fork_join`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_run_opcheck_wait_record`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_run_opcheck_wait_record_stream`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_stream_backward`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_stream_context_graph_break`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_stream_enter_exit`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_stream_enter_exit_graph_break`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_stream_input`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_stream_weakref`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`test_stream_with_mutation`**: [test_streams.py_docs.md](./test_streams.py_docs.md)

### Imports

- **`TEST_MULTIGPU`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`extract_graph`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`fork_stream`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`functools`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`get_current_stream`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`get_external_object_by_index`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`new_event`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`new_stream`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`opcheck`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`patch`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`re`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`record_event`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`requires_cuda`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`run_tests`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`torch`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`torch._dynamo.graph_bytecode_inputs`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`torch._dynamo.test_case`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`torch._dynamo.testing`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`torch._dynamo.variables.streams`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`torch.library`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`torch.testing._internal.common_cuda`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`unittest`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`unittest.mock`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`wait_stream`**: [test_streams.py_docs.md](./test_streams.py_docs.md)
- **`weakref`**: [test_streams.py_docs.md](./test_streams.py_docs.md)


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
