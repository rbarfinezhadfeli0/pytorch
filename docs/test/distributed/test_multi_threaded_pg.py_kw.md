# Keyword Index: `test/distributed/test_multi_threaded_pg.py`

## File Information

- **Original File**: [test/distributed/test_multi_threaded_pg.py](../../../test/distributed/test_multi_threaded_pg.py)
- **Documentation**: [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)
- **Folder**: `test/distributed`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MyFunc`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`TestCollectivesWithBaseClass`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`TestCollectivesWithWrapper`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)

### Functions

- **`_test_method`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`backward`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`forward`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`setUp`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`stuff_in_other_thread`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`tearDown`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_all_reduce`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_all_reduce_coalesced`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_all_reduce_ops`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_all_to_all`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_all_to_all_single_list`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_all_to_all_single_none`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_all_to_all_single_tensor`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_allgather`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_assert_equal_on_rank`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_broadcast`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_broadcast_object_list`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_bwd_sees_fwd_pg`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_collective_error_on_rank_non_zero`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_collective_error_on_rank_non_zero_all`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_collective_error_on_rank_zero`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_gather`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_reduce_scatter`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_scatter`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_skip`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_subpg`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`test_using_pg_from_another_thread`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`world_size`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)

### Imports

- **`IS_SANDCASTLE`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`ReduceOp`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`functools`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`operator`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`os`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`reduce`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`skip`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`sys`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`threading`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`torch`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`torch._C._distributed_c10d`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`torch.autograd`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`torch.distributed`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)
- **`unittest`**: [test_multi_threaded_pg.py_docs.md](./test_multi_threaded_pg.py_docs.md)


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
