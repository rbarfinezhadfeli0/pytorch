# Keyword Index: `torch/testing/_internal/distributed/multi_threaded_pg.py`

## File Information

- **Original File**: [torch/testing/_internal/distributed/multi_threaded_pg.py](../../../../../torch/testing/_internal/distributed/multi_threaded_pg.py)
- **Documentation**: [`multi_threaded_pg.py_docs.md`](./multi_threaded_pg.py_docs.md)
- **Folder**: `torch/testing/_internal/distributed`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AllGather`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`AllReduce`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`AllToAll`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`AllToAllBase`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`Broadcast`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`Collective`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`Gather`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`ProcessLocalGroup`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`ReduceScatter`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`Scatter`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`ThreadLocalWorld`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`class`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`from`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`of`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`or`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)

### Functions

- **`__init__`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`__repr__`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_allgather_base`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_create_threaded_pg`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_end_coll`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_get_world`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_install_threaded_pg`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_reduce_scatter_base`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_size_cumsum`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_start_coll`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_uninstall_threaded_pg`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`allgather`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`allgather_into_tensor_coalesced`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`allreduce`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`allreduce_coalesced`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`alltoall`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`alltoall_base`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`barrier`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`binop_reduce`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`bitwise_reduce`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`broadcast`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`default_pg`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`exception_handle`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`flatten_list`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`gather`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`getBackendName`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`group_count`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`group_name`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`join`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`pg_backend_config`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`pg_coalesce_state`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`pg_group_ranks`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`pg_map`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`pg_name`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`pg_names`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`pg_to_tag`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`reduce_scatter`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`reduce_scatter_tensor_coalesced`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`reset`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`ret_work`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`scatter`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`size`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`tags_to_pg`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`work`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)

### Imports

- **`Future`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`Optional`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_CollOp`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`_pytree`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`dataclass`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`dataclasses`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`functools`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`partial`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`sys`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`threading`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`torch`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`torch._C._distributed_c10d`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`torch.distributed`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`torch.futures`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`torch.utils`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`typing`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)
- **`weakref`**: [multi_threaded_pg.py_docs.md](./multi_threaded_pg.py_docs.md)


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
