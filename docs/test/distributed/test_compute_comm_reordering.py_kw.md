# Keyword Index: `test/distributed/test_compute_comm_reordering.py`

## File Information

- **Original File**: [test/distributed/test_compute_comm_reordering.py](../../../test/distributed/test_compute_comm_reordering.py)
- **Documentation**: [`test_compute_comm_reordering.py_docs.md`](./test_compute_comm_reordering.py_docs.md)
- **Folder**: `test/distributed`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestComputeCommReorderingMultiProc`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)

### Functions

- **`assert_pass`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`create_grouped_node_for_allreduce_and_its_deps`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`fn`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`func`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`get_snode_runtime_for_reorder_compute_test`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`get_world_trs`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`test_grouped_scheduler_node`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`test_inductor_default_comms_ordering`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`test_nccl_heuristics`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`test_raise_comms`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`test_reorder_compute_for_overlap`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`test_reorder_compute_for_overlap_custom_runtime_estimation`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`test_sink_waits`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`test_sink_waits_raise_comms`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`world_size`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)

### Imports

- **`FileCheck`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`HAS_GPU`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`get_devtype`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`ir`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`patch`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`run_and_get_triton_code`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`run_tests`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`same`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch._C`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch._dynamo`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch._dynamo.logging`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch._dynamo.test_case`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch._dynamo.utils`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch._inductor`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch._inductor.comm_analysis`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch._inductor.utils`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch.distributed._functional_collectives`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`unittest`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)
- **`unittest.mock`**: [test_compute_comm_reordering.py_docs.md](./test_compute_comm_reordering.py_docs.md)


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
