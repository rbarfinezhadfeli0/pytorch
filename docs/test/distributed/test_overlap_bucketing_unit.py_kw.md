# Keyword Index: `test/distributed/test_overlap_bucketing_unit.py`

## File Information

- **Original File**: [test/distributed/test_overlap_bucketing_unit.py](../../../test/distributed/test_overlap_bucketing_unit.py)
- **Documentation**: [`test_overlap_bucketing_unit.py_docs.md`](./test_overlap_bucketing_unit.py_docs.md)
- **Folder**: `test/distributed`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestOverlapPreservingBucketing`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)

### Functions

- **`build_collective_info`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`compute_ancestors`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`func`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`setUpClass`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`tearDownClass`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`test_can_bucket_all_reduce`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`test_can_bucket_independent_collectives`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`test_can_bucket_multidtype_collectives`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`test_cant_bucket_ag_with_rs_hiding_interval_between`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`test_cant_bucket_nested_hiding_intervals`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)

### Imports

- **`CollectiveInfo`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`FakeStore`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`FakeTensorMode`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`FileCheck`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`HAS_GPU`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`OrderedSet`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`TestCase`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`get_devtype`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`make_fx`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`requires_accelerator_dist_backend`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch._C`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch._dynamo`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch._dynamo.logging`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch._dynamo.test_case`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch._inductor.fx_passes.overlap_preserving_bucketer`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch._inductor.fx_passes.overlap_scheduling`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch._inductor.test_case`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch.distributed`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch.fx`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`torch.utils._ordered_set`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)
- **`unittest`**: [test_overlap_bucketing_unit.py_docs.md](./test_overlap_bucketing_unit.py_docs.md)


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
