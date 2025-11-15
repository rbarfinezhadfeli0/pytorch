# Keyword Index: `test/distributed/_composable/test_replicate_with_compiler.py`

## File Information

- **Original File**: [test/distributed/_composable/test_replicate_with_compiler.py](../../../../test/distributed/_composable/test_replicate_with_compiler.py)
- **Documentation**: [`test_replicate_with_compiler.py_docs.md`](./test_replicate_with_compiler.py_docs.md)
- **Folder**: `test/distributed/_composable`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DDP_TP_Test`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`MultiProcessInductorTestCase`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`Net`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`ReplicateTest`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)

### Functions

- **`__init__`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`_compiler_fn`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`_test_bucketing`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`_test_compile`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`bwd`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`compiler_fn`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`forward`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`inner_compiler`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`setUp`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`setup`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`tearDown`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_bucketing_coalesced_op`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_bucketing_concat_op`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_compile_backward_only`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_compile_bf16`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_compile_cpu`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_compile_cpu_no_sync`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_compile_fp16`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_compile_gpu`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_compile_gpu_ac`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`test_ddp_tp`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`world_size`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)

### Imports

- **`Callable`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`DistributedDataParallel`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`FakeStore`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`FileCheck`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`HAS_GPU`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`Optional`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`TestCase`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`_inductor`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`checkpoint`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`collections.abc`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`compiled_autograd`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`contextlib`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`copy`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`counters`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`deepcopy`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`functools`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`get_devtype`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`init_device_mesh`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`replicate`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`run_and_get_triton_code`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`run_tests`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch._C`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch._dynamo`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch._dynamo.utils`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch._inductor.test_case`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch._inductor.utils`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.distributed`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.distributed._composable.replicate`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.distributed.algorithms.ddp_comm_hooks`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.nn.parallel.distributed`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`torch.utils.checkpoint`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`typing`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)
- **`unittest`**: [test_replicate_with_compiler.py_docs.md](./test_replicate_with_compiler.py_docs.md)


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
