# Documentation: `test/distributed/test_nccl.py`

## File Metadata

- **Path**: `test/distributed/test_nccl.py`
- **Size**: 8,806 bytes (8.60 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.cuda
import torch.cuda.nccl as nccl
import torch.distributed as c10d
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    load_tests,
    NoTest,
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

nGPUs = torch.cuda.device_count()
if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


datatypes = [torch.float]
if (
    TEST_CUDA and c10d.is_nccl_available() and nccl.version() >= (2, 10)
) or TEST_WITH_ROCM:
    datatypes.append(torch.bfloat16)

# Broadcast (and alltoall) support float8, while reduce and allreduce do not support float8 currently
broadcast_dtypes = (
    datatypes + [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
    if TEST_WITH_ROCM
    else [torch.float8_e4m3fn, torch.float8_e5m2]
)


class TestNCCL(TestCase):
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_unique_id(self, device):
        uid = nccl.unique_id()
        self.assertIsInstance(uid, bytes)
        self.assertGreater(len(uid), 1)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*broadcast_dtypes)
    def test_broadcast(self, device, dtype):
        expected = torch.zeros(128).uniform_().to(dtype=dtype)
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            tensors.append(torch.zeros(128, dtype=dtype, device=device))

        nccl.broadcast(tensors)
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

        # Test with tuple
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            tensors.append(torch.zeros(128, dtype=dtype, device=device))

        nccl.broadcast(tuple(tensors))
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce(self, device, dtype):
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(128, dtype=dtype)
        for t in cpu_tensors:
            expected.add_(t)

        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tensors)

        self.assertEqual(tensors[0], expected)

        # Test with tuple
        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tuple(tensors))

        self.assertEqual(tensors[0], expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_reduce(self, device, dtype):
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(128, dtype=dtype)
        for t in cpu_tensors:
            expected.add_(t)

        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # Test with tuple.
        tensors = tuple(cpu_tensors[i].cuda(i) for i in range(nGPUs))
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # Test with set.
        tensors = {cpu_tensors[i].cuda(i) for i in range(nGPUs)}
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_collective_errors(self, device):
        t = torch.rand(10).cuda(0)
        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_reduce(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.broadcast(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_gather(t, t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce_scatter(t, t)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_gather(self, device, dtype):
        cpu_inputs = [torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)]
        expected = torch.cat(cpu_inputs, 0)

        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        nccl.all_gather(inputs, outputs)

        for tensor in outputs:
            self.assertEqual(tensor, expected)

        # Test with tuple.
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        nccl.all_gather(tuple(inputs), tuple(outputs))

        for tensor in outputs:
            self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce_scatter(self, device, dtype):
        in_size = 32 * nGPUs
        out_size = 32

        cpu_inputs = [
            torch.zeros(in_size).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(in_size, dtype=dtype)
        for t in cpu_inputs:
            expected.add_(t)
        expected = expected.view(nGPUs, 32)

        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        nccl.reduce_scatter(inputs, outputs)

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])

        # Test with tuple
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        nccl.reduce_scatter(tuple(inputs), tuple(outputs))

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])


@requires_cuda_p2p_access()
class NCCLSymmetricMemoryTest(MultiProcContinuousTest):
    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_alloc(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        def foo():
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)


instantiate_device_type_tests(TestNCCL, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestNCCL`, `NCCLSymmetricMemoryTest`

**Functions defined**: `test_unique_id`, `test_broadcast`, `test_reduce`, `test_all_reduce`, `test_collective_errors`, `test_all_gather`, `test_reduce_scatter`, `device`, `test_nccl_symmem_alloc`, `foo`

**Key imports**: sys, torch, torch.cuda, torch.cuda.nccl as nccl, torch.distributed as c10d, torch.distributed._symmetric_memory as symm_mem, TEST_CUDA, TEST_MULTIGPU


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch.cuda`
- `torch.cuda.nccl as nccl`
- `torch.distributed as c10d`
- `torch.distributed._symmetric_memory as symm_mem`
- `torch.testing._internal.common_cuda`: TEST_CUDA, TEST_MULTIGPU


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/test_nccl.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed`):

- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`test_c10d_logger.py_docs.md`](./test_c10d_logger.py_docs.md)
- [`test_dist2.py_docs.md`](./test_dist2.py_docs.md)
- [`test_c10d_functional_native.py_docs.md`](./test_c10d_functional_native.py_docs.md)
- [`test_c10d_object_collectives.py_docs.md`](./test_c10d_object_collectives.py_docs.md)
- [`test_c10d_spawn_ucc.py_docs.md`](./test_c10d_spawn_ucc.py_docs.md)
- [`test_c10d_ucc.py_docs.md`](./test_c10d_ucc.py_docs.md)
- [`test_serialization.py_docs.md`](./test_serialization.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_nccl.py_docs.md`
- **Keyword Index**: `test_nccl.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
