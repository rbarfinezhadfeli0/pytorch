# Documentation: `test/distributed/test_serialization.py`

## File Metadata

- **Path**: `test/distributed/test_serialization.py`
- **Size**: 5,614 bytes (5.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import os
import pickle
from io import BytesIO
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed._serialization import _streaming_load, _streaming_save
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor
from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase


DEBUG_ENV = "TORCH_SERIALIZATION_DEBUG"


class MyClass:
    def __init__(self, a: int) -> None:
        self.a = a

    def __eq__(self, other: "MyClass") -> bool:
        return self.a == other.a


class TestSerialization(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # disable debug asserts
        self._old_debug = os.environ.get(DEBUG_ENV)
        os.environ[DEBUG_ENV] = "0"

    def tearDown(self):
        if self._old_debug is not None:
            os.environ[DEBUG_ENV] = self._old_debug

    def test_scalar_tensor(self) -> None:
        tensor = torch.tensor(42, dtype=torch.int32)
        state_dict = {"scalar": tensor}
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_strided_tensor(self) -> None:
        base_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        strided_tensor = base_tensor[::2, ::2]
        state_dict = {"strided": strided_tensor}
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_tensor_with_offset(self) -> None:
        state_dict = {
            "offset": torch.arange(10, dtype=torch.float64)[2:],
            "strided": torch.arange(10, dtype=torch.float64)[2::2],
        }
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_nested_tensors(self) -> None:
        tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        tensor2 = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float64)
        state_dict = {"nested": {"tensor1": tensor1, "tensor2": tensor2}}
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_various_data_types(self) -> None:
        tensor_float32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor_int16 = torch.tensor([1, 2, 3], dtype=torch.int16)
        tensor_bool = torch.tensor([True, False, True], dtype=torch.bool)
        tensor_uint16 = torch.tensor([True, False, True], dtype=torch.uint16)
        state_dict = {
            "float32": tensor_float32,
            "int16": tensor_int16,
            "bool": tensor_bool,
            "uint16": tensor_uint16,
        }
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_empty_tensor(self) -> None:
        state_dict = {
            "empty": torch.zeros(0, 10),
        }

        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file, weights_only=False)
        self.assertEqual(result, state_dict)

    def test_dtensor(self) -> None:
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )

        device_mesh = DeviceMesh("cpu", 1)
        tensor = torch.randn(4, 4)
        dtensor = distribute_tensor(tensor, device_mesh, [])
        state_dict = dtensor
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = cast(DTensor, _streaming_load(file))
        torch.testing.assert_close(result.to_local(), state_dict.to_local())
        self.assertEqual(result._spec, state_dict._spec)

    def test_python_object(self) -> None:
        state_dict = {
            "obj": MyClass(42),
        }

        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file, weights_only=False)
        self.assertEqual(result, state_dict)

    def test_str_utf8(self) -> None:
        state_dict = {
            "obj": "Ü",
        }

        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        self.assertEqual(result, state_dict)

    def test_weights_only(self) -> None:
        state_dict = {
            "obj": MyClass(42),
        }

        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        with self.assertRaisesRegex(pickle.UnpicklingError, "not an allowed global"):
            _streaming_load(file)

        with self.assertRaisesRegex(RuntimeError, "explicit pickle_module"):
            _streaming_load(file, weights_only=True, pickle_module=pickle)

    @requires_cuda
    def test_cuda(self) -> None:
        device = torch.device("cuda:0")

        tensor = torch.tensor(42, dtype=torch.float, device=device)
        state_dict = {"scalar": tensor}
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)
        self.assertEqual(result["scalar"].device, device)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyClass`, `TestSerialization`

**Functions defined**: `__init__`, `__eq__`, `setUp`, `tearDown`, `test_scalar_tensor`, `test_strided_tensor`, `test_tensor_with_offset`, `test_nested_tensors`, `test_various_data_types`, `test_empty_tensor`, `test_dtensor`, `test_python_object`, `test_str_utf8`, `test_weights_only`, `test_cuda`

**Key imports**: os, pickle, BytesIO, cast, torch, torch.distributed as dist, _streaming_load, _streaming_save, DeviceMesh, distribute_tensor, DTensor, requires_cuda, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `pickle`
- `io`: BytesIO
- `typing`: cast
- `torch`
- `torch.distributed as dist`
- `torch.distributed._serialization`: _streaming_load, _streaming_save
- `torch.distributed.tensor`: DeviceMesh, distribute_tensor, DTensor
- `torch.testing._internal.common_utils`: requires_cuda, run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/test_serialization.py
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
- [`test_nccl.py_docs.md`](./test_nccl.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_serialization.py_docs.md`
- **Keyword Index**: `test_serialization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
