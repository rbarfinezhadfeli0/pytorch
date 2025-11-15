# Documentation: `docs/test/distributed/checkpoint/test_traverse.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/test_traverse.py_docs.md`
- **Size**: 8,925 bytes (8.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/test_traverse.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_traverse.py`
- **Size**: 5,588 bytes (5.46 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.distributed.checkpoint._traverse as _traverse
from torch.testing._internal.common_utils import run_tests, TestCase


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


class TestTraverse(TestCase):
    """
    Test class for util methods of _traverse
    """

    def test_traverse_shallow(self) -> None:
        state_dict = {
            "key0": 1,
            "key1": [1, 2],
            "key2": {1: 2, 2: 3},
            "key3": torch.tensor([1]),
        }

        data = {}

        def collect_data(path, value):
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertIn(("key0",), data)
        self.assertEqual(data[("key0",)], 1)

        self.assertIn(("key1",), data)
        self.assertEqual(data[("key1",)], [1, 2])

        self.assertIn(("key2", "1"), data)
        self.assertEqual(data[("key2", "1")], 2)
        self.assertIn(("key2", "2"), data)
        self.assertEqual(data[("key2", "2")], 3)

        self.assertIn(("key3",), data)
        self.assertEqual(data[("key3",)], torch.tensor([1]))

    def test_traverse_nested_list(self) -> None:
        state_dict = {
            "key1": [
                torch.tensor([1]),
                [33, torch.tensor([2]), [44, 55]],
                [66, 77],
            ],
        }

        data = {}

        def collect_data(path, value):
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertNotIn(("key1"), data)

        self.assertIn(("key1", 0), data)
        self.assertEqual(data[("key1", 0)], torch.tensor([1]))

        self.assertIn(("key1", 1, 0), data)
        self.assertEqual(data[("key1", 1, 0)], 33)

        self.assertIn(("key1", 1, 1), data)
        self.assertEqual(data[("key1", 1, 1)], torch.tensor([2]))

        self.assertIn(("key1", 1, 2), data)
        self.assertEqual(data[("key1", 1, 2)], [44, 55])
        self.assertNotIn(("key1", 1, 2, 0), data)

        self.assertIn(("key1", 2), data)
        self.assertEqual(data[("key1", 2)], [66, 77])

    def test_traverse_nested_dict(self) -> None:
        state_dict = {
            "key0": {"key1": 99, "key2": torch.tensor([1])},
        }

        data = {}

        def collect_data(path, value):
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertNotIn(("key0",), data)

        self.assertIn(("key0", "key1"), data)
        self.assertEqual(data[("key0", "key1")], 99)

        self.assertIn(("key0", "key2"), data)
        self.assertEqual(data[("key0", "key2")], torch.tensor([1]))

    def test_traverse_doesnt_ignore_intermediate_collections(self) -> None:
        state_dict: STATE_DICT_TYPE = {"key0": [{"key1": {"key2": torch.tensor([1])}}]}

        data = {}

        def collect_data(path, value):
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertIn(("key0", 0, "key1", "key2"), data)
        self.assertEqual(
            data[("key0", 0, "key1", "key2")],
            torch.tensor([1]),
        )

    def test_traverse_with_ordered_dict(self) -> None:
        state_dict = OrderedDict(
            {
                "key0": [
                    99,
                    torch.tensor([3]),
                ]
            }
        )

        data = {}

        def collect_data(path, value):
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertIn(("key0", 0), data)
        self.assertEqual(data[("key0", 0)], 99)

        self.assertIn(("key0", 1), data)
        self.assertEqual(data[("key0", 1)], torch.tensor([3]))

    def test_set_element(self) -> None:
        state_dict: STATE_DICT_TYPE = {}

        _traverse.set_element(state_dict, ("k",), 10)
        self.assertEqual(state_dict["k"], 10)

        _traverse.set_element(state_dict, ("k1", 2), 1)
        self.assertEqual(state_dict["k1"], [None, None, 1])

        _traverse.set_element(state_dict, ("k1", 1), 99)
        self.assertEqual(state_dict["k1"], [None, 99, 1])

        _traverse.set_element(state_dict, ("k1", 3), 88)
        self.assertEqual(state_dict["k1"], [None, 99, 1, 88])

        _traverse.set_element(state_dict, ("k2", "k3"), 3)
        self.assertEqual(state_dict["k2"], {"k3": 3})

        _traverse.set_element(state_dict, ("k2", "k4", 0, 0), 99)
        self.assertEqual(state_dict["k2"]["k4"][0], [99])

    def test_get_element(self) -> None:
        state_dict = {"a": [0, 1], "b": [2, {"c": "d"}]}
        self.assertEqual(_traverse.get_element(state_dict, ("a",)), [0, 1])
        self.assertEqual(_traverse.get_element(state_dict, ("b", 0)), 2)
        self.assertEqual(_traverse.get_element(state_dict, ("b", 1, "c")), "d")

        self.assertIsNone(_traverse.get_element(state_dict, ("c",)))
        self.assertIsNone(_traverse.get_element(state_dict, ("a", 33)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 88)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 0, 2)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 1, 2)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 1, "d")))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""    Test class for util methods of _traverse

This Python file contains 2 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTraverse`

**Functions defined**: `test_traverse_shallow`, `collect_data`, `test_traverse_nested_list`, `collect_data`, `test_traverse_nested_dict`, `collect_data`, `test_traverse_doesnt_ignore_intermediate_collections`, `collect_data`, `test_traverse_with_ordered_dict`, `collect_data`, `test_set_element`, `test_get_element`

**Key imports**: OrderedDict, TYPE_CHECKING, torch, torch.distributed.checkpoint._traverse as _traverse, run_tests, TestCase, STATE_DICT_TYPE


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: OrderedDict
- `typing`: TYPE_CHECKING
- `torch`
- `torch.distributed.checkpoint._traverse as _traverse`
- `torch.testing._internal.common_utils`: run_tests, TestCase
- `torch.distributed.checkpoint.metadata`: STATE_DICT_TYPE


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/checkpoint/test_traverse.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/checkpoint`):

- [`test_format_utils.py_docs.md`](./test_format_utils.py_docs.md)
- [`test_save_load_api.py_docs.md`](./test_save_load_api.py_docs.md)
- [`test_pg_transport.py_docs.md`](./test_pg_transport.py_docs.md)
- [`test_async_process_executor.py_docs.md`](./test_async_process_executor.py_docs.md)
- [`test_file_system_checkpoint.py_docs.md`](./test_file_system_checkpoint.py_docs.md)
- [`test_nested_dict.py_docs.md`](./test_nested_dict.py_docs.md)
- [`test_hf_storage.py_docs.md`](./test_hf_storage.py_docs.md)
- [`test_hf_safetensor_e2e.py_docs.md`](./test_hf_safetensor_e2e.py_docs.md)
- [`test_fsdp_optim_state.py_docs.md`](./test_fsdp_optim_state.py_docs.md)
- [`test_state_dict_stager.py_docs.md`](./test_state_dict_stager.py_docs.md)


## Cross-References

- **File Documentation**: `test_traverse.py_docs.md`
- **Keyword Index**: `test_traverse.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/checkpoint`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/checkpoint/test_traverse.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/checkpoint`):

- [`test_state_dict.py_docs.md_docs.md`](./test_state_dict.py_docs.md_docs.md)
- [`test_checkpoint.py_kw.md_docs.md`](./test_checkpoint.py_kw.md_docs.md)
- [`test_dtensor_checkpoint.py_docs.md_docs.md`](./test_dtensor_checkpoint.py_docs.md_docs.md)
- [`test_file_system_checkpoint_cpu.py_docs.md_docs.md`](./test_file_system_checkpoint_cpu.py_docs.md_docs.md)
- [`test_dedup_tensors.py_docs.md_docs.md`](./test_dedup_tensors.py_docs.md_docs.md)
- [`test_fsspec.py_kw.md_docs.md`](./test_fsspec.py_kw.md_docs.md)
- [`test_quantized_hf_storage.py_kw.md_docs.md`](./test_quantized_hf_storage.py_kw.md_docs.md)
- [`test_pg_transport.py_kw.md_docs.md`](./test_pg_transport.py_kw.md_docs.md)
- [`test_dedup_tensors.py_kw.md_docs.md`](./test_dedup_tensors.py_kw.md_docs.md)
- [`test_hsdp_checkpoint.py_docs.md_docs.md`](./test_hsdp_checkpoint.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_traverse.py_docs.md_docs.md`
- **Keyword Index**: `test_traverse.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
