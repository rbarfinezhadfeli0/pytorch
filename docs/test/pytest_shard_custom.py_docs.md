# Documentation: `test/pytest_shard_custom.py`

## File Metadata

- **Path**: `test/pytest_shard_custom.py`
- **Size**: 2,307 bytes (2.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
"""
Custom pytest shard plugin
https://github.com/AdamGleave/pytest-shard/blob/64610a08dac6b0511b6d51cf895d0e1040d162ad/pytest_shard/pytest_shard.py#L1
Modifications:
* shards are now 1 indexed instead of 0 indexed
* option for printing items in shard
"""

import hashlib

from _pytest.config.argparsing import Parser


def pytest_addoptions(parser: Parser):
    """Add options to control sharding."""
    group = parser.getgroup("shard")
    group.addoption(
        "--shard-id", dest="shard_id", type=int, default=1, help="Number of this shard."
    )
    group.addoption(
        "--num-shards",
        dest="num_shards",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    group.addoption(
        "--print-items",
        dest="print_items",
        action="store_true",
        default=False,
        help="Print out the items being tested in this shard.",
    )


class PytestShardPlugin:
    def __init__(self, config):
        self.config = config

    def pytest_report_collectionfinish(self, config, items) -> str:
        """Log how many and which items are tested in this shard."""
        msg = f"Running {len(items)} items in this shard"
        if config.getoption("print_items"):
            msg += ": " + ", ".join([item.nodeid for item in items])
        return msg

    def sha256hash(self, x: str) -> int:
        return int.from_bytes(hashlib.sha256(x.encode()).digest(), "little")

    def filter_items_by_shard(self, items, shard_id: int, num_shards: int):
        """Computes `items` that should be tested in `shard_id` out of `num_shards` total shards."""
        new_items = [
            item
            for item in items
            if self.sha256hash(item.nodeid) % num_shards == shard_id - 1
        ]
        return new_items

    def pytest_collection_modifyitems(self, config, items):
        """Mutate the collection to consist of just items to be tested in this shard."""
        shard_id = config.getoption("shard_id")
        shard_total = config.getoption("num_shards")
        if shard_id < 1 or shard_id > shard_total:
            raise ValueError(
                f"{shard_id} is not a valid shard ID out of {shard_total} total shards"
            )

        items[:] = self.filter_items_by_shard(items, shard_id, shard_total)

```



## High-Level Overview

"""Custom pytest shard pluginhttps://github.com/AdamGleave/pytest-shard/blob/64610a08dac6b0511b6d51cf895d0e1040d162ad/pytest_shard/pytest_shard.py#L1Modifications:* shards are now 1 indexed instead of 0 indexed* option for printing items in shard

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PytestShardPlugin`

**Functions defined**: `pytest_addoptions`, `__init__`, `pytest_report_collectionfinish`, `sha256hash`, `filter_items_by_shard`, `pytest_collection_modifyitems`

**Key imports**: hashlib, Parser


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `hashlib`
- `_pytest.config.argparsing`: Parser


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python test/pytest_shard_custom.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `pytest_shard_custom.py_docs.md`
- **Keyword Index**: `pytest_shard_custom.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
