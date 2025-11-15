# Documentation: `test/test_appending_byte_serializer.py`

## File Metadata

- **Path**: `test/test_appending_byte_serializer.py`
- **Size**: 2,710 bytes (2.65 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import dataclasses

from torch.testing._internal.common_utils import TestCase
from torch.utils._appending_byte_serializer import (
    AppendingByteSerializer,
    BytesReader,
    BytesWriter,
)


class TestAppendingByteSerializer(TestCase):
    def test_write_and_read_int(self) -> None:
        def int_serializer(writer: BytesWriter, i: int) -> None:
            writer.write_uint64(i)

        def int_deserializer(reader: BytesReader) -> int:
            return reader.read_uint64()

        s = AppendingByteSerializer(serialize_fn=int_serializer)

        data = [1, 2, 3, 4]
        s.extend(data)
        self.assertListEqual(
            data,
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=int_deserializer
            ),
        )

        data2 = [8, 9, 10, 11]
        s.extend(data2)
        self.assertListEqual(
            data + data2,
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=int_deserializer
            ),
        )

    def test_write_and_read_class(self) -> None:
        @dataclasses.dataclass(frozen=True, eq=True)
        class Foo:
            x: int
            y: str
            z: bytes

            @staticmethod
            def serialize(writer: BytesWriter, cls: "Foo") -> None:
                writer.write_uint64(cls.x)
                writer.write_str(cls.y)
                writer.write_bytes(cls.z)

            @staticmethod
            def deserialize(reader: BytesReader) -> "Foo":
                x = reader.read_uint64()
                y = reader.read_str()
                z = reader.read_bytes()
                return Foo(x, y, z)

        a = Foo(5, "ok", bytes([15]))
        b = Foo(10, "lol", bytes([25]))

        s = AppendingByteSerializer(serialize_fn=Foo.serialize)
        s.append(a)
        self.assertListEqual(
            [a],
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=Foo.deserialize
            ),
        )

        s.append(b)
        self.assertListEqual(
            [a, b],
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=Foo.deserialize
            ),
        )

    def test_checksum(self) -> None:
        writer = BytesWriter()
        writer.write_str("test")
        b = writer.to_bytes()
        b = bytearray(b)
        b[0:1] = b"\x00"
        b = bytes(b)

        with self.assertRaisesRegex(
            RuntimeError, r"Bytes object is corrupted, checksum does not match.*"
        ):
            BytesReader(b)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAppendingByteSerializer`, `Foo`

**Functions defined**: `test_write_and_read_int`, `int_serializer`, `int_deserializer`, `test_write_and_read_class`, `serialize`, `deserialize`, `test_checksum`

**Key imports**: dataclasses, TestCase, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `torch.testing._internal.common_utils`: TestCase
- `torch._inductor.test_case`: run_tests


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
python test/test_appending_byte_serializer.py
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

- **File Documentation**: `test_appending_byte_serializer.py_docs.md`
- **Keyword Index**: `test_appending_byte_serializer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
