# Documentation: `docs/test/export/test_schema.py_docs.md`

## File Metadata

- **Path**: `docs/test/export/test_schema.py_docs.md`
- **Size**: 16,432 bytes (16.05 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/export/test_schema.py`

## File Metadata

- **Path**: `test/export/test_schema.py`
- **Size**: 13,554 bytes (13.24 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: export"]
from torch._export.serde.schema_check import (
    _Commit,
    _diff_schema,
    check,
    SchemaUpdateError,
    update_schema,
)
from torch.testing._internal.common_utils import IS_FBCODE, run_tests, TestCase


class TestSchema(TestCase):
    def test_schema_compatibility(self):
        msg = """
Detected an invalidated change to export schema. Please run the following script to update the schema:
Example(s):
    python scripts/export/update_schema.py --prefix <path_to_torch_development_directory>
        """

        if IS_FBCODE:
            msg += """or
    buck run caffe2:export_update_schema -- --prefix /data/users/$USER/fbsource/fbcode/caffe2/
            """
        try:
            commit = update_schema()
        except SchemaUpdateError as e:
            self.fail(f"Failed to update schema: {e}\n{msg}")

        self.assertEqual(commit.checksum_head, commit.checksum_next, msg)

    def test_thrift_schema_unchanged(self):
        msg = """
Detected an unexpected change to schema.thrift. Please update schema.py instead and run the following script:
Example(s):
    python scripts/export/update_schema.py --prefix <path_to_torch_development_directory>
        """

        if IS_FBCODE:
            msg += """or
    buck run caffe2:export_update_schema -- --prefix /data/users/$USER/fbsource/fbcode/caffe2/
            """

        try:
            commit = update_schema()
        except SchemaUpdateError as e:
            self.fail(f"Failed to update schema: {e}\n{msg}")

        self.assertEqual(commit.thrift_checksum_head, commit.thrift_checksum_real, msg)
        self.assertEqual(commit.thrift_checksum_head, commit.thrift_checksum_next, msg)

    def test_schema_diff(self):
        additions, subtractions = _diff_schema(
            {
                "Type0": {"kind": "struct", "fields": {}},
                "Type2": {
                    "kind": "struct",
                    "fields": {
                        "field0": {"type": ""},
                        "field2": {"type": ""},
                        "field3": {"type": "", "default": "[]"},
                    },
                },
            },
            {
                "Type2": {
                    "kind": "struct",
                    "fields": {
                        "field1": {"type": "", "default": "0"},
                        "field2": {"type": "", "default": "[]"},
                        "field3": {"type": ""},
                    },
                },
                "Type1": {"kind": "struct", "fields": {}},
            },
        )

        self.assertEqual(
            additions,
            {
                "Type1": {"kind": "struct", "fields": {}},
                "Type2": {
                    "fields": {
                        "field1": {"type": "", "default": "0"},
                        "field2": {"default": "[]"},
                    },
                },
            },
        )
        self.assertEqual(
            subtractions,
            {
                "Type0": {"kind": "struct", "fields": {}},
                "Type2": {
                    "fields": {
                        "field0": {"type": ""},
                        "field3": {"default": "[]"},
                    },
                },
            },
        )

    def test_schema_check(self):
        # Adding field without default value
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                    "field1": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_next="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_head="",
            cpp_header="",
            cpp_header_path="",
            thrift_checksum_head="",
            thrift_checksum_real="",
            thrift_checksum_next="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [4, 1])

        # Removing field
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {},
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_next="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_head="",
            cpp_header="",
            cpp_header_path="",
            thrift_checksum_head="",
            thrift_checksum_real="",
            thrift_checksum_next="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [4, 1])

        # Adding field with default value
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                    "field1": {"type": "", "default": "[]"},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_next="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_head="",
            cpp_header="",
            cpp_header_path="",
            thrift_checksum_head="",
            thrift_checksum_real="",
            thrift_checksum_next="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [3, 3])

        # Changing field type
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": "int"},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }

        with self.assertRaises(SchemaUpdateError):
            _diff_schema(dst, src)

        # Adding new type.
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "Type1": {"kind": "struct", "fields": {}},
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_next="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_head="",
            cpp_header="",
            cpp_header_path="",
            thrift_checksum_head="",
            thrift_checksum_real="",
            thrift_checksum_next="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [3, 3])

        # Removing a type.
        dst = {
            "Type2": {
                "kind": "struct",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_next="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_head="",
            cpp_header="",
            cpp_header_path="",
            thrift_checksum_head="",
            thrift_checksum_real="",
            thrift_checksum_next="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [3, 3])

        # Adding new field in union.
        dst = {
            "Type2": {
                "kind": "union",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "union",
                "fields": {
                    "field0": {"type": ""},
                    "field1": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_next="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_head="",
            cpp_header="",
            cpp_header_path="",
            thrift_checksum_head="",
            thrift_checksum_real="",
            thrift_checksum_next="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [3, 3])

        # Removing a field in union.
        dst = {
            "Type2": {
                "kind": "union",
                "fields": {
                    "field0": {"type": ""},
                },
            },
            "SCHEMA_VERSION": [3, 2],
        }
        src = {
            "Type2": {
                "kind": "union",
                "fields": {},
            },
            "SCHEMA_VERSION": [3, 2],
        }

        additions, subtractions = _diff_schema(dst, src)

        commit = _Commit(
            result=src,
            checksum_next="",
            yaml_path="",
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_head="",
            cpp_header="",
            cpp_header_path="",
            thrift_checksum_head="",
            thrift_checksum_real="",
            thrift_checksum_next="",
            thrift_schema="",
            thrift_schema_path="",
        )
        next_version, _ = check(commit)
        self.assertEqual(next_version, [4, 1])

    def test_schema_comparison(self):
        import torch._export.serde.schema as schema

        sig = schema.ModuleCallSignature(
            inputs=[
                schema.Argument.create(as_none=True),
                schema.Argument.create(
                    as_sym_int=schema.SymIntArgument.create(as_name="s0")
                ),
            ],
            outputs=[
                schema.Argument.create(
                    as_sym_int=schema.SymIntArgument.create(as_name="s1")
                )
            ],
            in_spec="foo",
            out_spec="bar",
            forward_arg_names=["None", "symint"],
        )
        # same content as sig
        sig_same = schema.ModuleCallSignature(
            inputs=[
                schema.Argument.create(as_none=True),
                schema.Argument.create(
                    as_sym_int=schema.SymIntArgument.create(as_name="s0")
                ),
            ],
            outputs=[
                schema.Argument.create(
                    as_sym_int=schema.SymIntArgument.create(as_name="s1")
                )
            ],
            in_spec="foo",
            out_spec="bar",
            forward_arg_names=["None", "symint"],
        )
        # as_name of symint is different
        sig_diff = schema.ModuleCallSignature(
            inputs=[
                schema.Argument.create(as_none=True),
                schema.Argument.create(
                    as_sym_int=schema.SymIntArgument.create(as_name="s0")
                ),
            ],
            outputs=[
                schema.Argument.create(
                    as_sym_int=schema.SymIntArgument.create(as_name="s2")
                )
            ],
            in_spec="foo",
            out_spec="bar",
            forward_arg_names=["None", "symint"],
        )
        self.assertEqual(sig, sig_same)
        self.assertNotEqual(sig, sig_diff)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

msg = """Detected an invalidated change to export schema. Please run the following script to update the schema:Example(s):    python scripts/export/update_schema.py --prefix <path_to_torch_development_directory>

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSchema`

**Functions defined**: `test_schema_compatibility`, `test_thrift_schema_unchanged`, `test_schema_diff`, `test_schema_check`, `test_schema_comparison`

**Key imports**: IS_FBCODE, run_tests, TestCase, torch._export.serde.schema as schema


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.testing._internal.common_utils`: IS_FBCODE, run_tests, TestCase
- `torch._export.serde.schema as schema`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/export/test_schema.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)
- [`test_export.py_docs.md`](./test_export.py_docs.md)


## Cross-References

- **File Documentation**: `test_schema.py_docs.md`
- **Keyword Index**: `test_schema.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python docs/test/export/test_schema.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/export`):

- [`test_serialize.py_docs.md_docs.md`](./test_serialize.py_docs.md_docs.md)
- [`test_verifier.py_kw.md_docs.md`](./test_verifier.py_kw.md_docs.md)
- [`test_upgrader.py_kw.md_docs.md`](./test_upgrader.py_kw.md_docs.md)
- [`test_db.py_docs.md_docs.md`](./test_db.py_docs.md_docs.md)
- [`test_export.py_docs.md_docs.md`](./test_export.py_docs.md_docs.md)
- [`test_dynamic_shapes.py_kw.md_docs.md`](./test_dynamic_shapes.py_kw.md_docs.md)
- [`test_passes.py_kw.md_docs.md`](./test_passes.py_kw.md_docs.md)
- [`test_unflatten.py_docs.md_docs.md`](./test_unflatten.py_docs.md_docs.md)
- [`test_pass_infra.py_kw.md_docs.md`](./test_pass_infra.py_kw.md_docs.md)
- [`test_functionalized_assertions.py_kw.md_docs.md`](./test_functionalized_assertions.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_schema.py_docs.md_docs.md`
- **Keyword Index**: `test_schema.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
