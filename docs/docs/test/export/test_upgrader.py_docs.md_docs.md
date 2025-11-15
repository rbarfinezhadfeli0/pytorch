# Documentation: `docs/test/export/test_upgrader.py_docs.md`

## File Metadata

- **Path**: `docs/test/export/test_upgrader.py_docs.md`
- **Size**: 15,462 bytes (15.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/export/test_upgrader.py`

## File Metadata

- **Path**: `test/export/test_upgrader.py`
- **Size**: 11,344 bytes (11.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: export"]

import json

import torch
from torch.testing._internal.common_utils import TestCase


class TestUpgrader(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Register example upgraders dynamically
        torch._C._export.register_example_upgraders()

    def tearDown(self) -> None:
        # Clean up registered upgraders
        torch._C._export.deregister_example_upgraders()

    def test_nn_module_stack_transformation_from_v0(self):
        """Test that nn_module_stack strings are prepended with 'test_upgrader_' when upgrading from version 0"""

        # Create a mock JSON object that simulates version 0 schema
        # with nn_module_stack as a string that needs to be upgraded
        mock_json = {
            "schema_version": {"major": 0, "minor": 0},
            "graph_module": {
                "graph": {
                    "nodes": [
                        {
                            "target": "aten.add.Tensor",
                            "inputs": [],
                            "outputs": [],
                            "metadata": {
                                "nn_module_stack": "original_stack_info",
                                "other_field": "some_value",
                            },
                        },
                        {
                            "target": "aten.mul.Tensor",
                            "inputs": [],
                            "outputs": [],
                            "metadata": {
                                "nn_module_stack": "another_stack",
                                "stack_trace": "some trace",
                            },
                        },
                    ]
                }
            },
        }

        # Test the upgrader using the Python binding
        serialized_json = json.dumps(mock_json)
        upgraded_json_str = torch._C._export.upgrade(serialized_json, 2)
        upgraded_json = json.loads(upgraded_json_str)

        # Verify the schema version was updated (version 0 -> version 2 due to both v0 and v1 upgraders)
        self.assertEqual(upgraded_json["schema_version"]["major"], 2)
        self.assertEqual(upgraded_json["schema_version"]["minor"], 0)

        # Verify nn_module_stack was prepended with "test_upgrader_"
        nodes = upgraded_json["graph_module"]["graph"]["nodes"]

        # Check first node
        first_node_metadata = nodes[0]["metadata"]
        nn_stack = first_node_metadata["nn_module_stack"]
        self.assertIsInstance(nn_stack, str)
        self.assertEqual(nn_stack, "test_upgrader_original_stack_info")
        # Other metadata should be unchanged
        self.assertEqual(first_node_metadata["other_field"], "some_value")

        # Check second node
        second_node_metadata = nodes[1]["metadata"]
        nn_stack2 = second_node_metadata["nn_module_stack"]
        self.assertIsInstance(nn_stack2, str)
        self.assertEqual(nn_stack2, "test_upgrader_another_stack")
        # Other metadata should be unchanged
        self.assertEqual(second_node_metadata["stack_trace"], "some trace")

    def test_nn_module_stack_error_handling_invalid_type(self):
        """Test error handling when nn_module_stack is not a string"""

        # Test case: nn_module_stack is not a string
        mock_json_invalid_type = {
            "schema_version": {"major": 0, "minor": 0},
            "graph_module": {
                "graph": {
                    "nodes": [
                        {
                            "target": "aten.add.Tensor",
                            "inputs": [],
                            "outputs": [],
                            "metadata": {
                                "nn_module_stack": 42  # Invalid: should be string
                            },
                        }
                    ]
                }
            },
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "Error in upgrader 'version_0_upgrader_registered'",
        ):
            serialized_json = json.dumps(mock_json_invalid_type)
            torch._C._export.upgrade(serialized_json, 2)

    def test_nodes_without_metadata_handled_gracefully(self):
        """Test that nodes without metadata or nn_module_stack are handled gracefully"""

        mock_json = {
            "schema_version": {"major": 0, "minor": 0},
            "graph_module": {
                "graph": {
                    "nodes": [
                        {
                            "target": "aten.add.Tensor",
                            "inputs": [],
                            "outputs": [],
                            # No metadata field
                        },
                        {
                            "target": "aten.mul.Tensor",
                            "inputs": [],
                            "outputs": [],
                            "metadata": {
                                "stack_trace": "some trace"
                                # No nn_module_stack field
                            },
                        },
                    ]
                }
            },
        }

        # Should not raise an error
        serialized_json = json.dumps(mock_json)
        upgraded_json_str = torch._C._export.upgrade(serialized_json, 2)
        upgraded_json = json.loads(upgraded_json_str)

        # Verify the schema version was updated (version 0 -> version 2 due to both v0 and v1 upgraders)
        self.assertEqual(upgraded_json["schema_version"]["major"], 2)
        self.assertEqual(upgraded_json["schema_version"]["minor"], 0)

        # Verify nodes are unchanged
        nodes = upgraded_json["graph_module"]["graph"]["nodes"]
        self.assertEqual(len(nodes), 2)

        # First node should have no metadata
        self.assertNotIn("metadata", nodes[0])

        # Second node should have unchanged metadata
        self.assertEqual(nodes[1]["metadata"]["stack_trace"], "some trace")
        self.assertNotIn("nn_module_stack", nodes[1]["metadata"])

    def test_field_renaming_chain_from_v0_complete(self):
        """Test complete field renaming chain from v0: old_test_field -> new_test_field -> new_test_field2"""

        mock_json = {
            "schema_version": {"major": 0, "minor": 0},
            "graph_module": {
                "graph": {
                    "inputs": [],
                    "outputs": [],
                    "nodes": [
                        {
                            "target": "aten.add.Tensor",
                            "inputs": [],
                            "outputs": [],
                            "metadata": {"nn_module_stack": "test_stack"},
                        }
                    ],
                    "old_test_field": "original_value",
                    "existing_field": "existing_value",
                }
            },
        }

        # Test the upgrader using the Python binding
        serialized_json = json.dumps(mock_json)
        upgraded_json_str = torch._C._export.upgrade(serialized_json, 2)
        upgraded_json = json.loads(upgraded_json_str)

        # Verify the schema version was updated (version 0 -> version 2 due to both v0 and v1 upgraders)
        self.assertEqual(upgraded_json["schema_version"]["major"], 2)
        self.assertEqual(upgraded_json["schema_version"]["minor"], 0)

        # Verify complete field transformation: old_test_field -> new_test_field -> new_test_field2
        graph = upgraded_json["graph_module"]["graph"]
        self.assertIn("new_test_field2", graph)
        self.assertEqual(graph["new_test_field2"], "original_value")
        self.assertNotIn("old_test_field", graph)
        self.assertNotIn("new_test_field", graph)

        # Verify existing fields are preserved
        self.assertEqual(graph["existing_field"], "existing_value")
        self.assertIn("inputs", graph)
        self.assertIn("outputs", graph)
        self.assertIn("nodes", graph)

        # Verify the nn_module_stack was also upgraded by the other upgrader
        nodes = graph["nodes"]
        self.assertEqual(
            nodes[0]["metadata"]["nn_module_stack"], "test_upgrader_test_stack"
        )

    def test_field_renaming_chain_from_v0_missing_field(self):
        """Test that upgraders work gracefully when old_test_field doesn't exist"""

        mock_json = {
            "schema_version": {"major": 0, "minor": 0},
            "graph_module": {
                "graph": {
                    "inputs": [],
                    "outputs": [],
                    "nodes": [],
                    "existing_field": "existing_value",
                }
            },
        }

        # Test the upgrader using the Python binding
        serialized_json = json.dumps(mock_json)
        upgraded_json_str = torch._C._export.upgrade(serialized_json, 2)
        upgraded_json = json.loads(upgraded_json_str)

        # Verify the schema version was updated (version 0 -> version 2 due to both v0 and v1 upgraders)
        self.assertEqual(upgraded_json["schema_version"]["major"], 2)
        self.assertEqual(upgraded_json["schema_version"]["minor"], 0)

        # Verify no field transformations occurred since old_test_field didn't exist
        graph = upgraded_json["graph_module"]["graph"]
        self.assertNotIn("new_test_field2", graph)
        self.assertNotIn("new_test_field", graph)
        self.assertNotIn("old_test_field", graph)

        # Verify existing fields are preserved
        self.assertEqual(graph["existing_field"], "existing_value")
        self.assertIn("inputs", graph)
        self.assertIn("outputs", graph)
        self.assertIn("nodes", graph)

    def test_field_renaming_from_v1_partial_chain(self):
        """Test partial upgrade chain starting from v1: new_test_field -> new_test_field2"""

        mock_json = {
            "schema_version": {"major": 1, "minor": 0},
            "graph_module": {
                "graph": {
                    "inputs": [],
                    "outputs": [],
                    "nodes": [],
                    "new_test_field": "test_value",
                    "existing_field": "existing_value",
                }
            },
        }

        # Test the upgrader using the Python binding
        serialized_json = json.dumps(mock_json)
        upgraded_json_str = torch._C._export.upgrade(serialized_json, 2)
        upgraded_json = json.loads(upgraded_json_str)

        # Verify the schema version was updated (version 1 -> version 2 due to v1 upgrader only)
        self.assertEqual(upgraded_json["schema_version"]["major"], 2)
        self.assertEqual(upgraded_json["schema_version"]["minor"], 0)

        # Verify new_test_field was renamed to new_test_field2
        graph = upgraded_json["graph_module"]["graph"]
        self.assertIn("new_test_field2", graph)
        self.assertEqual(graph["new_test_field2"], "test_value")
        self.assertNotIn("new_test_field", graph)

        # Verify existing fields are preserved
        self.assertEqual(graph["existing_field"], "existing_value")
        self.assertIn("inputs", graph)
        self.assertIn("outputs", graph)
        self.assertIn("nodes", graph)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview

"""Test that nn_module_stack strings are prepended with 'test_upgrader_' when upgrading from version 0"""        # Create a mock JSON object that simulates version 0 schema        # with nn_module_stack as a string that needs to be upgraded        mock_json = {            "schema_version": {"major": 0, "minor": 0},            "graph_module": {                "graph": {                    "nodes": [                        {                            "target": "aten.add.Tensor",                            "inputs": [],                            "outputs": [],                            "metadata": {                                "nn_module_stack": "original_stack_info",                                "other_field": "some_value",                            },                        },                        {                            "target": "aten.mul.Tensor",                            "inputs": [],                            "outputs": [],                            "metadata": {                                "nn_module_stack": "another_stack",                                "stack_trace": "some trace",                            },                        },                    ]                }            },        }

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestUpgrader`

**Functions defined**: `setUp`, `tearDown`, `test_nn_module_stack_transformation_from_v0`, `test_nn_module_stack_error_handling_invalid_type`, `test_nodes_without_metadata_handled_gracefully`, `test_field_renaming_chain_from_v0_complete`, `test_field_renaming_chain_from_v0_missing_field`, `test_field_renaming_from_v1_partial_chain`

**Key imports**: json, torch, TestCase, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `torch`
- `torch.testing._internal.common_utils`: TestCase
- `torch._dynamo.test_case`: run_tests


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
python test/export/test_upgrader.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_schema.py_docs.md`](./test_schema.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)
- [`test_export.py_docs.md`](./test_export.py_docs.md)


## Cross-References

- **File Documentation**: `test_upgrader.py_docs.md`
- **Keyword Index**: `test_upgrader.py_kw.md`
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
python docs/test/export/test_upgrader.py_docs.md
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

- **File Documentation**: `test_upgrader.py_docs.md_docs.md`
- **Keyword Index**: `test_upgrader.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
