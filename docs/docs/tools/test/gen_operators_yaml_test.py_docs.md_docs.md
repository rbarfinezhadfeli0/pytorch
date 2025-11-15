# Documentation: `docs/tools/test/gen_operators_yaml_test.py_docs.md`

## File Metadata

- **Path**: `docs/tools/test/gen_operators_yaml_test.py_docs.md`
- **Size**: 10,959 bytes (10.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/test/gen_operators_yaml_test.py`

## File Metadata

- **Path**: `tools/test/gen_operators_yaml_test.py`
- **Size**: 8,021 bytes (7.83 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import unittest
from collections import defaultdict
from unittest.mock import Mock, patch

# pyrefly: ignore [import-error]
from gen_operators_yaml import (
    fill_output,
    get_parser_options,
    make_filter_from_options,
    verify_all_specified_present,
)


def _mock_options():
    options = argparse.Namespace()
    options.root_ops = "aten::add,aten::cat"
    options.training_root_ops = []
    options.output_path = "/tmp"
    options.dep_graph_yaml_path = "dummy_pytorch_op_deps.yaml"
    options.model_name = "test_model"
    options.model_versions = None
    options.model_assets = None
    options.model_backends = None
    options.models_yaml_path = None
    options.include_all_operators = False
    options.rule_name = "test_rule"
    options.not_include_all_overloads_static_root_ops = True
    options.not_include_all_overloads_closure_ops = True

    return options


def _mock_load_op_dep_graph():
    result = defaultdict(set)
    result["aten::add"] = {"aten::add", "aten::as_strided_"}
    result["aten::cat"] = {"aten::cat", "aten::as_strided_"}
    return dict(result)


class GenOperatorsYAMLTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_filter_creation(self) -> None:
        filter_func = make_filter_from_options(
            model_name="abc",
            model_versions=["100", "101"],
            model_assets=None,
            model_backends=None,
        )
        config = [
            {
                "model": {
                    "name": "abc",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 102,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
            {
                "model": {
                    "name": "abcd",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 101,
                    "asset": "asset-2",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
        ]

        filtered_configs = list(filter(filter_func, config))
        assert len(filtered_configs) == 2, (
            f"Expected 2 elements in filtered_configs, but got {len(filtered_configs)}"
        )

    def test_verification_success(self) -> None:
        filter_func = make_filter_from_options(
            model_name="abc",
            model_versions=["100", "101"],
            model_assets=["asset-1", "asset-2"],
            model_backends=None,
        )
        config = [
            {
                "model": {
                    "name": "abc",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 101,
                    "asset": "asset-2",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
        ]
        filtered_configs = list(filter(filter_func, config))
        try:
            verify_all_specified_present(
                model_assets=["asset-1", "asset-2"],
                model_versions=["100", "101"],
                selected_models_yaml=filtered_configs,
                rule_name="test",
                model_name="abc",
                new_style_rule=True,
            )
        except Exception:
            self.fail(
                "expected verify_all_specified_present to succeed instead it raised an exception"
            )

    def test_verification_fail(self) -> None:
        config = [
            {
                "model": {
                    "name": "abc",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 101,
                    "asset": "asset-2",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
        ]

        good_assets = ["asset-1", "asset-2"]
        good_versions = ["100", "101"]
        good_name = "abc"

        # Test bad asset
        filter_func_bad_asset = make_filter_from_options(
            model_name=good_name,
            model_versions=good_versions,
            model_assets=["asset-1", "asset-2", "asset-3"],
            model_backends=None,
        )
        filtered_configs_asset = list(filter(filter_func_bad_asset, config))
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=["asset-1", "asset-2", "asset-3"],
                model_versions=good_versions,
                selected_models_yaml=filtered_configs_asset,
                rule_name="test",
                model_name=good_name,
                new_style_rule=True,
            )

        # Test bad version
        filter_func_bad_version = make_filter_from_options(
            model_name=good_name,
            model_versions=["100", "101", "102"],
            model_assets=good_assets,
            model_backends=None,
        )
        filtered_configs_version = list(filter(filter_func_bad_version, config))
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=good_assets,
                model_versions=["100", "101", "102"],
                selected_models_yaml=filtered_configs_version,
                rule_name="test",
                model_name=good_name,
                new_style_rule=True,
            )

        # Test bad name
        filter_func_bad_name = make_filter_from_options(
            model_name="abcd",
            model_versions=good_versions,
            model_assets=good_assets,
            model_backends=None,
        )
        filtered_configs_name = list(filter(filter_func_bad_name, config))
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=good_assets,
                model_versions=good_versions,
                selected_models_yaml=filtered_configs_name,
                rule_name="test",
                model_name="abcd",
                new_style_rule=True,
            )

    @patch("gen_operators_yaml.parse_options", return_value=_mock_options())
    @patch(
        "gen_operators_yaml.load_op_dep_graph", return_value=_mock_load_op_dep_graph()
    )
    def test_fill_output_with_arguments_not_include_all_overloads(
        self, mock_parse_options: Mock, mock_load_op_dep_graph: Mock
    ) -> None:
        parser = argparse.ArgumentParser(description="Generate used operators YAML")
        options = get_parser_options(parser)

        model_dict = {
            "model_name": options.model_name,
            "asset_info": {},
            "is_new_style_rule": False,
        }
        output = {"debug_info": [json.dumps(model_dict)]}

        fill_output(output, options)

        # pyrefly: ignore [missing-attribute]
        for op_val in output["operators"].values():
            self.assertFalse(op_val["include_all_overloads"])

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GenOperatorsYAMLTest`

**Functions defined**: `_mock_options`, `_mock_load_op_dep_graph`, `setUp`, `test_filter_creation`, `test_verification_success`, `test_verification_fail`, `test_fill_output_with_arguments_not_include_all_overloads`

**Key imports**: argparse, json, unittest, defaultdict, Mock, patch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `json`
- `unittest`
- `collections`: defaultdict
- `unittest.mock`: Mock, patch


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
python tools/test/gen_operators_yaml_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/test`):

- [`test_upload_stats_lib.py_docs.md`](./test_upload_stats_lib.py_docs.md)
- [`test_codegen.py_docs.md`](./test_codegen.py_docs.md)
- [`linter_test_case.py_docs.md`](./linter_test_case.py_docs.md)
- [`test_upload_gate.py_docs.md`](./test_upload_gate.py_docs.md)
- [`test_gen_backend_stubs.py_docs.md`](./test_gen_backend_stubs.py_docs.md)
- [`test_gb_registry_linter.py_docs.md`](./test_gb_registry_linter.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_set_linter.py_docs.md`](./test_set_linter.py_docs.md)
- [`gen_oplist_test.py_docs.md`](./gen_oplist_test.py_docs.md)
- [`test_upload_test_stats.py_docs.md`](./test_upload_test_stats.py_docs.md)


## Cross-References

- **File Documentation**: `gen_operators_yaml_test.py_docs.md`
- **Keyword Index**: `gen_operators_yaml_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/test`, which is part of the **testing infrastructure**.



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
python docs/tools/test/gen_operators_yaml_test.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/test`):

- [`test_gen_backend_stubs.py_kw.md_docs.md`](./test_gen_backend_stubs.py_kw.md_docs.md)
- [`test_upload_stats_lib.py_kw.md_docs.md`](./test_upload_stats_lib.py_kw.md_docs.md)
- [`test_cmake.py_kw.md_docs.md`](./test_cmake.py_kw.md_docs.md)
- [`test_upload_test_stats.py_docs.md_docs.md`](./test_upload_test_stats.py_docs.md_docs.md)
- [`test_codegen_model.py_docs.md_docs.md`](./test_codegen_model.py_docs.md_docs.md)
- [`test_codegen.py_docs.md_docs.md`](./test_codegen.py_docs.md_docs.md)
- [`test_vulkan_codegen.py_kw.md_docs.md`](./test_vulkan_codegen.py_kw.md_docs.md)
- [`test_set_linter.py_docs.md_docs.md`](./test_set_linter.py_docs.md_docs.md)
- [`test_gb_registry_linter.py_kw.md_docs.md`](./test_gb_registry_linter.py_kw.md_docs.md)
- [`test_upload_test_stats.py_kw.md_docs.md`](./test_upload_test_stats.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `gen_operators_yaml_test.py_docs.md_docs.md`
- **Keyword Index**: `gen_operators_yaml_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
