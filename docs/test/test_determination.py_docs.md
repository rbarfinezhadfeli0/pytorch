# Documentation: `test/test_determination.py`

## File Metadata

- **Path**: `test/test_determination.py`
- **Size**: 4,328 bytes (4.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: ci"]

import os

import run_test

from torch.testing._internal.common_utils import run_tests, TestCase


class DummyOptions:
    verbose = False


class DeterminationTest(TestCase):
    # Test determination on a subset of tests
    TESTS = [
        "test_nn",
        "test_jit_profiling",
        "test_jit",
        "test_torch",
        "test_cpp_extensions_aot_ninja",
        "test_cpp_extensions_aot_no_ninja",
        "test_utils",
        "test_determination",
        "test_quantization",
    ]

    @classmethod
    def determined_tests(cls, changed_files):
        changed_files = [os.path.normpath(path) for path in changed_files]
        return [
            test
            for test in cls.TESTS
            if run_test.should_run_test(
                run_test.TARGET_DET_LIST, test, changed_files, DummyOptions()
            )
        ]

    def test_target_det_list_is_sorted(self):
        # We keep TARGET_DET_LIST sorted to minimize merge conflicts
        # but most importantly to allow us to comment on the absence
        # of a test. It would be very difficult to add a file right
        # next to a comment that says to keep it out of the list.
        self.assertListEqual(run_test.TARGET_DET_LIST, sorted(run_test.TARGET_DET_LIST))

    def test_config_change_only(self):
        """CI configs trigger all tests"""
        self.assertEqual(self.determined_tests([".ci/pytorch/test.sh"]), self.TESTS)

    def test_run_test(self):
        """run_test.py is imported by determination tests"""
        self.assertEqual(
            self.determined_tests(["test/run_test.py"]), ["test_determination"]
        )

    def test_non_code_change(self):
        """Non-code changes don't trigger any tests"""
        self.assertEqual(
            self.determined_tests(["CODEOWNERS", "README.md", "docs/doc.md"]), []
        )

    def test_cpp_file(self):
        """CPP files trigger all tests"""
        self.assertEqual(
            self.determined_tests(["aten/src/ATen/native/cpu/Activation.cpp"]),
            self.TESTS,
        )

    def test_test_file(self):
        """Test files trigger themselves and dependent tests"""
        self.assertEqual(
            self.determined_tests(["test/test_jit.py"]),
            ["test_jit_profiling", "test_jit"],
        )
        self.assertEqual(
            self.determined_tests(["test/jit/test_custom_operators.py"]),
            ["test_jit_profiling", "test_jit"],
        )
        self.assertEqual(
            self.determined_tests(
                ["test/quantization/eager/test_quantize_eager_ptq.py"]
            ),
            ["test_quantization"],
        )

    def test_test_internal_file(self):
        """testing/_internal files trigger dependent tests"""
        self.assertEqual(
            self.determined_tests(["torch/testing/_internal/common_quantization.py"]),
            [
                "test_jit_profiling",
                "test_jit",
                "test_quantization",
            ],
        )

    def test_torch_file(self):
        """Torch files trigger dependent tests"""
        self.assertEqual(
            # Many files are force-imported to all tests,
            # due to the layout of the project.
            self.determined_tests(["torch/onnx/utils.py"]),
            self.TESTS,
        )
        self.assertEqual(
            self.determined_tests(
                [
                    "torch/autograd/_functions/utils.py",
                    "torch/autograd/_functions/utils.pyi",
                ]
            ),
            ["test_utils"],
        )
        self.assertEqual(
            self.determined_tests(["torch/utils/cpp_extension.py"]),
            [
                "test_cpp_extensions_aot_ninja",
                "test_cpp_extensions_aot_no_ninja",
                "test_utils",
                "test_determination",
            ],
        )

    def test_new_folder(self):
        """New top-level Python folder triggers all tests"""
        self.assertEqual(self.determined_tests(["new_module/file.py"]), self.TESTS)

    def test_new_test_script(self):
        """New test script triggers nothing (since it's not in run_tests.py)"""
        self.assertEqual(self.determined_tests(["test/test_new_test_script.py"]), [])


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""CI configs trigger all tests"""        self.assertEqual(self.determined_tests([".ci/pytorch/test.sh"]), self.TESTS)    def test_run_test(self):

This Python file contains 2 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DummyOptions`, `DeterminationTest`

**Functions defined**: `determined_tests`, `test_target_det_list_is_sorted`, `test_config_change_only`, `test_run_test`, `test_non_code_change`, `test_cpp_file`, `test_test_file`, `test_test_internal_file`, `test_torch_file`, `test_new_folder`, `test_new_test_script`

**Key imports**: os, run_test, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `run_test`
- `torch.testing._internal.common_utils`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_determination.py
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

- **File Documentation**: `test_determination.py_docs.md`
- **Keyword Index**: `test_determination.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
