# Documentation: `test/mobile/test_upgraders.py`

## File Metadata

- **Path**: `test/mobile/test_upgraders.py`
- **Size**: 2,613 bytes (2.55 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: mobile"]

import io
from itertools import product
from pathlib import Path

import torch
import torch.utils.bundled_inputs
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_utils import run_tests, TestCase


pytorch_test_dir = Path(__file__).resolve().parents[1]


class TestLiteScriptModule(TestCase):
    def _save_load_mobile_module(self, script_module: torch.jit.ScriptModule):
        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
        )
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)
        return mobile_module

    def _try_fn(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return e

    def test_versioned_div_tensor(self):
        # noqa: F841
        def div_tensor_0_3(self, other):  # noqa: F841
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide(other)
            return self.divide(other, rounding_mode="trunc")

        model_path = (
            pytorch_test_dir
            / "cpp"
            / "jit"
            / "upgrader_models"
            / "test_versioned_div_tensor_v2.ptl"
        )
        _load_for_lite_interpreter(str(model_path))
        jit_module_v2 = torch.jit.load(str(model_path))
        self._save_load_mobile_module(jit_module_v2)
        vals = (2.0, 3.0, 2, 3)
        for val_a, val_b in product(vals, vals):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            def _helper(m, fn):
                m_results = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                if isinstance(m_results, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                else:
                    for result in m_results:
                        print("result: ", result)
                        print("fn_result: ", fn_result)
                        print(result == fn_result)
                        self.assertTrue(result.eq(fn_result))
                        # self.assertEqual(result, fn_result)

            # old operator should produce the same result as applying upgrader of torch.div op
            # _helper(mobile_module_v2, div_tensor_0_3)
            # latest operator should produce the same result as applying torch.div op
            # _helper(current_mobile_module, torch.div)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestLiteScriptModule`

**Functions defined**: `_save_load_mobile_module`, `_try_fn`, `test_versioned_div_tensor`, `div_tensor_0_3`, `_helper`

**Key imports**: io, product, Path, torch, torch.utils.bundled_inputs, _load_for_lite_interpreter, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/mobile`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `itertools`: product
- `pathlib`: Path
- `torch`
- `torch.utils.bundled_inputs`
- `torch.jit.mobile`: _load_for_lite_interpreter
- `torch.testing._internal.common_utils`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/mobile/test_upgraders.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/mobile`):

- [`test_upgrader_codegen.py_docs.md`](./test_upgrader_codegen.py_docs.md)
- [`test_quantize_fx_lite_script_module.py_docs.md`](./test_quantize_fx_lite_script_module.py_docs.md)
- [`test_upgrader_bytecode_table_example.cpp_docs.md`](./test_upgrader_bytecode_table_example.cpp_docs.md)
- [`test_lite_script_module.py_docs.md`](./test_lite_script_module.py_docs.md)
- [`test_lite_script_type.py_docs.md`](./test_lite_script_type.py_docs.md)
- [`test_bytecode.py_docs.md`](./test_bytecode.py_docs.md)


## Cross-References

- **File Documentation**: `test_upgraders.py_docs.md`
- **Keyword Index**: `test_upgraders.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
