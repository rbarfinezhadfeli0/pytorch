# Documentation: `test/inductor/test_compile.py`

## File Metadata

- **Path**: `test/inductor/test_compile.py`
- **Size**: 7,622 bytes (7.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import os
import shlex
import subprocess
import sys
from unittest import mock

import torch
from torch import _dynamo as dynamo, _inductor as inductor
from torch._inductor.codecache import write
from torch._inductor.cpp_builder import CppBuilder, CppOptions
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import gen_gm_and_inputs
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import HAS_CPU


_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"


def safe_command_output(cmd, timeout=30):
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            shell=isinstance(cmd, str),
        ).strip()
    except subprocess.CalledProcessError as e:
        return f"run failed（error code {e.returncode}）: {e.output.strip()}"
    except subprocess.TimeoutExpired:
        return "runt timeout"


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(10, 10)
        self.b = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.a(x))
        x = torch.sigmoid(self.b(x))
        return x


class MyModule2(MyModule):
    def forward(self, x):  # takes a dict of list
        a, b = x["key"]
        return {"result": super().forward(a) + b}


class MyModule3(MyModule):
    def forward(self, x):
        return (super().forward(x),)


class TestStandaloneInductor(TestCase):
    """
    These test check that you can call TorchInductor directly without
    going through TorchDynamo.
    """

    def test_inductor_via_fx(self):
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_tensor_return(self):
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_dict_input(self):
        mod = MyModule2().eval()
        inp = {"key": [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_make_fx(self):
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(make_fx(mod)(inp), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_bare_module(self):
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        # no FX graph at all (mod must return list/tuple in this case)
        mod_opt = inductor.compile(mod, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export1(self):
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        gm, _ = dynamo.export(mod, inp, aten_graph=True)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export2(self):
        mod = MyModule2().eval()
        inp = {"key": [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        gm, _ = dynamo.export(mod, inp)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_op_with_multiple_outputs(self):
        x1 = torch.randn((2, 512, 128))
        x2 = [128]
        x3 = torch.randn(128)
        x4 = torch.randn((128,))
        x5 = 1e-6
        mod, inp = gen_gm_and_inputs(
            torch.ops.aten.native_layer_norm.default, (x1, x2, x3, x4, x5), {}
        )
        mod_opt = inductor.compile(mod, inp)
        self.assertEqual(mod(*inp), mod_opt(*inp))

    @mock.patch.dict(os.environ, {"TORCHINDUCTOR_DEBUG_COMPILE": "1"})
    def test_inductor_generate_debug_compile(self):
        cpp_code = """
        int main(){
            return 0;
        }
        """

        _, source_path = write(
            cpp_code,
            "cpp",
        )
        build_option = CppOptions()
        cpp_builder = CppBuilder(
            name="test_compile",
            sources=source_path,
            output_dir=os.path.dirname(source_path),
            BuildOption=build_option,
        )
        cpp_builder.build()
        binary_path = cpp_builder.get_target_file_path()

        """
        When we turn on generate debug compile.
        On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
        On Linux, it should create some debug sections in binary file.
        """

        def check_linux_debug_section(module_path: str):
            check_cmd = shlex.split(f"readelf -S {module_path}")
            output = safe_command_output(check_cmd)
            has_debug_sym = ".debug_info" in output
            self.assertEqual(has_debug_sym, True)

        def check_windows_pdb_exist(module_path: str):
            file_name_no_ext = os.path.splitext(module_path)[0]
            file_name_pdb = f"{file_name_no_ext}.pdb"
            has_pdb_file = os.path.exists(file_name_pdb)
            self.assertEqual(has_pdb_file, True)

        if _IS_WINDOWS:
            check_windows_pdb_exist(binary_path)
        elif _IS_MACOS:
            pass  # MacOS not sure that if it should be works.
        else:
            check_linux_debug_section(binary_path)

    @mock.patch.dict(os.environ, {"TORCHINDUCTOR_DEBUG_SYMBOL": "1"})
    def test_inductor_generate_debug_symbol(self):
        cpp_code = """
        int main(){
            return 0;
        }
        """

        _, source_path = write(
            cpp_code,
            "cpp",
        )
        build_option = CppOptions()
        cpp_builder = CppBuilder(
            name="test_symbol",
            sources=source_path,
            output_dir=os.path.dirname(source_path),
            BuildOption=build_option,
        )
        cpp_builder.build()
        binary_path = cpp_builder.get_target_file_path()

        """
        When we turn on generate debug symbol.
        On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
        On Linux, it should create some debug sections in binary file.
        """

        def check_linux_debug_section(module_path: str):
            check_cmd = shlex.split(f"readelf -S {module_path}")
            output = safe_command_output(check_cmd)
            has_debug_sym = ".debug_info" in output
            self.assertEqual(has_debug_sym, True)

        def check_windows_pdb_exist(module_path: str):
            file_name_no_ext = os.path.splitext(module_path)[0]
            file_name_pdb = f"{file_name_no_ext}.pdb"
            has_pdb_file = os.path.exists(file_name_pdb)
            self.assertEqual(has_pdb_file, True)

        if _IS_WINDOWS:
            check_windows_pdb_exist(binary_path)
        elif _IS_MACOS:
            pass  # MacOS not sure that if it should be works.
        else:
            check_linux_debug_section(binary_path)


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()

```



## High-Level Overview


This Python file contains 4 class(es) and 19 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyModule`, `MyModule2`, `MyModule3`, `TestStandaloneInductor`

**Functions defined**: `safe_command_output`, `__init__`, `forward`, `forward`, `forward`, `test_inductor_via_fx`, `test_inductor_via_fx_tensor_return`, `test_inductor_via_fx_dict_input`, `test_inductor_via_make_fx`, `test_inductor_via_bare_module`, `test_inductor_via_export1`, `test_inductor_via_export2`, `test_inductor_via_op_with_multiple_outputs`, `test_inductor_generate_debug_compile`, `check_linux_debug_section`, `check_windows_pdb_exist`, `test_inductor_generate_debug_symbol`, `check_linux_debug_section`, `check_windows_pdb_exist`

**Key imports**: os, shlex, subprocess, sys, mock, torch, _dynamo as dynamo, _inductor as inductor, write, CppBuilder, CppOptions, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `shlex`
- `subprocess`
- `sys`
- `unittest`: mock
- `torch`
- `torch._inductor.codecache`: write
- `torch._inductor.cpp_builder`: CppBuilder, CppOptions
- `torch._inductor.test_case`: run_tests, TestCase
- `torch._inductor.utils`: gen_gm_and_inputs
- `torch.fx`: symbolic_trace
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.testing._internal.inductor_utils`: HAS_CPU


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_compile.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_compile.py_docs.md`
- **Keyword Index**: `test_compile.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
