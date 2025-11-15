# Documentation: `docs/test/cpp/aoti_inference/test.py_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/aoti_inference/test.py_docs.md`
- **Size**: 9,911 bytes (9.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/aoti_inference/test.py`

## File Metadata

- **Path**: `test/cpp/aoti_inference/test.py`
- **Size**: 7,028 bytes (6.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
import torch
import torch._inductor.config
from torch._export import aot_compile
from torch.export import Dim


torch.manual_seed(1337)


class Net(torch.nn.Module):
    def __init__(self, device, size=4):
        super().__init__()
        self.w_pre = torch.randn(size, size, device=device)
        self.w_add = torch.randn(size, size, device=device)

    def forward(self, x):
        w_transpose = torch.transpose(self.w_pre, 0, 1)
        w_relu = torch.nn.functional.relu(w_transpose)
        w = w_relu + self.w_add
        return torch.matmul(x, w)


class NetWithTensorConstants(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.randn(30, 1, device="cuda")

    def forward(self, x, y):
        z = self.w * x * y
        return z[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]]


data = {}
large_data = {}
cuda_alloc_data = {}
data_with_tensor_constants = {}


# Basice AOTI model test generation.
def generate_basic_tests():
    for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
        for use_runtime_constant_folding in [True, False]:
            if device == "cpu" and use_runtime_constant_folding:
                # We do not test runtime const folding for cpu mode.
                continue
            model = Net(device).to(device=device)
            x = torch.randn((4, 4), device=device)
            with torch.no_grad():
                ref_output = model(x)

            torch._dynamo.reset()
            with torch.no_grad():
                dim0_x = Dim("dim0_x", min=1, max=1024)
                dynamic_shapes = {"x": {0: dim0_x}}
                model_so_path = aot_compile(
                    model,
                    (x,),
                    dynamic_shapes=dynamic_shapes,
                    options={
                        "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                    },
                )
                # Also store a .pt2 file using the aoti_compile_and_package API
                pt2_package_path = torch._inductor.aoti_compile_and_package(
                    torch.export.export(
                        model,
                        (x,),
                        dynamic_shapes=dynamic_shapes,
                    ),
                    inductor_configs={
                        "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                    },
                )

            suffix = f"{device}"
            if use_runtime_constant_folding:
                suffix += "_use_runtime_constant_folding"
            data.update(
                {
                    f"model_so_path_{suffix}": model_so_path,
                    f"pt2_package_path_{suffix}": pt2_package_path,
                    f"inputs_{suffix}": [x],
                    f"outputs_{suffix}": [ref_output],
                    f"w_pre_{suffix}": model.w_pre,
                    f"w_add_{suffix}": model.w_add,
                }
            )


def generate_basic_tests_consts_cpp():
    backup_consts_asm_cfg: bool = (
        torch._inductor.config.aot_inductor.use_consts_asm_build
    )
    torch._inductor.config.aot_inductor.use_consts_asm_build = False

    # Test consts cpp build again.
    generate_basic_tests()

    torch._inductor.config.aot_inductor.use_consts_asm_build = backup_consts_asm_cfg


def generate_large_tests():
    device = "cuda"
    model = Net(device, size=4096).to(device=device)
    x = torch.randn((4096, 4096), device=device)
    with torch.no_grad():
        ref_output = model(x)

    torch._dynamo.reset()
    for use_runtime_constant_folding in [True, False]:
        with torch.no_grad():
            model_so_path = aot_compile(
                model,
                (x,),
                options={
                    "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                },
            )
            # Also store a .pt2 file using the aoti_compile_and_package API
            pt2_package_path = torch._inductor.aoti_compile_and_package(
                torch.export.export(
                    model,
                    (x,),
                ),
                inductor_configs={
                    "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                },
            )

        suffix = "_use_runtime_constant_folding" if use_runtime_constant_folding else ""
        large_data.update(
            {  # noqa: F541
                f"model_so_path{suffix}": model_so_path,
                f"pt2_package_path{suffix}": pt2_package_path,
                "inputs": [x],
                "outputs": [ref_output],
                "w_pre": model.w_pre,
                "w_add": model.w_add,
            }
        )


def generate_cuda_alloc_test():
    device = "cuda"
    model = Net(device, size=4096).to(device=device)
    x = torch.randn((4096, 4096), device=device)
    with torch.no_grad():
        ref_output = model(x)

    torch._dynamo.reset()
    with torch.no_grad():
        model_so_path = aot_compile(
            model,
            (x,),
            options={"aot_inductor.weight_use_caching_allocator": True},
        )

    cuda_alloc_data.update(
        {  # noqa: F541
            "model_so_path": model_so_path,
            "inputs": [x],
            "outputs": [ref_output],
            "w_pre": model.w_pre,
            "w_add": model.w_add,
        }
    )


# AOTI model which will create additional tensors during autograd.
def generate_test_with_additional_tensors():
    if not torch.cuda.is_available():
        return

    model = NetWithTensorConstants()
    x = torch.randn((30, 1), device="cuda")
    y = torch.randn((30, 1), device="cuda")
    with torch.no_grad():
        ref_output = model(x, y)

    torch._dynamo.reset()
    with torch.no_grad():
        model_so_path = aot_compile(model, (x, y))
        # Also store a .pt2 file using the aoti_compile_and_package API
        pt2_package_path = torch._inductor.aoti_compile_and_package(
            torch.export.export(model, (x, y))
        )

    data_with_tensor_constants.update(
        {
            "model_so_path": model_so_path,
            "pt2_package_path": pt2_package_path,
            "inputs": [x, y],
            "outputs": [ref_output],
            "w": model.w,
        }
    )


generate_basic_tests()
generate_basic_tests_consts_cpp()
generate_large_tests()
generate_test_with_additional_tensors()
generate_cuda_alloc_test()


# Use this to communicate tensors to the cpp code
class Serializer(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        for key in data:
            setattr(self, key, data[key])


torch.jit.script(Serializer(data)).save("data.pt")
torch.jit.script(Serializer(large_data)).save("large_data.pt")
torch.jit.script(Serializer(data_with_tensor_constants)).save(
    "data_with_tensor_constants.pt"
)
torch.jit.script(Serializer(cuda_alloc_data)).save("cuda_alloc_data.pt")

```



## High-Level Overview


This Python file contains 3 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Net`, `NetWithTensorConstants`, `Serializer`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `generate_basic_tests`, `generate_basic_tests_consts_cpp`, `generate_large_tests`, `generate_cuda_alloc_test`, `generate_test_with_additional_tensors`, `__init__`

**Key imports**: torch, torch._inductor.config, aot_compile, Dim


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/aoti_inference`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor.config`
- `torch._export`: aot_compile
- `torch.export`: Dim


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python test/cpp/aoti_inference/test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/aoti_inference`):

- [`aoti_custom_class.h_docs.md`](./aoti_custom_class.h_docs.md)
- [`standalone_test.cpp_docs.md`](./standalone_test.cpp_docs.md)
- [`generate_lowered_cpu.py_docs.md`](./generate_lowered_cpu.py_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test.cpp_docs.md`](./test.cpp_docs.md)
- [`compile_model.py_docs.md`](./compile_model.py_docs.md)
- [`aoti_custom_class.cpp_docs.md`](./aoti_custom_class.cpp_docs.md)
- [`standalone_compile.sh_docs.md`](./standalone_compile.sh_docs.md)


## Cross-References

- **File Documentation**: `test.py_docs.md`
- **Keyword Index**: `test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/aoti_inference`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/aoti_inference`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/cpp/aoti_inference/test.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/aoti_inference`):

- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`standalone_test.cpp_docs.md_docs.md`](./standalone_test.cpp_docs.md_docs.md)
- [`compile_model.py_kw.md_docs.md`](./compile_model.py_kw.md_docs.md)
- [`standalone_test.cpp_kw.md_docs.md`](./standalone_test.cpp_kw.md_docs.md)
- [`standalone_compile.sh_kw.md_docs.md`](./standalone_compile.sh_kw.md_docs.md)
- [`aoti_custom_class.cpp_kw.md_docs.md`](./aoti_custom_class.cpp_kw.md_docs.md)
- [`aoti_custom_class.h_kw.md_docs.md`](./aoti_custom_class.h_kw.md_docs.md)
- [`aoti_custom_class.cpp_docs.md_docs.md`](./aoti_custom_class.cpp_docs.md_docs.md)
- [`test.cpp_kw.md_docs.md`](./test.cpp_kw.md_docs.md)
- [`generate_lowered_cpu.py_kw.md_docs.md`](./generate_lowered_cpu.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test.py_docs.md_docs.md`
- **Keyword Index**: `test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
