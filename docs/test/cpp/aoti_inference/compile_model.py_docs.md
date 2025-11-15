# Documentation: `test/cpp/aoti_inference/compile_model.py`

## File Metadata

- **Path**: `test/cpp/aoti_inference/compile_model.py`
- **Size**: 2,378 bytes (2.32 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Can be **executed as a standalone script**.

## Original Source

```python
import torch
from torch.export import Dim


# custom op that loads the aot-compiled model
AOTI_CUSTOM_OP_LIB = "libaoti_custom_class.so"
torch.classes.load_library(AOTI_CUSTOM_OP_LIB)


class TensorSerializer(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        for key in data:
            setattr(self, key, data[key])


class SimpleModule(torch.nn.Module):
    """
    a simple module to be compiled
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 6)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        a = self.fc(x)
        b = self.relu(a)
        return b


class MyAOTIModule(torch.nn.Module):
    """
    a wrapper nn.Module that instantiates its forward method
    on MyAOTIClass
    """

    def __init__(self, lib_path, device):
        super().__init__()
        self.aoti_custom_op = torch.classes.aoti.MyAOTIClass(
            lib_path,
            device,
        )

    def forward(self, *x):
        outputs = self.aoti_custom_op.forward(x)
        return tuple(outputs)


def make_script_module(lib_path, device, *inputs):
    m = MyAOTIModule(lib_path, device)
    # sanity check
    m(*inputs)
    return torch.jit.trace(m, inputs)


def compile_model(device, data):
    module = SimpleModule().to(device)
    x = torch.randn((4, 4), device=device)
    inputs = (x,)
    # make batch dimension
    batch_dim = Dim("batch", min=1, max=1024)
    dynamic_shapes = {
        "x": {0: batch_dim},
    }
    with torch.no_grad():
        # aot-compile the module into a .so pointed by lib_path
        lib_path = torch._export.aot_compile(
            module, inputs, dynamic_shapes=dynamic_shapes
        )
    script_module = make_script_module(lib_path, device, *inputs)
    aoti_script_model = f"script_model_{device}.pt"
    script_module.save(aoti_script_model)

    # save sample inputs and ref output
    with torch.no_grad():
        ref_output = module(*inputs)
    data.update(
        {
            f"inputs_{device}": list(inputs),
            f"outputs_{device}": [ref_output],
        }
    )


def main():
    data = {}
    for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
        compile_model(device, data)
    torch.jit.script(TensorSerializer(data)).save("script_data.pt")


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""    a simple module to be compiled

This Python file contains 3 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TensorSerializer`, `SimpleModule`, `MyAOTIModule`

**Functions defined**: `__init__`, `__init__`, `forward`, `__init__`, `forward`, `make_script_module`, `compile_model`, `main`

**Key imports**: torch, Dim


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/aoti_inference`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
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
python test/cpp/aoti_inference/compile_model.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/aoti_inference`):

- [`aoti_custom_class.h_docs.md`](./aoti_custom_class.h_docs.md)
- [`test.py_docs.md`](./test.py_docs.md)
- [`standalone_test.cpp_docs.md`](./standalone_test.cpp_docs.md)
- [`generate_lowered_cpu.py_docs.md`](./generate_lowered_cpu.py_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test.cpp_docs.md`](./test.cpp_docs.md)
- [`aoti_custom_class.cpp_docs.md`](./aoti_custom_class.cpp_docs.md)
- [`standalone_compile.sh_docs.md`](./standalone_compile.sh_docs.md)


## Cross-References

- **File Documentation**: `compile_model.py_docs.md`
- **Keyword Index**: `compile_model.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
