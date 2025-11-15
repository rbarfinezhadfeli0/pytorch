# Documentation: `docs/test/mobile/model_test/builtin_ops.py_docs.md`

## File Metadata

- **Path**: `docs/test/mobile/model_test/builtin_ops.py_docs.md`
- **Size**: 5,447 bytes (5.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/mobile/model_test/builtin_ops.py`

## File Metadata

- **Path**: `test/mobile/model_test/builtin_ops.py`
- **Size**: 2,978 bytes (2.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
import torch


# https://pytorch.org/docs/stable/jit_builtin_functions.html#builtin-functions


class TSBuiltinOpsModule(torch.nn.Module):
    def forward(self):
        x = torch.tensor(1)
        y = torch.tensor(0.5)
        b = float(1)
        l = ["1", "2", "test", "a{}b"]
        d = {"key": 1}
        d2 = {0: 100}
        return len(
            # type
            bool(x),
            bool(x.item()),
            int(y),
            int(y.item()),
            float(x),
            float(x.item()),
            # math
            x & x,
            bool(x) & bool(x),
            int(x) & int(x),
            x | x,
            bool(x) | bool(x),
            int(x) | int(x),
            x << x,
            int(x) << int(x),
            x >> x,
            int(x) >> int(x),
            x ^ x,
            bool(x) ^ bool(x),
            int(x) ^ int(x),
            b * float(x),
            b * int(x),
            b + float(x),
            b - float(x),
            x.item() + y.item(),
            x.item() - y.item(),
            x.item() * y.item(),
            x.item() / y.item(),
            float(x) < float(y),
            float(x) <= float(y),
            float(x) > float(y),
            float(x) > int(y),
            float(x) >= float(y),
            float(x) >= int(y),
            float(x) == float(y),
            float(x) == int(y),
            float(x) != float(y),
            int(x) != float(y),
            float(x) / float(y),
            int(x) / int(y),
            max(x),
            max(x.item(), y.item()),
            max(int(x), int(y)),
            max(float(x), float(y)),
            min(x),
            min(x.item(), y.item()),
            min(int(x), int(y)),
            min(float(x), float(y)),
            int(l[0]),
            float(l[0]),
            # string
            str(torch.tensor(1)),
            l[2].find("t"),
            l[2].replace("t", "x"),
            l[2].lower(),
            l[2].startswith("t"),
            l[2].split("t"),
            l[2].strip(),
            l[2].rstrip(),
            l[2].lstrip(),
            l[2][slice(2)],
            l[3].format("x"),
            ord(l[2][0]),
            len(torch.randn(3)),
            len(l),
            len(l[2]),
            len(d),
            len(d2),
        )


class TSCollectionOpsModule(torch.nn.Module):
    def forward(self):
        s = "abcde"
        # list
        l = ["1", "2", "test"]
        l.reverse()
        l.reverse()
        l[1] = "3"
        l.extend(["4"])
        # str dict
        d = {"key": 1}
        d.clear()
        d.update({"key": 0})
        if "key" in d:
            d["key"] = 2
        #  int dict
        d2 = {0: 100}
        if 0 in d2:
            d2.clear()
            d2[0] = 100

        return len(
            s[torch.tensor(1)],
            d["key"],
            d2[0],
            d.keys(),
            d.items(),
            d.values(),
            d2.values(),
            l.pop(),
        )

```



## High-Level Overview


This Python file contains 2 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TSBuiltinOpsModule`, `TSCollectionOpsModule`

**Functions defined**: `forward`, `forward`

**Key imports**: torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/mobile/model_test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/mobile/model_test/builtin_ops.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/mobile/model_test`):

- [`torchvision_models.py_docs.md`](./torchvision_models.py_docs.md)
- [`gen_test_model.py_docs.md`](./gen_test_model.py_docs.md)
- [`update_production_ops.py_docs.md`](./update_production_ops.py_docs.md)
- [`math_ops.py_docs.md`](./math_ops.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`nn_ops.py_docs.md`](./nn_ops.py_docs.md)
- [`model_ops.yaml_docs.md`](./model_ops.yaml_docs.md)
- [`quantization_ops.py_docs.md`](./quantization_ops.py_docs.md)
- [`android_api_module.py_docs.md`](./android_api_module.py_docs.md)


## Cross-References

- **File Documentation**: `builtin_ops.py_docs.md`
- **Keyword Index**: `builtin_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/mobile/model_test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/mobile/model_test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python docs/test/mobile/model_test/builtin_ops.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/mobile/model_test`):

- [`tensor_ops.py_kw.md_docs.md`](./tensor_ops.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`sampling_ops.py_docs.md_docs.md`](./sampling_ops.py_docs.md_docs.md)
- [`torchvision_models.py_docs.md_docs.md`](./torchvision_models.py_docs.md_docs.md)
- [`android_api_module.py_kw.md_docs.md`](./android_api_module.py_kw.md_docs.md)
- [`android_api_module.py_docs.md_docs.md`](./android_api_module.py_docs.md_docs.md)
- [`torchvision_models.py_kw.md_docs.md`](./torchvision_models.py_kw.md_docs.md)
- [`tensor_ops.py_docs.md_docs.md`](./tensor_ops.py_docs.md_docs.md)
- [`math_ops.py_kw.md_docs.md`](./math_ops.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `builtin_ops.py_docs.md_docs.md`
- **Keyword Index**: `builtin_ops.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
