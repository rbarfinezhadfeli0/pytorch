# Documentation: `docs/tools/dynamo/verify_dynamo.py_docs.md`

## File Metadata

- **Path**: `docs/tools/dynamo/verify_dynamo.py_docs.md`
- **Size**: 9,136 bytes (8.92 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/dynamo/verify_dynamo.py`

## File Metadata

- **Path**: `tools/dynamo/verify_dynamo.py`
- **Size**: 6,710 bytes (6.55 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
import os
import re
import subprocess
import sys
import traceback
import warnings


MIN_CUDA_VERSION = "11.6"
MIN_ROCM_VERSION = "5.4"
MIN_PYTHON_VERSION = (3, 10)


class VerifyDynamoError(BaseException):
    pass


def check_python():
    if sys.version_info < MIN_PYTHON_VERSION:
        raise VerifyDynamoError(
            f"Python version not supported: {sys.version_info} "
            f"- minimum requirement: {MIN_PYTHON_VERSION}"
        )
    return sys.version_info


def check_torch():
    import torch

    return torch.__version__


# based on torch/utils/cpp_extension.py
def get_cuda_version():
    from torch.torch_version import TorchVersion
    from torch.utils import cpp_extension

    CUDA_HOME = cpp_extension._find_cuda_home()
    if not CUDA_HOME:
        raise VerifyDynamoError(cpp_extension.CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    cuda_version_str = (
        subprocess.check_output([nvcc, "--version"])
        .strip()
        .decode(*cpp_extension.SUBPROCESS_DECODE_ARGS)
    )
    cuda_version = re.search(r"release (\d+[.]\d+)", cuda_version_str)
    if cuda_version is None:
        raise VerifyDynamoError("CUDA version not found in `nvcc --version` output")

    cuda_str_version = cuda_version.group(1)
    return TorchVersion(cuda_str_version)


def get_rocm_version():
    from torch.torch_version import TorchVersion
    from torch.utils import cpp_extension

    ROCM_HOME = cpp_extension._find_rocm_home()
    if not ROCM_HOME:
        raise VerifyDynamoError(
            "ROCM was not found on the system, please set ROCM_HOME environment variable"
        )

    hipcc = os.path.join(ROCM_HOME, "bin", "hipcc")
    hip_version_str = (
        subprocess.check_output([hipcc, "--version"])
        .strip()
        .decode(*cpp_extension.SUBPROCESS_DECODE_ARGS)
    )
    hip_version = re.search(r"HIP version: (\d+[.]\d+)", hip_version_str)

    if hip_version is None:
        raise VerifyDynamoError("HIP version not found in `hipcc --version` output")

    hip_str_version = hip_version.group(1)

    return TorchVersion(hip_str_version)


def check_cuda():
    import torch
    from torch.torch_version import TorchVersion

    if not torch.cuda.is_available() or torch.version.hip is not None:
        return None

    torch_cuda_ver = TorchVersion(torch.version.cuda)

    # check if torch cuda version matches system cuda version
    cuda_ver = get_cuda_version()
    if cuda_ver != torch_cuda_ver:
        # raise VerifyDynamoError(
        warnings.warn(
            f"CUDA version mismatch, `torch` version: {torch_cuda_ver}, env version: {cuda_ver}"
        )

    if torch_cuda_ver < MIN_CUDA_VERSION:
        # raise VerifyDynamoError(
        warnings.warn(
            f"(`torch`) CUDA version not supported: {torch_cuda_ver} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )
    if cuda_ver < MIN_CUDA_VERSION:
        # raise VerifyDynamoError(
        warnings.warn(
            f"(env) CUDA version not supported: {cuda_ver} "
            f"- minimum requirement: {MIN_CUDA_VERSION}"
        )

    return cuda_ver if torch.version.hip is None else "None"


def check_rocm():
    import torch
    from torch.torch_version import TorchVersion

    if not torch.cuda.is_available() or torch.version.hip is None:
        return None

    # Extracts main ROCm version from full string
    torch_rocm_ver = TorchVersion(".".join(list(torch.version.hip.split(".")[0:2])))

    # check if torch rocm version matches system rocm version
    rocm_ver = get_rocm_version()
    if rocm_ver != torch_rocm_ver:
        warnings.warn(
            f"ROCm version mismatch, `torch` version: {torch_rocm_ver}, env version: {rocm_ver}"
        )
    if torch_rocm_ver < MIN_ROCM_VERSION:
        warnings.warn(
            f"(`torch`) ROCm version not supported: {torch_rocm_ver} "
            f"- minimum requirement: {MIN_ROCM_VERSION}"
        )
    if rocm_ver < MIN_ROCM_VERSION:
        warnings.warn(
            f"(env) ROCm version not supported: {rocm_ver} "
            f"- minimum requirement: {MIN_ROCM_VERSION}"
        )

    return rocm_ver if torch.version.hip else "None"


def check_dynamo(backend, device, err_msg) -> None:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available -- skipping CUDA check on {backend} backend\n")
        return

    try:
        import torch._dynamo as dynamo

        if device == "cuda":
            from torch.utils._triton import has_triton

            if not has_triton():
                print(
                    f"WARNING: CUDA available but triton cannot be used. "
                    f"Your GPU may not be supported. "
                    f"Skipping CUDA check on {backend} backend\n"
                )
                return

        dynamo.reset()

        @dynamo.optimize(backend, nopython=True)
        def fn(x):
            return x + x

        class Module(torch.nn.Module):
            def forward(self, x):
                return x + x

        mod = Module()
        opt_mod = dynamo.optimize(backend, nopython=True)(mod)

        for f in (fn, opt_mod):
            x = torch.randn(10, 10).to(device)
            x.requires_grad = True
            y = f(x)
            torch.testing.assert_close(y, x + x)
            z = y.sum()
            z.backward()
            torch.testing.assert_close(x.grad, 2 * torch.ones_like(x))
    except Exception:
        sys.stderr.write(traceback.format_exc() + "\n" + err_msg + "\n\n")
        sys.exit(1)


_SANITY_CHECK_ARGS = (
    ("eager", "cpu", "CPU eager sanity check failed"),
    ("eager", "cuda", "CUDA eager sanity check failed"),
    ("aot_eager", "cpu", "CPU aot_eager sanity check failed"),
    ("aot_eager", "cuda", "CUDA aot_eager sanity check failed"),
    ("inductor", "cpu", "CPU inductor sanity check failed"),
    (
        "inductor",
        "cuda",
        "CUDA inductor sanity check failed\n"
        + "NOTE: Please check that you installed the correct hash/version of `triton`",
    ),
)


def main() -> None:
    python_ver = check_python()
    torch_ver = check_torch()
    cuda_ver = check_cuda()
    rocm_ver = check_rocm()
    print(
        f"Python version: {python_ver.major}.{python_ver.minor}.{python_ver.micro}\n"
        f"`torch` version: {torch_ver}\n"
        f"CUDA version: {cuda_ver}\n"
        f"ROCM version: {rocm_ver}\n"
    )
    for args in _SANITY_CHECK_ARGS:
        if sys.version_info >= (3, 15):
            warnings.warn("Dynamo not yet supported in Python 3.15.")
        check_dynamo(*args)
    print("All required checks passed")


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `VerifyDynamoError`, `Module`

**Functions defined**: `check_python`, `check_torch`, `get_cuda_version`, `get_rocm_version`, `check_cuda`, `check_rocm`, `check_dynamo`, `fn`, `forward`, `main`

**Key imports**: os, re, subprocess, sys, traceback, warnings, torch, TorchVersion, cpp_extension, TorchVersion


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/dynamo`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `re`
- `subprocess`
- `sys`
- `traceback`
- `warnings`
- `torch`
- `torch.torch_version`: TorchVersion
- `torch.utils`: cpp_extension
- `torch._dynamo as dynamo`
- `torch.utils._triton`: has_triton


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/dynamo`):

- [`gb_id_mapping.py_docs.md`](./gb_id_mapping.py_docs.md)


## Cross-References

- **File Documentation**: `verify_dynamo.py_docs.md`
- **Keyword Index**: `verify_dynamo.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/dynamo`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/dynamo`):

- [`verify_dynamo.py_kw.md_docs.md`](./verify_dynamo.py_kw.md_docs.md)
- [`gb_id_mapping.py_kw.md_docs.md`](./gb_id_mapping.py_kw.md_docs.md)
- [`gb_id_mapping.py_docs.md_docs.md`](./gb_id_mapping.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `verify_dynamo.py_docs.md_docs.md`
- **Keyword Index**: `verify_dynamo.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
