# Documentation: `mypy.ini`

## File Metadata

- **Path**: `mypy.ini`
- **Size**: 6,245 bytes (6.10 KB)
- **Type**: Source File (.ini)
- **Extension**: `.ini`

## File Purpose

This is a source file (.ini) that is part of the PyTorch project.

## Original Source

```
# This is the PyTorch mypy.ini file (note: don't change this line! -
# test_run_mypy in test/test_type_hints.py uses this string)

[mypy]
plugins = mypy_plugins/check_mypy_version.py, mypy_plugins/sympy_mypy_plugin.py

cache_dir = .mypy_cache/normal
allow_redefinition = True
warn_unused_configs = True
warn_redundant_casts = True
show_error_codes = True
show_column_numbers = True
check_untyped_defs = True
disallow_untyped_defs = True
disallow_untyped_decorators = True
follow_imports = normal
local_partial_types = True
enable_error_code = possibly-undefined

# do not re-enable this:
# https://github.com/pytorch/pytorch/pull/60006#issuecomment-866130657
warn_unused_ignores = False

#
# Note: test/ still has syntax errors so can't be added
#
# Typing tests is low priority, but enabling type checking on the
# untyped test functions (using `--check-untyped-defs`) is still
# high-value because it helps test the typing.
#

files =
    torch,
    caffe2,
    test/test_bundled_images.py,
    test/test_bundled_inputs.py,
    test/test_complex.py,
    test/test_datapipe.py,
    test/test_futures.py,
    test/test_numpy_interop.py,
    test/test_torch.py,
    test/test_type_hints.py,
    test/test_type_info.py,
    test/test_utils.py

#
# `exclude` is a regex, not a list of paths like `files` (sigh)
#
exclude = torch/include/|torch/csrc/|torch/distributed/elastic/agent/server/api.py|torch/testing/_internal|torch/distributed/fsdp/fully_sharded_data_parallel.py

python_version = 3.11


#
# Extension modules without stubs.
#

[mypy-torch.for_onnx.onnx]
ignore_missing_imports = True

[mypy-torch.ao.quantization.experimental.apot_utils]
ignore_missing_imports = True

[mypy-torch.ao.quantization.experimental.quantizer]
ignore_missing_imports = True

[mypy-torch.ao.quantization.experimental.observer]
ignore_missing_imports = True

[mypy-torch.ao.quantization.experimental.APoT_tensor]
ignore_missing_imports = True

[mypy-torch.ao.quantization.experimental.fake_quantize_function]
ignore_missing_imports = True

[mypy-torch.ao.quantization.experimental.fake_quantize]
ignore_missing_imports = True

[mypy-torch.ao.quantization.pt2e._affine_quantization]
ignore_errors = True

#
# Files with various errors. Mostly real errors, possibly some false
# positives as well.
#

[mypy-test_torch]
check_untyped_defs = False

# Excluded from mypy due to OpInfos being annoying to type
[mypy-torch.testing._internal.common_methods_invocations.*]
ignore_errors = True

[mypy-torch.testing._internal.hypothesis_utils.*]
ignore_errors = True

[mypy-torch.testing._internal.common_quantization.*]
ignore_errors = True

[mypy-torch.testing._internal.generated.*]
ignore_errors = True

[mypy-torch.testing._internal.distributed.*]
ignore_errors = True

[mypy-torch.nn.modules.pooling]
ignore_errors = True

[mypy-torch.nn.parallel._functions]
ignore_errors = True

[mypy-torch._appdirs]
ignore_errors = True

[mypy-torch.multiprocessing.pool]
ignore_errors = True

[mypy-torch.overrides]
ignore_errors = True

#
# Files with 'type: ignore' comments that are needed if checked with mypy-strict.ini
#

[mypy-tools.render_junit]
warn_unused_ignores = False

[mypy-tools.generate_torch_version]
warn_unused_ignores = False

#
# Adding type annotations to caffe2 is probably not worth the effort
# only work on this if you have a specific reason for it, otherwise
# leave these ignores as they are.
#

[mypy-caffe2.python.*]
ignore_errors = True

[mypy-caffe2.proto.*]
ignore_errors = True

[mypy-caffe2.distributed.store_ops_test_util]
ignore_errors = True

[mypy-caffe2.experiments.*]
ignore_errors = True

[mypy-caffe2.contrib.*]
ignore_errors = True

[mypy-caffe2.quantization.server.*]
ignore_errors = True

#
# Third party dependencies that don't have types.
#

[mypy-triton.*]
ignore_missing_imports = True

[mypy-tensorflow.*]
ignore_missing_imports = True

[mypy-tensorboard.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True

[mypy-numpy.*]
ignore_missing_imports = True

[mypy-sympy]
ignore_missing_imports = True

[mypy-sympy.*]
ignore_missing_imports = True

[mypy-hypothesis.*]
ignore_missing_imports = True

[mypy-tqdm.*]
ignore_missing_imports = True

[mypy-multiprocessing.*]
ignore_missing_imports = True

[mypy-setuptools.*]
ignore_missing_imports = True

[mypy-distutils.*]
ignore_missing_imports = True

[mypy-nvd3.*]
ignore_missing_imports = True

[mypy-future.utils]
ignore_missing_imports = True

[mypy-past.builtins]
ignore_missing_imports = True

[mypy-numba.*]
ignore_missing_imports = True

[mypy-PIL.*]
ignore_missing_imports = True

[mypy-moviepy.*]
ignore_missing_imports = True

[mypy-cv2.*]
ignore_missing_imports = True

[mypy-torchvision.*]
ignore_missing_imports = True

[mypy-pycuda.*]
ignore_missing_imports = True

[mypy-tensorrt.*]
ignore_missing_imports = True

[mypy-tornado.*]
ignore_missing_imports = True

[mypy-pydot.*]
ignore_missing_imports = True

[mypy-networkx.*]
ignore_missing_imports = True

[mypy-scipy.*]
ignore_missing_imports = True

[mypy-IPython.*]
ignore_missing_imports = True

[mypy-google.protobuf.textformat]
ignore_missing_imports = True

[mypy-lmdb.*]
ignore_missing_imports = True

[mypy-mpi4py.*]
ignore_missing_imports = True

[mypy-skimage.*]
ignore_missing_imports = True

[mypy-librosa.*]
ignore_missing_imports = True

[mypy-mypy.*]
ignore_missing_imports = True

[mypy-xml.*]
ignore_missing_imports = True

[mypy-boto3.*]
ignore_missing_imports = True

[mypy-dill.*]
ignore_missing_imports = True

[mypy-usort.*]
ignore_missing_imports = True

[mypy-torch._inductor.*]
disallow_any_generics = True

[mypy-torch._dynamo.*]
disallow_any_generics = True

[mypy-cutlass_library.*]
ignore_missing_imports = True

[mypy-deeplearning.*]
ignore_missing_imports = True

[mypy-einops.*]
ignore_missing_imports = True

[mypy-libfb.*]
ignore_missing_imports = True

[mypy-torch.*.fb.*]
ignore_missing_imports = True

[mypy-torch.fb.*]
ignore_missing_imports = True

[mypy-torch_xla.*]
ignore_missing_imports = True

#
# Third party dependencies that are optional.
#

[mypy-onnx.*]
ignore_missing_imports = True

[mypy-onnxruntime.*]
ignore_missing_imports = True

[mypy-onnxscript.*]
ignore_missing_imports = True

[mypy-redis]
ignore_missing_imports = True

```



## High-Level Overview

This file is part of the PyTorch framework located at ``.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `root`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`root`):

- [`AGENTS.md_docs.md`](./AGENTS.md_docs.md)
- [`pytest.ini_docs.md`](./pytest.ini_docs.md)
- [`codex_setup.sh_docs.md`](./codex_setup.sh_docs.md)
- [`pt_template_srcs.bzl_docs.md`](./pt_template_srcs.bzl_docs.md)
- [`aten.bzl_docs.md`](./aten.bzl_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`buckbuild.bzl_docs.md`](./buckbuild.bzl_docs.md)
- [`Dockerfile_docs.md`](./Dockerfile_docs.md)
- [`.bc-linter.yml_docs.md`](./.bc-linter.yml_docs.md)
- [`setup.py_docs.md`](./setup.py_docs.md)


## Cross-References

- **File Documentation**: `mypy.ini_docs.md`
- **Keyword Index**: `mypy.ini_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
