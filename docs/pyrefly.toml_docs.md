# Documentation: `pyrefly.toml`

## File Metadata

- **Path**: `pyrefly.toml`
- **Size**: 4,787 bytes (4.67 KB)
- **Type**: TOML Configuration
- **Extension**: `.toml`

## File Purpose

This is a toml configuration that is part of the PyTorch project.

## Original Source

```
# A Pyrefly configuration for PyTorch
# Based on https://github.com/pytorch/pytorch/blob/main/mypy.ini
python-version = "3.12"

project-includes = [
    "torch",
    "caffe2",
    "tools",
    "test/test_bundled_images.py",
    "test/test_bundled_inputs.py",
    "test/test_complex.py",
    "test/test_datapipe.py",
    # "test/test_futures.py", # uncomment when enabling pyrefly
    "test/test_numpy_interop.py",
    # We exclude test_torch.py because it is full of errors, but most functions lack type signatures,
    # and mypy.ini specifies `check_untyped_defs = False` for this file.
    # If you check even the unannotated stuff mypy produces 322 errors.
    # "test/test_torch.py",
    "test/test_type_hints.py",
    "test/test_type_info.py",
    # "test/test_utils.py", # uncomment when enabling pyrefly
]
project-excludes = [
  # ==== below will be enabled directory by directory ====
  # ==== to test Pyrefly on a specific directory, simply comment it out ====
  "torch/_inductor/codegen/triton.py",
  "tools/linter/adapters/test_device_bias_linter.py",
  "tools/code_analyzer/gen_operators_yaml.py",
  "torch/_inductor/runtime/triton_heuristics.py",
  "torch/_inductor/runtime/triton_helpers.py",
  "torch/_inductor/runtime/halide_helpers.py",
  "torch/utils/tensorboard/summary.py",
  # formatting issues, will turn on after adjusting where suppressions can be
  # in import statements
  "torch/distributed/flight_recorder/components/types.py",
  "torch/linalg/__init__.py",
  "torch/package/importer.py",
  "torch/package/_package_pickler.py",
  "torch/jit/annotations.py",
  "torch/utils/data/datapipes/_typing.py",
  "torch/nn/functional.py",
  "torch/_export/utils.py",
  "torch/fx/experimental/unification/multipledispatch/__init__.py",
  "torch/nn/modules/__init__.py",
  "torch/nn/modules/rnn.py", # only remove when parsing errors are fixed
  "torch/_inductor/codecache.py",
  "torch/distributed/elastic/metrics/__init__.py",
  "torch/_inductor/fx_passes/bucketing.py",
  # ====
  "torch/onnx/_internal/exporter/_torchlib/ops/nn.py",
  "torch/include/**",
  "torch/csrc/**",
  "torch/distributed/elastic/agent/server/api.py",
  "torch/testing/_internal/**",
  "torch/distributed/fsdp/fully_sharded_data_parallel.py",
  "torch/ao/quantization/pt2e/_affine_quantization.py",
  "torch/nn/modules/pooling.py",
  "torch/nn/parallel/_functions.py",
  "torch/_appdirs.py",
  "torch/multiprocessing/pool.py",
  "torch/overrides.py",
  "*/__pycache__/**",
  "*/.*",
]
ignore-missing-imports = [
    "torch._C._jit_tree_views.*",
    "torch.for_onnx.onnx.*",
    "torch.ao.quantization.experimental.apot_utils.*",
    "torch.ao.quantization.experimental.quantizer.*",
    "torch.ao.quantization.experimental.observer.*",
    "torch.ao.quantization.experimental.APoT_tensor.*",
    "torch.ao.quantization.experimental.fake_quantize_function.*",
    "torch.ao.quantization.experimental.fake_quantize.*",
    "triton.*",
    "tensorflow.*",
    "tensorboard.*",
    "matplotlib.*",
    "numpy.*",
    "sympy.*",
    "hypothesis.*",
    "tqdm.*",
    "multiprocessing.*",
    "setuptools.*",
    "distutils.*",
    "nvd3.*",
    "future.utils.*",
    "past.builtins.*",
    "numba.*",
    "PIL.*",
    "moviepy.*",
    "cv2.*",
    "torchvision.*",
    "pycuda.*",
    "tensorrt.*",
    "tornado.*",
    "pydot.*",
    "networkx.*",
    "scipy.*",
    "IPython.*",
    "google.protobuf.textformat.*",
    "lmdb.*",
    "mpi4py.*",
    "skimage.*",
    "librosa.*",
    "mypy.*",
    "xml.*",
    "boto3.*",
    "dill.*",
    "usort.*",
    "cutlass_library.*",
    "deeplearning.*",
    "einops.*",
    "libfb.*",
    "torch.fb.*",
    "torch.*.fb.*",
    "torch_xla.*",
    "onnx.*",
    "onnxruntime.*",
    "onnxscript.*",
    "redis.*",
]
# By default, mypy does not check untyped definitions.
# However, mypy has a configuration called check_untyped_defs which is used
# to typecheck the interior of untyped functions.
untyped-def-behavior = "check-and-infer-return-any"
# In lots of places they define their attributes in `_init` or similar.
# https://github.com/pytorch/pytorch/blob/75f3e5a88df60caef27fd9c9df3fd51161378fcc/torch/fx/experimental/symbolic_shapes.py#L3632C1-L3633C1
errors.implicitly-defined-attribute = false
# In many methods that are overridden, parameters are renamed.
# We can come up with a codemod for this in the future
errors.bad-param-name-override = false
# Mypy doesn't require that imports are explicitly imported, so be compatible with that.
# Might be a good idea to turn this on in future.
errors.implicit-import = false
errors.deprecated = false # re-enable after we've fix import formatting
permissive-ignores = true
replace-imports-with-any = ["!sympy.printing.*", "sympy.*", "onnxscript.onnx_opset.*"]
search-path = ["tools/experimental"]

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

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

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

- **File Documentation**: `pyrefly.toml_docs.md`
- **Keyword Index**: `pyrefly.toml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
