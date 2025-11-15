# Documentation: `docs/tools/test/test_vulkan_codegen.py_docs.md`

## File Metadata

- **Path**: `docs/tools/test/test_vulkan_codegen.py_docs.md`
- **Size**: 7,802 bytes (7.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/test/test_vulkan_codegen.py`

## File Metadata

- **Path**: `tools/test/test_vulkan_codegen.py`
- **Size**: 3,749 bytes (3.66 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
import tempfile
import unittest

from tools.gen_vulkan_spv import DEFAULT_ENV, SPVGenerator


####################
# Data for testing #
####################

test_shader = """
#version 450 core

#define FORMAT ${FORMAT}
#define PRECISION ${PRECISION}
#define OP(X) ${OPERATOR}

$def is_int(dtype):
$   return dtype in {"int", "int32", "int8"}

$def is_uint(dtype):
$   return dtype in {"uint", "uint32", "uint8"}

$if is_int(DTYPE):
  #define VEC4_T ivec4
$elif is_uint(DTYPE):
  #define VEC4_T uvec4
$else:
  #define VEC4_T vec4

$if not INPLACE:
  $if is_int(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;
  $elif is_uint(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly uimage3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION usampler3D uInput;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
$else:
  $if is_int(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict iimage3D uOutput;
  $elif is_uint(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict uimage3D uOutput;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  $if not INPLACE:
    VEC4_T v = texelFetch(uInput, pos, 0);
  $else:
    VEC4_T v = imageLoad(uOutput, pos);
  $for i in range(ITER[0]):
    for (int i = 0; i < ${ITER[1]}; ++i) {
        v = OP(v + i);
    }
  imageStore(uOutput, pos, OP(v));
}

"""

test_params_yaml = """
test_shader:
  parameter_names_with_default_values:
    DTYPE: float
    INPLACE: false
    OPERATOR: X + 3
    ITER: !!python/tuple [3, 5]
  generate_variant_forall:
    INPLACE:
      - VALUE: false
        SUFFIX: ""
      - VALUE: true
        SUFFIX: inplace
    DTYPE:
      - VALUE: int8
      - VALUE: float
  shader_variants:
    - NAME: test_shader_1
    - NAME: test_shader_3
      OPERATOR: X - 1
      ITER: !!python/tuple [3, 2]
      generate_variant_forall:
        DTYPE:
        - VALUE: float
        - VALUE: int

"""

##############
# Unit Tests #
##############


class TestVulkanSPVCodegen(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()

        with open(f"{self.tmpdir.name}/test_shader.glsl,", "w") as f:
            f.write(test_shader)

        with open(f"{self.tmpdir.name}/test_params.yaml", "w") as f:
            f.write(test_params_yaml)

        self.tmpoutdir = tempfile.TemporaryDirectory()

        self.generator = SPVGenerator(
            src_dir_paths=self.tmpdir.name, env=DEFAULT_ENV, glslc_path=None
        )

    def cleanUp(self) -> None:
        self.tmpdir.cleanup()
        self.tmpoutdir.cleanup()

    def testOutputMap(self) -> None:
        # Each shader variant will produce variants generated based on all possible combinations
        # of the DTYPE and INPLACE parameters. test_shader_3 has fewer generated variants due to
        # a custom specified generate_variant_forall field.
        expected_output_shaders = {
            "test_shader_1_float",
            "test_shader_1_inplace_float",
            "test_shader_1_inplace_int8",
            "test_shader_1_int8",
            "test_shader_3_float",
            "test_shader_3_int",
        }

        actual_output_shaders = set(self.generator.output_shader_map.keys())

        self.assertEqual(expected_output_shaders, actual_output_shaders)

```



## High-Level Overview

test_shader = """#version 450 core#define FORMAT ${FORMAT}#define PRECISION ${PRECISION}#define OP(X) ${OPERATOR}$def is_int(dtype):$   return dtype in {"int", "int32", "int8"}$def is_uint(dtype):$   return dtype in {"uint", "uint32", "uint8"}$if is_int(DTYPE):  #define VEC4_T ivec4$elif is_uint(DTYPE):  #define VEC4_T uvec4$else:  #define VEC4_T vec4$if not INPLACE:  $if is_int(DTYPE):    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;    layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;  $elif is_uint(DTYPE):    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly uimage3D uOutput;    layout(set = 0, binding = 1) uniform PRECISION usampler3D uInput;  $else:    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;$else:  $if is_int(DTYPE):    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict iimage3D uOutput;  $elif is_uint(DTYPE):    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict uimage3D uOutput;  $else:    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestVulkanSPVCodegen`

**Functions defined**: `is_int`, `is_uint`, `setUp`, `cleanUp`, `testOutputMap`

**Key imports**: tempfile, unittest, DEFAULT_ENV, SPVGenerator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `tempfile`
- `unittest`
- `tools.gen_vulkan_spv`: DEFAULT_ENV, SPVGenerator


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python tools/test/test_vulkan_codegen.py
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

- **File Documentation**: `test_vulkan_codegen.py_docs.md`
- **Keyword Index**: `test_vulkan_codegen.py_kw.md`
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

*No specific patterns automatically detected.*


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
python docs/tools/test/test_vulkan_codegen.py_docs.md
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

- **File Documentation**: `test_vulkan_codegen.py_docs.md_docs.md`
- **Keyword Index**: `test_vulkan_codegen.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
