# Documentation: `docs/third_party/substitution.bzl_docs.md`

## File Metadata

- **Path**: `docs/third_party/substitution.bzl_docs.md`
- **Size**: 4,941 bytes (4.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `third_party/substitution.bzl`

## File Metadata

- **Path**: `third_party/substitution.bzl`
- **Size**: 2,625 bytes (2.56 KB)
- **Type**: Source File (.bzl)
- **Extension**: `.bzl`

## File Purpose

This is a source file (.bzl) that is part of the PyTorch project.

## Original Source

```
# This Bazel rules file is derived from https://github.com/tensorflow/tensorflow/blob/master/third_party/common.bzl

# Rule for simple expansion of template files. This performs a simple
# search over the template file for the keys in substitutions,
# and replaces them with the corresponding values.
#
# Typical usage:
#   load("/tools/build_rules/template_rule", "template_rule")
#   template_rule(
#       name = "ExpandMyTemplate",
#       src = "my.template",
#       out = "my.txt",
#       substitutions = {
#         "$VAR1": "foo",
#         "$VAR2": "bar",
#       }
#   )
#
# Args:
#   name: The name of the rule.
#   template: The template file to expand
#   out: The destination of the expanded file
#   substitutions: A dictionary mapping strings to their substitutions

def template_rule_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )

template_rule = rule(
    attrs = {
        "out": attr.output(mandatory = True),
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "substitutions": attr.string_dict(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = template_rule_impl,
)

# Header template rule is an extension of template substitution rule
# That also makes this header a valid dependency for cc_library
# From https://stackoverflow.com/a/55407399
def header_template_rule_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )
    return [
        # create a provider which says that this
        # out file should be made available as a header
        CcInfo(compilation_context = cc_common.create_compilation_context(

            # pass out the include path for finding this header
            system_includes = depset([ctx.attr.include, ctx.outputs.out.dirname, ctx.bin_dir.path]),

            # and the actual header here.
            headers = depset([ctx.outputs.out]),
        )),
    ]

header_template_rule = rule(
    attrs = {
        "include": attr.string(),
        "out": attr.output(mandatory = True),
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "substitutions": attr.string_dict(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = header_template_rule_impl,
)

```



## High-Level Overview

This file is part of the PyTorch framework located at `third_party`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `third_party`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`third_party`):

- [`glog.buck.bzl_docs.md`](./glog.buck.bzl_docs.md)
- [`generate-xnnpack-wrappers.py_docs.md`](./generate-xnnpack-wrappers.py_docs.md)
- [`generate-cpuinfo-wrappers.py_docs.md`](./generate-cpuinfo-wrappers.py_docs.md)
- [`xpu.txt_docs.md`](./xpu.txt_docs.md)
- [`kineto.buck.bzl_docs.md`](./kineto.buck.bzl_docs.md)
- [`xnnpack.buck.bzl_docs.md`](./xnnpack.buck.bzl_docs.md)
- [`xnnpack_wrapper_defs.bzl_docs.md`](./xnnpack_wrapper_defs.bzl_docs.md)
- [`eigen_pin.txt_docs.md`](./eigen_pin.txt_docs.md)
- [`LICENSES_BUNDLED.txt_docs.md`](./LICENSES_BUNDLED.txt_docs.md)
- [`sleef.bzl_docs.md`](./sleef.bzl_docs.md)


## Cross-References

- **File Documentation**: `substitution.bzl_docs.md`
- **Keyword Index**: `substitution.bzl_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/third_party`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/third_party`, which is part of the PyTorch project infrastructure.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/third_party`):

- [`substitution.bzl_kw.md_docs.md`](./substitution.bzl_kw.md_docs.md)
- [`xnnpack_buck_shim.bzl_kw.md_docs.md`](./xnnpack_buck_shim.bzl_kw.md_docs.md)
- [`LICENSES_BUNDLED.txt_kw.md_docs.md`](./LICENSES_BUNDLED.txt_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`kineto.buck.bzl_docs.md_docs.md`](./kineto.buck.bzl_docs.md_docs.md)
- [`generate-cpuinfo-wrappers.py_kw.md_docs.md`](./generate-cpuinfo-wrappers.py_kw.md_docs.md)
- [`xnnpack_buck_shim.bzl_docs.md_docs.md`](./xnnpack_buck_shim.bzl_docs.md_docs.md)
- [`eigen_pin.txt_docs.md_docs.md`](./eigen_pin.txt_docs.md_docs.md)
- [`build_bundled.py_kw.md_docs.md`](./build_bundled.py_kw.md_docs.md)
- [`generate-cpuinfo-wrappers.py_docs.md_docs.md`](./generate-cpuinfo-wrappers.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `substitution.bzl_docs.md_docs.md`
- **Keyword Index**: `substitution.bzl_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
