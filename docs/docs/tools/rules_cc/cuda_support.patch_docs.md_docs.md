# Documentation: `docs/tools/rules_cc/cuda_support.patch_docs.md`

## File Metadata

- **Path**: `docs/tools/rules_cc/cuda_support.patch_docs.md`
- **Size**: 5,087 bytes (4.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/rules_cc/cuda_support.patch`

## File Metadata

- **Path**: `tools/rules_cc/cuda_support.patch`
- **Size**: 3,250 bytes (3.17 KB)
- **Type**: Source File (.patch)
- **Extension**: `.patch`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```
diff --git cc/private/toolchain/unix_cc_configure.bzl cc/private/toolchain/unix_cc_configure.bzl
index ba992fc..e4e8364 100644
--- cc/private/toolchain/unix_cc_configure.bzl
+++ cc/private/toolchain/unix_cc_configure.bzl
@@ -27,6 +27,7 @@ load(
     "which",
     "write_builtin_include_directory_paths",
 )
+load("@rules_cuda//cuda:toolchain.bzl", "cuda_compiler_deps")
 
 def _field(name, value):
     """Returns properly indented top level crosstool field."""
@@ -397,7 +398,7 @@ def configure_unix_toolchain(repository_ctx, cpu_value, overriden_tools):
     cxx_opts = split_escaped(get_env_var(
         repository_ctx,
         "BAZEL_CXXOPTS",
-        "-std=c++0x",
+        "-std=c++17",
         False,
     ), ":")
 
@@ -463,7 +464,7 @@ def configure_unix_toolchain(repository_ctx, cpu_value, overriden_tools):
             )),
             "%{cc_compiler_deps}": get_starlark_list([":builtin_include_directory_paths"] + (
                 [":cc_wrapper"] if darwin else []
-            )),
+            ) + cuda_compiler_deps()),
             "%{cc_toolchain_identifier}": cc_toolchain_identifier,
             "%{compile_flags}": get_starlark_list(
                 [
diff --git cc/private/toolchain/unix_cc_toolchain_config.bzl cc/private/toolchain/unix_cc_toolchain_config.bzl
index c3cf3ba..1744eb4 100644
--- cc/private/toolchain/unix_cc_toolchain_config.bzl
+++ cc/private/toolchain/unix_cc_toolchain_config.bzl
@@ -25,6 +25,7 @@ load(
     "variable_with_value",
     "with_feature_set",
 )
+load("@rules_cuda//cuda:toolchain.bzl", "cuda_toolchain_config")
 
 all_compile_actions = [
     ACTION_NAMES.c_compile,
@@ -580,7 +581,8 @@ def _impl(ctx):
                 ],
                 flag_groups = [
                     flag_group(
-                        flags = ["-iquote", "%{quote_include_paths}"],
+                        # -isystem because there is an nvcc thing where it doesn't forward -iquote to host compiler.
+                        flags = ["-isystem", "%{quote_include_paths}"],
                         iterate_over = "quote_include_paths",
                     ),
                     flag_group(
@@ -1152,10 +1154,15 @@ def _impl(ctx):
             unfiltered_compile_flags_feature,
         ]
 
+    cuda = cuda_toolchain_config(
+        cuda_toolchain_info = ctx.attr._cuda_toolchain_info,
+        compiler_path = ctx.attr.tool_paths["gcc"],
+    )
+
     return cc_common.create_cc_toolchain_config_info(
         ctx = ctx,
-        features = features,
-        action_configs = action_configs,
+        features = features + cuda.features,
+        action_configs = action_configs + cuda.action_configs,
         cxx_builtin_include_directories = ctx.attr.cxx_builtin_include_directories,
         toolchain_identifier = ctx.attr.toolchain_identifier,
         host_system_name = ctx.attr.host_system_name,
@@ -1192,6 +1199,9 @@ cc_toolchain_config = rule(
         "tool_paths": attr.string_dict(),
         "toolchain_identifier": attr.string(mandatory = True),
         "unfiltered_compile_flags": attr.string_list(),
+        "_cuda_toolchain_info": attr.label(
+            default = Label("@rules_cuda//cuda:cuda_toolchain_info"),
+        ),
     },
     provides = [CcToolchainConfigInfo],
 )

```



## High-Level Overview

This file is part of the PyTorch framework located at `tools/rules_cc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/rules_cc`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/rules_cc`):



## Cross-References

- **File Documentation**: `cuda_support.patch_docs.md`
- **Keyword Index**: `cuda_support.patch_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/rules_cc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/rules_cc`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/rules_cc`):

- [`cuda_support.patch_kw.md_docs.md`](./cuda_support.patch_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cuda_support.patch_docs.md_docs.md`
- **Keyword Index**: `cuda_support.patch_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
