# Documentation: `docs/tools/test/test_codegen_model.py_docs.md`

## File Metadata

- **Path**: `docs/tools/test/test_codegen_model.py_docs.md`
- **Size**: 11,028 bytes (10.77 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/test/test_codegen_model.py`

## File Metadata

- **Path**: `tools/test/test_codegen_model.py`
- **Size**: 7,399 bytes (7.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: codegen"]

import textwrap
import unittest
from typing import cast

import expecttest
import yaml

import torchgen.dest as dest
import torchgen.gen as gen
from torchgen.gen import LineLoader, parse_native_yaml_struct
from torchgen.model import (
    Annotation,
    BaseOperatorName,
    CustomClassType,
    DispatchKey,
    NativeFunctionsGroup,
    Type,
)


class TestCodegenModel(expecttest.TestCase):
    def assertParseErrorInline(self, yaml_str: str, expect: str) -> None:
        es = yaml.load(yaml_str, Loader=LineLoader)
        try:
            parse_native_yaml_struct(es, set())
        except AssertionError as e:
            # hack to strip out the context
            msg, _ = str(e).split("  in ", 2)
            self.assertExpectedInline("\n".join(textwrap.wrap(msg)), expect, skip=1)
            return
        self.fail(msg="Did not raise when expected to")

    def assertUfuncErrorInline(self, yaml_str: str, expect: str) -> None:
        # parse a single structured group out of the yaml to g
        es = yaml.load(yaml_str, Loader=LineLoader)
        parsed_yaml = parse_native_yaml_struct(es, set())
        native_functions, backend_indices = (
            parsed_yaml.native_functions,
            parsed_yaml.backend_indices,
        )
        grouped_native_functions = gen.get_grouped_native_functions(native_functions)
        assert len(grouped_native_functions) == 1
        g = grouped_native_functions[0]
        assert isinstance(g, NativeFunctionsGroup)
        assert g.out.ufunc_inner_loop
        # this is not ufunc codegen per se, but it does some basic sanity tests for
        # ufunc generation
        gen.compute_meta_function_declaration(g)
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CPU])
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CUDA])
        try:
            # the real kahuna
            dest.compute_ufunc_cpu(g)
            dest.compute_ufunc_cpu_kernel(g)
            dest.compute_ufunc_cuda(g)
        except AssertionError as e:
            # hack to strip out the context
            msg, _ = str(e).split("  in ", 2)
            self.assertExpectedInline("\n".join(textwrap.wrap(msg)), expect, skip=1)
            return
        self.fail(msg="Did not raise when expected to")

    # NB: indent is hardcoded to be two here, so format your yaml accordingly
    binop_out = (
        "func: binop.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"
    )
    ti_binop_out = f"""{binop_out}
  structured: True
  structured_inherits: TensorIteratorBase"""
    ti_binop = """func: binop(Tensor self, Tensor other) -> Tensor
  structured_delegate: binop.out
"""

    ti_unop_out = """func: unop.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase"""
    ti_unop = """func: unop(Tensor self) -> Tensor
  structured_delegate: unop.out
"""

    def test_nonstructured_ufunc(self) -> None:
        yaml_str = f"""\
- {self.binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
"""
        self.assertParseErrorInline(
            yaml_str,
            """\
ufunc must be structured""",
        )

    def test_overlapping_ufunc_and_dispatch(self) -> None:
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
  dispatch:
    CPU: binop_cpu
"""
        self.assertParseErrorInline(
            yaml_str,
            """\
ufunc should not have explicit dispatch entry for CPU""",
        )

    # See https://github.com/pytorch/pytorch/pull/65851#discussion_r810238456
    @unittest.expectedFailure
    def test_scalaronly_shadowed(self) -> None:
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
    ScalarOnly: binop (Bool)
"""
        self.assertParseErrorInline(
            yaml_str,
            """\
""",
        )

    def test_conflicting_ufunc(self) -> None:
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
    ScalarOnly: binop_scalar (Bool)
- {self.ti_binop}
"""
        self.assertUfuncErrorInline(
            yaml_str,
            """\
ScalarOnly and Generic must have same ufunc name""",
        )

    def test_invalid_cudafunctoronself_for_binary_op(self) -> None:
        yaml_str = f"""\
- {self.ti_unop_out}
  ufunc_inner_loop:
    Generic: unop (All)
    CUDAFunctorOnSelf: unop_self_cuda (All)
- {self.ti_unop}
"""
        self.assertUfuncErrorInline(
            yaml_str,
            """\
cannot use CUDAFunctorOnSelf on non-binary function""",
        )

    def test_parse_custom_class_type(self) -> None:
        custom_class_name = "namespace_foo.class_bar"
        custom_class_name_with_prefix = f"__torch__.torch.classes.{custom_class_name}"
        custom_class_type = cast(
            CustomClassType, Type.parse(custom_class_name_with_prefix)
        )
        self.assertTrue(isinstance(custom_class_type, CustomClassType))
        self.assertEqual(custom_class_name, custom_class_type.class_name)
        self.assertEqual(custom_class_name_with_prefix, str(custom_class_type))


class TestAnnotation(expecttest.TestCase):
    def test_single_alias_no_write(self) -> None:
        a = Annotation.parse("a")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertFalse(a.is_write)
        self.assertEqual(a.alias_set_after, ())

    def test_single_alias_is_write(self) -> None:
        a = Annotation.parse("a!")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, ())

    def test_single_alias_is_write_to_wildcard(self) -> None:
        a = Annotation.parse("a! -> *")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, tuple("*"))

    def test_alias_set(self) -> None:
        a = Annotation.parse("a|b")
        self.assertEqual(a.alias_set, ("a", "b"))

    def test_alias_set_is_write_raises_exception(self) -> None:
        with self.assertRaisesRegex(
            AssertionError, r"alias set larger than 1 is not mutable"
        ):
            Annotation.parse("a|b!")

    def test_single_alias_is_write_to_alias_set(self) -> None:
        a = Annotation.parse("a! -> a|b")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, ("a", "b"))

    def test_before_and_after_alias_set_larger_than_1_raises_exception(self) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            r"before alias set and after alias set cannot be larger than 1 at the same time",
        ):
            Annotation.parse("a|b -> c|d")


class TestBaseOperatorName(expecttest.TestCase):
    def test_base_operator_name_with_ns_has_same_attributes_as_the_one_without_ns(
        self,
    ) -> None:
        op = "aten::__lshift__"
        op_without_ns = "__lshift__"

        op_name = BaseOperatorName.parse(op)
        op_name_without_ns = BaseOperatorName.parse(op_without_ns)

        self.assertEqual(op_name.base, op_name_without_ns.base)
        self.assertEqual(op_name.inplace, op_name_without_ns.inplace)
        self.assertEqual(op_name.dunder_method, op_name_without_ns.dunder_method)


if __name__ == "__main__":
    unittest.main()

```



## High-Level Overview


This Python file contains 3 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCodegenModel`, `TestAnnotation`, `TestBaseOperatorName`

**Functions defined**: `assertParseErrorInline`, `assertUfuncErrorInline`, `test_nonstructured_ufunc`, `test_overlapping_ufunc_and_dispatch`, `test_scalaronly_shadowed`, `test_conflicting_ufunc`, `test_invalid_cudafunctoronself_for_binary_op`, `test_parse_custom_class_type`, `test_single_alias_no_write`, `test_single_alias_is_write`, `test_single_alias_is_write_to_wildcard`, `test_alias_set`, `test_alias_set_is_write_raises_exception`, `test_single_alias_is_write_to_alias_set`, `test_before_and_after_alias_set_larger_than_1_raises_exception`, `test_base_operator_name_with_ns_has_same_attributes_as_the_one_without_ns`

**Key imports**: textwrap, unittest, cast, expecttest, yaml, torchgen.dest as dest, torchgen.gen as gen, LineLoader, parse_native_yaml_struct


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `textwrap`
- `unittest`
- `typing`: cast
- `expecttest`
- `yaml`
- `torchgen.dest as dest`
- `torchgen.gen as gen`
- `torchgen.gen`: LineLoader, parse_native_yaml_struct


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python tools/test/test_codegen_model.py
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

- **File Documentation**: `test_codegen_model.py_docs.md`
- **Keyword Index**: `test_codegen_model.py_kw.md`
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

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/tools/test/test_codegen_model.py_docs.md
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
- [`test_codegen.py_docs.md_docs.md`](./test_codegen.py_docs.md_docs.md)
- [`test_vulkan_codegen.py_kw.md_docs.md`](./test_vulkan_codegen.py_kw.md_docs.md)
- [`test_set_linter.py_docs.md_docs.md`](./test_set_linter.py_docs.md_docs.md)
- [`test_gb_registry_linter.py_kw.md_docs.md`](./test_gb_registry_linter.py_kw.md_docs.md)
- [`test_upload_test_stats.py_kw.md_docs.md`](./test_upload_test_stats.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_codegen_model.py_docs.md_docs.md`
- **Keyword Index**: `test_codegen_model.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
