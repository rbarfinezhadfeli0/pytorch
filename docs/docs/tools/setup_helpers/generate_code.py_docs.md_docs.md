# Documentation: `docs/tools/setup_helpers/generate_code.py_docs.md`

## File Metadata

- **Path**: `docs/tools/setup_helpers/generate_code.py_docs.md`
- **Size**: 11,934 bytes (11.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/setup_helpers/generate_code.py`

## File Metadata

- **Path**: `tools/setup_helpers/generate_code.py`
- **Size**: 9,133 bytes (8.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

import yaml


try:
    # use faster C loader if available
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeLoader as YamlLoader  # type: ignore[assignment, misc]


NATIVE_FUNCTIONS_PATH = "aten/src/ATen/native/native_functions.yaml"
TAGS_PATH = "aten/src/ATen/native/tags.yaml"


def generate_code(
    gen_dir: Path,
    native_functions_path: str | None = None,
    tags_path: str | None = None,
    install_dir: str | None = None,
    subset: str | None = None,
    disable_autograd: bool = False,
    force_schema_registration: bool = False,
    operator_selector: Any = None,
) -> None:
    from tools.autograd.gen_annotated_fn_args import gen_annotated
    from tools.autograd.gen_autograd import gen_autograd, gen_autograd_python

    from torchgen.selective_build.selector import SelectiveBuilder

    # Build ATen based Variable classes
    if install_dir is None:
        install_dir = os.fspath(gen_dir / "torch/csrc")
        python_install_dir = os.fspath(gen_dir / "torch/testing/_internal/generated")
    else:
        python_install_dir = install_dir
    autograd_gen_dir = os.path.join(install_dir, "autograd", "generated")
    for d in (autograd_gen_dir, python_install_dir):
        os.makedirs(d, exist_ok=True)
    autograd_dir = os.fspath(Path(__file__).parent.parent / "autograd")

    if subset == "pybindings" or not subset:
        gen_autograd_python(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            tags_path or TAGS_PATH,
            autograd_gen_dir,
            autograd_dir,
        )

    if operator_selector is None:
        operator_selector = SelectiveBuilder.get_nop_selector()

    if subset == "libtorch" or not subset:
        gen_autograd(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            tags_path or TAGS_PATH,
            autograd_gen_dir,
            autograd_dir,
            disable_autograd=disable_autograd,
            operator_selector=operator_selector,
        )

    if subset == "python" or not subset:
        gen_annotated(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            tags_path or TAGS_PATH,
            python_install_dir,
            autograd_dir,
        )


def get_selector_from_legacy_operator_selection_list(
    selected_op_list_path: str,
) -> Any:
    with open(selected_op_list_path) as f:
        # strip out the overload part
        # It's only for legacy config - do NOT copy this code!
        selected_op_list = {
            opname.split(".", 1)[0] for opname in yaml.load(f, Loader=YamlLoader)
        }

    # Internal build doesn't use this flag any more. Only used by OSS
    # build now. Every operator should be considered a root operator
    # (hence generating unboxing code for it, which is consistent with
    # the current behavior), and also be considered as used for
    # training, since OSS doesn't support training on mobile for now.
    #
    is_root_operator = True
    is_used_for_training = True

    from torchgen.selective_build.selector import SelectiveBuilder

    selector = SelectiveBuilder.from_legacy_op_registration_allow_list(
        selected_op_list,
        is_root_operator,
        is_used_for_training,
    )

    return selector


def get_selector(
    selected_op_list_path: str | None,
    operators_yaml_path: str | None,
) -> Any:
    # cwrap depends on pyyaml, so we can't import it earlier
    REPO_ROOT = Path(__file__).absolute().parents[2]
    sys.path.insert(0, str(REPO_ROOT))

    from torchgen.selective_build.selector import SelectiveBuilder

    assert not (
        selected_op_list_path is not None and operators_yaml_path is not None
    ), (
        "Expected at most one of selected_op_list_path and "
        + "operators_yaml_path to be set."
    )

    if selected_op_list_path is None and operators_yaml_path is None:
        return SelectiveBuilder.get_nop_selector()
    elif selected_op_list_path is not None:
        return get_selector_from_legacy_operator_selection_list(selected_op_list_path)
    else:
        return SelectiveBuilder.from_yaml_path(cast(str, operators_yaml_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Autogenerate code")
    parser.add_argument("--native-functions-path")
    parser.add_argument("--tags-path")
    parser.add_argument(
        "--gen-dir",
        type=Path,
        default=Path("."),
        help="Root directory where to install files. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--install-dir",
        "--install_dir",
        help=(
            "Deprecated. Use --gen-dir instead. The semantics are different, do not change "
            "blindly."
        ),
    )
    parser.add_argument(
        "--subset",
        help='Subset of source files to generate. Can be "libtorch" or "pybindings". Generates both when omitted.',
    )
    parser.add_argument(
        "--disable-autograd",
        default=False,
        action="store_true",
        help="It can skip generating autograd related code when the flag is set",
    )
    parser.add_argument(
        "--selected-op-list-path",
        help="Path to the YAML file that contains the list of operators to include for custom build.",
    )
    parser.add_argument(
        "--operators-yaml-path",
        "--operators_yaml_path",
        help="Path to the model YAML file that contains the list of operators to include for custom build.",
    )
    parser.add_argument(
        "--force-schema-registration",
        "--force_schema_registration",
        action="store_true",
        help="force it to generate schema-only registrations for ops that are not"
        "listed on --selected-op-list",
    )
    parser.add_argument(
        "--gen-lazy-ts-backend",
        "--gen_lazy_ts_backend",
        action="store_true",
        help="Enable generation of the torch::lazy TorchScript backend",
    )
    parser.add_argument(
        "--per-operator-headers",
        "--per_operator_headers",
        action="store_true",
        help="Build lazy tensor ts backend with per-operator ATen headers, must match how ATen was built",
    )
    options = parser.parse_args()

    # Path: aten/src/ATen
    aten_path = os.path.dirname(os.path.dirname(options.native_functions_path))
    operator_selector = get_selector(
        options.selected_op_list_path, options.operators_yaml_path
    )

    generate_code(
        options.gen_dir,
        options.native_functions_path,
        options.tags_path,
        options.install_dir,
        options.subset,
        options.disable_autograd,
        options.force_schema_registration,
        # options.selected_op_list
        operator_selector=operator_selector,
    )

    # Generate the python bindings for functionalization's `ViewMeta` classes.
    from torchgen.gen_functionalization_type import (
        gen_functionalization_view_meta_classes,
    )

    functionalization_templates_dir = os.path.join(aten_path, "templates")
    install_dir = options.install_dir or os.fspath(options.gen_dir / "torch/csrc")
    functionalization_install_dir = os.path.join(
        install_dir, "functionalization", "generated"
    )

    os.makedirs(functionalization_install_dir, exist_ok=True)
    assert os.path.isdir(functionalization_install_dir)
    assert os.path.isdir(functionalization_templates_dir)

    gen_functionalization_view_meta_classes(
        options.native_functions_path or NATIVE_FUNCTIONS_PATH,
        options.tags_path or TAGS_PATH,
        selector=operator_selector,
        install_dir=functionalization_install_dir,
        template_dir=functionalization_templates_dir,
    )

    if options.gen_lazy_ts_backend:
        ts_backend_yaml = os.path.join(aten_path, "native/ts_native_functions.yaml")
        ts_native_functions = "torch/csrc/lazy/ts_backend/ts_native_functions.cpp"
        ts_node_base = "torch/csrc/lazy/ts_backend/ts_node.h"
        lazy_install_dir = os.path.join(install_dir, "lazy", "generated")
        os.makedirs(lazy_install_dir, exist_ok=True)

        assert os.path.isfile(ts_backend_yaml), (
            f"Unable to access ts_backend_yaml: {ts_backend_yaml}"
        )
        assert os.path.isfile(ts_native_functions), (
            f"Unable to access {ts_native_functions}"
        )
        from torchgen.dest.lazy_ir import GenTSLazyIR
        from torchgen.gen_lazy_tensor import run_gen_lazy_tensor

        run_gen_lazy_tensor(
            aten_path=aten_path,
            source_yaml=ts_backend_yaml,
            backend_name="TorchScript",
            output_dir=lazy_install_dir,
            dry_run=False,
            impl_path=ts_native_functions,
            node_base="TsNode",
            node_base_hdr=ts_node_base,
            build_in_tree=True,
            lazy_ir_generator=GenTSLazyIR,
            per_operator_headers=options.per_operator_headers,
            gen_forced_fallback_code=True,
        )


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `generate_code`, `get_selector_from_legacy_operator_selection_list`, `get_selector`, `main`

**Key imports**: annotations, argparse, os, sys, Path, Any, cast, yaml, CSafeLoader as YamlLoader, SafeLoader as YamlLoader  , gen_annotated


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/setup_helpers`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `os`
- `sys`
- `pathlib`: Path
- `typing`: Any, cast
- `yaml`
- `tools.autograd.gen_annotated_fn_args`: gen_annotated
- `tools.autograd.gen_autograd`: gen_autograd, gen_autograd_python
- `torchgen.selective_build.selector`: SelectiveBuilder
- `it earlier`
- `torchgen.dest.lazy_ir`: GenTSLazyIR
- `torchgen.gen_lazy_tensor`: run_gen_lazy_tensor


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`tools/setup_helpers`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`gen_version_header.py_docs.md`](./gen_version_header.py_docs.md)
- [`gen_unboxing.py_docs.md`](./gen_unboxing.py_docs.md)
- [`env.py_docs.md`](./env.py_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`cmake_utils.py_docs.md`](./cmake_utils.py_docs.md)
- [`cmake.py_docs.md`](./cmake.py_docs.md)
- [`gen.py_docs.md`](./gen.py_docs.md)


## Cross-References

- **File Documentation**: `generate_code.py_docs.md`
- **Keyword Index**: `generate_code.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/setup_helpers`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/setup_helpers`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/tools/setup_helpers`):

- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`gen_version_header.py_kw.md_docs.md`](./gen_version_header.py_kw.md_docs.md)
- [`cmake_utils.py_docs.md_docs.md`](./cmake_utils.py_docs.md_docs.md)
- [`gen.py_kw.md_docs.md`](./gen.py_kw.md_docs.md)
- [`generate_code.py_kw.md_docs.md`](./generate_code.py_kw.md_docs.md)
- [`build.bzl_kw.md_docs.md`](./build.bzl_kw.md_docs.md)
- [`cmake_utils.py_kw.md_docs.md`](./cmake_utils.py_kw.md_docs.md)
- [`gen_unboxing.py_kw.md_docs.md`](./gen_unboxing.py_kw.md_docs.md)
- [`env.py_docs.md_docs.md`](./env.py_docs.md_docs.md)
- [`env.py_kw.md_docs.md`](./env.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `generate_code.py_docs.md_docs.md`
- **Keyword Index**: `generate_code.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
