# Documentation: `docs/torch/utils/_get_clean_triton.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/_get_clean_triton.py_docs.md`
- **Size**: 9,573 bytes (9.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/_get_clean_triton.py`

## File Metadata

- **Path**: `torch/utils/_get_clean_triton.py`
- **Size**: 6,985 bytes (6.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def remove_triton_function_declaration(source_code: str) -> str:
    remove_head = re.sub(r"(\n.+\s\'\'\'\n)", "\n", source_code)
    remove_tail = re.sub(r"(\'\'\'\,.+)", "\n", remove_head)
    return remove_tail


def remove_async_compile(source_code: str) -> str:
    remove_top_level = str.replace(source_code, "async_compile = AsyncCompile()", "")
    remove_compile = str.replace(remove_top_level, "async_compile.wait(globals())", "")
    remove_del = str.replace(remove_compile, "del async_compile", "")
    return remove_del


def rename_kernels(source_code: str) -> str:
    pattern = r"(\w+)\s*=\s*async_compile\.triton\('triton_',\s"
    triton_kernel_decl = "def triton_"
    matches = [
        (match.end(), match.group(1))
        for match in re.finditer(pattern, source_code, re.DOTALL)
    ]

    # Starting from the last match to avoid issues with shifting indices after replacements
    for end_index, captured_string in reversed(matches):
        # Find the index of the next "B" after the current match
        index_of_B = source_code.find(triton_kernel_decl, end_index)
        if index_of_B != -1:
            # Replace the triton_kernel_decl with the captured string
            source_code = (
                source_code[:index_of_B]
                + f"def {captured_string}"
                + source_code[index_of_B + len(triton_kernel_decl) :]
            )
        else:
            # If triton_kernel_decl is not found after the current match, continue to the next
            continue

    return source_code


def merge_params(original_params: list[str], new_params: list[str]) -> list[str]:
    for idx in range(len(new_params)):
        if new_params[idx] == "T":
            new_params[idx] = original_params[idx]
    return new_params


def add_launch_params(
    original: str, kernel_to_params: dict[str, tuple[str, str]]
) -> str:
    # Regex to match the function call in the original string
    pattern = r"(\w+)\.run\((.*)\)"

    def replace(match) -> str:
        # Extract parts from the regex match
        func_name = match.group(1)
        params = match.group(2)
        new_params, grid = kernel_to_params[func_name]
        new_params = merge_params(params.split(", "), new_params.split(", "))

        # Format the new function call
        new_string = f"{func_name}[{grid}]({', '.join(new_params)})"
        return new_string

    transformed = re.sub(pattern, replace, original)

    remove_inductor_wrappers = re.sub(
        r"@triton_heuristics[^@]*@triton.jit",
        r"@triton.jit",
        transformed,
        flags=re.DOTALL,
    )

    return remove_inductor_wrappers


def process_file(
    input_filename: str, output_filename: str, auto_generate_params: bool = True
) -> str:
    with open(input_filename) as file:
        source_code = file.read()

    transformed_code = source_code
    if "def triton_(" in source_code:
        raise RuntimeError(
            "Need to run original Pytorch code generating kernels with TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1"
        )
    # transformed_code = rename_kernels(transformed_code)
    transformed_code = remove_triton_function_declaration(transformed_code)
    transformed_code = remove_async_compile(transformed_code)

    launch_params_filename = f"{input_filename}.launch_params"

    # Auto-generate launch_params if they don't exist and auto_generate_params is True
    if not os.path.exists(launch_params_filename) and auto_generate_params:
        print(f"Launch params file {launch_params_filename} not found. Generating...")
        try:
            # Set environment variable and run the input file
            env = os.environ.copy()
            env["TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"] = "1"

            result = subprocess.run(
                [sys.executable, input_filename],
                env=env,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(input_filename) or ".",
            )

            if result.returncode != 0:
                print(f"Error running {input_filename}:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                raise RuntimeError(
                    f"Failed to generate launch params. Command failed with return code {result.returncode}"
                )

            print(f"Successfully generated {launch_params_filename}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to generate launch params by running {input_filename}: {str(e)}"
            ) from e

    if not os.path.exists(launch_params_filename):
        raise RuntimeError(
            f"Missing {launch_params_filename}. Run `TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1 python {input_filename}` first."
        )

    with open(launch_params_filename) as f:
        launch_params_meta = f.readlines()

    split_params = [i.split("|") for i in launch_params_meta]
    kernel_args_grid = {a.strip(): (b.strip(), c.strip()) for a, b, c in split_params}
    transformed_code = add_launch_params(transformed_code, kernel_args_grid)

    with open(output_filename, "w") as file:
        file.write(transformed_code)
    print(f"Successfully generated {output_filename}")
    return transformed_code


def get_clean_triton(
    input_path: Path,
    output_path: Path = Path("triton_only_repro.py"),
    auto_generate_params: bool = True,
):
    """Run experiments and output results to file

    Args:
        input_path (Optional[Path]): Path to inductor generated output codede
        output_path (Optional[Path]): Path to write out the new python file
        auto_generate_params (bool): Whether to automatically generate launch_params if missing
    """
    return process_file(str(input_path), str(output_path), auto_generate_params)


if __name__ == "__main__":
    """Sample usage:
    # Running sweep
    python _get_clean_triton.py output_code.py

    # To disable auto-generation of launch params:
    python _get_clean_triton.py output_code.py --no-auto-generate
    """
    parser = argparse.ArgumentParser(
        description="Clean Inductor generated code to remove Inductor dependencies"
    )

    # Add the arguments
    parser.add_argument(
        "input_path", type=Path, help="Path to inductor generated output code"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("triton_only_repro.py"),
        help="Path to write out the clean triton output",
    )
    parser.add_argument(
        "--no-auto-generate",
        action="store_true",
        help="Disable automatic generation of launch_params file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    result = get_clean_triton(
        args.input_path, args.output_path, not args.no_auto_generate
    )

```



## High-Level Overview


This Python file contains 0 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `remove_triton_function_declaration`, `remove_async_compile`, `rename_kernels`, `merge_params`, `add_launch_params`, `replace`, `process_file`, `triton_`, `get_clean_triton`

**Key imports**: argparse, os, re, subprocess, sys, Path


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `os`
- `re`
- `subprocess`
- `sys`
- `pathlib`: Path


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/utils`):

- [`_zip.py_docs.md`](./_zip.py_docs.md)
- [`weak.py_docs.md`](./weak.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_cpp_embed_headers.py_docs.md`](./_cpp_embed_headers.py_docs.md)
- [`_cpp_extension_versioner.py_docs.md`](./_cpp_extension_versioner.py_docs.md)
- [`module_tracker.py_docs.md`](./module_tracker.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`_content_store.py_docs.md`](./_content_store.py_docs.md)
- [`_triton.py_docs.md`](./_triton.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `_get_clean_triton.py_docs.md`
- **Keyword Index**: `_get_clean_triton.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/utils`):

- [`show_pickle.py_docs.md_docs.md`](./show_pickle.py_docs.md_docs.md)
- [`file_baton.py_docs.md_docs.md`](./file_baton.py_docs.md_docs.md)
- [`_filelock.py_kw.md_docs.md`](./_filelock.py_kw.md_docs.md)
- [`_config_module.py_docs.md_docs.md`](./_config_module.py_docs.md_docs.md)
- [`cpp_extension.py_docs.md_docs.md`](./cpp_extension.py_docs.md_docs.md)
- [`checkpoint.py_docs.md_docs.md`](./checkpoint.py_docs.md_docs.md)
- [`module_tracker.py_kw.md_docs.md`](./module_tracker.py_kw.md_docs.md)
- [`dlpack.py_docs.md_docs.md`](./dlpack.py_docs.md_docs.md)
- [`_import_utils.py_kw.md_docs.md`](./_import_utils.py_kw.md_docs.md)
- [`_traceback.py_kw.md_docs.md`](./_traceback.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_get_clean_triton.py_docs.md_docs.md`
- **Keyword Index**: `_get_clean_triton.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
