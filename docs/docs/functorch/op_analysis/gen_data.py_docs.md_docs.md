# Documentation: `docs/functorch/op_analysis/gen_data.py_docs.md`

## File Metadata

- **Path**: `docs/functorch/op_analysis/gen_data.py_docs.md`
- **Size**: 7,637 bytes (7.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `functorch/op_analysis/gen_data.py`

## File Metadata

- **Path**: `functorch/op_analysis/gen_data.py`
- **Size**: 5,697 bytes (5.56 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import csv
from collections import defaultdict

import yaml

import torch


def get_ops_for_key(key):
    # Needs modified PyTorch C++ code to work
    if key is None:
        ops = torch._C._dispatch_get_registrations_for_dispatch_key()
    else:
        ops = torch._C._dispatch_get_registrations_for_dispatch_key(key)
    cleaned_ops = []
    for i in ops:
        if "aten::" not in i:
            continue
        cleaned_ops.append(i[6:].strip())
    return set(cleaned_ops)


def gen_data(special_op_lists, analysis_name):
    all_ops = get_ops_for_key(None)
    composite_ops = get_ops_for_key("CompositeImplicitAutograd")
    noncomposite_ops = all_ops - composite_ops
    with open("../../aten/src/ATen/native/native_functions.yaml") as f:
        ops = yaml.load(f.read(), Loader=yaml.CLoader)

    with open("annotated_ops") as f:
        annotated_ops = {a.strip(): b.strip() for a, b in csv.reader(f)}

    uniq_ops = []
    uniq_names = set()
    overload_types = defaultdict(list)
    cnt = 0
    for op in ops:
        func_str = op["func"]
        name = func_str[: func_str.index("(")]
        if "." in name:
            uniq_name = name[: name.index(".")]
            overload_types[name[name.index(".") + 1 :]].append(name)
        else:
            uniq_name = name
        op["name"] = uniq_name
        full_name = func_str[: func_str.index("(")]
        op["full_name"] = full_name
        ret_type = func_str[func_str.index("->") + 3 :]
        op["ret_type"] = ret_type
        cnt += 1
        if uniq_name in uniq_names:
            continue
        uniq_names.add(uniq_name)
        uniq_ops.append(op)

    def annotate_ops(ops, is_unique):
        categorization = defaultdict(int)
        for op in ops:
            if op["name"][-1] == "_":
                categorization["inplace"] += 1
                op["meta"] = "inplace"
                continue
            if not is_unique and "a!" in op["func"].lower():
                categorization["out"] += 1
                op["meta"] = "out"
                continue
            if "conv" in op["name"]:
                categorization["conv"] += 1
                op["meta"] = "conv"
                continue
            if "pool" in op["name"]:
                categorization["pool"] += 1
                op["meta"] = "pool"
                continue
            if "backward" in op["name"]:
                categorization["backward"] += 1
                op["meta"] = "backward"
                continue
            if op["name"][0] == "_" and op["name"][1] != "_":
                categorization["private"] += 1
                op["meta"] = "private"
                continue
            if "batch_norm" in op["name"]:
                categorization["batch_norm"] += 1
                op["meta"] = "batch_norm"
                continue
            if "Tensor" not in op["func"] or "Tensor" not in op["ret_type"]:
                categorization["non_tensor"] += 1
                op["meta"] = "non_tensor"
                continue
            if (
                "cudnn" in op["name"]
                or "mkldnn" in op["name"]
                or "miopen" in op["name"]
                or "native" in op["name"]
                or "thnn" in op["name"]
                or "slow" in op["name"]
            ):
                categorization["backend"] += 1
                op["meta"] = "backend"
                continue
            if op["name"] in annotated_ops:
                categorization["core"] += 1
                op["meta"] = "core " + annotated_ops[op["name"]]
                continue
            categorization["core"] += 1
            op["meta"] = "core unknown"
        return categorization

    annotate_ops(ops, is_unique=False)
    with open(f"{analysis_name}", "w") as f:
        for op in ops:
            info = [
                op["full_name"],
                op["meta"],
                op["full_name"] not in noncomposite_ops,
            ] + [check(op) for check in special_op_lists]
            f.write(",".join([str(i) for i in info]) + "\n")


def name_check(lst):
    return lambda x: x["name"] in lst


def full_name_check(lst):
    return lambda x: x["full_name"] in lst


# Generates batching rule data
gen_data([full_name_check(get_ops_for_key("FuncTorchBatched"))], "vmap.txt")


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


def remove_prefix(input_string, prefix):
    if prefix and input_string.startswith(prefix):
        return input_string[len(prefix) :]
    return input_string


if True:
    with open("run_ops.txt") as f:
        opinfo_ops = [remove_suffix(i.strip(), ".default") for i in f]
    with open("count_ops.txt") as f:
        opinfo_counts = [i.strip() for i in f]
        opinfo_counts = defaultdict(int, dict(zip(opinfo_ops, opinfo_counts)))

    def count_fn(x):
        return opinfo_counts[x["full_name"]]

    with open("run_decompositions.txt") as f:
        decomposed_ops = [remove_suffix(i.strip(), ".default") for i in f]

    with open("public_api") as f:
        ref_api = [i.strip() for i in f]

    def has_ref_impl(x):
        name = x["name"]
        for prefix in ["linalg_", "special_"]:
            name = remove_prefix(name, prefix)
        prefixes = ["nn.functional", "fft", "special", "linalg"]
        return (
            any(f"{prefix}.{name}" in ref_api for prefix in prefixes) or name in ref_api
        )

    gen_data(
        [
            full_name_check(opinfo_ops),
            full_name_check(decomposed_ops),
            count_fn,
            has_ref_impl,
        ],
        "decompositions.txt",
    )

```



## High-Level Overview


This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_ops_for_key`, `gen_data`, `annotate_ops`, `name_check`, `full_name_check`, `remove_suffix`, `remove_prefix`, `count_fn`, `has_ref_impl`

**Key imports**: csv, defaultdict, yaml, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `functorch/op_analysis`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `csv`
- `collections`: defaultdict
- `yaml`
- `torch`


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

Files in the same folder (`functorch/op_analysis`):



## Cross-References

- **File Documentation**: `gen_data.py_docs.md`
- **Keyword Index**: `gen_data.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/functorch/op_analysis`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/functorch/op_analysis`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/functorch/op_analysis`):

- [`gen_data.py_kw.md_docs.md`](./gen_data.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `gen_data.py_docs.md_docs.md`
- **Keyword Index**: `gen_data.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
