# Documentation: `docs/test/cpp_api_parity/functional_impl_check.py_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_api_parity/functional_impl_check.py_docs.md`
- **Size**: 14,514 bytes (14.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp_api_parity/functional_impl_check.py`

## File Metadata

- **Path**: `test/cpp_api_parity/functional_impl_check.py`
- **Size**: 11,624 bytes (11.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# The purpose of this test is to check that we have implementation parity between
# a Python `torch.nn.functional` function and its corresponding C++ `torch::nn::functional`
# function. Concretely, this test does the following:
#
# 1. Get a test params dict from common_nn.py, run forward pass on the Python functional
# created using the test params.
#
# 2. Serialize the Python functional's forward input arguments, deserialize them
# in C++ and use them as input for the C++ functional's forward pass.
#
# 3. Run the forward pass on the C++ functional, and serialize the C++ functional's
# forward output.
#
# 4. Compare Python/C++ functional's forward output. If they are the same, then we
# have implementation parity between Python/C++ module.

import os
import pprint
import re
import tempfile
from string import Template

import torch
from cpp_api_parity.sample_functional import SAMPLE_FUNCTIONAL_CPP_SOURCE
from cpp_api_parity.utils import (
    add_test,
    compile_cpp_code_inline,
    compute_arg_dict,
    compute_cpp_args_construction_stmts_and_forward_arg_symbols,
    compute_temp_file_path,
    decorate_test_fn,
    generate_error_msg,
    is_torch_nn_functional_test,
    move_python_tensors_to_device,
    serialize_arg_dict_as_script_module,
    set_python_tensors_requires_grad,
    TORCH_NN_COMMON_TEST_HARNESS,
    TorchNNFunctionalTestParams,
    try_remove_folder,
)


# Expected substitutions:
#
# ${functional_variant_name}  (e.g. `BCELoss_no_reduce`)
# ${cpp_args_construction_stmts}
# ${cpp_function_call}
TORCH_NN_FUNCTIONAL_TEST_FORWARD = Template(
    """
void ${functional_variant_name}_test_forward(
    const std::string& arg_dict_file_path,
    const std::string& forward_output_file_path) {
  pybind11::gil_scoped_release no_gil;

  namespace F = torch::nn::functional;

  // Declare arguments
  auto arg_dict = load_dict_from_file(arg_dict_file_path);
  ${cpp_args_construction_stmts};

  // Some functionals (such as `F::rrelu`) create random tensors in their call path.
  // To make sure the random tensors created are the same in Python/C++, we need
  // to set the RNG seed manually.
  torch::manual_seed(0);

  // Run function with arguments
  auto cpp_output = ${cpp_function_call};

  // Save the output into a file to be compared in Python later
  write_ivalue_to_file(torch::IValue(cpp_output), forward_output_file_path);
}
"""
)


def run_forward(unit_test_class, test_params):
    device = test_params.device

    inputs = set_python_tensors_requires_grad(
        move_python_tensors_to_device(
            [arg_value for _, arg_value in test_params.arg_dict["input"]], device
        )
    )
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict["target"]], device
    )
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict["extra_args"]], device
    )

    # Some functionals (such as `F.rrelu`) create random tensors in their call path.
    # To make sure the random tensors created are the same in Python/C++, we need
    # to set the RNG seed manually.
    torch.manual_seed(0)
    python_output = test_params.test_instance.constructor()(*inputs)

    return python_output


def test_forward(unit_test_class, test_params):
    functional_variant_name = test_params.functional_variant_name
    cpp_tmp_folder = test_params.cpp_tmp_folder
    # Remove the temporary folder if it exists already
    try_remove_folder(cpp_tmp_folder)
    os.mkdir(cpp_tmp_folder)

    # Run forward on Python functional
    python_output = run_forward(unit_test_class, test_params)

    # Save Python arguments to be used from C++ function
    arg_dict_file_path = compute_temp_file_path(
        cpp_tmp_folder, functional_variant_name, "arg_dict"
    )
    serialize_arg_dict_as_script_module(test_params.arg_dict).save(arg_dict_file_path)

    cpp_test_name = f"{test_params.functional_variant_name}_test_forward"
    cpp_test_fn = getattr(
        unit_test_class.functional_impl_check_cpp_module, cpp_test_name
    )

    def run_cpp_test_fn_and_check_output():
        forward_output_file_path = compute_temp_file_path(
            cpp_tmp_folder, functional_variant_name, "forward_output"
        )

        cpp_test_fn(arg_dict_file_path, forward_output_file_path)
        cpp_output = torch.load(forward_output_file_path)

        # Check that forward outputs are equal
        unit_test_class.assertEqual(
            python_output,
            cpp_output,
            msg=generate_error_msg("forward output", cpp_output, python_output),
        )

    run_cpp_test_fn_and_check_output()

    # Remove temporary folder that stores C++ outputs
    try_remove_folder(cpp_tmp_folder)


def compute_functional_name(test_params_dict):
    def camel_case_to_snake_case(camel_case_str):
        return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_str).lower()

    if "cpp_options_args" in test_params_dict:
        # Expected format for `cpp_options_args`: `F::FunctionalFuncOptions(...)`
        # Example output: `binary_cross_entropy`
        return camel_case_to_snake_case(
            test_params_dict["cpp_options_args"]
            .split("(")[0]
            .replace("F::", "")
            .replace("FuncOptions", "")
        )
    elif "cpp_function_call" in test_params_dict:
        # Expected format for `cpp_function_call`: `F::functional_name(...)`
        # Example output: `binary_cross_entropy`
        return test_params_dict["cpp_function_call"].split("(")[0].replace("F::", "")
    else:
        raise RuntimeError(
            "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n"
            f"{pprint.pformat(test_params_dict)}"
        )


def compute_cpp_function_call(test_params_dict, arg_dict, functional_name):
    if "cpp_function_call" in test_params_dict:
        return test_params_dict["cpp_function_call"]
    elif "cpp_options_args" in test_params_dict:
        cpp_forward_args_symbols = [
            arg_name
            for arg_name, _ in arg_dict["input"]
            + arg_dict["target"]
            + arg_dict["extra_args"]
        ]
        return "F::{}({}, {})".format(
            functional_name,
            ", ".join(cpp_forward_args_symbols),
            test_params_dict["cpp_options_args"],
        )
    else:
        raise RuntimeError(
            "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n"
            f"{pprint.pformat(test_params_dict)}"
        )


def process_test_params_for_functional(test_params_dict, device, test_instance_class):
    test_instance = test_instance_class(**test_params_dict)
    functional_name = compute_functional_name(test_params_dict)
    assert test_instance.get_name().startswith("test_")
    # Example output: `BCELoss_no_reduce_cuda`
    functional_variant_name = test_instance.get_name()[5:] + (
        ("_" + device) if device != "cpu" else ""
    )
    arg_dict = compute_arg_dict(test_params_dict, test_instance)

    return TorchNNFunctionalTestParams(
        functional_name=functional_name,
        functional_variant_name=functional_variant_name,
        test_instance=test_instance,
        cpp_function_call=compute_cpp_function_call(
            test_params_dict, arg_dict, functional_name
        ),
        arg_dict=arg_dict,
        has_parity=test_params_dict.get("has_parity", True),
        device=device,
        cpp_tmp_folder=tempfile.mkdtemp(),
    )


def write_test_to_test_class(
    unit_test_class, test_params_dict, test_instance_class, parity_table, devices
):
    assert is_torch_nn_functional_test(test_params_dict)

    assert (
        "cpp_options_args" in test_params_dict
        or "cpp_function_call" in test_params_dict
    ), (
        "To enable C++ API parity test, "
        "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n"
        f"{pprint.pformat(test_params_dict)}. \n"
        "If you are interested in adding the C++ API parity test, please see:\n"
        "NOTE [How to check NN module / functional API parity between Python and C++ frontends]. \n"
        "If not, please add `test_cpp_api_parity=False` to the test params dict and file an issue about this."
    )

    assert not (
        "cpp_options_args" in test_params_dict
        and "cpp_function_call" in test_params_dict
    ), (
        "Only one of `cpp_options_args` and `cpp_function_call` entries "
        f"should be present in test params dict:\n{pprint.pformat(test_params_dict)}"
    )

    functional_name = compute_functional_name(test_params_dict)

    assert hasattr(torch.nn.functional, functional_name), (
        f"`torch.nn.functional` doesn't have function `{functional_name}`. "
        f"(Discovered while processing\n{pprint.pformat(test_params_dict)}.)"
    )

    functional_full_name = "F::" + functional_name

    assert functional_full_name in parity_table["torch::nn::functional"], (
        f"Please add `{functional_full_name}` entry to `torch::nn::functional` "
        "section of `test/cpp_api_parity/parity-tracker.md`. "
        f"(Discovered while processing\n{pprint.pformat(test_params_dict)}.)"
    )

    for device in devices:
        test_params = process_test_params_for_functional(
            test_params_dict=test_params_dict,
            device=device,
            test_instance_class=test_instance_class,
        )
        try_remove_folder(test_params.cpp_tmp_folder)
        unit_test_name = (
            f"test_torch_nn_functional_{test_params.functional_variant_name}"
        )
        unit_test_class.functional_test_params_map[unit_test_name] = test_params

        def test_fn(self):
            test_forward(
                unit_test_class=self,
                test_params=unit_test_class.functional_test_params_map[
                    self._testMethodName
                ],
            )

        test_fn = decorate_test_fn(
            test_fn=test_fn,
            test_cuda=test_params_dict.get("test_cuda", True),
            has_impl_parity=parity_table["torch::nn::functional"][functional_full_name][
                0
            ]
            and test_params_dict.get("has_parity", True),
            device=device,
        )

        add_test(unit_test_class, unit_test_name, test_fn)


def generate_test_cpp_sources(test_params, template):
    (
        cpp_args_construction_stmts,
        _,
    ) = compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params)

    test_cpp_sources = template.substitute(
        functional_variant_name=test_params.functional_variant_name,
        cpp_args_construction_stmts=";\n  ".join(cpp_args_construction_stmts),
        cpp_function_call=test_params.cpp_function_call,
    )
    return test_cpp_sources


# Build all C++ tests together, instead of once per test.
def build_cpp_tests(unit_test_class, print_cpp_source=False):
    assert len(unit_test_class.functional_test_params_map) > 0
    cpp_sources = TORCH_NN_COMMON_TEST_HARNESS + SAMPLE_FUNCTIONAL_CPP_SOURCE
    functions = []
    for test_params in unit_test_class.functional_test_params_map.values():
        cpp_sources += generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_FUNCTIONAL_TEST_FORWARD
        )
        functions.append(f"{test_params.functional_variant_name}_test_forward")
    if print_cpp_source:
        print(cpp_sources)

    cpp_module = compile_cpp_code_inline(
        name="functional_impl_check", cpp_sources=cpp_sources, functions=functions
    )
    unit_test_class.functional_impl_check_cpp_module = cpp_module

```



## High-Level Overview

"""void ${functional_variant_name}_test_forward(

This Python file contains 0 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `run_forward`, `test_forward`, `run_cpp_test_fn_and_check_output`, `compute_functional_name`, `camel_case_to_snake_case`, `compute_cpp_function_call`, `process_test_params_for_functional`, `write_test_to_test_class`, `test_fn`, `generate_test_cpp_sources`, `build_cpp_tests`

**Key imports**: os, pprint, re, tempfile, Template, torch, SAMPLE_FUNCTIONAL_CPP_SOURCE


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_api_parity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `pprint`
- `re`
- `tempfile`
- `string`: Template
- `torch`
- `cpp_api_parity.sample_functional`: SAMPLE_FUNCTIONAL_CPP_SOURCE


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python test/cpp_api_parity/functional_impl_check.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_api_parity`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`sample_functional.py_docs.md`](./sample_functional.py_docs.md)
- [`module_impl_check.py_docs.md`](./module_impl_check.py_docs.md)
- [`sample_module.py_docs.md`](./sample_module.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`parity-tracker.md_docs.md`](./parity-tracker.md_docs.md)
- [`parity_table_parser.py_docs.md`](./parity_table_parser.py_docs.md)


## Cross-References

- **File Documentation**: `functional_impl_check.py_docs.md`
- **Keyword Index**: `functional_impl_check.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_api_parity`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_api_parity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python docs/test/cpp_api_parity/functional_impl_check.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_api_parity`):

- [`parity_table_parser.py_docs.md_docs.md`](./parity_table_parser.py_docs.md_docs.md)
- [`module_impl_check.py_docs.md_docs.md`](./module_impl_check.py_docs.md_docs.md)
- [`module_impl_check.py_kw.md_docs.md`](./module_impl_check.py_kw.md_docs.md)
- [`parity-tracker.md_kw.md_docs.md`](./parity-tracker.md_kw.md_docs.md)
- [`sample_module.py_kw.md_docs.md`](./sample_module.py_kw.md_docs.md)
- [`parity_table_parser.py_kw.md_docs.md`](./parity_table_parser.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`sample_module.py_docs.md_docs.md`](./sample_module.py_docs.md_docs.md)
- [`sample_functional.py_docs.md_docs.md`](./sample_functional.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `functional_impl_check.py_docs.md_docs.md`
- **Keyword Index**: `functional_impl_check.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
