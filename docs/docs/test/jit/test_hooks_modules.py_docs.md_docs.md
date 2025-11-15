# Documentation: `docs/test/jit/test_hooks_modules.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_hooks_modules.py_docs.md`
- **Size**: 21,428 bytes (20.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_hooks_modules.py`

## File Metadata

- **Path**: `test/jit/test_hooks_modules.py`
- **Size**: 18,309 bytes (17.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

from typing import List, Tuple

import torch


class SubmoduleNoForwardInputs(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self):
        assert self.name == "inner_mod_name"


class ModuleNoForwardInputs(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleNoForwardInputs(submodule_name)

    def forward(self):
        self.submodule()


class SubmoduleForwardSingleInput(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def foo(self, input: str):
        return input

    def forward(self, input: str):
        input = input + "_inner_mod"
        input = self.foo(input)
        return input


class ModuleForwardSingleInput(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        input = input + "_outermod"
        return self.submodule(input)


class ModuleDirectforwardSubmodCall(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        input = input + "_outermod"
        return self.submodule.forward(input)


class SuboduleForwardMultipleInputs(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        output2 = input2 + "_"
        return input1, output2


class ModuleForwardMultipleInputs(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SuboduleForwardMultipleInputs(submodule_name)

    def forward(self, input1: List[str], input2: str):
        input1.append(self.name)
        return self.submodule(input1, input2)


class SubmoduleForwardTupleInput(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input: Tuple[int]):
        input_access = input[0]  # noqa: F841
        return (1,)


class ModuleForwardTupleInput(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardTupleInput(submodule_name)

    def forward(self, input: Tuple[int]):
        input_access = input[0]  # noqa: F841
        return self.submodule((1,))


# Modules for JIT forward hook and pre-hooks python and cpp tests
def create_module_no_forward_input():
    # Use to test module level hooks with no forward input
    m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[()]) -> None:
        assert self.name == "outer_mod_name"

    def forward_hook(self, input: Tuple[()], output: None):
        assert self.name == "outer_mod_name"

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_submodule_no_forward_input():
    # Use to test submodule level hooks with no forward input
    m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[()]) -> None:
        assert self.name == "inner_mod_name"

    def forward_hook(self, input: Tuple[()], output: None):
        assert self.name == "inner_mod_name"

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_module_forward_multiple_inputs():
    # Use to test module level hooks with forward having multiple
    # inputs and returns
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "outer_mod_name"
        assert input[0][0] == "a"
        return ["pre_hook_override_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_override_name"
        output2 = output[1] + "fh"
        return output[0], output2

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_multiple_hooks_multiple_inputs():
    # Use to test that module level hooks with multiple inputs execute
    # in correct order and pass correct information between each other
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "outer_mod_name"
        assert input[0][0] == "a"
        return ["pre_hook_override_name"], "pre_hook_override"

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_override_name"
        return ["pre_hook_override_name2"], "pre_hook_override"

    def forward_hook1(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_override_name2"
        output2 = output[1] + "fh1"
        return output[0], output2

    def forward_hook2(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_override_name2"
        assert output[1] == "pre_hook_override_fh1"
        output2 = output[1] + "_fh2"
        return output[0], output2

    m.register_forward_pre_hook(pre_hook1)
    m.register_forward_pre_hook(pre_hook2)
    m.register_forward_hook(forward_hook1)
    m.register_forward_hook(forward_hook2)

    return m


def create_module_forward_single_input():
    # Use to test module level hooks for forward with single input
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "outer_mod_name"
        assert input[0] == "a"
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("pre_hook_override_name",)
        output = output + "_fh"
        return output

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_same_hook_repeated():
    # Use to test module can run same hook multiple times
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "outer_mod_name"
        input_change = input[0] + "_ph"
        return (input_change,)

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("a_ph_ph",)
        output = output + "_fh"
        return output

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_hook_return_nothing():
    # Use to test module level hooks that return nothing
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> None:
        assert self.name == "outer_mod_name"
        assert input[0] == "a"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("a",)

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_multiple_hooks_single_input():
    # Use to test that modules can run multiple hooks with single input
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "outer_mod_name"
        assert input[0] == "a"
        return ("pre_hook_override_name1",)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "outer_mod_name"
        assert input[0] == "pre_hook_override_name1"
        return ("pre_hook_override_name2",)

    def forward_hook1(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("pre_hook_override_name2",)
        assert output == "pre_hook_override_name2_outermod_inner_mod"
        output = output + "_fh1"
        return output, output

    def forward_hook2(self, input: Tuple[str], output: Tuple[str, str]):
        assert self.name == "outer_mod_name"
        assert input == ("pre_hook_override_name2",)
        assert output[0] == "pre_hook_override_name2_outermod_inner_mod_fh1"
        output = output[0] + "_fh2"
        return output

    m.register_forward_pre_hook(pre_hook1)
    m.register_forward_pre_hook(pre_hook2)
    m.register_forward_hook(forward_hook1)
    m.register_forward_hook(forward_hook2)

    return m


def create_submodule_forward_multiple_inputs():
    # Use to test that submodules can run hooks that have multiple forward inputs
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "inner_mod_name"
        assert input[0][1] == "outer_mod_name"
        return ["pre_hook_override_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        assert self.name == "inner_mod_name"
        assert input[0][0] == "pre_hook_override_name"
        output2 = output[1] + "fh"
        return output[0], output2

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_multiple_hooks_multiple_inputs():
    # Use to test that submodules can run multiple hooks with multiple
    # forward inputs
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "inner_mod_name"
        assert input[1] == "no_pre_hook"
        return ["pre_hook_override_name"], "pre_hook_override1"

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "inner_mod_name"
        assert input[1] == "pre_hook_override1"
        return ["pre_hook_override_name"], "pre_hook_override2"

    def forward_hook1(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        assert self.name == "inner_mod_name"
        assert input[1] == "pre_hook_override2"
        assert output[1] == "pre_hook_override2_"
        output2 = output[1] + "fh1"
        return output[0], output2, output2

    def forward_hook2(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str, str]
    ):
        assert self.name == "inner_mod_name"
        assert input[1] == "pre_hook_override2"
        assert output[1] == "pre_hook_override2_fh1"
        output2 = output[1] + "_fh2"
        return output[0], output2

    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)

    return m


def create_submodule_forward_single_input():
    # Use to test that submodules can run hooks with a single argument
    # passed to forward
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name",)
        return output

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_to_call_directly_with_hooks():
    # Use to test that submodules have their hooks invoked when called
    # directly
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name",)
        return output + "_fh"

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_same_hook_repeated():
    # Use to test that submodules can run same hooks multiple times
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        changed = input[0] + "_ph"
        return (changed,)

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("a_outermod_ph_ph",)
        return output + "_fh"

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_hook_return_nothing():
    # Use to test that submodules can run hooks that return nothing
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> None:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("a_outermod",)

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_multiple_hooks_single_input():
    # Use to test that submodules can run multiple hooks that have a single input
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"
        return ("pre_hook_override_name",)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        assert input[0] == "pre_hook_override_name"
        return ("pre_hook_override_name2",)

    def forward_hook1(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name2",)
        assert output == "pre_hook_override_name2_inner_mod"
        return output + "_fwh1"

    def forward_hook2(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name2",)
        assert output == "pre_hook_override_name2_inner_mod_fwh1"
        return output

    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)

    return m


def create_forward_tuple_input():
    # Use to test case where forward is passed a single tuple for input.
    # This is different because eager always wraps pre-hook return arguments
    # in a tuple when the returned pre-hook result isn't a tuple
    # (to allow the result to be passed to another pre-hook if needed).
    # The eager behavior doesn't wrap the single tuple input pre-hook return in a
    # tuple as it should. To get consistent behavior between single tuple inputs and
    # the rest of the possible forward inputs, pre-hooks need to
    # wrap single tuple inputs returns in another tuple. This is
    # enforced by the schema checker.
    m = ModuleForwardTupleInput("outer_mod_name", "inner_mod_name")

    def pre_hook_outermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        # 'return (11,)' doesn't work with eager, inner tuple lost
        return ((11,),)

    def pre_hook_innermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        # 'return (22,)' doesn't work with eager, inner tuple lost
        return ((22,),)

    def forward_hook_outermod(self, input: Tuple[Tuple[int]], output: int):
        return (11,)

    def forward_hook_innermod(self, input: Tuple[Tuple[int]], output: Tuple[int]):
        return 22

    m.register_forward_pre_hook(pre_hook_outermod)
    m.submodule.register_forward_pre_hook(pre_hook_innermod)
    m.register_forward_hook(forward_hook_outermod)
    m.submodule.register_forward_hook(forward_hook_innermod)

    return m


def create_submodule_forward_single_input_return_not_tupled():
    # Use to check that submodules can return modified inputs
    # that aren't wrapped in a tuple (to match eager behavior)
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> str:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"
        # return is wrapped in tuple in other test cases
        return "pre_hook_override_name"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name",)
        output = output + "_fh"
        return output

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


if __name__ == "__main__":
    raise RuntimeError(
        "This file is a collection of utils, it should be imported not executed directly"
    )

```



## High-Level Overview


This Python file contains 9 class(es) and 80 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SubmoduleNoForwardInputs`, `ModuleNoForwardInputs`, `SubmoduleForwardSingleInput`, `ModuleForwardSingleInput`, `ModuleDirectforwardSubmodCall`, `SuboduleForwardMultipleInputs`, `ModuleForwardMultipleInputs`, `SubmoduleForwardTupleInput`, `ModuleForwardTupleInput`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `__init__`, `foo`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `create_module_no_forward_input`

**Key imports**: List, Tuple, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: List, Tuple
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python test/jit/test_hooks_modules.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit`):

- [`test_dataclasses.py_docs.md`](./test_dataclasses.py_docs.md)
- [`test_recursive_script.py_docs.md`](./test_recursive_script.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_python_builtins.py_docs.md`](./test_python_builtins.py_docs.md)
- [`test_functional_blocks.py_docs.md`](./test_functional_blocks.py_docs.md)
- [`mydecorator.py_docs.md`](./mydecorator.py_docs.md)
- [`test_union.py_docs.md`](./test_union.py_docs.md)
- [`test_python_bindings.py_docs.md`](./test_python_bindings.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)


## Cross-References

- **File Documentation**: `test_hooks_modules.py_docs.md`
- **Keyword Index**: `test_hooks_modules.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python docs/test/jit/test_hooks_modules.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/jit`):

- [`test_attr.py_kw.md_docs.md`](./test_attr.py_kw.md_docs.md)
- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_dataclasses.py_docs.md_docs.md`](./test_dataclasses.py_docs.md_docs.md)
- [`test_aten_pow.py_kw.md_docs.md`](./test_aten_pow.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_graph_rewrite_passes.py_kw.md_docs.md`](./test_graph_rewrite_passes.py_kw.md_docs.md)
- [`test_module_containers.py_kw.md_docs.md`](./test_module_containers.py_kw.md_docs.md)
- [`test_complex.py_kw.md_docs.md`](./test_complex.py_kw.md_docs.md)
- [`test_types.py_kw.md_docs.md`](./test_types.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_hooks_modules.py_docs.md_docs.md`
- **Keyword Index**: `test_hooks_modules.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
