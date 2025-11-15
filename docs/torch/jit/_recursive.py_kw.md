# Keyword Index: `torch/jit/_recursive.py`

## File Information

- **Original File**: [torch/jit/_recursive.py](../../../torch/jit/_recursive.py)
- **Documentation**: [`_recursive.py_docs.md`](./_recursive.py_docs.md)
- **Folder**: `torch/jit`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ConcreteTypeStore`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`SourceContext`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`are`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`def`**: [_recursive.py_docs.md](./_recursive.py_docs.md)

### Functions

- **`__contains__`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`__init__`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`__len__`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`_check_no_signature`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`_compile_and_register_class`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`_forward`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`_get_valid_constant`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`add_python_attr_to_scripted_model`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`check_module_initialized`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`compile_unbound_method`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`create_hooks_from_stubs`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`create_methods_and_properties_from_stubs`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`create_script_class`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`create_script_module`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`create_script_module_impl`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`get_annotations`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`get_cls_annotations`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`get_hook_stubs`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`get_module_concrete_type`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`get_or_create_concrete_type`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`get_overload_annotations`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`get_overload_name_mapping`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`get_properties_names`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`get_property_stubs`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`ignore_overloaded`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`infer_concrete_type_builder`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`infer_interface_methods_to_compile`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`infer_methods_to_compile`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`infer_type`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`init_fn`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`interface_script`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`jit_ignored_properties`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`lazy_bind`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`lazy_binding_method`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`make_stub`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`make_stub_from_method`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`make_stubs_for_overloads`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`make_stubs_from_exported_methods`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`script_model_defines_attr`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`try_compile_fn`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`wrap_cpp_class`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`wrap_cpp_module`**: [_recursive.py_docs.md](./_recursive.py_docs.md)

### Imports

- **`AttributeTypeIsSupportedChecker`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`Module`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`_add_script_class`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`_find_builtin`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`collections`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`fake_range`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`functools`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`inspect`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`textwrap`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`torch`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`torch._jit_internal`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`torch._sources`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`torch.jit._builtins`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`torch.jit._check`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`torch.jit._state`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`torch.jit.frontend`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`torch.nn`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`types`**: [_recursive.py_docs.md](./_recursive.py_docs.md)
- **`warnings`**: [_recursive.py_docs.md](./_recursive.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
