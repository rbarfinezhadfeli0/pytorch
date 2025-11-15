# Documentation: `torch/_dynamo/graph_break_registry.json`

## File Metadata

- **Path**: `torch/_dynamo/graph_break_registry.json`
- **Size**: 151,210 bytes (147.67 KB)
- **Type**: JSON Configuration
- **Extension**: `.json`

## File Purpose

This is a json configuration that is part of the PyTorch project.

## Original Source

```json
{
  "GB0000": [
    {
      "Gb_type": "All __torch_function__ overrides returned NotImplemented due to TypeError from user code",
      "Context": "fn={fn}, args={args}, kwargs={kwargs}",
      "Explanation": "All __torch_function__ overrides for for function {fn} returned NotImplemented",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0001": [
    {
      "Gb_type": "Argument of `as_subclass` must be a non-dispatcher-style tensor subclass",
      "Context": "{self}.as_subclass({cls})",
      "Explanation": "Currently not supported",
      "Hints": [
        "Avoid this call or move it outside `torch.compile` regione",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0002": [
    {
      "Gb_type": "Assertion failed on symbolic shapes",
      "Context": "str(sym_expr)",
      "Explanation": "",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0003": [
    {
      "Gb_type": "Attempt to trace generator",
      "Context": "",
      "Explanation": "Generators cannot be compiled directly with `torch.compile`.",
      "Hints": [
        "Call a generator from inside of a non-generator Python function and ",
        "compile that function instead.",
        "This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround."
      ]
    }
  ],
  "GB0004": [
    {
      "Gb_type": "Attempted super().__delattr__() on an object without mutation tracking",
      "Context": "call_method {self} {name}",
      "Explanation": "Dynamo needs to track mutations on an object before `super().__delattr__` can be used on it. But the object ({self.objvar}) doesn't have attribute mutation tracking enabled.",
      "Hints": [
        "Ensure the object is tracked by Dynamo's side effect system.",
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0005": [
    {
      "Gb_type": "Attempted to a str() method implemented in C/C++",
      "Context": "",
      "Explanation": "{type(arg.value)} has a C/C++ based str method. This is not supported.",
      "Hints": [
        "Write the str method in Python"
      ]
    }
  ],
  "GB0006": [
    {
      "Gb_type": "Attempted to call a super() attribute that is not a function or method",
      "Context": "call_method {self} {name}",
      "Explanation": "Dynamo does not know how to trace the call `super().{name}()` because `super().{name}` is not a function or method attribute.",
      "Hints": [
        "Ensure the attribute accessed via `super()` is a standard method or function."
      ]
    }
  ],
  "GB0007": [
    {
      "Gb_type": "Attempted to call function marked as skipped",
      "Context": "module: {module_name}, qualname: {qualname}, skip reason: {reason}",
      "Explanation": "explanation",
      "Hints": []
    }
  ],
  "GB0008": [
    {
      "Gb_type": "Attempted to inline function marked as skipped",
      "Context": "qualname: {fn_qualname}, name: {func.get_name()}, filename: `{func.get_filename()}`, skip reason: {result.reason}",
      "Explanation": "Dynamo developers have intentionally marked that the function `{fn_qualname}` should not be traced.",
      "Hints": []
    }
  ],
  "GB0009": [
    {
      "Gb_type": "Attempted to inline function marked as skipped (SkipFunctionVariable)",
      "Context": "Attempted to inline a SkipFunctionVariable {func}",
      "Explanation": "Attempted to inline a function that was previously determined to be marked as intentionally skipped.",
      "Hints": []
    }
  ],
  "GB0010": [
    {
      "Gb_type": "Attempted to read a deleted variable",
      "Context": "item: {item}, name: {name}",
      "Explanation": "",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0011": [
    {
      "Gb_type": "Attempted to read undefined local variable",
      "Context": "LOAD_FAST {name}",
      "Explanation": "Could not find a local variable with name `{name}`",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0012": [
    {
      "Gb_type": "Attempted to read undefined local variable (implicit)",
      "Context": "LOAD_FAST {name}",
      "Explanation": "Could not find an implicit local variable with name `{name}`",
      "Hints": [
        "This happens in dict/list comprehensions",
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0013": [
    {
      "Gb_type": "Attempted to represent unregistered RemovableHandle",
      "Context": "",
      "Explanation": "Dynamo attempted to build a representation of a torch.utils.hooks.RemovableHandle, which is not supported. This happens because the RemovableHandle was created in another frame.",
      "Hints": []
    }
  ],
  "GB0014": [
    {
      "Gb_type": "Attempted to wrap RNN, GRU, or LSTM",
      "Context": "str(value)",
      "Explanation": "Dynamo does not support RNN, GRU, or LSTM.",
      "Hints": [
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0015": [
    {
      "Gb_type": "Attempted to wrap sparse Tensor",
      "Context": "",
      "Explanation": "torch.compile does not support sparse Tensors",
      "Hints": [
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0016": [
    {
      "Gb_type": "Attempted to wrap strided NestedTensor",
      "Context": "",
      "Explanation": "torch.compile does not support strided NestedTensor",
      "Hints": []
    }
  ],
  "GB0017": [
    {
      "Gb_type": "Attempted to wrap torch._higher_order_ops.invoke_subgraph",
      "Context": "",
      "Explanation": "Directly using invoke_subgraph is not supported. Use nested_compile_region",
      "Hints": []
    }
  ],
  "GB0018": [
    {
      "Gb_type": "Attempted to wrap unbacked SymInt",
      "Context": "",
      "Explanation": "Unbacked SymInt input is not supported yet.",
      "Hints": [
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0019": [
    {
      "Gb_type": "AutogradFunctionContextVariable escaped Dynamo-traced region",
      "Context": "",
      "Explanation": "We cannot reconstruct a torch.autograd.Function's context object.",
      "Hints": []
    }
  ],
  "GB0020": [
    {
      "Gb_type": "BUILD_STRING key conflict",
      "Context": "format_string_parts: {format_string_parts}, kwargs: {kwargs}, part.sym_kwargs: {part.sym_kwargs}",
      "Explanation": "Failed to build format string due to key conflict",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0021": [
    {
      "Gb_type": "BUILD_STRING type error",
      "Context": "str(part)",
      "Explanation": "Format string part type is not correct - expected constant or format string.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0022": [
    {
      "Gb_type": "Bad import result",
      "Context": "typestr(value)",
      "Explanation": "Import result is not a Python module.",
      "Hints": []
    }
  ],
  "GB0023": [
    {
      "Gb_type": "Builtin `operator.*` comparison with constant `self` failed",
      "Context": "call_method {self} {name} {args} {kwargs}",
      "Explanation": "\"Failed to compare {self} with {other}, \"                     + f\"because {other} is not a Python constant or its mutation check fails.\"",
      "Hints": []
    }
  ],
  "GB0024": [
    {
      "Gb_type": "CLEANUP_THROW with StopIteration",
      "Context": "",
      "Explanation": "Received StopIteration when handling generator.throw/close. This is not supported.",
      "Hints": []
    }
  ],
  "GB0025": [
    {
      "Gb_type": "Call to `torch._dynamo.graph_break()`",
      "Context": "Called `torch._dynamo.graph_break()` with args `{args}`, kwargs `{kwargs}`",
      "Explanation": "User-inserted graph break. Message: {graph_break_msg}",
      "Hints": [
        "Remove the `torch._dynamo.graph_break()` call."
      ]
    }
  ],
  "GB0026": [
    {
      "Gb_type": "Calling subclass default constructor with more than tensor argument",
      "Context": "{self.value}(args={args}, kwargs={kwargs})",
      "Explanation": "Currently not supported",
      "Hints": [
        "Avoid this constructor call or move it outside ",
        "`torch.compile` regione",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0027": [
    {
      "Gb_type": "Cannot check Tensor object identity without its fake value",
      "Context": "str(fake_tensor)",
      "Explanation": "TensorVariable is missing a fake example_value.",
      "Hints": [
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0028": [
    {
      "Gb_type": "Caught non-Exception value",
      "Context": "str(exc_instance)",
      "Explanation": "Except expects to receive an object of Exception type but received {exc_instance}.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0029": [
    {
      "Gb_type": "Compilation of intermediate hooks requires compiled autograd",
      "Context": "var_getattr {self} {name}",
      "Explanation": "Dynamo must be in compiled_autograd to register hooks.",
      "Hints": []
    }
  ],
  "GB0030": [
    {
      "Gb_type": "ComptimeContext graph break",
      "Context": "msg",
      "Explanation": "Manually triggered ComptimeContext graph break with message {msg}.",
      "Hints": []
    }
  ],
  "GB0031": [
    {
      "Gb_type": "Custom __getattribute__ in nn.Module attribute access",
      "Context": "var_getattr {self} {name}",
      "Explanation": "Dynamo does not support checking key existence on `nn.Module` instances that have a custom `__getattribute__` method defined.",
      "Hints": [
        "Avoid defining `__getattribute__` in your module.",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0032": [
    {
      "Gb_type": "Custom __getattribute__ in nn.Module dict key check",
      "Context": "has_key_in_generic_dict {self} {key}",
      "Explanation": "Dynamo does not support checking key existence on `nn.Module` instances that have a custom `__getattribute__` method defined.",
      "Hints": [
        "Avoid defining `__getattribute__` in your module.",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0033": [
    {
      "Gb_type": "Data dependent operator",
      "Context": "str(cause.func)",
      "Explanation": "Operator `{cause.func}` has a non-Tensor output whose value is dependent on the data of Tensor inputs.",
      "Hints": []
    }
  ],
  "GB0034": [
    {
      "Gb_type": "Data-dependent assertion failed (cannot compile partial graph)",
      "Context": "value: {value}",
      "Explanation": "Dynamo has determined when encountering a data-dependent assert failure that it should not compile the partial graph.",
      "Hints": [
        "Use `torch._assert()` to raise a hard AssertionError when the check fails. ",
        "This error will propagate back the user code ",
        "that called the compiled function (i.e. Dynamo will not trace any exception handling).",
        "Remove the assert statement.",
        "Move the assert statement outside of any context managers in order to graph break with ",
        "partial graph compilation (if fullgraph=False).",
        "This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround."
      ]
    }
  ],
  "GB0035": [
    {
      "Gb_type": "Data-dependent branching with non-constant __bool__",
      "Context": "method: {x}, result: {result}",
      "Explanation": "Attempted to perform data-dependent branching on a user-defined object with a __bool__ method that did not return a constant.",
      "Hints": []
    }
  ],
  "GB0036": [
    {
      "Gb_type": "Dynamic shape operator",
      "Context": "str(cause.func)",
      "Explanation": "Operator `{cause.func}`'s output shape depends on input Tensor data.",
      "Hints": [
        "Enable tracing of dynamic shape operators with ",
        "`torch._dynamo.config.capture_dynamic_output_shape_ops = True`"
      ]
    }
  ],
  "GB0037": [
    {
      "Gb_type": "Dynamic shape operator (no meta kernel)",
      "Context": "str(cause.func)",
      "Explanation": "Operator `{cause.func}` does not have a meta kernel that supports dynamic output shapes",
      "Hints": [
        "Please report an issue to PyTorch"
      ]
    }
  ],
  "GB0038": [
    {
      "Gb_type": "Dynamic slicing with Tensor arguments",
      "Context": "SliceVariable start: {start}, stop: {stop}, step: {step}",
      "Explanation": "Creating slices with Tensor arguments is not supported. e.g. `l[:x]`, where `x` is a 1-element tensor.",
      "Hints": [
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0039": [
    {
      "Gb_type": "Dynamo cache limit exceeded",
      "Context": "Limit type: {limit_type}",
      "Explanation": "Dynamo attempted to recompile the code object too many times, exceeding the {limit_type} cache size limit.Giving up on compiling as the compile time tradeoff is likely not worth the performance gain.",
      "Hints": []
    }
  ],
  "GB0040": [
    {
      "Gb_type": "Encountered aliasing during higher order op tracing",
      "Context": "context",
      "Explanation": "Higher order ops do not support aliasing. Found in {source_target.name()}",
      "Hints": [
        "Replace `return input` with `return input.clone()` to avoid aliasing.",
        "Consider using the debug context to change user code to avoid aliasing.",
        "Please open an issue."
      ]
    }
  ],
  "GB0041": [
    {
      "Gb_type": "Encountered input mutation during higher order op tracing",
      "Context": "context",
      "Explanation": "Higher order ops do not support input mutation. Found in {source_target.name()}",
      "Hints": [
        "Consider using the debug context to change user code to avoid mutation.",
        "Please open an issue."
      ]
    }
  ],
  "GB0042": [
    {
      "Gb_type": "Encountered non user function variable during invoke_subgraph HOP tracing",
      "Context": "str(fn_vt)",
      "Explanation": "invoke_subgraph does not support non user function variable",
      "Hints": [
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0043": [
    {
      "Gb_type": "Encountered non-PT2-compliant op",
      "Context": "",
      "Explanation": "msg +   + err_epilogue",
      "Hints": []
    }
  ],
  "GB0044": [
    {
      "Gb_type": "Encountered strided NestedTensor in automatic dynamic dim determination",
      "Context": "",
      "Explanation": "torch.compile does not support strided NestedTensor",
      "Hints": []
    }
  ],
  "GB0045": [
    {
      "Gb_type": "Encountered tensor.is_inference() during tracing",
      "Context": "",
      "Explanation": "tensor.is_inference() is not supported",
      "Hints": [
        "This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround."
      ]
    }
  ],
  "GB0046": [
    {
      "Gb_type": "Encountered torch.is_inference_mode_enabled during tracing",
      "Context": "",
      "Explanation": "torch.is_inference_mode_enabled() is not supported",
      "Hints": [
        "This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround."
      ]
    }
  ],
  "GB0047": [
    {
      "Gb_type": "Encountered unconverted argument when attempting to inline",
      "Context": "func: {func}, arg: {v}",
      "Explanation": "An argument to an inlined function was not successfully converted to a VariableTracker.",
      "Hints": [
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0048": [
    {
      "Gb_type": "Error getting associated real value",
      "Context": "call_id {self}",
      "Explanation": "Dynamo encountered an error while trying to get the associated real value.",
      "Hints": []
    }
  ],
  "GB0049": [
    {
      "Gb_type": "Error when attempting to resolve op packet",
      "Context": "",
      "Explanation": "str(e)",
      "Hints": []
    }
  ],
  "GB0050": [
    {
      "Gb_type": "Exception with bad expected type",
      "Context": "str(expected_exc_types)",
      "Explanation": "`except ...` has unsupported type {expected_exc_types}.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0051": [
    {
      "Gb_type": "Exception with non-type expectation",
      "Context": "str(expected_type)",
      "Explanation": "`except ...` expects a non-type: {expected_type}.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0052": [
    {
      "Gb_type": "Excessive RestartAnalysis() calls",
      "Context": "",
      "Explanation": "Dynamo attempted to trace the same frame 100+ times. Giving up on compiling as the compile time tradeoff is likely not worth the performance gain.",
      "Hints": []
    }
  ],
  "GB0053": [
    {
      "Gb_type": "FSDP with use_orig_params=False",
      "Context": "",
      "Explanation": "Dynamo only supports FSDP with use_orig_params=True",
      "Hints": []
    }
  ],
  "GB0054": [
    {
      "Gb_type": "Failed to construct Enum variable",
      "Context": "value: {value_vt}, allowed enum values: {list(cls_type)}",
      "Explanation": "Attempted to construct an Enum value that is non-constant (e.g. int, string) or is not an acceptable value for the Enum. Acceptable values for Enum `{cls_type}`: {list(cls_type)}.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0055": [
    {
      "Gb_type": "Failed to convert args/kwargs to proxy",
      "Context": "call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}",
      "Explanation": "Missing `as_proxy()` implementation for some arg/kwarg.",
      "Hints": []
    }
  ],
  "GB0056": [
    {
      "Gb_type": "Failed to mutate tensor data attribute",
      "Context": "setattr({obj}, {name}, {val})",
      "Explanation": "Dyanmo only supports mutating `.data` of tensor created outside `torch.compile` region",
      "Hints": [
        "Don't mutate `.data` on this tensor, or move ",
        "the mutation out of `torch.compile` region"
      ]
    }
  ],
  "GB0057": [
    {
      "Gb_type": "Failed to raise exception",
      "Context": "str(exc)",
      "Explanation": "Attempted to raise a non-Exception type/value.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0058": [
    {
      "Gb_type": "Failed to set tensor attribute",
      "Context": "setattr({obj}, {name}, {val})",
      "Explanation": "Dyanmo doesn't support setting these tensor attributes",
      "Hints": [
        "Don't mutate attribute '{name}' on tensors, or ",
        "move the mutation out of `torch.compile` region"
      ]
    }
  ],
  "GB0059": [
    {
      "Gb_type": "Failed to trace builtin operator",
      "Context": "builtin {fn.__name__} {arg_types} {has_kwargs}",
      "Explanation": "Dynamo does not know how to trace builtin operator `{fn.__name__}` with argument types {real_arg_types} (has_kwargs {has_kwargs})",
      "Hints": [
        "Avoid calling builtin `{fn.__name__}` with argument types {real_arg_types}. ",
        "Consider using an equivalent alternative function/method to `{fn.__name__}`.",
        "If you are attempting to call a logging function (e.g. `print`), ",
        "you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.",
        "Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0060": [
    {
      "Gb_type": "Failed to trace unittest method",
      "Context": "function: unittest.TestCase.{name}",
      "Explanation": "Dynamo does not know how to trace unittest method `{name}` ",
      "Hints": [
        "Avoid calling `TestCase.{name}`. ",
        "Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0061": [
    {
      "Gb_type": "Failed to unpack object for BUILD_LIST_UNPACK",
      "Context": "str(seq)",
      "Explanation": "{seq} cannot be unpacked into a list for the BUILD_LIST_UNPACK bytecode (`[*x, *y, ...]`).",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0062": [
    {
      "Gb_type": "Failed to unpack object for UNPACK_EX",
      "Context": "str(seq)",
      "Explanation": "{seq} cannot be unpacked into a list for the UNPACK_EX bytecode.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0063": [
    {
      "Gb_type": "Failed to unpack object for UNPACK_SEQUENCE",
      "Context": "str(seq)",
      "Explanation": "{seq} cannot be unpacked into a list for the UNPACK_SEQUENCE bytecode (i.e. `a, b, c = d`).",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0064": [
    {
      "Gb_type": "Fake tensor propagation exception",
      "Context": "str(e.reason)",
      "Explanation": "msg",
      "Hints": []
    }
  ],
  "GB0065": [
    {
      "Gb_type": "Graph break in inlined function",
      "Context": "",
      "Explanation": "Graph breaks in an inlined call are not supported.",
      "Hints": []
    }
  ],
  "GB0066": [
    {
      "Gb_type": "Graph break under GenericContextWrappingVariable",
      "Context": "Active generic context managers: {self.active_generic_context_managers}",
      "Explanation": "Attempted to graph break in an active context manager(s) that doesn't support graph breaking.",
      "Hints": [
        "Move the offending context manager(s) to outside the compiled region.",
        "This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one."
      ]
    }
  ],
  "GB0067": [
    {
      "Gb_type": "HigherOrderOperator: Mutating a variable not in the current scope (SideEffects)",
      "Context": "",
      "Explanation": "This is not supported.",
      "Hints": []
    }
  ],
  "GB0068": [
    {
      "Gb_type": "Illegal method invocation in strict mode",
      "Context": "call_method {self} {name} {args} {kwargs}",
      "Explanation": "Dynamo currently does not support this method ({name}) invocation in strict mode.",
      "Hints": []
    }
  ],
  "GB0069": [
    {
      "Gb_type": "Import failure",
      "Context": "module_name: {module_name}, fromlist: {fromlist}, level={level}",
      "Explanation": "Failure when attempting to import.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0070": [
    {
      "Gb_type": "Indexing list with non-scalar tensor",
      "Context": "call_method {self} {name} {args} {kwargs}",
      "Explanation": "Attempted to index list-like object with tensor with > 1 element.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0071": [
    {
      "Gb_type": "Inline attempt with __self__",
      "Context": "str(func)",
      "Explanation": "Attempted to inline a function with the `__self__` attribute. Dynamo is expected to decompose method calls into function calls with a `self` argument.",
      "Hints": []
    }
  ],
  "GB0072": [
    {
      "Gb_type": "Inplace op on input tensor",
      "Context": "",
      "Explanation": "Attempted to trace an inplace view op on input tensor {typestr(self.value)}.",
      "Hints": [
        "Ensure you do not modify input tensor in place.",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0073": [
    {
      "Gb_type": "Invoking an nn.Module inside a HigherOrderOperator",
      "Context": "",
      "Explanation": "This is not supported.",
      "Hints": []
    }
  ],
  "GB0074": [
    {
      "Gb_type": "Invoking an nn.Module inside a higher order operator",
      "Context": "Higher order op name: {self.source_target}",
      "Explanation": "This is not supported.",
      "Hints": []
    }
  ],
  "GB0075": [
    {
      "Gb_type": "LOAD_BUILD_CLASS bytecode not supported",
      "Context": "",
      "Explanation": "Dynamo does not support tracing classes that are defined in the compiled region.",
      "Hints": [
        "Move the class definition out of the compiled region.",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0076": [
    {
      "Gb_type": "LOAD_FAST_CHECK on uninitialized variable",
      "Context": "inst.argval",
      "Explanation": "Attempted to load uninitialized local variable {inst.argval}",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0077": [
    {
      "Gb_type": "Length mismatch when unpacking object for UNPACK_SEQUENCE",
      "Context": "expected length: {inst.argval}, actual: {len(val)}",
      "Explanation": "{seq} unpacked to a list for the UNPACK_SEQUENCE bytecode (i.e. `a, b, c = d`) with unexpected length.",
      "Hints": [
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0078": [
    {
      "Gb_type": "Limitation of `nonstrict_trace",
      "Context": "{self}",
      "Explanation": "msg",
      "Hints": [
        "make sure definition of {fn_name} is outside ",
        "`torch.compile` region"
      ]
    }
  ],
  "GB0079": [
    {
      "Gb_type": "Missing CALL_INTRINSIC_1 handler",
      "Context": "CALL_INTRINSIC_1 operand: {inst.argval}",
      "Explanation": "No handler implemented for CALL_INTRINSIC_1 {inst.argval} instruction.",
      "Hints": [
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0080": [
    {
      "Gb_type": "Missing FakeTensor example value",
      "Context": "str(node)",
      "Explanation": "`FakeTensor` example value was required for {node} but not available.",
      "Hints": [
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0081": [
    {
      "Gb_type": "Missing attribute when running call_method node",
      "Context": "",
      "Explanation": "make_error_message(\"attribute not defined\")",
      "Hints": []
    }
  ],
  "GB0082": [
    {
      "Gb_type": "Missing bytecode handler",
      "Context": "{opname} with args {args}",
      "Explanation": "Dynamo does not know how to handle the bytecode instruction `{opname}`.",
      "Hints": [
        "Do not trace code that produces the `{opname}` bytecode instruction ",
        "(see https://docs.python.org/3/library/dis.html for bytecode semantics).",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0083": [
    {
      "Gb_type": "Module-level backwards hooks require compiled autograd.",
      "Context": "",
      "Explanation": "",
      "Hints": [
        "Enable compiled autograd by setting torch._dynamo.config.compiled_autograd = True."
      ]
    }
  ],
  "GB0084": [
    {
      "Gb_type": "Non-constant attribute given to `super().__delattr__()`",
      "Context": "call_method {self} {name}",
      "Explanation": "Dynamo requires the attribute name passed to `super().__delattr__(...)` to be a constant (string).",
      "Hints": [
        "Ensure the attribute name is a string literal or a constant variable."
      ]
    }
  ],
  "GB0085": [
    {
      "Gb_type": "Non-function or method in subclass of torch.autograd.Function",
      "Context": "call_apply {self} {args} {kwargs}",
      "Explanation": "Dynamo requires the `forward` attribute of a `torch.autograd.Function` subclass to be a standard Python function or method. Found type `{type(fn).__name__}` instead.",
      "Hints": [
        "Ensure the `forward` method is defined as a regular ",
        "function or instance method."
      ]
    }
  ],
  "GB0086": [
    {
      "Gb_type": "Not a Python constant",
      "Context": "guard_as_python_constant {self}",
      "Explanation": "Failed to convert {self} into a Python constant.",
      "Hints": []
    }
  ],
  "GB0087": [
    {
      "Gb_type": "NotImplementedError/UnsupportedFakeTensorException when running FX node",
      "Context": "",
      "Explanation": "make_error_message(e)",
      "Hints": []
    }
  ],
  "GB0088": [
    {
      "Gb_type": "Observed exception",
      "Context": "raised exception {curr_exc.python_type_name()}({curr_exc.args})",
      "Explanation": "observed_exn_gb_explanation",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0089": [
    {
      "Gb_type": "Observed exception (EXCEPT_HANDLER)",
      "Context": "str(raised_exception)",
      "Explanation": "observed_exn_gb_explanation                                 + \" This graph break is unexpected.\"",
      "Hints": [
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0090": [
    {
      "Gb_type": "Operator does not support running with fake tensors",
      "Context": "unsupported operator: {cause.func}",
      "Explanation": "",
      "Hints": [
        "{import_suggestion}see ",
        "https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0",
        " for how to fix"
      ]
    }
  ],
  "GB0091": [
    {
      "Gb_type": "Read uninitialized cell",
      "Context": "str(cellvar)",
      "Explanation": "Attempted to read a cell variable that has not been populated yet.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0092": [
    {
      "Gb_type": "Reconstruction failure",
      "Context": "str(value)",
      "Explanation": "Dynamo has no bytecode reconstruction implemented for sourceless variable {value}.",
      "Hints": [
        "If Dynamo is attempting to trace a return statement and your code is attempting to return a variable ",
        "that Dynamo cannot reconstruct, then remove it from the return statement.",
        "Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have ",
        "reconstruction rules may be fundamentally unreconstructable.",
        "This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one."
      ]
    }
  ],
  "GB0093": [
    {
      "Gb_type": "Reconstruction failure: source.reconstruct not implemented",
      "Context": "str(source)",
      "Explanation": "Dynamo has no bytecode reconstruction implemented for {type(source)} variable {source}.",
      "Hints": [
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0094": [
    {
      "Gb_type": "SEND with bad type",
      "Context": "TOS type: {typestr(tos)}",
      "Explanation": "Attempted to SEND with unsupported type {typestr(tos)}.",
      "Hints": []
    }
  ],
  "GB0095": [
    {
      "Gb_type": "Set Exception object `__traceback__` attribute to not-`None`",
      "Context": "call_setattr {self} {name}",
      "Explanation": "Dynamo does not support setting the attribute '__traceback__' on tracked exception objects to anything other than None.",
      "Hints": [
        "Avoid setting '__traceback__' on exception objects ",
        "within traced code, or set it to None."
      ]
    }
  ],
  "GB0096": [
    {
      "Gb_type": "Should not compile partial graph (STORE_ATTR)",
      "Context": "",
      "Explanation": "Dynamo has determined when encountering an unsupported STORE_ATTR instruction (i.e. `obj.attr = val`) that it should not compile the partial graph.",
      "Hints": []
    }
  ],
  "GB0097": [
    {
      "Gb_type": "Side effect on existing deque with limited maxlen",
      "Context": "",
      "Explanation": "This is not supported.",
      "Hints": [
        "Don't use a deque with `maxlen` specified."
      ]
    }
  ],
  "GB0098": [
    {
      "Gb_type": "Skip calling `torch.compiler.disable()`d function",
      "Context": "str(self.value)",
      "Explanation": "Skip calling function `{self.value}` since it was wrapped with `torch.compiler.disable` (reason: {msg})",
      "Hints": [
        "Remove the `torch.compiler.disable` call"
      ]
    }
  ],
  "GB0099": [
    {
      "Gb_type": "Skip inlining `torch.compiler.disable()`d function",
      "Context": "str(func.get_function())",
      "Explanation": "Skip inlining function {func.get_function()} since it was wrapped with `torch.compiler.disable` (reason: {msg})",
      "Hints": [
        "Remove the `torch.compiler.disable` call"
      ]
    }
  ],
  "GB0100": [
    {
      "Gb_type": "Storing Tensor hook handle in globals",
      "Context": "name",
      "Explanation": "This is not supported.",
      "Hints": []
    }
  ],
  "GB0101": [
    {
      "Gb_type": "Storing Tensor hook handle in globals (inline call)",
      "Context": "inst.argval",
      "Explanation": "This is not supported.",
      "Hints": []
    }
  ],
  "GB0102": [
    {
      "Gb_type": "Strict mode banned op",
      "Context": "var_getattr {self} {name}",
      "Explanation": "Getattr invocation '{name}' in strict mode is not supported.",
      "Hints": [
        "Remove `{name}` from the list of banned ops by ",
        "setting `torch._dynamo.config._autograd_backward_strict_mode_banned_ops`."
      ]
    }
  ],
  "GB0103": [
    {
      "Gb_type": "Tensor subclass overridden method call",
      "Context": "{name}",
      "Explanation": "`torch.compile` currently can't trace this",
      "Hints": [
        "Avoid calling {name} of tensor subclass in torch.compile region",
        "Renaming method `{name}` of type {self.class_type}",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0104": [
    {
      "Gb_type": "Tensor with grad_fn()",
      "Context": "var_getattr {self} grad_fn",
      "Explanation": "Dynamo does not support tracing tensors with a grad_fn directly.",
      "Hints": []
    }
  ],
  "GB0105": [
    {
      "Gb_type": "Tensor.numpy() with trace_numpy=False",
      "Context": "call_method {self} numpy",
      "Explanation": "`Tensor.numpy()` was called, but the `trace_numpy` configuration was manually disabled.",
      "Hints": [
        "Set `torch._dynamo.config.trace_numpy = True` to allow ",
        "Dynamo to trace through NumPy."
      ]
    }
  ],
  "GB0106": [
    {
      "Gb_type": "Tensor.numpy() without NumPy installed",
      "Context": "call_method {self} numpy",
      "Explanation": "`Tensor.numpy()` was called, but the NumPy library is not available in the current environment.",
      "Hints": [
        "Ensure NumPy is installed in your Python environment."
      ]
    }
  ],
  "GB0107": [
    {
      "Gb_type": "Tensor.random_ op",
      "Context": "Tensor.{name}(args={args}, kwargs={kwargs})",
      "Explanation": "This is currently not supported.",
      "Hints": [
        "Use the out-of-place version of this op",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0108": [
    {
      "Gb_type": "Tensor.retain_grad() with AOTDispatcher",
      "Context": "var_getattr {self} retain_grad",
      "Explanation": "`Tensor.retain_grad()` does not work with AOTDispatcher.",
      "Hints": []
    }
  ],
  "GB0109": [
    {
      "Gb_type": "Tensor.tolist() with non-integer tensor",
      "Context": "call_method {self} to_list",
      "Explanation": "Dynamo currently does not support tracing `tolist()` on non-integer tensors.",
      "Hints": [
        "Ensure the input tensor to `tolist()` is an integer ",
        "type (e.g., int8, int16, int32, int64)."
      ]
    }
  ],
  "GB0110": [
    {
      "Gb_type": "Tensor.uniform_ op called with `from` keyword",
      "Context": "Tensor.{name}(args={args}, kwargs={kwargs})",
      "Explanation": "This is currently not supported.",
      "Hints": [
        "Avoid using the `from` keyword.",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0111": [
    {
      "Gb_type": "TypeError from user code",
      "Context": "call_function({self.value}, {args}, {kwargs})",
      "Explanation": "msg",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0112": [
    {
      "Gb_type": "TypeError when making fake tensor call",
      "Context": "TypeError {node.target}: {cause}",
      "Explanation": "",
      "Hints": []
    }
  ],
  "GB0113": [
    {
      "Gb_type": "Unable to resolve super getattr",
      "Context": "",
      "Explanation": "Dynamo failed to trace attribute `{name}` accessed via `super()` (for type `{self.typevar}` and object `{self.objvar}`) because the resolved attribute type is not supported.",
      "Hints": [
        "Ensure the attribute exists in the parent class.",
        "Check the arguments passed to `super()`."
      ]
    }
  ],
  "GB0114": [
    {
      "Gb_type": "Unexpected failure during itertools.accumulate() iteration",
      "Context": "call_function {self} {args} {kwargs}",
      "Explanation": "Unexpected failure in invoking function during accumulate. Failed running func {func}({item}{acc})",
      "Hints": [
        "This graph break may be difficult to debug. Please report an issue to PyTorch for assistance."
      ]
    }
  ],
  "GB0115": [
    {
      "Gb_type": "Unexpected failure during itertools.groupby() iteration",
      "Context": "call_function {self} {args} {kwargs}",
      "Explanation": "Unexpected failure in invoking function during groupby",
      "Hints": [
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0116": [
    {
      "Gb_type": "Unexpected type in sourceless builder",
      "Context": "{value_type.__module__}.{value_type.__qualname__}",
      "Explanation": "SourcelessBuilder.create does not know how to wrap {value_type}",
      "Hints": [
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0117": [
    {
      "Gb_type": "Unhandled args for method",
      "Context": "call_method {self} {name} {args} {kwargs}",
      "Explanation": "Dynamo encountered an error while calling the method `{name}`.",
      "Hints": []
    }
  ],
  "GB0118": [
    {
      "Gb_type": "Unimplemented next() call",
      "Context": "next({self})",
      "Explanation": "This abstract method must be implemented",
      "Hints": [
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0119": [
    {
      "Gb_type": "Uninitialized nn.Module",
      "Context": "typestr(value)",
      "Explanation": "Attempted to trace an uninitialized nn.Module of type {typestr(value)}.",
      "Hints": [
        "Ensure your nn.Module instance has called `super().__init__()`.",
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0120": [
    {
      "Gb_type": "Unreachable sub-generator code",
      "Context": "",
      "Explanation": "Should only be encountered while implementing generator support.",
      "Hints": []
    }
  ],
  "GB0121": [
    {
      "Gb_type": "UnspecializedNNModuleVariable missing method",
      "Context": "call_method: {self} {name} {args} {kwargs}",
      "Explanation": "Dynamo does not support tracing method {name} of nn.Module {self.value}",
      "Hints": [
        "Dynamo does not really define unspecialized nn.Module very well.",
        "This graph break may be difficult to debug. Please report an issue to PyTorch for assistance."
      ]
    }
  ],
  "GB0122": [
    {
      "Gb_type": "Unsupported SourceType",
      "Context": "MutationType.__init__ {self} {typ}",
      "Explanation": "Dynamo does not support the type `{typ}`",
      "Hints": [
        "This branch is not supposed to be reachable.",
        "This is likely to be a Dynamo bug. Please report an issue to PyTorch."
      ]
    }
  ],
  "GB0123": [
    {
      "Gb_type": "Unsupported Tensor.backward() call",
      "Context": "call_method {self} backward {args} {kwargs}",
      "Explanation": "Dynamo currently does not support tracing `Tensor.backward()`.",
      "Hints": [
        "This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround."
      ]
    }
  ],
  "GB0124": [
    {
      "Gb_type": "Unsupported Tensor.item() call with capture_scalar_outputs=False",
      "Context": "call_method {self} item {args} {kwargs}",
      "Explanation": "Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.",
      "Hints": [
        "Set `torch._dynamo.config.capture_scalar_outputs = True` ",
        "or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` ",
        "to include these operations in the captured graph."
      ]
    }
  ],
  "GB0125": [
    {
      "Gb_type": "Unsupported Tensor.requires_grad_() call",
      "Context": "call_method {self} requires_grad_",
      "Explanation": "Dynamo does not support changes to a Tensor's `requires_grad` through calling `requires_grad_()`.",
      "Hints": []
    }
  ],
  "GB0126": [
    {
      "Gb_type": "Unsupported Tensor.resize_() call",
      "Context": "call_method {self} resize_ {args} {kwargs}",
      "Explanation": "Dynamo currently does not support tracing `Tensor.resize_()`.",
      "Hints": []
    }
  ],
  "GB0127": [
    {
      "Gb_type": "Unsupported Tensor.resize_as_() call",
      "Context": "call_method {self} resize_as_ {args} {kwargs}",
      "Explanation": "Dynamo currently does not support tracing `Tensor.resize_as_()`.",
      "Hints": []
    }
  ],
  "GB0128": [
    {
      "Gb_type": "Unsupported Tensor.set_() call",
      "Context": "call_method {self} set_ {args} {kwargs}",
      "Explanation": "Dynamo currently does not support tracing `Tensor.set_()` overloads that include more than one argument.",
      "Hints": [
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0129": [
    {
      "Gb_type": "Unsupported Tensor.sparse_resize_() call",
      "Context": "call_method {self} sparse_resize_ {args} {kwargs}",
      "Explanation": "Dynamo currently does not support tracing `Tensor.sparse_resize_()`.",
      "Hints": []
    }
  ],
  "GB0130": [
    {
      "Gb_type": "Unsupported Tensor.sparse_resize_and_clear_() call",
      "Context": "call_method {self} sparse_resize_and_clear_ {args} {kwargs}",
      "Explanation": "Dynamo currently does not support tracing `Tensor.sparse_resize_and_clear_()`.",
      "Hints": []
    }
  ],
  "GB0131": [
    {
      "Gb_type": "Unsupported __setitem__/__setattr__ inline attempt",
      "Context": "code name: {code.co_name}, args: {args}",
      "Explanation": "Attempted to inline {code.co_name} where first argument (self) is not a user-defined object.",
      "Hints": []
    }
  ],
  "GB0132": [
    {
      "Gb_type": "Unsupported `func` in itertools.accumulate",
      "Context": "call_function {self} {args} {kwargs}",
      "Explanation": "Dynamo does not know how to get the function to use for itertools.accumulate. itertools.accumulate expects the `func` as the second argument or as a keyword argument.",
      "Hints": [
        "Dynamo has detected that tracing the code will result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled."
      ]
    }
  ],
  "GB0133": [
    {
      "Gb_type": "Unsupported arguments for itertools.accumulate",
      "Context": "call_function {self} {args} {kwargs}",
      "Explanation": "Dynamo does not know how to trace itertools.accumulate with args: {args} and kwargs: {kwargs}. itertools.accumulate expects an iterable, an optional binary function for accumulation, and an optional initial value to set the starting state.",
      "Hints": [
        "Make sure the arguments to itertools.accumulate are correct.",
        "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues."
      ]
    }
  ],
  "GB0134": [
    {
      "Gb_type": "Unsupported argum
```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/_dynamo`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `graph_break_registry.json_docs.md`
- **Keyword Index**: `graph_break_registry.json_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
