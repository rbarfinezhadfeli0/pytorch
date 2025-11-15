# Documentation: `docs/docs/source/accelerator/hooks.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/accelerator/hooks.md_docs.md`
- **Size**: 11,940 bytes (11.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/accelerator/hooks.md`

## File Metadata

- **Path**: `docs/source/accelerator/hooks.md`
- **Size**: 9,860 bytes (9.63 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Accelerator Hooks

## Background

OpenReg hooks provide a mechanism for integrating custom accelerator devices into PyTorch's runtime system. OpenReg (Open Registration) is PyTorch's extensibility framework that allows accelerator vendors to register custom device backends without modifying PyTorch core code.

## Design

The following tables list all hooks that accelerator vendors need to implement when integrating a new device backend. These hooks are categorized into two priority levels:

- **High Priority Hooks**: Core APIs that PyTorch runtime directly depends on. Accelerator vendors are recommended to implement all high priority hooks to ensure full PyTorch compatibility and enable basic device functionality.

- **Low Priority Hooks**: Device management and utility APIs that PyTorch does not directly depend on. These hooks enhance user experience and multi-device support but are *optional*. Accelerator vendors can choose to implement them based on their specific requirements and use cases.

### High Priority Hooks

| Hook Method                        | Description                                               | Application Scenario                                                             |
| ---------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `init()`                           | Initializes the accelerator runtime and device contexts   | Set up necessary state when PyTorch first accesses the device                    |
| `hasPrimaryContext(DeviceIndex)`   | Checks if a primary context exists for the device         | Determine whether device initialization has occurred                             |
| `getDefaultGenerator(DeviceIndex)` | Returns the default random number generator for a device  | Access the device's primary RNG for reproducible random operations               |
| `getNewGenerator(DeviceIndex)`     | Creates a new independent random number generator         | Create isolated RNG instances for parallel operations                            |
| `getDeviceFromPtr(void*)`          | Determines which device a memory pointer belongs to       | Identify the accelerator device associated with a memory allocation              |
| `getPinnedMemoryAllocator()`       | Returns an allocator for pinned (page-locked) host memory | Allocate host memory that can be efficiently transferred to/from the accelerator |
| `isPinnedPtr(void*)`               | Checks if a pointer points to pinned memory               | Validate memory types before performing operations                               |

### Low Priority Hooks

| Hook Method                        | Description                                                                  | Application Scenario                                                 |
| ---------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `isBuilt()`                        | Returns whether the accelerator backend is built/compiled into the extension | Check whether the accelerator library is available at compile time   |
| `isAvailable()`                    | Returns whether the accelerator hardware is available at runtime             | Verify whether accelerator devices can be detected and initialized   |
| `deviceCount()`                    | Returns the number of available accelerator devices                          | Enumerate all available accelerator devices for device selection     |
| `setCurrentDevice(DeviceIndex)`    | Sets the active device for the current thread                                | Switch the current thread's context to a specific accelerator device |
| `getCurrentDevice()`               | Returns the currently active device index                                    | Query which accelerator device is active in the current thread       |
| `exchangeDevice(DeviceIndex)`      | Atomically exchanges the current device and returns the previous one         | Temporarily switch devices and restore the previous device afterward |
| `maybeExchangeDevice(DeviceIndex)` | Conditionally exchanges device only if the index is valid                    | Safely attempt device switching with validation                      |

## Implementation

We can just take `getDefaultGenerator` as an implementation example:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK EXAMPLES
    :end-before: LITERALINCLUDE END: OPENREG HOOK EXAMPLES
    :linenos:
```

In this implementation:

1. **Override the base interface**: The `getDefaultGenerator` method overrides the virtual method from `at::PrivateUse1HooksInterface`.

2. **Delegate to device-specific implementation**: It calls `getDefaultOpenRegGenerator(device_index)`, which manages a per-device generator instance.

3. **Return device-specific generator**: The returned `at::Generator` wraps an `OpenRegGeneratorImpl` that implements device-specific random number generation.

This pattern applies to all hooks: override the interface method, validate inputs, delegate to your device-specific API, and return results in PyTorch's expected format.

## Integration Example

The following sections demonstrate how PyTorch integrates with accelerator hooks when accessing the default random number generator. The example traces the complete flow from user-facing Python code down to the device-specific implementation.

### Layer 1: User Code

User code initiates the operation by calling `manual_seed` to set the random seed for reproducible results:

```python
import torch
torch.openreg.manual_seed(42)
```

### Layer 2: Extension Python API

The Python API layer handles device management and calls into the C++ extension (defined in [`torch_openreg/openreg/random.py`][random.py]):

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py
    :language: python
    :start-after: LITERALINCLUDE START: OPENREG MANUAL SEED
    :end-before: LITERALINCLUDE END: OPENREG MANUAL SEED
    :linenos:
```

The `manual_seed` function gets the current device index and calls `torch_openreg._C._get_default_generator(idx)` to obtain the device-specific generator, then sets the seed on it.

### Layer 3: Python/C++ Bridge

The C++ extension exposes `_getDefaultGenerator` to Python, which bridges to PyTorch's core runtime:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR
    :end-before: LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR
    :linenos:
    :emphasize-lines: 10-11
```

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG MODULE METHODS
    :end-before: LITERALINCLUDE END: OPENREG MODULE METHODS
    :linenos:
    :emphasize-lines: 3
```

This function unpacks the device index from Python, creates a `PrivateUse1` device object, and calls `at::globalContext().defaultGenerator()`. PyTorch's context then dispatches to the registered hooks.

### Layer 4: PyTorch Core Context

PyTorch's Context class dispatches to the appropriate accelerator hooks ([`aten/src/ATen/Context.h`][Context.h]):

```{eval-rst}
.. literalinclude:: ../../../aten/src/ATen/Context.h
    :language: c++
    :lines: 60-103
    :linenos:
    :emphasize-lines: 8-9, 24-25
```

This layered architecture enables PyTorch to remain device-agnostic while delegating hardware-specific operations to accelerator implementations. The hooks are registered once at module load time:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK REGISTER
    :end-before: LITERALINCLUDE END: OPENREG HOOK REGISTER
    :linenos:
    :emphasize-lines: 4
```

### Layer 5: Accelerator Hooks

The hooks interface provides the abstraction that PyTorch uses to delegate to device-specific implementations:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK EXAMPLES
    :end-before: LITERALINCLUDE END: OPENREG HOOK EXAMPLES
    :linenos:
```

The `getDefaultGenerator` hook method overrides the base interface and delegates to `getDefaultOpenRegGenerator`, which manages the actual generator instances.

### Layer 6: Device-Specific Implementation

The device-specific implementation manages per-device generator instances:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR IMPL
    :end-before: LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR IMPL
    :linenos:
```

This function maintains a static vector of generators (one per device), initializes them on first access, validates the device index, and returns the appropriate generator instance.

[random.py]: https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py#L48-L53 "random.py"
[Context.h]: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/Context.h#L61-L102 "Context.h"
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source/accelerator`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source/accelerator`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`docs/source/accelerator`):

- [`autoload.md_docs.md`](./autoload.md_docs.md)
- [`amp.md_docs.md`](./amp.md_docs.md)
- [`device.md_docs.md`](./device.md_docs.md)
- [`operators.md_docs.md`](./operators.md_docs.md)
- [`index.md_docs.md`](./index.md_docs.md)


## Cross-References

- **File Documentation**: `hooks.md_docs.md`
- **Keyword Index**: `hooks.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source/accelerator`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source/accelerator`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`docs/docs/source/accelerator`):

- [`operators.md_docs.md_docs.md`](./operators.md_docs.md_docs.md)
- [`device.md_docs.md_docs.md`](./device.md_docs.md_docs.md)
- [`index.md_docs.md_docs.md`](./index.md_docs.md_docs.md)
- [`amp.md_docs.md_docs.md`](./amp.md_docs.md_docs.md)
- [`autoload.md_kw.md_docs.md`](./autoload.md_kw.md_docs.md)
- [`device.md_kw.md_docs.md`](./device.md_kw.md_docs.md)
- [`operators.md_kw.md_docs.md`](./operators.md_kw.md_docs.md)
- [`hooks.md_kw.md_docs.md`](./hooks.md_kw.md_docs.md)
- [`amp.md_kw.md_docs.md`](./amp.md_kw.md_docs.md)
- [`index.md_kw.md_docs.md`](./index.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `hooks.md_docs.md_docs.md`
- **Keyword Index**: `hooks.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
