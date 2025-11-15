# Documentation: `docs/torch/distributed/argparse_util.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/argparse_util.py_docs.md`
- **Size**: 7,538 bytes (7.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/argparse_util.py`

## File Metadata

- **Path**: `torch/distributed/argparse_util.py`
- **Size**: 3,903 bytes (3.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from argparse import Action


class env(Action):
    """
    Get argument values from ``PET_{dest}`` before defaulting to the given ``default`` value.

    For flags (e.g. ``--standalone``)
    use ``check_env`` instead.

    .. note:: when multiple option strings are specified, ``dest`` is
              the longest option string (e.g. for ``"-f", "--foo"``
              the env var to set is ``PET_FOO`` not ``PET_F``)

    Example:
    ::

     parser.add_argument("-f", "--foo", action=env, default="bar")

     ./program                                      -> args.foo="bar"
     ./program -f baz                               -> args.foo="baz"
     ./program --foo baz                            -> args.foo="baz"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
     PET_FOO="env_bar" ./program --foo baz -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"

     parser.add_argument("-f", "--foo", action=env, required=True)

     ./program                                      -> fails
     ./program -f baz                               -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
    """

    def __init__(self, dest, default=None, required=False, **kwargs) -> None:
        env_name = f"PET_{dest.upper()}"
        default = os.environ.get(env_name, default)

        # ``required`` means that it NEEDS to be present  in the command-line args
        # rather than "this option requires a value (either set explicitly or default"
        # so if we found default then we don't "require" it to be in the command-line
        # so set it to False
        if default:
            required = False

        super().__init__(dest=dest, default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class check_env(Action):
    """
    Check whether the env var ``PET_{dest}`` exists before defaulting to the given ``default`` value.

    Equivalent to
    ``store_true`` argparse built-in action except that the argument can
    be omitted from the commandline if the env var is present and has a
    non-zero value.

    .. note:: it is redundant to pass ``default=True`` for arguments
              that use this action because a flag should be ``True``
              when present and ``False`` otherwise.

    Example:
    ::

     parser.add_argument("--verbose", action=check_env)

     ./program                                  -> args.verbose=False
     ./program --verbose                        -> args.verbose=True
     PET_VERBOSE=1 ./program           -> args.verbose=True
     PET_VERBOSE=0 ./program           -> args.verbose=False
     PET_VERBOSE=0 ./program --verbose -> args.verbose=True

    Anti-pattern (don't do this):

    ::

     parser.add_argument("--verbose", action=check_env, default=True)

     ./program                                  -> args.verbose=True
     ./program --verbose                        -> args.verbose=True
     PET_VERBOSE=1 ./program           -> args.verbose=True
     PET_VERBOSE=0 ./program           -> args.verbose=False

    """

    def __init__(self, dest, default=False, **kwargs) -> None:
        env_name = f"PET_{dest.upper()}"
        default = bool(int(os.environ.get(env_name, "1" if default else "0")))
        super().__init__(dest=dest, const=True, default=default, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)

```



## High-Level Overview

"""    Get argument values from ``PET_{dest}`` before defaulting to the given ``default`` value.    For flags (e.g. ``--standalone``)    use ``check_env`` instead.    .. note:: when multiple option strings are specified, ``dest`` is              the longest option string (e.g. for ``"-f", "--foo"``              the env var to set is ``PET_FOO`` not ``PET_F``)    Example:    ::     parser.add_argument("-f", "--foo", action=env, default="bar")     ./program                                      -> args.foo="bar"     ./program -f baz                               -> args.foo="baz"     ./program --foo baz                            -> args.foo="baz"     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"     PET_FOO="env_bar" ./program --foo baz -> args.foo="baz"     PET_FOO="env_bar" ./program           -> args.foo="env_bar"     parser.add_argument("-f", "--foo", action=env, required=True)     ./program                                      -> fails     ./program -f baz                               -> args.foo="baz"     PET_FOO="env_bar" ./program           -> args.foo="env_bar"     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `env`, `check_env`

**Functions defined**: `__init__`, `__call__`, `__init__`, `__call__`

**Key imports**: os, Action


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `argparse`: Action


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_mesh_layout.py_docs.md`](./_mesh_layout.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`c10d_logger.py_docs.md`](./c10d_logger.py_docs.md)
- [`_functional_collectives.py_docs.md`](./_functional_collectives.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`CONTRIBUTING.md_docs.md`](./CONTRIBUTING.md_docs.md)
- [`_functional_collectives_impl.py_docs.md`](./_functional_collectives_impl.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)


## Cross-References

- **File Documentation**: `argparse_util.py_docs.md`
- **Keyword Index**: `argparse_util.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/distributed`):

- [`_mesh_layout.py_docs.md_docs.md`](./_mesh_layout.py_docs.md_docs.md)
- [`run.py_docs.md_docs.md`](./run.py_docs.md_docs.md)
- [`device_mesh.py_docs.md_docs.md`](./device_mesh.py_docs.md_docs.md)
- [`_composable_state.py_docs.md_docs.md`](./_composable_state.py_docs.md_docs.md)
- [`run.py_kw.md_docs.md`](./run.py_kw.md_docs.md)
- [`_dist2.py_kw.md_docs.md`](./_dist2.py_kw.md_docs.md)
- [`_state_dict_utils.py_kw.md_docs.md`](./_state_dict_utils.py_kw.md_docs.md)
- [`rendezvous.py_kw.md_docs.md`](./rendezvous.py_kw.md_docs.md)
- [`rendezvous.py_docs.md_docs.md`](./rendezvous.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `argparse_util.py_docs.md_docs.md`
- **Keyword Index**: `argparse_util.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
