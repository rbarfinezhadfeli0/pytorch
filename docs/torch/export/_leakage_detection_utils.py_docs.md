# Documentation: `torch/export/_leakage_detection_utils.py`

## File Metadata

- **Path**: `torch/export/_leakage_detection_utils.py`
- **Size**: 3,501 bytes (3.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import gc
import types
import typing
import weakref

import torch


"""
These functions are used to detect potential fake tensor leakage when using PT2 export.
See NOTE [export non-strict fake tensor leak detection]

There are some complications that made this logic overly complicated:
1) Python 3.10 and Python 3.12 have different ways of implementing referrer so
   we need to account for whether it is ref.__dict__ or the real ref object

2) There are some internal PT2 references to fake tensors like `TrackedFake`
3) closures, generators, and bound methods can hold fake tensors.
4) global object can hold onto a fake tensor

In general, these utils are our last resort to detect fake tensors. if the leak happens
within the model attributes, we have a separate mechanism to detect. This tool relies a bit
on garbage collector internal details, so I think it is unsafe to turn on by default, hence
this tool should be used as debugging tool.
"""


# Things we never want to flag as leaks
_SKIP_TYPES = (
    types.FrameType,
    types.ModuleType,
)


def _is_globals_or_locals(obj: typing.Any) -> bool:
    # These comparisons only make sense within this frame; still cheap to check.
    return obj is globals() or obj is locals()


def _is_tracked_fake(obj: typing.Any) -> bool:
    return isinstance(obj, torch.fx.experimental.symbolic_shapes.TrackedFake)


def _is_gm_meta_like_dict(d: dict, o: typing.Any) -> bool:
    # Hope gm.meta was a custom dict we can assert on
    return d.get("val") is o


def _dict_is_attr_of_tracked_fake(d: dict) -> bool:
    """
    Python 3.10 quirk: sometimes the referrer is obj.__dict__ instead of obj.
    Check if this dict is exactly the __dict__ of a TrackedFake.
    """
    for parent in gc.get_referrers(d):
        if (
            hasattr(parent, "__dict__")
            and parent.__dict__ is d
            and _is_tracked_fake(parent)
        ):
            return True
    return False


def find_legit_leaks_from_referrers(active_fakes: weakref.WeakSet) -> weakref.WeakSet:
    legit_leak: weakref.WeakSet = weakref.WeakSet()

    # This is so that we don't falsely flag generator to be holding fake tensor
    fake_list = list(active_fakes)
    fake_list_id = id(fake_list)

    for act in fake_list:
        # Track by id to avoid processing duplicate referrers
        seen = set()
        # Assume it's a leak unless we find only ignorable referrers
        flagged = False

        for r in gc.get_referrers(act):
            rid = id(r)
            if rid in seen:
                continue
            seen.add(rid)

            # Skip our own fake_list
            if rid == fake_list_id:
                continue

            # Fast-path: skip obvious non-owners
            if _is_globals_or_locals(r):
                continue
            if isinstance(r, _SKIP_TYPES):
                continue
            if _is_tracked_fake(r):
                # TrackedFake should be ignored
                continue

            # Handle dicts carefully (Python 3.10 sometimes shows __dict__)
            if isinstance(r, dict):
                if _is_gm_meta_like_dict(r, act):
                    continue
                if _dict_is_attr_of_tracked_fake(r):
                    continue
                flagged = True
                break

            # Any other referrer we don't explicitly whitelist counts as a leak
            flagged = True
            break

        if flagged:
            legit_leak.add(act)

    return legit_leak

```



## High-Level Overview

"""These functions are used to detect potential fake tensor leakage when using PT2 export.See NOTE [export non-strict fake tensor leak detection]There are some complications that made this logic overly complicated:1) Python 3.10 and Python 3.12 have different ways of implementing referrer so   we need to account for whether it is ref.__dict__ or the real ref object2) There are some internal PT2 references to fake tensors like `TrackedFake`3) closures, generators, and bound methods can hold fake tensors.4) global object can hold onto a fake tensorIn general, these utils are our last resort to detect fake tensors. if the leak happenswithin the model attributes, we have a separate mechanism to detect. This tool relies a biton garbage collector internal details, so I think it is unsafe to turn on by default, hencethis tool should be used as debugging tool.

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_is_globals_or_locals`, `_is_tracked_fake`, `_is_gm_meta_like_dict`, `_dict_is_attr_of_tracked_fake`, `find_legit_leaks_from_referrers`

**Key imports**: gc, types, typing, weakref, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `gc`
- `types`
- `typing`
- `weakref`
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

Files in the same folder (`torch/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_remove_auto_functionalized_pass.py_docs.md`](./_remove_auto_functionalized_pass.py_docs.md)
- [`exported_program.py_docs.md`](./exported_program.py_docs.md)
- [`_wrapper_utils.py_docs.md`](./_wrapper_utils.py_docs.md)
- [`_unlift.py_docs.md`](./_unlift.py_docs.md)
- [`_trace.py_docs.md`](./_trace.py_docs.md)
- [`_swap.py_docs.md`](./_swap.py_docs.md)
- [`_tree_utils.py_docs.md`](./_tree_utils.py_docs.md)
- [`_safeguard.py_docs.md`](./_safeguard.py_docs.md)
- [`_remove_effect_tokens_pass.py_docs.md`](./_remove_effect_tokens_pass.py_docs.md)


## Cross-References

- **File Documentation**: `_leakage_detection_utils.py_docs.md`
- **Keyword Index**: `_leakage_detection_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
