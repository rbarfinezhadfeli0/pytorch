# Documentation: `docs/torch/_inductor/autoheuristic/learned_heuristic_controller.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/autoheuristic/learned_heuristic_controller.py_docs.md`
- **Size**: 7,005 bytes (6.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/autoheuristic/learned_heuristic_controller.py`

## File Metadata

- **Path**: `torch/_inductor/autoheuristic/learned_heuristic_controller.py`
- **Size**: 4,317 bytes (4.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import importlib
import inspect
import pkgutil
from collections import defaultdict
from typing import Any, Optional

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import LearnedHeuristic


def find_and_instantiate_subclasses(
    package_name: str, base_class: Any
) -> list[LearnedHeuristic]:
    instances = []

    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            module_basename = module_name.split(".")[-1]
            if not module_basename.startswith("_"):
                # learned heuristics start with an underscore
                continue
            module = importlib.import_module(module_name)

            # look for classes that are subclasses of base_class
            for _name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, base_class)
                    and obj != base_class
                ):
                    instance = obj()
                    instances.append(instance)
        except Exception as e:
            print(f"Error processing module {module_name}: {e}")

    return instances


class LearnedHeuristicController:
    """
    Class that finds and instantiates all learned heuristics. It also provides
    a way to get the decision of a learned heuristic.
    """

    existing_heuristics: dict[str, list[LearnedHeuristic]] = defaultdict(list)
    """
    A dictionary that stores all the learned heuristics for each optimization.
    The key is the optimization name, and the value is a list of LearnedHeuristic objects.
    """

    heuristics_initialized: bool = False
    """
    A flag that indicates whether the learned heuristics have been initialized.
    Set to true when the get_decision() function is called for the first time.
    """

    def __init__(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> None:
        self.metadata = metadata
        self.context = context

    def get_heuristics(self, name: str) -> list[LearnedHeuristic]:
        """
        Returns a list of learned heuristics for the given optimization name.
        """

        if not LearnedHeuristicController.heuristics_initialized:
            # learned heuristics are generated into the following package
            learned_heuristics_package = "torch._inductor.autoheuristic.artifacts"

            # learned heuristics have to be of type LearnedHeuristic
            base_class = LearnedHeuristic
            found_heuristics = find_and_instantiate_subclasses(
                learned_heuristics_package, base_class
            )

            for learned_heuristic in found_heuristics:
                opt_name = learned_heuristic.get_name()
                LearnedHeuristicController.existing_heuristics[opt_name].append(
                    learned_heuristic
                )
            LearnedHeuristicController.heuristics_initialized = True

        return LearnedHeuristicController.existing_heuristics[name]

    def get_decision(self) -> Optional[Choice]:
        """
        Returns the decision made by the learned heuristic or None if no heuristic was found or the heuristic is unsure
        which choice to make.
        """

        heuristics = self.get_heuristics(self.metadata.name)
        for heuristic in heuristics:
            if heuristic.check_precondition(self.metadata, self.context):
                return heuristic.get_decision(self.context, self.metadata.choices)
        return None

    def get_decisions_ranked(self, top_k: int) -> Optional[list[Choice]]:
        heuristics = self.get_heuristics(self.metadata.name)
        for heuristic in heuristics:
            if heuristic.check_precondition(self.metadata, self.context):
                choices = heuristic.get_decisions_ranked(self.context)
                if choices is None:
                    return None
                avail_choices = [
                    choice for choice in choices if choice in self.metadata.choices
                ]
                return avail_choices[:top_k]
        return None

```



## High-Level Overview

"""    Class that finds and instantiates all learned heuristics. It also provides    a way to get the decision of a learned heuristic.

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LearnedHeuristicController`

**Functions defined**: `find_and_instantiate_subclasses`, `__init__`, `get_heuristics`, `get_decision`, `get_decisions_ranked`

**Key imports**: importlib, inspect, pkgutil, defaultdict, Any, Optional, LearnedHeuristic


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/autoheuristic`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `importlib`
- `inspect`
- `pkgutil`
- `collections`: defaultdict
- `typing`: Any, Optional
- `torch._inductor.autoheuristic.learnedheuristic_interface`: LearnedHeuristic


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

Files in the same folder (`torch/_inductor/autoheuristic`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`learnedheuristic_interface.py_docs.md`](./learnedheuristic_interface.py_docs.md)
- [`autoheuristic_utils.py_docs.md`](./autoheuristic_utils.py_docs.md)
- [`autoheuristic.py_docs.md`](./autoheuristic.py_docs.md)


## Cross-References

- **File Documentation**: `learned_heuristic_controller.py_docs.md`
- **Keyword Index**: `learned_heuristic_controller.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/autoheuristic`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/autoheuristic`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

Files in the same folder (`docs/torch/_inductor/autoheuristic`):

- [`learnedheuristic_interface.py_kw.md_docs.md`](./learnedheuristic_interface.py_kw.md_docs.md)
- [`autoheuristic_utils.py_docs.md_docs.md`](./autoheuristic_utils.py_docs.md_docs.md)
- [`autoheuristic_utils.py_kw.md_docs.md`](./autoheuristic_utils.py_kw.md_docs.md)
- [`learned_heuristic_controller.py_kw.md_docs.md`](./learned_heuristic_controller.py_kw.md_docs.md)
- [`learnedheuristic_interface.py_docs.md_docs.md`](./learnedheuristic_interface.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`autoheuristic.py_kw.md_docs.md`](./autoheuristic.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`autoheuristic.py_docs.md_docs.md`](./autoheuristic.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `learned_heuristic_controller.py_docs.md_docs.md`
- **Keyword Index**: `learned_heuristic_controller.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
