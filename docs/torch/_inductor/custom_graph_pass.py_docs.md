# Documentation: `torch/_inductor/custom_graph_pass.py`

## File Metadata

- **Path**: `torch/_inductor/custom_graph_pass.py`
- **Size**: 5,881 bytes (5.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, Optional, TypeAlias, Union

import torch.fx.graph


class CustomGraphPass(ABC):
    """
    Implement this interface for custom Graph passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    passes are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom passes would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom pass
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    ** IMPORTANT ** If your custom pass's behavior depends on some external state, then
    you'll need to implement something more complicated (or disable caching).

    EXAMPLE:

    class MyCustomGraphPass(CustomGraphPass):
        def __call__(self, graph: torch.fx.graph.Graph) -> None:
            # my custom graph optimization pass
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """

    @abstractmethod
    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """


class CustomGraphModulePass(ABC):
    """
    Implement this interface for custom Graph passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    passes are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom passes would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom pass
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.
    """

    @abstractmethod
    def __call__(self, gm: torch.fx.GraphModule) -> None:
        """
        Implementation of the custom pass.
        """

    @abstractmethod
    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """


CustomGraphPassType: TypeAlias = Optional[
    Union[CustomGraphPass, Callable[[torch.fx.graph.Graph], None]]
]


@lru_cache(1)
def get_hash_for_files(paths: tuple[str], extra: str = "") -> bytes:
    """
    Helper to compute a unique string by hashing the contents of a list of files.
    """
    hasher = hashlib.sha256()
    hasher.update(extra.encode("utf-8"))
    for path in paths:
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.digest()


class CustomPartitionerFn(ABC):
    """
    Implement this interface for custom partitioner:

    1) The __call__() method contains the implementation of the custom partitioner.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    partitioner are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom partitioner would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom partitioner
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    EXAMPLE:

    from torch._inductor.custom_graph_pass import get_hash_for_files

    class MyCustomPartitionerFn(CustomPartitionerFn):
        def __call__(
            self,
            gm: torch.fx.GraphModule,
            joint_inputs: Sequence[object],
            **kwargs: Any
        ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
            # my custom partitioner implementation
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        """
        Implementation of the custom partitioner.
        """

    @abstractmethod
    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom partitioner implementation.
        Return None to skip inductor code caching entirely.
        """


CustomPartitionerFnType: TypeAlias = Optional[CustomPartitionerFn]

```



## High-Level Overview

"""    Implement this interface for custom Graph passes:    1) The __call__() method contains the implementation of the custom pass.    2) The uuid() method enables inductor to cache compiled graphs when your custom    passes are applied. This method can return any identifier as long as it uniquely    identifies your implementation (and can be pickled). The caching logic includes this    identifier in its key calculation, i.e., any new value will effectively invalidate    existing entries. We expect custom passes would typically depend purely on the    textual representation of the implementation. In that case, we recommend using the    'get_hash_for_files' helper below to compute a unique hash from the contents of a    static list of source files, i.e., the source(s) containing the custom pass    implementation. That approach ensures that any change to the implementation will    mean a new uuid.    ** IMPORTANT ** If your custom pass's behavior depends on some external state, then    you'll need to implement something more complicated (or disable caching).    EXAMPLE:    class MyCustomGraphPass(CustomGraphPass):        def __call__(self, graph: torch.fx.graph.Graph) -> None:            # my custom graph optimization pass            #     ...        def uuid(self) -> Optional[Any]:            return get_hash_for_files((__file__,))

This Python file contains 5 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CustomGraphPass`, `MyCustomGraphPass`, `CustomGraphModulePass`, `CustomPartitionerFn`, `MyCustomPartitionerFn`

**Functions defined**: `__call__`, `uuid`, `__call__`, `uuid`, `__call__`, `uuid`, `get_hash_for_files`, `__call__`, `uuid`, `__call__`, `uuid`

**Key imports**: hashlib, ABC, abstractmethod, Callable, Sequence, lru_cache, Any, Optional, TypeAlias, Union, torch.fx.graph, get_hash_for_files


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `hashlib`
- `abc`: ABC, abstractmethod
- `collections.abc`: Callable, Sequence
- `functools`: lru_cache
- `typing`: Any, Optional, TypeAlias, Union
- `torch.fx.graph`
- `torch._inductor.custom_graph_pass`: get_hash_for_files


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `custom_graph_pass.py_docs.md`
- **Keyword Index**: `custom_graph_pass.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
