# Documentation: `docs/torch/utils/data/datapipes/dataframe/datapipes.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/dataframe/datapipes.py_docs.md`
- **Size**: 7,350 bytes (7.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/dataframe/datapipes.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/dataframe/datapipes.py`
- **Size**: 4,626 bytes (4.52 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import random
from typing import Any

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe


__all__ = [
    "ConcatDataFramesPipe",
    "DataFramesAsTuplesPipe",
    "ExampleAggregateAsDataFrames",
    "FilterDataFramesPipe",
    "PerRowDataFramesPipe",
    "ShuffleDataFramesPipe",
]


@functional_datapipe("_dataframes_as_tuples")
class DataFramesAsTuplesPipe(IterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for df in self.source_datapipe:
            # for record in df.to_records(index=False):
            yield from df_wrapper.iterate(df)


@functional_datapipe("_dataframes_per_row", enable_df_api_tracing=True)
class PerRowDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for df in self.source_datapipe:
            # TODO(VitalyFedyunin): Replacing with TorchArrow only API, as we are dropping pandas as followup
            for i in range(len(df)):
                yield df[i : i + 1]


@functional_datapipe("_dataframes_concat", enable_df_api_tracing=True)
class ConcatDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, batch=3) -> None:
        self.source_datapipe = source_datapipe
        self.n_batch = batch

    def __iter__(self):
        buffer = []
        for df in self.source_datapipe:
            buffer.append(df)
            if len(buffer) == self.n_batch:
                yield df_wrapper.concat(buffer)
                buffer = []
        if buffer:
            yield df_wrapper.concat(buffer)


@functional_datapipe("_dataframes_shuffle", enable_df_api_tracing=True)
class ShuffleDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self):
        size = None
        all_buffer: list[Any] = []
        for df in self.source_datapipe:
            if size is None:
                size = df_wrapper.get_len(df)
            all_buffer.extend(
                df_wrapper.get_item(df, i) for i in range(df_wrapper.get_len(df))
            )
        random.shuffle(all_buffer)
        buffer = []
        for df in all_buffer:
            buffer.append(df)
            if len(buffer) == size:
                yield df_wrapper.concat(buffer)
                buffer = []
        if buffer:
            yield df_wrapper.concat(buffer)


@functional_datapipe("_dataframes_filter", enable_df_api_tracing=True)
class FilterDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, filter_fn) -> None:
        self.source_datapipe = source_datapipe
        self.filter_fn = filter_fn

    def __iter__(self):
        size = None
        all_buffer = []
        filter_res = []
        # pyrefly: ignore [bad-assignment]
        for df in self.source_datapipe:
            if size is None:
                size = len(df.index)
            for i in range(len(df.index)):
                all_buffer.append(df[i : i + 1])
                filter_res.append(self.filter_fn(df.iloc[i]))

        buffer = []
        for df, res in zip(all_buffer, filter_res, strict=True):
            if res:
                buffer.append(df)
                if len(buffer) == size:
                    yield df_wrapper.concat(buffer)
                    buffer = []
        if buffer:
            yield df_wrapper.concat(buffer)


@functional_datapipe("_to_dataframes_pipe", enable_df_api_tracing=True)
class ExampleAggregateAsDataFrames(DFIterDataPipe):
    def __init__(self, source_datapipe, dataframe_size=10, columns=None) -> None:
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.dataframe_size = dataframe_size

    def _as_list(self, item):
        try:
            return list(item)
        except (
            Exception
        ):  # TODO(VitalyFedyunin): Replace with better iterable exception
            return [item]

    def __iter__(self):
        aggregate = []
        for item in self.source_datapipe:
            aggregate.append(self._as_list(item))
            if len(aggregate) == self.dataframe_size:
                yield df_wrapper.create_dataframe(aggregate, columns=self.columns)
                aggregate = []
        if len(aggregate) > 0:
            yield df_wrapper.create_dataframe(aggregate, columns=self.columns)

```



## High-Level Overview


This Python file contains 6 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DataFramesAsTuplesPipe`, `PerRowDataFramesPipe`, `ConcatDataFramesPipe`, `ShuffleDataFramesPipe`, `FilterDataFramesPipe`, `ExampleAggregateAsDataFrames`

**Functions defined**: `__init__`, `__iter__`, `__init__`, `__iter__`, `__init__`, `__iter__`, `__init__`, `__iter__`, `__init__`, `__iter__`, `__init__`, `_as_list`, `__iter__`

**Key imports**: random, Any, functional_datapipe, dataframe_wrapper as df_wrapper, DFIterDataPipe, IterDataPipe


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/dataframe`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `random`
- `typing`: Any
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.dataframe`: dataframe_wrapper as df_wrapper
- `torch.utils.data.datapipes.datapipe`: DFIterDataPipe, IterDataPipe


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

Files in the same folder (`torch/utils/data/datapipes/dataframe`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`dataframe_wrapper.py_docs.md`](./dataframe_wrapper.py_docs.md)
- [`structures.py_docs.md`](./structures.py_docs.md)
- [`dataframes.py_docs.md`](./dataframes.py_docs.md)


## Cross-References

- **File Documentation**: `datapipes.py_docs.md`
- **Keyword Index**: `datapipes.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/data/datapipes/dataframe`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/data/datapipes/dataframe`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/utils/data/datapipes/dataframe`):

- [`dataframes.py_docs.md_docs.md`](./dataframes.py_docs.md_docs.md)
- [`dataframes.py_kw.md_docs.md`](./dataframes.py_kw.md_docs.md)
- [`datapipes.py_kw.md_docs.md`](./datapipes.py_kw.md_docs.md)
- [`dataframe_wrapper.py_docs.md_docs.md`](./dataframe_wrapper.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`structures.py_docs.md_docs.md`](./structures.py_docs.md_docs.md)
- [`structures.py_kw.md_docs.md`](./structures.py_kw.md_docs.md)
- [`dataframe_wrapper.py_kw.md_docs.md`](./dataframe_wrapper.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `datapipes.py_docs.md_docs.md`
- **Keyword Index**: `datapipes.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
