# Documentation: `torch/utils/data/datapipes/dataframe/dataframe_wrapper.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/dataframe/dataframe_wrapper.py`
- **Size**: 3,288 bytes (3.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Any


_pandas: Any = None
_WITH_PANDAS: bool | None = None


def _try_import_pandas() -> bool:
    try:
        import pandas  # type: ignore[import]

        global _pandas
        _pandas = pandas
        return True
    except ImportError:
        return False


# pandas used only for prototyping, will be shortly replaced with TorchArrow
def _with_pandas() -> bool:
    global _WITH_PANDAS
    if _WITH_PANDAS is None:
        _WITH_PANDAS = _try_import_pandas()
    return _WITH_PANDAS


class PandasWrapper:
    @classmethod
    def create_dataframe(cls, data, columns):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return _pandas.DataFrame(data, columns=columns)  # type: ignore[union-attr]

    @classmethod
    def is_dataframe(cls, data):
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.frame.DataFrame)  # type: ignore[union-attr]

    @classmethod
    def is_column(cls, data):
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.series.Series)  # type: ignore[union-attr]

    @classmethod
    def iterate(cls, data):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        yield from data.itertuples(index=False)

    @classmethod
    def concat(cls, buffer):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return _pandas.concat(buffer)  # type: ignore[union-attr]

    @classmethod
    def get_item(cls, data, idx):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return data[idx : idx + 1]

    @classmethod
    def get_len(cls, df):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return len(df.index)

    @classmethod
    def get_columns(cls, df):
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return list(df.columns.values.tolist())


# When you build own implementation just override it with dataframe_wrapper.set_df_wrapper(new_wrapper_class)
default_wrapper = PandasWrapper


def get_df_wrapper():
    return default_wrapper


def set_df_wrapper(wrapper) -> None:
    global default_wrapper
    default_wrapper = wrapper


def create_dataframe(data, columns=None):
    wrapper = get_df_wrapper()
    return wrapper.create_dataframe(data, columns)


def is_dataframe(data):
    wrapper = get_df_wrapper()
    return wrapper.is_dataframe(data)


def get_columns(data):
    wrapper = get_df_wrapper()
    return wrapper.get_columns(data)


def is_column(data):
    wrapper = get_df_wrapper()
    return wrapper.is_column(data)


def concat(buffer):
    wrapper = get_df_wrapper()
    return wrapper.concat(buffer)


def iterate(data):
    wrapper = get_df_wrapper()
    return wrapper.iterate(data)


def get_item(data, idx):
    wrapper = get_df_wrapper()
    return wrapper.get_item(data, idx)


def get_len(df):
    wrapper = get_df_wrapper()
    return wrapper.get_len(df)

```



## High-Level Overview


This Python file contains 1 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PandasWrapper`

**Functions defined**: `_try_import_pandas`, `_with_pandas`, `create_dataframe`, `is_dataframe`, `is_column`, `iterate`, `concat`, `get_item`, `get_len`, `get_columns`, `get_df_wrapper`, `set_df_wrapper`, `create_dataframe`, `is_dataframe`, `get_columns`, `is_column`, `concat`, `iterate`, `get_item`, `get_len`

**Key imports**: Any, pandas  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/dataframe`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `pandas  `


## Code Patterns & Idioms

### Common Patterns

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
- [`datapipes.py_docs.md`](./datapipes.py_docs.md)
- [`structures.py_docs.md`](./structures.py_docs.md)
- [`dataframes.py_docs.md`](./dataframes.py_docs.md)


## Cross-References

- **File Documentation**: `dataframe_wrapper.py_docs.md`
- **Keyword Index**: `dataframe_wrapper.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
