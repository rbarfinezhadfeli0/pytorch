# Documentation: `tools/stats/utilization_stats_lib.py`

## File Metadata

- **Path**: `tools/stats/utilization_stats_lib.py`
- **Size**: 3,287 bytes (3.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

#  pyrefly: ignore [missing-import]
from dataclasses_json import DataClassJsonMixin  # type: ignore[import-not-found]


_DATA_MODEL_VERSION = 1.5


# data model for test log usage
@dataclass
class UtilizationStats:
    avg: Optional[float] = None
    max: Optional[float] = None
    raw: Optional[list[float]] = None


@dataclass
class UtilizationMetadata(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    level: str
    workflow_id: str
    job_id: str
    workflow_name: str
    job_name: str
    usage_collect_interval: float
    data_model_version: float
    start_at: int
    gpu_count: Optional[int] = None
    cpu_count: Optional[int] = None
    gpu_type: Optional[str] = None
    error: Optional[str] = None


@dataclass
class GpuUsage(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    uuid: Optional[str] = None
    util_percent: Optional[UtilizationStats] = None
    mem_util_percent: Optional[UtilizationStats] = None
    allocated_mem_percent: Optional[UtilizationStats] = None
    allocated_mem_value: Optional[UtilizationStats] = None
    total_mem_value: Optional[float] = None


@dataclass
class RecordData(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    cpu: Optional[UtilizationStats] = None
    memory: Optional[UtilizationStats] = None
    gpu_usage: Optional[list[GpuUsage]] = None


@dataclass
class UtilizationRecord(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    level: str
    timestamp: int
    data: Optional[RecordData] = None
    cmd_names: Optional[list[str]] = None
    error: Optional[str] = None
    log_duration: Optional[str] = None
    logs: Optional[list[str]] = None


# the db schema related to this is:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_utilization_metadata_schema.sql
@dataclass
class OssCiSegmentV1(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    level: str
    name: str
    start_at: int
    end_at: int
    extra_info: dict[str, str]


@dataclass
class OssCiUtilizationMetadataV1:
    created_at: int
    repo: str
    workflow_id: int
    run_attempt: int
    job_id: int
    workflow_name: str
    job_name: str
    usage_collect_interval: float
    data_model_version: str
    gpu_count: int
    cpu_count: int
    gpu_type: str
    start_at: int
    end_at: int
    segments: list[OssCiSegmentV1]
    tags: list[str] = field(default_factory=list)


# this data model is for the time series data:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_time_series_schema.sql
@dataclass
class OssCiUtilizationTimeSeriesV1:
    created_at: int
    type: str
    tags: list[str]
    time_stamp: int
    repo: str
    workflow_id: int
    run_attempt: int
    job_id: int
    workflow_name: str
    job_name: str
    json_data: str


def getDataModelVersion() -> float:
    return _DATA_MODEL_VERSION


def getTsNow() -> int:
    ts = datetime.now().timestamp()
    return int(ts)


@dataclass
class WorkflowInfo:
    workflow_run_id: int
    workflow_name: str
    job_id: int
    run_attempt: int
    job_name: str
    repo: str = "pytorch/pytorch"

```



## High-Level Overview


This Python file contains 9 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `UtilizationStats`, `UtilizationMetadata`, `GpuUsage`, `RecordData`, `UtilizationRecord`, `OssCiSegmentV1`, `OssCiUtilizationMetadataV1`, `OssCiUtilizationTimeSeriesV1`, `WorkflowInfo`

**Functions defined**: `getDataModelVersion`, `getTsNow`

**Key imports**: dataclass, field, datetime, Optional, DataClassJsonMixin  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/stats`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`: dataclass, field
- `datetime`: datetime
- `typing`: Optional
- `dataclasses_json`: DataClassJsonMixin  


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`tools/stats`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`upload_sccache_stats.py_docs.md`](./upload_sccache_stats.py_docs.md)
- [`upload_external_contrib_stats.py_docs.md`](./upload_external_contrib_stats.py_docs.md)
- [`check_disabled_tests.py_docs.md`](./check_disabled_tests.py_docs.md)
- [`upload_metrics.py_docs.md`](./upload_metrics.py_docs.md)
- [`import_test_stats.py_docs.md`](./import_test_stats.py_docs.md)
- [`upload_artifacts.py_docs.md`](./upload_artifacts.py_docs.md)
- [`upload_test_stats_intermediate.py_docs.md`](./upload_test_stats_intermediate.py_docs.md)
- [`export_test_times.py_docs.md`](./export_test_times.py_docs.md)


## Cross-References

- **File Documentation**: `utilization_stats_lib.py_docs.md`
- **Keyword Index**: `utilization_stats_lib.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
