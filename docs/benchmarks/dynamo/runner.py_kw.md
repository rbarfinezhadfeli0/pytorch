# Keyword Index: `benchmarks/dynamo/runner.py`

## File Information

- **Original File**: [benchmarks/dynamo/runner.py](../../../benchmarks/dynamo/runner.py)
- **Documentation**: [`runner.py_docs.md`](./runner.py_docs.md)
- **Folder**: `benchmarks/dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DashboardUpdater`**: [runner.py_docs.md](./runner.py_docs.md)
- **`ParsePerformanceLogs`**: [runner.py_docs.md](./runner.py_docs.md)
- **`Parser`**: [runner.py_docs.md](./runner.py_docs.md)
- **`RegressionDetector`**: [runner.py_docs.md](./runner.py_docs.md)
- **`RegressionTracker`**: [runner.py_docs.md](./runner.py_docs.md)
- **`SummaryStatDiffer`**: [runner.py_docs.md](./runner.py_docs.md)
- **`class`**: [runner.py_docs.md](./runner.py_docs.md)

### Functions

- **`__init__`**: [runner.py_docs.md](./runner.py_docs.md)
- **`archive`**: [runner.py_docs.md](./runner.py_docs.md)
- **`archive_data`**: [runner.py_docs.md](./runner.py_docs.md)
- **`build_summary`**: [runner.py_docs.md](./runner.py_docs.md)
- **`clean_batch_sizes`**: [runner.py_docs.md](./runner.py_docs.md)
- **`comment_on_gh`**: [runner.py_docs.md](./runner.py_docs.md)
- **`comp_time`**: [runner.py_docs.md](./runner.py_docs.md)
- **`default_archive_name`**: [runner.py_docs.md](./runner.py_docs.md)
- **`diff`**: [runner.py_docs.md](./runner.py_docs.md)
- **`env_var`**: [runner.py_docs.md](./runner.py_docs.md)
- **`exec_summary_df`**: [runner.py_docs.md](./runner.py_docs.md)
- **`exec_summary_text`**: [runner.py_docs.md](./runner.py_docs.md)
- **`extract`**: [runner.py_docs.md](./runner.py_docs.md)
- **`extract_df`**: [runner.py_docs.md](./runner.py_docs.md)
- **`find_last_2_with_filenames`**: [runner.py_docs.md](./runner.py_docs.md)
- **`find_last_k`**: [runner.py_docs.md](./runner.py_docs.md)
- **`flag_accuracy`**: [runner.py_docs.md](./runner.py_docs.md)
- **`flag_bad_entries`**: [runner.py_docs.md](./runner.py_docs.md)
- **`flag_compilation_latency`**: [runner.py_docs.md](./runner.py_docs.md)
- **`flag_compression_ratio`**: [runner.py_docs.md](./runner.py_docs.md)
- **`flag_speedup`**: [runner.py_docs.md](./runner.py_docs.md)
- **`gen_comment`**: [runner.py_docs.md](./runner.py_docs.md)
- **`gen_summary_files`**: [runner.py_docs.md](./runner.py_docs.md)
- **`generate_commands`**: [runner.py_docs.md](./runner.py_docs.md)
- **`generate_comment`**: [runner.py_docs.md](./runner.py_docs.md)
- **`generate_csv_name`**: [runner.py_docs.md](./runner.py_docs.md)
- **`generate_diff`**: [runner.py_docs.md](./runner.py_docs.md)
- **`generate_dropdown_comment`**: [runner.py_docs.md](./runner.py_docs.md)
- **`generate_executive_summary`**: [runner.py_docs.md](./runner.py_docs.md)
- **`generate_warnings`**: [runner.py_docs.md](./runner.py_docs.md)
- **`geomean`**: [runner.py_docs.md](./runner.py_docs.md)
- **`get_archive_name`**: [runner.py_docs.md](./runner.py_docs.md)
- **`get_date`**: [runner.py_docs.md](./runner.py_docs.md)
- **`get_metric_title`**: [runner.py_docs.md](./runner.py_docs.md)
- **`get_mode`**: [runner.py_docs.md](./runner.py_docs.md)
- **`get_passing_entries`**: [runner.py_docs.md](./runner.py_docs.md)
- **`get_skip_tests`**: [runner.py_docs.md](./runner.py_docs.md)
- **`has_header`**: [runner.py_docs.md](./runner.py_docs.md)
- **`memory`**: [runner.py_docs.md](./runner.py_docs.md)
- **`parse`**: [runner.py_docs.md](./runner.py_docs.md)
- **`parse_args`**: [runner.py_docs.md](./runner.py_docs.md)
- **`parse_logs`**: [runner.py_docs.md](./runner.py_docs.md)
- **`passrate`**: [runner.py_docs.md](./runner.py_docs.md)
- **`percentage`**: [runner.py_docs.md](./runner.py_docs.md)
- **`plot_graph`**: [runner.py_docs.md](./runner.py_docs.md)
- **`prepare_message`**: [runner.py_docs.md](./runner.py_docs.md)
- **`print_commit_hash`**: [runner.py_docs.md](./runner.py_docs.md)
- **`read_csv`**: [runner.py_docs.md](./runner.py_docs.md)
- **`update`**: [runner.py_docs.md](./runner.py_docs.md)
- **`update_lookup_file`**: [runner.py_docs.md](./runner.py_docs.md)
- **`upload_graphs`**: [runner.py_docs.md](./runner.py_docs.md)

### Imports

- **`abspath`**: [runner.py_docs.md](./runner.py_docs.md)
- **`argparse`**: [runner.py_docs.md](./runner.py_docs.md)
- **`collections`**: [runner.py_docs.md](./runner.py_docs.md)
- **`dataclasses`**: [runner.py_docs.md](./runner.py_docs.md)
- **`datetime`**: [runner.py_docs.md](./runner.py_docs.md)
- **`defaultdict`**: [runner.py_docs.md](./runner.py_docs.md)
- **`functools`**: [runner.py_docs.md](./runner.py_docs.md)
- **`git`**: [runner.py_docs.md](./runner.py_docs.md)
- **`glob`**: [runner.py_docs.md](./runner.py_docs.md)
- **`gmean`**: [runner.py_docs.md](./runner.py_docs.md)
- **`importlib`**: [runner.py_docs.md](./runner.py_docs.md)
- **`io`**: [runner.py_docs.md](./runner.py_docs.md)
- **`itertools`**: [runner.py_docs.md](./runner.py_docs.md)
- **`logging`**: [runner.py_docs.md](./runner.py_docs.md)
- **`matplotlib`**: [runner.py_docs.md](./runner.py_docs.md)
- **`matplotlib.pyplot`**: [runner.py_docs.md](./runner.py_docs.md)
- **`numpy`**: [runner.py_docs.md](./runner.py_docs.md)
- **`os`**: [runner.py_docs.md](./runner.py_docs.md)
- **`os.path`**: [runner.py_docs.md](./runner.py_docs.md)
- **`pandas`**: [runner.py_docs.md](./runner.py_docs.md)
- **`platform`**: [runner.py_docs.md](./runner.py_docs.md)
- **`randint`**: [runner.py_docs.md](./runner.py_docs.md)
- **`random`**: [runner.py_docs.md](./runner.py_docs.md)
- **`rcParams`**: [runner.py_docs.md](./runner.py_docs.md)
- **`re`**: [runner.py_docs.md](./runner.py_docs.md)
- **`scipy.stats`**: [runner.py_docs.md](./runner.py_docs.md)
- **`shutil`**: [runner.py_docs.md](./runner.py_docs.md)
- **`subprocess`**: [runner.py_docs.md](./runner.py_docs.md)
- **`sys`**: [runner.py_docs.md](./runner.py_docs.md)
- **`tabulate`**: [runner.py_docs.md](./runner.py_docs.md)
- **`tempfile`**: [runner.py_docs.md](./runner.py_docs.md)
- **`torch`**: [runner.py_docs.md](./runner.py_docs.md)
- **`torch._dynamo`**: [runner.py_docs.md](./runner.py_docs.md)


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
