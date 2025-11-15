# Keyword Index: `torch/distributed/elastic/timer/file_based_local_timer.py`

## File Information

- **Original File**: [torch/distributed/elastic/timer/file_based_local_timer.py](../../../../../torch/distributed/elastic/timer/file_based_local_timer.py)
- **Documentation**: [`file_based_local_timer.py_docs.md`](./file_based_local_timer.py_docs.md)
- **Folder**: `torch/distributed/elastic/timer`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FileTimerClient`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`FileTimerRequest`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`FileTimerServer`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)

### Functions

- **`__eq__`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`__init__`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`_get_requests`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`_get_scopes`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`_open_non_blocking`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`_reap_worker`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`_retry`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`_run_watchdog`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`_send_request`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`_watchdog_loop`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`acquire`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`clear_timers`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`get_expired_timers`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`get_last_progress_time`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`is_process_running`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`register_timers`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`release`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`run_once`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`start`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`stop`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`to_json`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`wrapper`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)

### Imports

- **`Callable`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`Optional`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`ParamSpec`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`TimerClient`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`collections.abc`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`get_logger`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`io`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`json`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`os`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`select`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`signal`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`sys`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`threading`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`time`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`torch.distributed.elastic.timer.api`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`torch.distributed.elastic.timer.debug_info_logging`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`torch.distributed.elastic.utils.logging`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`typing`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)
- **`typing_extensions`**: [file_based_local_timer.py_docs.md](./file_based_local_timer.py_docs.md)


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
