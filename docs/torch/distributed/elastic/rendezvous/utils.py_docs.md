# Documentation: `torch/distributed/elastic/rendezvous/utils.py`

## File Metadata

- **Path**: `torch/distributed/elastic/rendezvous/utils.py`
- **Size**: 8,417 bytes (8.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ipaddress
import random
import re
import socket
import time
import weakref
from collections.abc import Callable
from datetime import timedelta
from threading import Event, Thread
from typing import Any, Optional, Union


__all__ = ["parse_rendezvous_endpoint"]


def _parse_rendezvous_config(config_str: str) -> dict[str, str]:
    """Extract key-value pairs from a rendezvous configuration string.

    Args:
        config_str:
            A string in format <key1>=<value1>,...,<keyN>=<valueN>.
    """
    config: dict[str, str] = {}

    config_str = config_str.strip()
    if not config_str:
        return config

    key_values = config_str.split(",")
    for kv in key_values:
        key, *values = kv.split("=", 1)

        key = key.strip()
        if not key:
            raise ValueError(
                "The rendezvous configuration string must be in format "
                "<key1>=<value1>,...,<keyN>=<valueN>."
            )

        value: Optional[str]
        if values:
            value = values[0].strip()
        else:
            value = None
        if not value:
            raise ValueError(
                f"The rendezvous configuration option '{key}' must have a value specified."
            )

        config[key] = value
    return config


def _try_parse_port(port_str: str) -> Optional[int]:
    """Try to extract the port number from ``port_str``."""
    if port_str and re.match(r"^[0-9]{1,5}$", port_str):
        return int(port_str)
    return None


def parse_rendezvous_endpoint(
    endpoint: Optional[str], default_port: int
) -> tuple[str, int]:
    """Extract the hostname and the port number from a rendezvous endpoint.

    Args:
        endpoint:
            A string in format <hostname>[:<port>].
        default_port:
            The port number to use if the endpoint does not include one.

    Returns:
        A tuple of hostname and port number.
    """
    if endpoint is not None:
        endpoint = endpoint.strip()

    if not endpoint:
        return ("localhost", default_port)

    # An endpoint that starts and ends with brackets represents an IPv6 address.
    if endpoint[0] == "[" and endpoint[-1] == "]":
        host, *rest = endpoint, *[]
    else:
        host, *rest = endpoint.rsplit(":", 1)

    # Sanitize the IPv6 address.
    if len(host) > 1 and host[0] == "[" and host[-1] == "]":
        host = host[1:-1]

    if len(rest) == 1:
        port = _try_parse_port(rest[0])
        if port is None or port >= 2**16:
            raise ValueError(
                f"The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                "between 0 and 65536."
            )
    else:
        port = default_port

    if not re.match(r"^[\w\.:-]+$", host):
        raise ValueError(
            f"The hostname of the rendezvous endpoint '{endpoint}' must be a dot-separated list of "
            "labels, an IPv4 address, or an IPv6 address."
        )

    return host, port


def _matches_machine_hostname(host: str) -> bool:
    """Indicate whether ``host`` matches the hostname of this machine.

    This function compares ``host`` to the hostname as well as to the IP
    addresses of this machine. Note that it may return a false negative if this
    machine has CNAME records beyond its FQDN or IP addresses assigned to
    secondary NICs.
    """
    if host == "localhost":
        return True

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None

    if addr and addr.is_loopback:
        return True

    try:
        host_addr_list = socket.getaddrinfo(
            host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME
        )
    except (ValueError, socket.gaierror) as _:
        host_addr_list = []

    host_ip_list = [host_addr_info[4][0] for host_addr_info in host_addr_list]

    this_host = socket.gethostname()
    if host == this_host:
        return True

    addr_list = socket.getaddrinfo(
        this_host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME
    )
    for addr_info in addr_list:
        # If we have an FQDN in the addr_info, compare it to `host`.
        if addr_info[3] and addr_info[3] == host:
            return True

        # Otherwise if `host` represents an IP address, compare it to our IP
        # address.
        if addr and addr_info[4][0] == str(addr):
            return True

        # If the IP address matches one of the provided host's IP addresses
        if addr_info[4][0] in host_ip_list:
            return True

    return False


def _delay(seconds: Union[float, tuple[float, float]]) -> None:
    """Suspend the current thread for ``seconds``.

    Args:
        seconds:
            Either the delay, in seconds, or a tuple of a lower and an upper
            bound within which a random delay will be picked.
    """
    if isinstance(seconds, tuple):
        seconds = random.uniform(*seconds)
    # Ignore delay requests that are less than 10 milliseconds.
    if seconds >= 0.01:
        time.sleep(seconds)


class _PeriodicTimer:
    """Represent a timer that periodically runs a specified function.

    Args:
        interval:
            The interval, in seconds, between each run.
        function:
            The function to run.
    """

    # The state of the timer is hold in a separate context object to avoid a
    # reference cycle between the timer and the background thread.
    class _Context:
        interval: float
        function: Callable[..., None]
        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        stop_event: Event

    _name: Optional[str]
    _thread: Optional[Thread]
    _finalizer: Optional[weakref.finalize]

    # The context that is shared between the timer and the background thread.
    _ctx: _Context

    def __init__(
        self,
        interval: timedelta,
        function: Callable[..., None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._name = None

        self._ctx = self._Context()
        self._ctx.interval = interval.total_seconds()
        self._ctx.function = function  # type: ignore[assignment]
        self._ctx.args = args or ()
        self._ctx.kwargs = kwargs or {}
        self._ctx.stop_event = Event()

        self._thread = None
        self._finalizer = None

    @property
    def name(self) -> Optional[str]:
        """Get the name of the timer."""
        return self._name

    def set_name(self, name: str) -> None:
        """Set the name of the timer.

        The specified name will be assigned to the background thread and serves
        for debugging and troubleshooting purposes.
        """
        if self._thread:
            raise RuntimeError("The timer has already started.")

        self._name = name

    def start(self) -> None:
        """Start the timer."""
        if self._thread:
            raise RuntimeError("The timer has already started.")

        self._thread = Thread(
            target=self._run,
            name=self._name or "PeriodicTimer",
            args=(self._ctx,),
            daemon=True,
        )

        # We avoid using a regular finalizer (a.k.a. __del__) for stopping the
        # timer as joining a daemon thread during the interpreter shutdown can
        # cause deadlocks. The weakref.finalize is a superior alternative that
        # provides a consistent behavior regardless of the GC implementation.
        self._finalizer = weakref.finalize(
            self, self._stop_thread, self._thread, self._ctx.stop_event
        )

        # We do not attempt to stop our background thread during the interpreter
        # shutdown. At that point we do not even know whether it still exists.
        self._finalizer.atexit = False

        self._thread.start()

    def cancel(self) -> None:
        """Stop the timer at the next opportunity."""
        if self._finalizer:
            self._finalizer()

    @staticmethod
    def _run(ctx) -> None:
        while not ctx.stop_event.wait(ctx.interval):
            ctx.function(*ctx.args, **ctx.kwargs)

    @staticmethod
    def _stop_thread(thread, stop_event):
        stop_event.set()

        thread.join()

```



## High-Level Overview

"""Extract key-value pairs from a rendezvous configuration string.    Args:        config_str:            A string in format <key1>=<value1>,...,<keyN>=<valueN>.

This Python file contains 2 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_PeriodicTimer`, `_Context`

**Functions defined**: `_parse_rendezvous_config`, `_try_parse_port`, `parse_rendezvous_endpoint`, `_matches_machine_hostname`, `_delay`, `__init__`, `name`, `set_name`, `start`, `cancel`, `_run`, `_stop_thread`

**Key imports**: ipaddress, random, re, socket, time, weakref, Callable, timedelta, Event, Thread, Any, Optional, Union


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/elastic/rendezvous`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `ipaddress`
- `random`
- `re`
- `socket`
- `time`
- `weakref`
- `collections.abc`: Callable
- `datetime`: timedelta
- `threading`: Event, Thread
- `typing`: Any, Optional, Union


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/distributed/elastic/rendezvous`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`etcd_rendezvous_backend.py_docs.md`](./etcd_rendezvous_backend.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`dynamic_rendezvous.py_docs.md`](./dynamic_rendezvous.py_docs.md)
- [`etcd_server.py_docs.md`](./etcd_server.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`etcd_rendezvous.py_docs.md`](./etcd_rendezvous.py_docs.md)
- [`etcd_store.py_docs.md`](./etcd_store.py_docs.md)
- [`c10d_rendezvous_backend.py_docs.md`](./c10d_rendezvous_backend.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
