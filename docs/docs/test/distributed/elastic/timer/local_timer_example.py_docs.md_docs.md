# Documentation: `docs/test/distributed/elastic/timer/local_timer_example.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/elastic/timer/local_timer_example.py_docs.md`
- **Size**: 6,877 bytes (6.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/distributed/elastic/timer/local_timer_example.py`

## File Metadata

- **Path**: `test/distributed/elastic/timer/local_timer_example.py`
- **Size**: 4,180 bytes (4.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import multiprocessing as mp
import signal
import time

import torch.distributed.elastic.timer as timer
import torch.multiprocessing as torch_mp
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
)


def _happy_function(rank, mp_queue):
    timer.configure(timer.LocalTimerClient(mp_queue))
    with timer.expires(after=1):
        time.sleep(0.5)


def _stuck_function(rank, mp_queue):
    timer.configure(timer.LocalTimerClient(mp_queue))
    with timer.expires(after=1):
        time.sleep(5)


# timer is not supported on these platforms
if not (IS_WINDOWS or IS_MACOS or IS_ARM64):

    class LocalTimerExample(TestCase):
        """
        Demonstrates how to use LocalTimerServer and LocalTimerClient
        to enforce expiration of code-blocks.

        Since torch multiprocessing's ``start_process`` method currently
        does not take the multiprocessing context as parameter argument
        there is no way to create the mp.Queue in the correct
        context BEFORE spawning child processes. Once the ``start_process``
        API is changed in torch, then re-enable ``test_torch_mp_example``
        unittest. As of now this will SIGSEGV.
        """

        @skip_but_pass_in_sandcastle_if(
            TEST_WITH_DEV_DBG_ASAN, "test is asan incompatible"
        )
        def test_torch_mp_example(self):
            # in practice set the max_interval to a larger value (e.g. 60 seconds)
            mp_queue = mp.get_context("spawn").Queue()
            server = timer.LocalTimerServer(mp_queue, max_interval=0.01)
            server.start()

            world_size = 8

            # all processes should complete successfully
            # since start_process does NOT take context as parameter argument yet
            # this method WILL FAIL (hence the test is disabled)
            torch_mp.spawn(
                fn=_happy_function, args=(mp_queue,), nprocs=world_size, join=True
            )

            with self.assertRaises(Exception):
                # torch.multiprocessing.spawn kills all sub-procs
                # if one of them gets killed
                torch_mp.spawn(
                    fn=_stuck_function, args=(mp_queue,), nprocs=world_size, join=True
                )

            server.stop()

        @skip_but_pass_in_sandcastle_if(
            TEST_WITH_DEV_DBG_ASAN, "test is asan incompatible"
        )
        def test_example_start_method_spawn(self):
            self._run_example_with(start_method="spawn")

        # @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, "test is asan incompatible")
        # def test_example_start_method_forkserver(self):
        #     self._run_example_with(start_method="forkserver")

        def _run_example_with(self, start_method):
            spawn_ctx = mp.get_context(start_method)
            mp_queue = spawn_ctx.Queue()
            server = timer.LocalTimerServer(mp_queue, max_interval=0.01)
            server.start()

            world_size = 8
            processes = []
            for i in range(world_size):
                if i % 2 == 0:
                    p = spawn_ctx.Process(target=_stuck_function, args=(i, mp_queue))
                else:
                    p = spawn_ctx.Process(target=_happy_function, args=(i, mp_queue))
                p.start()
                processes.append(p)

            for i in range(world_size):
                p = processes[i]
                p.join()
                if i % 2 == 0:
                    self.assertEqual(-signal.SIGKILL, p.exitcode)
                else:
                    self.assertEqual(0, p.exitcode)

            server.stop()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""        Demonstrates how to use LocalTimerServer and LocalTimerClient        to enforce expiration of code-blocks.

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LocalTimerExample`

**Functions defined**: `_happy_function`, `_stuck_function`, `test_torch_mp_example`, `test_example_start_method_spawn`, `test_example_start_method_forkserver`, `_run_example_with`

**Key imports**: logging, multiprocessing as mp, signal, time, torch.distributed.elastic.timer as timer, torch.multiprocessing as torch_mp


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/timer`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `multiprocessing as mp`
- `signal`
- `time`
- `torch.distributed.elastic.timer as timer`
- `torch.multiprocessing as torch_mp`


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

This is a test file. Run it with:

```bash
python test/distributed/elastic/timer/local_timer_example.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic/timer`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`api_test.py_docs.md`](./api_test.py_docs.md)
- [`local_timer_test.py_docs.md`](./local_timer_test.py_docs.md)
- [`file_based_local_timer_test.py_docs.md`](./file_based_local_timer_test.py_docs.md)


## Cross-References

- **File Documentation**: `local_timer_example.py_docs.md`
- **Keyword Index**: `local_timer_example.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/elastic/timer`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/elastic/timer`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python docs/test/distributed/elastic/timer/local_timer_example.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/elastic/timer`):

- [`file_based_local_timer_test.py_kw.md_docs.md`](./file_based_local_timer_test.py_kw.md_docs.md)
- [`local_timer_example.py_kw.md_docs.md`](./local_timer_example.py_kw.md_docs.md)
- [`file_based_local_timer_test.py_docs.md_docs.md`](./file_based_local_timer_test.py_docs.md_docs.md)
- [`local_timer_test.py_docs.md_docs.md`](./local_timer_test.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`local_timer_test.py_kw.md_docs.md`](./local_timer_test.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`api_test.py_kw.md_docs.md`](./api_test.py_kw.md_docs.md)
- [`api_test.py_docs.md_docs.md`](./api_test.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `local_timer_example.py_docs.md_docs.md`
- **Keyword Index**: `local_timer_example.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
