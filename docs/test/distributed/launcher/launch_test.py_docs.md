# Documentation: `test/distributed/launcher/launch_test.py`

## File Metadata

- **Path**: `test/distributed/launcher/launch_test.py`
- **Size**: 2,892 bytes (2.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import tempfile
import unittest
from contextlib import closing

import torch.distributed.launch as launch
from torch.distributed.elastic.utils import get_socket_with_port
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # set a sentinel env var on the parent proc
        # this should be present on the child and gets
        # asserted in ``bin/test_script.py``
        os.environ["TEST_SENTINEL_PARENT"] = "FOOBAR"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_without_env(self):
        nnodes = 1
        nproc_per_node = 4
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--master-addr=localhost",
            f"--master-port={master_port}",
            "--node-rank=0",
            path("bin/test_script_local_rank.py"),
        ]
        launch.main(args)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_DEV_DBG_ASAN, "test incompatible with dev/dbg asan"
    )
    def test_launch_with_env(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--master-addr=localhost",
            f"--master-port={master_port}",
            "--node-rank=0",
            "--use-env",
            path("bin/test_script.py"),
            f"--touch-file-dir={self.test_dir}",
        ]
        launch.main(args)
        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LaunchTest`

**Functions defined**: `path`, `setUp`, `tearDown`, `test_launch_without_env`, `test_launch_with_env`

**Key imports**: os, shutil, tempfile, unittest, closing, torch.distributed.launch as launch, get_socket_with_port


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/launcher`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `shutil`
- `tempfile`
- `unittest`
- `contextlib`: closing
- `torch.distributed.launch as launch`
- `torch.distributed.elastic.utils`: get_socket_with_port


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
python test/distributed/launcher/launch_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/launcher`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`api_test.py_docs.md`](./api_test.py_docs.md)
- [`test_api.py_docs.md`](./test_api.py_docs.md)
- [`script_deviceid.py_docs.md`](./script_deviceid.py_docs.md)


## Cross-References

- **File Documentation**: `launch_test.py_docs.md`
- **Keyword Index**: `launch_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
