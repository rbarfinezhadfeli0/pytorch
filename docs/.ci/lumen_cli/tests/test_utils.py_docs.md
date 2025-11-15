# Documentation: `.ci/lumen_cli/tests/test_utils.py`

## File Metadata

- **Path**: `.ci/lumen_cli/tests/test_utils.py`
- **Size**: 4,594 bytes (4.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
import os
import tempfile
import unittest
from pathlib import Path

from cli.lib.common.utils import temp_environ, working_directory  # <-- replace import


class EnvIsolatedTestCase(unittest.TestCase):
    """Base class that snapshots os.environ and CWD for isolation."""

    def setUp(self):
        import os
        import tempfile

        self._env_backup = dict(os.environ)

        # Snapshot/repair CWD if it's gone
        try:
            self._cwd_backup = os.getcwd()
        except FileNotFoundError:
            # If CWD no longer exists, switch to a safe place and record that
            self._cwd_backup = tempfile.gettempdir()
            os.chdir(self._cwd_backup)

        # Create a temporary directory for the test to run in
        self._temp_dir = tempfile.mkdtemp()
        os.chdir(self._temp_dir)

    def tearDown(self):
        import os
        import shutil
        import tempfile

        # Restore cwd first (before cleaning up temp dir)
        try:
            os.chdir(self._cwd_backup)
        except OSError:
            os.chdir(tempfile.gettempdir())

        # Clean up temporary directory
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors

        # Restore env
        to_del = set(os.environ.keys()) - set(self._env_backup.keys())
        for k in to_del:
            os.environ.pop(k, None)
        for k, v in self._env_backup.items():
            os.environ[k] = v


class TestTempEnviron(EnvIsolatedTestCase):
    def test_sets_and_restores_new_var(self):
        var = "TEST_TMP_ENV_NEW"
        self.assertNotIn(var, os.environ)

        with temp_environ({var: "123"}):
            self.assertEqual(os.environ[var], "123")

        self.assertNotIn(var, os.environ)  # removed after exit

    def test_overwrites_and_restores_existing_var(self):
        var = "TEST_TMP_ENV_OVERWRITE"
        os.environ[var] = "orig"

        with temp_environ({var: "override"}):
            self.assertEqual(os.environ[var], "override")

        self.assertEqual(os.environ[var], "orig")  # restored

    def test_multiple_vars_and_missing_cleanup(self):
        v1, v2 = "TEST_ENV_V1", "TEST_ENV_V2"
        os.environ.pop(v1, None)
        os.environ[v2] = "keep"

        with temp_environ({v1: "a", v2: "b"}):
            self.assertEqual(os.environ[v1], "a")
            self.assertEqual(os.environ[v2], "b")

        self.assertNotIn(v1, os.environ)  # newly-added -> removed
        self.assertEqual(os.environ[v2], "keep")  # pre-existing -> restored

    def test_restores_even_on_exception(self):
        var = "TEST_TMP_ENV_EXCEPTION"
        self.assertNotIn(var, os.environ)

        with self.assertRaises(RuntimeError):
            with temp_environ({var: "x"}):
                self.assertEqual(os.environ[var], "x")
                raise RuntimeError("boom")

        self.assertNotIn(var, os.environ)  # removed after exception


class TestWorkingDirectory(EnvIsolatedTestCase):
    def test_changes_and_restores(self):
        start = Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "wd"
            target.mkdir()

            with working_directory(str(target)):
                self.assertEqual(Path.cwd().resolve(), target.resolve())

        self.assertEqual(Path.cwd(), start)

    def test_noop_when_empty_path(self):
        start = Path.cwd()
        with working_directory(""):
            self.assertEqual(Path.cwd(), start)
        self.assertEqual(Path.cwd(), start)

    def test_restores_on_exception(self):
        start = Path.cwd()

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "wd_exc"
            target.mkdir()

            with self.assertRaises(ValueError):
                with working_directory(str(target)):
                    # Normalize both sides to handle /var -> /private/var
                    self.assertEqual(Path.cwd().resolve(), target.resolve())
                    raise ValueError("boom")

        self.assertEqual(Path.cwd().resolve(), start.resolve())

    def test_raises_for_missing_dir(self):
        start = Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "does_not_exist"
            with self.assertRaises(FileNotFoundError):
                # os.chdir should raise before yielding
                with working_directory(str(missing)):
                    pass
        self.assertEqual(Path.cwd(), start)


if __name__ == "__main__":
    unittest.main(verbosity=2)

```



## High-Level Overview

"""Base class that snapshots os.environ and CWD for isolation."""    def setUp(self):        import os        import tempfile        self._env_backup = dict(os.environ)        # Snapshot/repair CWD if it's gone        try:            self._cwd_backup = os.getcwd()        except FileNotFoundError:            # If CWD no longer exists, switch to a safe place and record that            self._cwd_backup = tempfile.gettempdir()            os.chdir(self._cwd_backup)        # Create a temporary directory for the test to run in        self._temp_dir = tempfile.mkdtemp()        os.chdir(self._temp_dir)    def tearDown(self):        import os        import shutil        import tempfile        # Restore cwd first (before cleaning up temp dir)        try:            os.chdir(self._cwd_backup)        except OSError:            os.chdir(tempfile.gettempdir())        # Clean up temporary directory        try:            shutil.rmtree(self._temp_dir, ignore_errors=True)        except Exception:            pass  # Ignore cleanup errors        # Restore env        to_del = set(os.environ.keys()) - set(self._env_backup.keys())        for k in to_del:            os.environ.pop(k, None)

This Python file contains 4 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EnvIsolatedTestCase`, `TestTempEnviron`, `TestWorkingDirectory`

**Functions defined**: `setUp`, `tearDown`, `test_sets_and_restores_new_var`, `test_overwrites_and_restores_existing_var`, `test_multiple_vars_and_missing_cleanup`, `test_restores_even_on_exception`, `test_changes_and_restores`, `test_noop_when_empty_path`, `test_restores_on_exception`, `test_raises_for_missing_dir`

**Key imports**: os, tempfile, unittest, Path, temp_environ, working_directory  , class EnvIsolatedTestCase, os, tempfile, os, shutil


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/lumen_cli/tests`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `tempfile`
- `unittest`
- `pathlib`: Path
- `cli.lib.common.utils`: temp_environ, working_directory  
- `class EnvIsolatedTestCase`
- `shutil`


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

This is a test file. Run it with:

```bash
python .ci/lumen_cli/tests/test_utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.ci/lumen_cli/tests`):

- [`test_cli_helper.py_docs.md`](./test_cli_helper.py_docs.md)
- [`test_app.py_docs.md`](./test_app.py_docs.md)
- [`test_docker_helper.py_docs.md`](./test_docker_helper.py_docs.md)
- [`test_vllm.py_docs.md`](./test_vllm.py_docs.md)
- [`test_run_plan.py_docs.md`](./test_run_plan.py_docs.md)
- [`test_envs_helper.py_docs.md`](./test_envs_helper.py_docs.md)
- [`test_path_helper.py_docs.md`](./test_path_helper.py_docs.md)


## Cross-References

- **File Documentation**: `test_utils.py_docs.md`
- **Keyword Index**: `test_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
