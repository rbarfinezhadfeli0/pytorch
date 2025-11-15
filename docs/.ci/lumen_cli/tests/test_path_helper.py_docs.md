# Documentation: `.ci/lumen_cli/tests/test_path_helper.py`

## File Metadata

- **Path**: `.ci/lumen_cli/tests/test_path_helper.py`
- **Size**: 3,959 bytes (3.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# test_path_utils.py
# Run: pytest -q

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cli.lib.common.path_helper import (
    copy,
    ensure_dir_exists,
    force_create_dir,
    get_path,
    is_path_exist,
    remove_dir,
)


class TestPathHelper(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    # -------- get_path --------
    def test_get_path_returns_path_for_str(self):
        # Use relative path to avoid absolute-ness
        rel_str = "sub/f.txt"
        os.chdir(self.tmp_path)
        p = get_path(rel_str, resolve=False)
        self.assertIsInstance(p, Path)
        self.assertFalse(p.is_absolute())
        self.assertEqual(str(p), rel_str)

    def test_get_path_resolves(self):
        rel_str = "sub/f.txt"
        p = get_path(str(self.tmp_path / rel_str), resolve=True)
        self.assertTrue(p.is_absolute())
        self.assertTrue(str(p).endswith(rel_str))

    def test_get_path_with_path_input(self):
        p_in = self.tmp_path / "sub/f.txt"
        p_out = get_path(p_in, resolve=False)
        self.assertTrue(str(p_out) == str(p_in))

    def test_get_path_with_none_raises(self):
        with self.assertRaises(ValueError):
            get_path(None)  # type: ignore[arg-type]

    def test_get_path_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            get_path(123)  # type: ignore[arg-type]

    # -------- ensure_dir_exists / force_create_dir / remove_dir --------
    def test_ensure_dir_exists_creates_and_is_idempotent(self):
        d = self.tmp_path / "made"
        ensure_dir_exists(d)
        self.assertTrue(d.exists() and d.is_dir())
        ensure_dir_exists(d)

    def test_force_create_dir_clears_existing(self):
        d = self.tmp_path / "fresh"
        (d / "inner").mkdir(parents=True)
        (d / "inner" / "f.txt").write_text("x")
        force_create_dir(d)
        self.assertTrue(d.exists())
        self.assertEqual(list(d.iterdir()), [])

    def test_remove_dir_none_is_noop(self):
        remove_dir(None)  # type: ignore[arg-type]

    def test_remove_dir_nonexistent_is_noop(self):
        ghost = self.tmp_path / "ghost"
        remove_dir(ghost)

    def test_remove_dir_accepts_str(self):
        d = self.tmp_path / "to_rm"
        d.mkdir()
        remove_dir(str(d))
        self.assertFalse(d.exists())

    # -------- copy --------
    def test_copy_file_to_file(self):
        src = self.tmp_path / "src.txt"
        dst = self.tmp_path / "out" / "dst.txt"
        src.write_text("hello")
        copy(src, dst)
        self.assertEqual(dst.read_text(), "hello")

    def test_copy_dir_to_new_dir(self):
        src = self.tmp_path / "srcdir"
        (src / "a").mkdir(parents=True)
        (src / "a" / "f.txt").write_text("content")
        dst = self.tmp_path / "destdir"
        copy(src, dst)
        self.assertEqual((dst / "a" / "f.txt").read_text(), "content")

    def test_copy_dir_into_existing_dir_overwrite_true_merges(self):
        src = self.tmp_path / "srcdir"
        dst = self.tmp_path / "destdir"
        (src / "x").mkdir(parents=True)
        (src / "x" / "new.txt").write_text("new")
        dst.mkdir()
        (dst / "existing.txt").write_text("old")
        copy(src, dst)
        self.assertEqual((dst / "existing.txt").read_text(), "old")
        self.assertEqual((dst / "x" / "new.txt").read_text(), "new")

    def test_is_str_path_exist(self):
        p = self.tmp_path / "x.txt"
        p.write_text("1")
        self.assertTrue(is_path_exist(str(p)))
        self.assertTrue(is_path_exist(p))
        self.assertFalse(is_path_exist(str(self.tmp_path / "missing")))
        self.assertFalse(is_path_exist(self.tmp_path / "missing"))
        self.assertFalse(is_path_exist(""))


if __name__ == "__main__":
    unittest.main()

```



## High-Level Overview


This Python file contains 1 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPathHelper`

**Functions defined**: `setUp`, `tearDown`, `test_get_path_returns_path_for_str`, `test_get_path_resolves`, `test_get_path_with_path_input`, `test_get_path_with_none_raises`, `test_get_path_invalid_type_raises`, `test_ensure_dir_exists_creates_and_is_idempotent`, `test_force_create_dir_clears_existing`, `test_remove_dir_none_is_noop`, `test_remove_dir_nonexistent_is_noop`, `test_remove_dir_accepts_str`, `test_copy_file_to_file`, `test_copy_dir_to_new_dir`, `test_copy_dir_into_existing_dir_overwrite_true_merges`, `test_is_str_path_exist`

**Key imports**: os, unittest, Path, TemporaryDirectory


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/lumen_cli/tests`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest`
- `pathlib`: Path
- `tempfile`: TemporaryDirectory


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
python .ci/lumen_cli/tests/test_path_helper.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.ci/lumen_cli/tests`):

- [`test_cli_helper.py_docs.md`](./test_cli_helper.py_docs.md)
- [`test_app.py_docs.md`](./test_app.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_docker_helper.py_docs.md`](./test_docker_helper.py_docs.md)
- [`test_vllm.py_docs.md`](./test_vllm.py_docs.md)
- [`test_run_plan.py_docs.md`](./test_run_plan.py_docs.md)
- [`test_envs_helper.py_docs.md`](./test_envs_helper.py_docs.md)


## Cross-References

- **File Documentation**: `test_path_helper.py_docs.md`
- **Keyword Index**: `test_path_helper.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
