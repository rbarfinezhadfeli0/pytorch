# Documentation: `docs/test/test_hub.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_hub.py_docs.md`
- **Size**: 15,546 bytes (15.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_hub.py`

## File Metadata

- **Path**: `test/test_hub.py`
- **Size**: 12,329 bytes (12.04 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: hub"]

import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

import torch
import torch.hub as hub
from torch.testing._internal.common_utils import (
    IS_SANDCASTLE,
    retry,
    run_tests,
    TestCase,
)


def sum_of_state_dict(state_dict):
    s = 0
    for v in state_dict.values():
        s += v.sum()
    return s


SUM_OF_HUB_EXAMPLE = 431080
TORCHHUB_EXAMPLE_RELEASE_URL = (
    "https://github.com/ailzhang/torchhub_example/releases/download/0.1/mnist_init_ones"
)


@unittest.skipIf(IS_SANDCASTLE, "Sandcastle cannot ping external")
class TestHub(TestCase):
    def setUp(self):
        super().setUp()
        self.previous_hub_dir = torch.hub.get_dir()
        self.tmpdir = tempfile.TemporaryDirectory("hub_dir")
        torch.hub.set_dir(self.tmpdir.name)
        self.trusted_list_path = os.path.join(torch.hub.get_dir(), "trusted_list")

    def tearDown(self):
        super().tearDown()
        torch.hub.set_dir(self.previous_hub_dir)  # probably not needed, but can't hurt
        self.tmpdir.cleanup()

    def _assert_trusted_list_is_empty(self):
        with open(self.trusted_list_path) as f:
            assert not f.readlines()

    def _assert_in_trusted_list(self, line):
        with open(self.trusted_list_path) as f:
            assert line in (l.strip() for l in f)

    @retry(Exception, tries=3)
    def test_load_from_github(self):
        hub_model = hub.load(
            "ailzhang/torchhub_example",
            "mnist",
            source="github",
            pretrained=True,
            verbose=False,
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_local_dir(self):
        local_dir = hub._get_cache_or_reload(
            "ailzhang/torchhub_example",
            force_reload=False,
            trust_repo=True,
            calling_fn=None,
        )
        hub_model = hub.load(
            local_dir, "mnist", source="local", pretrained=True, verbose=False
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_branch(self):
        hub_model = hub.load(
            "ailzhang/torchhub_example:ci/test_slash",
            "mnist",
            pretrained=True,
            verbose=False,
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_get_set_dir(self):
        previous_hub_dir = torch.hub.get_dir()
        with tempfile.TemporaryDirectory("hub_dir") as tmpdir:
            torch.hub.set_dir(tmpdir)
            self.assertEqual(torch.hub.get_dir(), tmpdir)
            self.assertNotEqual(previous_hub_dir, tmpdir)

            hub_model = hub.load(
                "ailzhang/torchhub_example", "mnist", pretrained=True, verbose=False
            )
            self.assertEqual(
                sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE
            )
            assert os.path.exists(
                os.path.join(tmpdir, "ailzhang_torchhub_example_master")
            )

        # Test that set_dir properly calls expanduser()
        # non-regression test for https://github.com/pytorch/pytorch/issues/69761
        new_dir = os.path.join("~", "hub")
        torch.hub.set_dir(new_dir)
        self.assertEqual(torch.hub.get_dir(), os.path.expanduser(new_dir))

    @retry(Exception, tries=3)
    def test_list_entrypoints(self):
        entry_lists = hub.list("ailzhang/torchhub_example", trust_repo=True)
        self.assertObjectIn("mnist", entry_lists)

    @retry(Exception, tries=3)
    def test_download_url_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "temp")
            hub.download_url_to_file(TORCHHUB_EXAMPLE_RELEASE_URL, f, progress=False)
            loaded_state = torch.load(f)
            self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)
            # Check that the downloaded file has default file permissions
            f_ref = os.path.join(tmpdir, "reference")
            open(f_ref, "w").close()
            expected_permissions = oct(os.stat(f_ref).st_mode & 0o777)
            actual_permissions = oct(os.stat(f).st_mode & 0o777)
            assert actual_permissions == expected_permissions

    @retry(Exception, tries=3)
    def test_load_state_dict_from_url(self):
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL)
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

        # with name
        file_name = "the_file_name"
        loaded_state = hub.load_state_dict_from_url(
            TORCHHUB_EXAMPLE_RELEASE_URL, file_name=file_name
        )
        expected_file_path = os.path.join(torch.hub.get_dir(), "checkpoints", file_name)
        self.assertTrue(os.path.exists(expected_file_path))
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

        # with safe weight_only
        loaded_state = hub.load_state_dict_from_url(
            TORCHHUB_EXAMPLE_RELEASE_URL, weights_only=True
        )
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_legacy_zip_checkpoint(self):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            hub_model = hub.load(
                "ailzhang/torchhub_example", "mnist_zip", pretrained=True, verbose=False
            )
            self.assertEqual(
                sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE
            )
            assert any(
                "will be deprecated in favor of default zipfile" in str(w) for w in ws
            )

    # Test the default zipfile serialization format produced by >=1.6 release.
    @retry(Exception, tries=3)
    def test_load_zip_1_6_checkpoint(self):
        hub_model = hub.load(
            "ailzhang/torchhub_example",
            "mnist_zip_1_6",
            pretrained=True,
            verbose=False,
            trust_repo=True,
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_hub_parse_repo_info(self):
        # If the branch is specified we just parse the input and return
        self.assertEqual(torch.hub._parse_repo_info("a/b:c"), ("a", "b", "c"))
        # For torchvision, the default branch is main
        self.assertEqual(
            torch.hub._parse_repo_info("pytorch/vision"), ("pytorch", "vision", "main")
        )
        # For the torchhub_example repo, the default branch is still master
        self.assertEqual(
            torch.hub._parse_repo_info("ailzhang/torchhub_example"),
            ("ailzhang", "torchhub_example", "master"),
        )

    @retry(Exception, tries=3)
    def test_load_commit_from_forked_repo(self):
        with self.assertRaisesRegex(ValueError, "If it's a commit from a forked repo"):
            torch.hub.load("pytorch/vision:4e2c216", "resnet18")

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="")
    def test_trust_repo_false_emptystring(self, patched_input):
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="no")
    def test_trust_repo_false_no(self, patched_input):
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="y")
    def test_trusted_repo_false_yes(self, patched_input):
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False)
        self._assert_in_trusted_list("ailzhang_torchhub_example")
        patched_input.assert_called_once()

        # Loading a second time with "check", we don't ask for user input
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        patched_input.assert_not_called()

        # Loading again with False, we still ask for user input
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False)
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="no")
    def test_trust_repo_check_no(self, patched_input):
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check"
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check"
            )
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="y")
    def test_trust_repo_check_yes(self, patched_input):
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        self._assert_in_trusted_list("ailzhang_torchhub_example")
        patched_input.assert_called_once()

        # Loading a second time with "check", we don't ask for user input
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        patched_input.assert_not_called()

    @retry(Exception, tries=3)
    def test_trust_repo_true(self):
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=True)
        self._assert_in_trusted_list("ailzhang_torchhub_example")

    @retry(Exception, tries=3)
    def test_trust_repo_builtin_trusted_owners(self):
        torch.hub.load("pytorch/vision", "resnet18", trust_repo="check")
        self._assert_trusted_list_is_empty()

    @retry(Exception, tries=3)
    def test_trust_repo_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=None
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (
                "You are about to download and run code from an untrusted repository"
                in str(w[-1].message)
            )

        self._assert_trusted_list_is_empty()

    @retry(Exception, tries=3)
    def test_trust_repo_legacy(self):
        # We first download a repo and then delete the allowlist file
        # Then we check that the repo is indeed trusted without a prompt,
        # because it was already downloaded in the past.
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=True)
        os.remove(self.trusted_list_path)

        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")

        self._assert_trusted_list_is_empty()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 25 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestHub`

**Functions defined**: `sum_of_state_dict`, `setUp`, `tearDown`, `_assert_trusted_list_is_empty`, `_assert_in_trusted_list`, `test_load_from_github`, `test_load_from_local_dir`, `test_load_from_branch`, `test_get_set_dir`, `test_list_entrypoints`, `test_download_url_to_file`, `test_load_state_dict_from_url`, `test_load_legacy_zip_checkpoint`, `test_load_zip_1_6_checkpoint`, `test_hub_parse_repo_info`, `test_load_commit_from_forked_repo`, `test_trust_repo_false_emptystring`, `test_trust_repo_false_no`, `test_trusted_repo_false_yes`, `test_trust_repo_check_no`

**Key imports**: os, tempfile, unittest, warnings, patch, torch, torch.hub as hub


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `tempfile`
- `unittest`
- `warnings`
- `unittest.mock`: patch
- `torch`
- `torch.hub as hub`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_hub.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_hub.py_docs.md`
- **Keyword Index**: `test_hub.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_hub.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_hub.py_docs.md_docs.md`
- **Keyword Index**: `test_hub.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
