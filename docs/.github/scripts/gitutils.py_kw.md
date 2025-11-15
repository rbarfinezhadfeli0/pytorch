# Keyword Index: `.github/scripts/gitutils.py`

## File Information

- **Original File**: [.github/scripts/gitutils.py](../../../.github/scripts/gitutils.py)
- **Documentation**: [`gitutils.py_docs.md`](./gitutils.py_docs.md)
- **Folder**: `.github/scripts`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GitCommit`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`GitRepo`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`PeekableIterator`**: [gitutils.py_docs.md](./gitutils.py_docs.md)

### Functions

- **`__contains__`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`__init__`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`__iter__`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`__next__`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`__repr__`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`_check_output`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`_run_git`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`_shasum`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`amend_commit_message`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`are_ghstack_branches_in_sync`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`branches_containing_ref`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`checkout`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`cherry_pick`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`cherry_pick_commits`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`clone_repo`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`commit_message`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`commits_resolving_gh_pr`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`compute_branch_diffs`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`create_branch_and_checkout`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`current_branch`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`decorator`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`diff`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`fetch`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`fuzzy_list_to_dict`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`get_commit`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`get_git_remote_name`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`get_git_repo_dir`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`get_merge_base`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`gh_owner_and_name`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`head_hash`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`is_commit_hash`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`parse_fuller_format`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`patch_id`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`patterns_to_regex`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`peek`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`push`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`remote_url`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`retries_decorator`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`rev_parse`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`revert`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`revlist`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`show_ref`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`wrapper`**: [gitutils.py_docs.md](./gitutils.py_docs.md)

### Imports

- **`Any`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`Callable`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`CalledProcessError`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`Path`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`collections`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`collections.abc`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`datetime`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`defaultdict`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`functools`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`hashlib`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`os`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`pathlib`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`re`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`subprocess`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`tempfile`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`typing`**: [gitutils.py_docs.md](./gitutils.py_docs.md)
- **`wraps`**: [gitutils.py_docs.md](./gitutils.py_docs.md)


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
