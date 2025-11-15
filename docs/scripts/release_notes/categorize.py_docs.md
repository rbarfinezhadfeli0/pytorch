# Documentation: `scripts/release_notes/categorize.py`

## File Metadata

- **Path**: `scripts/release_notes/categorize.py`
- **Size**: 7,143 bytes (6.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import os
import textwrap
from pathlib import Path

import common

# Imports for working with classi
from classifier import (
    CategoryConfig,
    CommitClassifier,
    CommitClassifierInputs,
    get_author_map,
    get_file_map,
    XLMR_BASE,
)
from commitlist import CommitList
from common import get_commit_data_cache, topics

import torch


class Categorizer:
    def __init__(self, path, category="Uncategorized", use_classifier: bool = False):
        self.cache = get_commit_data_cache()
        self.commits = CommitList.from_existing(path)
        if use_classifier:
            print("Using a classifier to aid with categorization.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            classifier_config = CategoryConfig(common.categories)
            author_map = get_author_map(
                Path("results/classifier"), regen_data=False, assert_stored=True
            )
            file_map = get_file_map(
                Path("results/classifier"), regen_data=False, assert_stored=True
            )
            self.classifier = CommitClassifier(
                XLMR_BASE, author_map, file_map, classifier_config
            ).to(device)
            self.classifier.load_state_dict(
                torch.load(Path("results/classifier/commit_classifier.pt"))
            )
            self.classifier.eval()
        else:
            self.classifier = None
        # Special categories: 'Uncategorized'
        # All other categories must be real
        self.category = category

    def categorize(self):
        commits = self.commits.filter(category=self.category)
        total_commits = len(self.commits.commits)
        already_done = total_commits - len(commits)
        i = 0
        while i < len(commits):
            cur_commit = commits[i]
            next_commit = commits[i + 1] if i + 1 < len(commits) else None
            jump_to = self.handle_commit(
                cur_commit, already_done + i + 1, total_commits, commits
            )

            # Increment counter
            if jump_to is not None:
                i = jump_to
            elif next_commit is None:
                i = len(commits)
            else:
                i = commits.index(next_commit)

    def features(self, commit):
        return self.cache.get(commit.commit_hash)

    def potential_reverts_of(self, commit, commits):
        submodule_update_str = [
            "Update TensorPipe submodule",
            "Updating submodules",
            "Automated submodule update",
        ]
        if any(a in commit.title for a in submodule_update_str):
            return []

        features = self.features(commit)
        if "Reverted" in features.labels:
            reasons = {"GithubBot": "Reverted"}
        else:
            reasons = {}

        index = commits.index(commit)
        # -8 to remove the (#35011)
        cleaned_title = commit.title[:-10]
        # NB: the index + 2 is sketch
        reasons.update(
            {
                (index + 2 + delta): cand
                for delta, cand in enumerate(commits[index + 1 :])
                if cleaned_title in cand.title
                and commit.commit_hash != cand.commit_hash
            }
        )
        return reasons

    def handle_commit(self, commit, i, total, commits):
        potential_reverts = self.potential_reverts_of(commit, commits)
        if potential_reverts:
            potential_reverts = f"!!!POTENTIAL REVERTS!!!: {potential_reverts}"
        else:
            potential_reverts = ""

        features = self.features(commit)
        if self.classifier is not None:
            # Some commits don't have authors:
            author = features.author if features.author else "Unknown"
            files = " ".join(features.files_changed)
            classifier_input = CommitClassifierInputs(
                title=[features.title], files=[files], author=[author]
            )
            classifier_category = self.classifier.get_most_likely_category_name(
                classifier_input
            )[0]

        else:
            classifier_category = commit.category

        breaking_alarm = ""
        if "module: bc-breaking" in features.labels:
            breaking_alarm += "\n!!!!!! BC BREAKING !!!!!!"

        if "module: deprecation" in features.labels:
            breaking_alarm += "\n!!!!!! DEPRECATION !!!!!!"

        os.system("clear")
        view = textwrap.dedent(
            f"""\
[{i}/{total}]
================================================================================
{features.title}

{potential_reverts} {breaking_alarm}

{features.body}

Files changed: {features.files_changed}

Labels: {features.labels}

Current category: {commit.category}

Select from: {", ".join(common.categories)}

        """
        )
        print(view)
        cat_choice = None
        while cat_choice is None:
            print("Enter category: ")
            value = input(f"{classifier_category} ").strip()
            if len(value) == 0:
                # The user just pressed enter and likes the default value
                cat_choice = classifier_category
                continue
            choices = [cat for cat in common.categories if cat.startswith(value)]
            if len(choices) != 1:
                print(f"Possible matches: {choices}, try again")
                continue
            cat_choice = choices[0]
        print(f"\nSelected: {cat_choice}")
        print(f"\nCurrent topic: {commit.topic}")
        print(f"""Select from: {", ".join(topics)}""")
        topic_choice = None
        while topic_choice is None:
            value = input("topic> ").strip()
            if len(value) == 0:
                topic_choice = commit.topic
                continue
            choices = [cat for cat in topics if cat.startswith(value)]
            if len(choices) != 1:
                print(f"Possible matches: {choices}, try again")
                continue
            topic_choice = choices[0]
        print(f"\nSelected: {topic_choice}")
        self.update_commit(commit, cat_choice, topic_choice)
        return None

    def update_commit(self, commit, category, topic):
        assert category in common.categories
        assert topic in topics
        commit.category = category
        commit.topic = topic
        self.commits.write_result()


def main():
    parser = argparse.ArgumentParser(description="Tool to help categorize commits")
    parser.add_argument(
        "--category",
        type=str,
        default="Uncategorized",
        help='Which category to filter by. "Uncategorized", None, or a category name',
    )
    parser.add_argument(
        "--file",
        help="The location of the commits CSV",
        default="results/commitlist.csv",
    )
    parser.add_argument(
        "--use_classifier",
        action="store_true",
        help="Whether or not to use a classifier to aid in categorization.",
    )

    args = parser.parse_args()
    categorizer = Categorizer(args.file, args.category, args.use_classifier)
    categorizer.categorize()


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Categorizer`

**Functions defined**: `__init__`, `categorize`, `features`, `potential_reverts_of`, `handle_commit`, `update_commit`, `main`

**Key imports**: argparse, os, textwrap, Path, common, CommitList, get_commit_data_cache, topics, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `scripts/release_notes`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `os`
- `textwrap`
- `pathlib`: Path
- `common`
- `commitlist`: CommitList
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`scripts/release_notes`):

- [`test_release_notes.py_docs.md`](./test_release_notes.py_docs.md)
- [`commitlist.py_docs.md`](./commitlist.py_docs.md)
- [`apply_categories.py_docs.md`](./apply_categories.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`classifier.py_docs.md`](./classifier.py_docs.md)
- [`requirements.txt_docs.md`](./requirements.txt_docs.md)


## Cross-References

- **File Documentation**: `categorize.py_docs.md`
- **Keyword Index**: `categorize.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
