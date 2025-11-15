# Documentation: `.github/scripts/test_runner_determinator.py`

## File Metadata

- **Path**: `.github/scripts/test_runner_determinator.py`
- **Size**: 13,144 bytes (12.84 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
from unittest import main, TestCase
from unittest.mock import Mock, patch

import runner_determinator as rd


USER_BRANCH = "somebranch"
EXCEPTION_BRANCH = "main"


class TestRunnerDeterminatorIssueParser(TestCase):
    def test_parse_settings(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertTupleEqual(
            rd.Experiment(rollout_perc=0, default=False),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_settings_with_invalid_experiment_name_skips_experiment(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            -badExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,-badExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertNotIn("-badExp", settings.experiments)

    def test_parse_settings_in_code_block(self) -> None:
        settings_text = """

        ```
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 0
                default: false
        ```

        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertTupleEqual(
            rd.Experiment(rollout_perc=0, default=False),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_all_branches_setting(self) -> None:
        settings_text = """
        ```
        experiments:
            lf:
                rollout_perc: 25
                all_branches: true
            otherExp:
                all_branches: True
                rollout_perc: 0
        ```

        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        settings = rd.parse_settings(settings_text)

        self.assertTupleEqual(
            rd.Experiment(rollout_perc=25, all_branches=True),
            settings.experiments["lf"],
            "lf settings not parsed correctly",
        )
        self.assertTrue(settings.experiments["otherExp"].all_branches)
        self.assertTupleEqual(
            rd.Experiment(rollout_perc=0, all_branches=True),
            settings.experiments["otherExp"],
            "otherExp settings not parsed correctly",
        )

    def test_parse_users(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        users = rd.parse_users(settings_text)
        self.assertDictEqual(
            {"User1": ["lf"], "User2": ["lf", "otherExp"]},
            users,
            "Users not parsed correctly",
        )

    def test_parse_users_without_settings(self) -> None:
        settings_text = """

        @User1,lf
        @User2,lf,otherExp

        """

        users = rd.parse_users(settings_text)
        self.assertDictEqual(
            {"User1": ["lf"], "User2": ["lf", "otherExp"]},
            users,
            "Users not parsed correctly",
        )


class TestRunnerDeterminatorGetRunnerPrefix(TestCase):
    def test_opted_in_user(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        prefix = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf.", prefix, "Runner prefix not correct for User1")

    def test_explicitly_opted_out_user(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,-lf
        @User2,lf,otherExp

        """
        prefix = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for User1")

    def test_explicitly_opted_in_and_out_user_should_opt_out(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,-lf,lf
        @User2,lf,otherExp

        """
        prefix = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for User1")

    def test_opted_in_user_two_experiments(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        prefix = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for User2")

    def test_opted_in_user_two_experiments_default(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        prefix = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf.", prefix, "Runner prefix not correct for User2")

    def test_opted_in_user_two_experiments_default_exp(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        prefix = rd.get_runner_prefix(
            settings_text, ["User2"], USER_BRANCH, frozenset(["lf", "otherExp"])
        )
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for User2")

    def test_opted_in_user_two_experiments_default_exp_2(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        prefix = rd.get_runner_prefix(
            settings_text, ["User2"], USER_BRANCH, frozenset(["otherExp"])
        )
        self.assertEqual("otherExp.", prefix, "Runner prefix not correct for User2")

    @patch("random.uniform", return_value=50)
    def test_opted_out_user(self, mock_uniform: Mock) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """
        prefix = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout(self, mock_uniform: Mock) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into both experiments by the 10% rollout
        prefix = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout_excl_nondefault(
        self, mock_uniform: Mock
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into default experiments by the 10% rollout
        prefix = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("lf.", prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=10)
    def test_opted_out_user_was_pulled_in_by_rollout_filter_exp(
        self, mock_uniform: Mock
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 25
            otherExp:
                rollout_perc: 25
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into default experiments by the 10% rollout
        prefix = rd.get_runner_prefix(
            settings_text, ["User3"], USER_BRANCH, frozenset(["otherExp"])
        )
        self.assertEqual("otherExp.", prefix, "Runner prefix not correct for user")

    @patch("random.uniform", return_value=25)
    def test_opted_out_user_was_pulled_out_by_rollout_filter_exp(
        self, mock_uniform: Mock
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 10
            otherExp:
                rollout_perc: 50
                default: false
        ---

        Users:
        @User1,lf
        @User2,lf,otherExp

        """

        # User3 is opted out, but is pulled into default experiments by the 10% rollout
        prefix = rd.get_runner_prefix(settings_text, ["User3"], USER_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for user")

    def test_lf_prefix_always_comes_first(self) -> None:
        settings_text = """
        experiments:
            otherExp:
                rollout_perc: 0
            lf:
                rollout_perc: 0
        ---

        Users:
        @User1,lf
        @User2,otherExp,lf

        """

        prefix = rd.get_runner_prefix(settings_text, ["User2"], USER_BRANCH)
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for user")

    def test_ignores_commented_users(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
        ---

        Users:
        #@User1,lf
        @User2,lf,otherExp

        """

        prefix = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for user")

    def test_ignores_extra_experiments(self) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 0
            otherExp:
                rollout_perc: 0
            foo:
                rollout_perc: 0
        ---

        Users:
        @User1,lf,otherExp,foo

        """

        prefix = rd.get_runner_prefix(settings_text, ["User1"], USER_BRANCH)
        self.assertEqual("lf.otherExp.", prefix, "Runner prefix not correct for user")

    def test_disables_experiment_on_exception_branches_when_not_explicitly_opted_in(
        self,
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
        ---

        Users:
        @User,lf,otherExp

        """

        prefix = rd.get_runner_prefix(settings_text, ["User1"], EXCEPTION_BRANCH)
        self.assertEqual("", prefix, "Runner prefix not correct for user")

    def test_allows_experiment_on_exception_branches_when_explicitly_opted_in(
        self,
    ) -> None:
        settings_text = """
        experiments:
            lf:
                rollout_perc: 100
                all_branches: true
        ---

        Users:
        @User,lf,otherExp

        """

        prefix = rd.get_runner_prefix(settings_text, ["User1"], EXCEPTION_BRANCH)
        self.assertEqual("lf.", prefix, "Runner prefix not correct for user")


if __name__ == "__main__":
    main()

```



## High-Level Overview

settings_text = """        experiments:            lf:                rollout_perc: 25            otherExp:                rollout_perc: 0                default: false        ---        Users:        @User1,lf        @User2,lf,otherExp

This Python file contains 2 class(es) and 23 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestRunnerDeterminatorIssueParser`, `TestRunnerDeterminatorGetRunnerPrefix`

**Functions defined**: `test_parse_settings`, `test_parse_settings_with_invalid_experiment_name_skips_experiment`, `test_parse_settings_in_code_block`, `test_parse_all_branches_setting`, `test_parse_users`, `test_parse_users_without_settings`, `test_opted_in_user`, `test_explicitly_opted_out_user`, `test_explicitly_opted_in_and_out_user_should_opt_out`, `test_opted_in_user_two_experiments`, `test_opted_in_user_two_experiments_default`, `test_opted_in_user_two_experiments_default_exp`, `test_opted_in_user_two_experiments_default_exp_2`, `test_opted_out_user`, `test_opted_out_user_was_pulled_in_by_rollout`, `test_opted_out_user_was_pulled_in_by_rollout_excl_nondefault`, `test_opted_out_user_was_pulled_in_by_rollout_filter_exp`, `test_opted_out_user_was_pulled_out_by_rollout_filter_exp`, `test_lf_prefix_always_comes_first`, `test_ignores_commented_users`

**Key imports**: main, TestCase, Mock, patch, runner_determinator as rd


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`: main, TestCase
- `unittest.mock`: Mock, patch
- `runner_determinator as rd`


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
python .github/scripts/test_runner_determinator.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.github/scripts`):

- [`convert_lintrunner_annotations_to_github.py_docs.md`](./convert_lintrunner_annotations_to_github.py_docs.md)
- [`gitutils.py_docs.md`](./gitutils.py_docs.md)
- [`collect_ciflow_labels.py_docs.md`](./collect_ciflow_labels.py_docs.md)
- [`generate_docker_release_matrix.py_docs.md`](./generate_docker_release_matrix.py_docs.md)
- [`github_utils.py_docs.md`](./github_utils.py_docs.md)
- [`filter_test_configs.py_docs.md`](./filter_test_configs.py_docs.md)
- [`trymerge.py_docs.md`](./trymerge.py_docs.md)
- [`comment_on_pr.py_docs.md`](./comment_on_pr.py_docs.md)
- [`generate_binary_build_matrix.py_docs.md`](./generate_binary_build_matrix.py_docs.md)


## Cross-References

- **File Documentation**: `test_runner_determinator.py_docs.md`
- **Keyword Index**: `test_runner_determinator.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
