# Documentation: `.github/scripts/README.md`

## File Metadata

- **Path**: `.github/scripts/README.md`
- **Size**: 3,042 bytes (2.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```markdown
# pytorch/.github

> NOTE: This README contains information for the `.github` directory but cannot be located there because it will overwrite the
repo README.

This directory contains workflows and scripts to support our CI infrastructure that runs on GitHub Actions.

## Workflows

- Pull CI (`pull.yml`) is run on PRs and on main.
- Trunk CI (`trunk.yml`) is run on trunk to validate incoming commits. Trunk jobs are usually more expensive to run so we do not run them on PRs unless specified.
- Scheduled CI (`periodic.yml`) is a subset of trunk CI that is run every few hours on main.
- Binary CI is run to package binaries for distribution for all platforms.

## Templates

Templates written in [Jinja](https://jinja.palletsprojects.com/en/3.0.x/) are located in the `.github/templates` directory
and used to generate workflow files for binary jobs found in the `.github/workflows/` directory. These are also a
couple of utility templates used to discern common utilities that can be used amongst different templates.

### (Re)Generating workflow files

You will need `jinja2` in order to regenerate the workflow files which can be installed using:
```bash
pip install -r .github/requirements/regenerate-requirements.txt
```

Workflows can be generated / regenerated using the following command:
```bash
.github/regenerate.sh
```

### Adding a new generated binary workflow

New generated binary workflows can be added in the `.github/scripts/generate_ci_workflows.py` script. You can reference
examples from that script in order to add the workflow to the stream that is relevant to what you particularly
care about.

Different parameters can be used to achieve different goals, i.e. running jobs on a cron, running only on trunk, etc.

#### ciflow (trunk)

The label `ciflow/trunk` can be used to run `trunk` only workflows. This is especially useful if trying to re-land a PR that was
reverted for failing a `non-default` workflow.

## Infra

Currently most of our self hosted runners are hosted on AWS, for a comprehensive list of available runner types you
can reference `.github/scale-config.yml`.

Exceptions to AWS for self hosted:
* ROCM runners

### Adding new runner types

New runner types can be added by committing changes to `.github/scale-config.yml`. Example: https://github.com/pytorch/pytorch/pull/70474

> NOTE: New runner types can only be used once the changes to `.github/scale-config.yml` have made their way into the default branch

### Testing [pytorch/builder](https://github.com/pytorch/builder) changes

In order to test changes to the builder scripts:

1. Specify your builder PR's branch and repo as `builder_repo` and  `builder_branch` in [`.github/templates/common.yml.j2`](https://github.com/pytorch/pytorch/blob/32356aaee6a77e0ae424435a7e9da3d99e7a4ca5/.github/templates/common.yml.j2#LL10C26-L10C32).
2. Regenerate workflow files with `.github/regenerate.sh` (see above).
3. Submit fake PR to PyTorch. If changing binaries build, add an appropriate label like `ciflow/binaries` to trigger the builds.

```



## High-Level Overview

This file is part of the PyTorch framework located at `.github/scripts`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Test files for this module may be located in the `test/` directory.

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
- [`test_runner_determinator.py_docs.md`](./test_runner_determinator.py_docs.md)
- [`trymerge.py_docs.md`](./trymerge.py_docs.md)
- [`comment_on_pr.py_docs.md`](./comment_on_pr.py_docs.md)
- [`generate_binary_build_matrix.py_docs.md`](./generate_binary_build_matrix.py_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md`
- **Keyword Index**: `README.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
