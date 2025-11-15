# Documentation: `.github/ISSUE_TEMPLATE/release-feature-request.yml`

## File Metadata

- **Path**: `.github/ISSUE_TEMPLATE/release-feature-request.yml`
- **Size**: 3,677 bytes (3.59 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: 🚀 New Feature for Release
description: Submit a Release highlight for proposed Feature
labels: ["release-feature-request"]

body:
- type: textarea
  attributes:
    label: New Feature for Release
    description: >
      Example: “A torch.special module, analogous to SciPy's special module.”
- type: input
  id: contact
  attributes:
    label: Point(s) of contact
    description: How can we get in touch with you if we need more info?
    placeholder: ex. github username
  validations:
    required: false
- type: dropdown
  attributes:
    label: Release Mode (pytorch/pytorch features only)
    description: |
      If "out-of-tree", please include the GH repo name
    options:
      - In-tree
      - Out-of-tree
  validations:
    required: true
- type: textarea
  attributes:
    label: Out-Of-Tree Repo
    description: >
      please include the GH repo name
  validations:
    required: false
- type: textarea
  attributes:
    label: Description and value to the user
    description: >
      Please provide a brief description of the feature and how it will benefit the user.
  validations:
    required: false
- type: textarea
  attributes:
    label: Link to design doc, GitHub issues, past submissions, etc
  validations:
    required: false
- type: textarea
  attributes:
    label: What feedback adopters have provided
    description: >
      Please list users/teams that have tried the feature and provided feedback. If that feedback motivated material changes (API, doc, etc..), a quick overview of the changes and the status (planned, in progress, implemented) would be helpful as well.
  validations:
    required: false
- type: dropdown
  attributes:
    label: Plan for documentations / tutorials
    description: |
      Select One of the following options
    options:
      - Tutorial exists
      - Will submit a PR to pytorch/tutorials
      - Will submit a PR to a repo
      - Tutorial is not needed
  validations:
    required: true
- type: textarea
  attributes:
    label: Additional context for tutorials
    description: >
      Please provide a link for existing tutorial or link to a repo or context for why tutorial is not needed.
  validations:
    required: false
- type: dropdown
  attributes:
    label: Marketing/Blog Coverage
    description: |
      Are you requesting feature Inclusion in the release blogs?
    options:
      - "Yes"
      - "No"
  validations:
    required: true
- type: textarea
  attributes:
    label: Are you requesting other marketing assistance with this feature?
    description: >
      E.g. supplementary blogs, social media amplification, etc.
  validations:
    required: false
- type: textarea
  attributes:
    label: Release Version
    description: >
      Please include release version for marketing coverage.
  validations:
    required: false
- type: textarea
  attributes:
    label: OS / Platform / Compute Coverage
    description: >
      Please list the platforms supported by the proposed feature. If the feature supports all the platforms, write "all". Goal of this section is to clearly share if this feature works in all PyTorch configurations or is it limited to only certain platforms/configurations (e.g. CPU only, GPU only, Linux only, etc...)
  validations:
    required: false
- type: textarea
  attributes:
    label: Testing Support (CI, test cases, etc..)
    description: >
      Please provide an overview of test coverage. This includes unit testing and integration testing, but if E2E validation testing has been done to show that the feature works for a certain set of use cases or models please mention that as well.
  validations:
    required: false

```



## High-Level Overview

This file is part of the PyTorch framework located at `.github/ISSUE_TEMPLATE`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/ISSUE_TEMPLATE`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`.github/ISSUE_TEMPLATE`):

- [`ci-sev.md_docs.md`](./ci-sev.md_docs.md)
- [`config.yml_docs.md`](./config.yml_docs.md)
- [`disable-ci-jobs.md_docs.md`](./disable-ci-jobs.md_docs.md)
- [`disable-autorevert.md_docs.md`](./disable-autorevert.md_docs.md)
- [`documentation.yml_docs.md`](./documentation.yml_docs.md)
- [`feature-request.yml_docs.md`](./feature-request.yml_docs.md)
- [`pt2-bug-report.yml_docs.md`](./pt2-bug-report.yml_docs.md)
- [`bug-report.yml_docs.md`](./bug-report.yml_docs.md)


## Cross-References

- **File Documentation**: `release-feature-request.yml_docs.md`
- **Keyword Index**: `release-feature-request.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
