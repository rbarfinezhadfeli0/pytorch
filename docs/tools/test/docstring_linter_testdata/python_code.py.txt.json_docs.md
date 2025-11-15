# Documentation: python_code.py.txt.json

## File Metadata
- **Path**: `tools/test/docstring_linter_testdata/python_code.py.txt.json`
- **Size**: 4938 bytes
- **Lines**: 57
- **Extension**: .json
- **Type**: Regular file

## Original Source

```json
[
  {
    "char": 0,
    "code": "DOCSTRING_LINTER",
    "description": null,
    "line": 17,
    "name": "No docstring found for class 'LongWithoutDocstring' (6 lines)",
    "original": null,
    "path": "tools/test/docstring_linter_testdata/python_code.py.txt",
    "replacement": null,
    "severity": "error"
  },
  {
    "char": 0,
    "code": "DOCSTRING_LINTER",
    "description": null,
    "line": 24,
    "name": "docstring found for class 'LongWithShortDocstring' (6 lines) was too short (10 characters, needed 16)",
    "original": null,
    "path": "tools/test/docstring_linter_testdata/python_code.py.txt",
    "replacement": null,
    "severity": "error"
  },
  {
    "char": 0,
    "code": "DOCSTRING_LINTER",
    "description": null,
    "line": 54,
    "name": "No docstring found for function 'long_without_docstring' (7 lines)",
    "original": null,
    "path": "tools/test/docstring_linter_testdata/python_code.py.txt",
    "replacement": null,
    "severity": "error"
  },
  {
    "char": 4,
    "code": "DOCSTRING_LINTER",
    "description": null,
    "line": 71,
    "name": "No docstring found for function 'needs_docs' (12 lines). If the method overrides a method on a parent class, adding the `@typing_extensions.override` decorator will make this error go away.",
    "original": null,
    "path": "tools/test/docstring_linter_testdata/python_code.py.txt",
    "replacement": null,
    "severity": "error"
  },
  {
    "char": null,
    "code": "DOCSTRING_LINTER",
    "description": null,
    "line": null,
    "name": "Suggested fixes for docstring_linter",
    "original": "class ShortWithDocstring:\n    \"\"\"This docstring, while short, is enough\"\"\"\n    pass\n\n\nclass Short:\n    pass\n\n\nclass LongWithDocstring:\n    \"\"\"This docstring, while short, is enough\"\"\"\n\n    def short1(self):\n        pass\n\n\nclass LongWithoutDocstring:\n    # A comment isn't a docstring\n\n    def short1(self):\n        pass\n\n\nclass LongWithShortDocstring:\n    \"\"\"TODO\"\"\"\n\n    def short1(self):\n        pass\n\n\nclass _Protected:\n    \"\"\"TODO\"\"\"\n\n    def short1(self):\n        pass\n\n\ndef short():\n    #\n    #\n    #\n    pass\n\n\ndef long():\n    \"\"\"This docstring, while short, is enough\"\"\"\n    #\n    #\n    #\n    #\n    pass\n\n\ndef long_without_docstring():\n    #\n    #\n    #\n    #\n    pass\n\n\nclass ImpossibleCombo(\n    set,\n    tuple,\n    int,\n):\n    # We could have comments\n    # before the doc comment\n    \"\"\"This docstring, while short, is enough\"\"\"\n\n    def needs_docs(self):\n        def not_short():\n            class Long:\n                a = 1\n                b = 1\n                c = 1\n                d = 1\n                e = 1\n\n            class Short:\n                pass\n\n\n@override  # Won't work!\nclass NotDocstring:\n    def short1(self):\n        pass\n\n    \"\"\"This is not a docstring\"\"\"\n\n    @override\n    def long_with_override(self):\n        #\n        #\n        #\n        #\n        pass\n\n    def short2(self):\n        pass\n\n    def short3(self):\n        pass\n\n\n\ndef long_with_omit():  # noqa: docstring_linter\n    #\n    #\n    #\n    #\n    pass\n",
    "path": "tools/test/docstring_linter_testdata/python_code.py.txt",
    "replacement": "class ShortWithDocstring:\n    \"\"\"This docstring, while short, is enough\"\"\"\n    pass\n\n\nclass Short:\n    pass\n\n\nclass LongWithDocstring:\n    \"\"\"This docstring, while short, is enough\"\"\"\n\n    def short1(self):\n        pass\n\n\nclass LongWithoutDocstring:\n    # A comment isn't a docstring\n\n    def short1(self):\n        pass\n\n\nclass LongWithShortDocstring:\n    \"\"\"TODO\"\"\"\n\n    def short1(self):\n        pass\n\n\nclass _Protected:\n    \"\"\"TODO\"\"\"\n\n    def short1(self):\n        pass\n\n\ndef short():\n    #\n    #\n    #\n    pass\n\n\ndef long():\n    \"\"\"This docstring, while short, is enough\"\"\"\n    #\n    #\n    #\n    #\n    pass\n\n\ndef long_without_docstring():\n    #\n    #\n    #\n    #\n    pass\n\n\nclass ImpossibleCombo(\n    set,\n    tuple,\n    int,\n):\n    # We could have comments\n    # before the doc comment\n    \"\"\"This docstring, while short, is enough\"\"\"\n\n    def needs_docs(self):\n        def not_short():\n            class Long:\n                a = 1\n                b = 1\n                c = 1\n                d = 1\n                e = 1\n\n            class Short:\n                pass\n\n\n@override  # Won't work!\nclass NotDocstring:\n    def short1(self):\n        pass\n\n    \"\"\"This is not a docstring\"\"\"\n\n    @override\n    def long_with_override(self):\n        #\n        #\n        #\n        #\n        pass\n\n    def short2(self):\n        pass\n\n    def short3(self):\n        pass\n\n\n\ndef long_with_omit():  # noqa: docstring_linter\n    #\n    #\n    #\n    #\n    pass\n",
    "severity": "error"
  }
]

```

## High-Level Overview

This file is part of the PyTorch repository. It is a configuration file.

## Detailed Walkthrough


## Key Components

The file contains 443 words across 57 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4938 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
