# Documentation: `test/HowToWriteTestsUsingFileCheck.md`

## File Metadata

- **Path**: `test/HowToWriteTestsUsingFileCheck.md`
- **Size**: 4,452 bytes (4.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```markdown
# How to write tests using FileCheck

## What is FileCheck

FileCheck can be seen as an advanced version of grep. We use it for writing
small annotated unit tests for optimization passes. FileCheck used in PyTorch is
inspired by [LLVM FileCheck
Tool](https://llvm.org/docs/CommandGuide/FileCheck.html), but is not the same.
FileCheck is available for writing both C++ and python tests.

## How does it work

Let's look at a test written with FileCheck. The following test verifies that
CSE pass removes one out of two similar `aten::mul` nodes. Here is how the test
looks like:

```python
def test_cse():
    input_str = """graph(%a : Tensor, %b : Tensor):
      # CHECK: aten::mul
      %x : Tensor = aten::mul(%a, %b)
      # Check that the second aten::mul is removed by CSE.
      # CHECK-NOT: aten::mul
      %y : Tensor = aten::mul(%a, %b)
      # CHECK: return
      return (%x, %y)
      """
    parsed = parse_ir(input_str)
    optimized = run_cse(parsed)
    FileCheck().run(input_str, optimized)
```

Let's look in detail at how it works. First, the input string is parsed by
`parse_ir`. At that stage all annotations are ignored since they are written in
comments, so this is what parser essentially sees:

```
graph(%a : Tensor, %b : Tensor):
      %x : Tensor = aten::mul(%a, %b)
      %y : Tensor = aten::mul(%a, %b)
      return (%x, %y)
```

We then run CSE on the parsed IR and expect it to remove the second `aten::mul`,
which is redundant. After CSE our IR looks like this:

```
graph(%a : Tensor, %b : Tensor):
      %x : Tensor = aten::mul(%a, %b)
      return (%x, %x)
```

And now we run `FileCheck` passing to it both original input string and the
optimized IR. From the input string `FileCheck` ignores everything except `#
CHECK` pragmas and essentially it sees the input string like this:

```
      # CHECK: aten::mul       (1)
      # CHECK-NOT: aten::mul   (2)
      # CHECK: return          (3)
```

It then checks that the optimized IR satisfies the specified annotations. It
first finds string `%x : Tensor = aten::mul(%a, %b)` matching the annotation (1),
then it finds string `return (%x, %x)` matching the annotation (3), and since
there were no lines matching `aten::mul` after the match (1) and before the
match (3), the annotation (2) is also satisfied.

One could also register FileCheck annotations using a builder API. To generate
annotations from the example above one would write:
```python
      FileCheck().check("aten::mul")     \
                 .check_not("aten::mul") \
                 .check("return")        \
                 .run(optimized)
```

## Supported pragmas

* `CHECK: <pattern>`
  Scans the input until `PATTERN` is found. Fails if the pattern is not found.
* `CHECK-NEXT: <pattern>`
  Scans the input on the line immediately following the previous CHECK until
  `PATTERN` is found. Fails if the pattern is not found on that line.
* `CHECK-NOT: <pattern>`
  Scans the input and fails if `PATTERN` is found on any line. The scan stops when
  a match for a next `CHECK` is found.
* `CHECK-SAME: <pattern>`
  Checks that PATTERN is found in the line of the last match.
* `CHECK-COUNT-<num>: <pattern>`
  Scans the input and succeeds when a line containing at least `NUM` entries of
  `PATTERN` is found.
* `CHECK-COUNT-EXACTLY-<num>: <pattern>`
  Scans the input and succeeds when a line containing exactly `NUM` entries of
  `PATTERN` is found.
* `CHECK-DAG: <pattern>`
  Works similar to the usual `CHECK` pragma, but also matches if there exists a
  way to reorder the CHECK-DAG pragmas to satisfy all patterns.
  For example the following pattern:
  ```
  # CHECK: foo
  # CHECK-DAG: bar
  # CHECK-DAG: ham
  # CHECK: end
  ```
  would match the following input (note that `ham` and `bar` are swapped):
  ```
  foo
  ham
  bar
  end
  ```
* `CHECK-SOURCE-HIGHLIGHTED: <pattern>`
  Check for highlighted source ranges. This is useful when writing tests regarding generated error messages that require source code highlighting.
  For example the following pattern:
  ```
  # CHECK-SOURCE-HIGHLIGHTED: raise Exception("raised exception
  ```
  would match the following input:
  ```
  def method_that_raises() -> torch.Tensor:
      raise Exception("raised exception")  # noqa: TRY002
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
  builtins.Exception: raised exception
  ```
* `CHECK-REGEX: <pattern>`
  Scans the input until `PATTERN` is matched, accepts RE syntax for std::regex.

```



## High-Level Overview

This file is part of the PyTorch framework located at `test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python test/HowToWriteTestsUsingFileCheck.md
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

- **File Documentation**: `HowToWriteTestsUsingFileCheck.md_docs.md`
- **Keyword Index**: `HowToWriteTestsUsingFileCheck.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
