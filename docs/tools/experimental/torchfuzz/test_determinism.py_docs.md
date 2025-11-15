# Documentation: `tools/experimental/torchfuzz/test_determinism.py`

## File Metadata

- **Path**: `tools/experimental/torchfuzz/test_determinism.py`
- **Size**: 4,592 bytes (4.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
"""Test to verify fuzzer produces deterministic output with same seed."""

import subprocess
import sys
from pathlib import Path


def run_fuzzer_with_seed(seed):
    """Run the fuzzer with a specific seed and return the generated code."""
    cmd = [sys.executable, "fuzzer.py", "--seed", str(seed), "--template", "unbacked"]

    # Clear the output directory first
    torchfuzz_dir = Path("/tmp/torchfuzz")
    if torchfuzz_dir.exists():
        for f in torchfuzz_dir.glob("*.py"):
            f.unlink()

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent
    )

    # Always attempt to read the generated file even if execution failed.
    if result.returncode != 0:
        print(f"Fuzzer failed with return code {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")

    # Prefer to compare the exact Program Source that the fuzzer printed in stdout,
    # which reflects the executed code even if files are overwritten between runs.
    src_block = None
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "=== Program Source ===":
            # Collect until the next delimiter line of === or the end
            j = i + 1
            block_lines = []
            while j < len(lines) and not lines[j].startswith("==="):
                block_lines.append(lines[j])
                j += 1
            src_block = "\n".join(block_lines)
            break

    if src_block:
        return src_block

    # Fallback: parse the exact path the fuzzer ran from stdout: "Running: /tmp/torchfuzz/fuzz_XXXX.py"
    path = None
    for line in lines:
        if line.startswith("Running: ") and line.strip().endswith(".py"):
            path = line.split("Running: ", 1)[1].strip()
            break

    if path is None:
        # Fallback: pick the most recently modified fuzz_*.py in /tmp/torchfuzz
        py_files = sorted(
            torchfuzz_dir.glob("fuzz_*.py"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not py_files:
            print("No Python files generated in /tmp/torchfuzz/")
            return None
        path = str(py_files[0])

    # Read the content of the generated file that was actually executed
    with open(path) as f:
        return f.read()


def test_deterministic_output():
    """Test that the fuzzer produces identical output for the same seed."""
    seed = 13157  # Use the seed mentioned in the user's issue
    num_runs = 3

    outputs = []

    print(f"Running fuzzer {num_runs} times with seed {seed}...")

    for i in range(num_runs):
        print(f"Run {i + 1}...")
        output = run_fuzzer_with_seed(seed)
        if output is None:
            print(f"Failed to get output from run {i + 1}")
            return False
        outputs.append(output)

    # Compare all outputs
    first_output = outputs[0]
    all_identical = all(output == first_output for output in outputs[1:])

    if all_identical:
        print("✓ SUCCESS: All outputs are identical!")
        print(f"Generated code length: {len(first_output)} characters")
        return True
    else:
        print("✗ FAILURE: Outputs differ between runs!")

        # Show differences for debugging
        for i, output in enumerate(outputs[1:], 2):
            if output != first_output:
                print(f"\nDifferences between run 1 and run {i}:")

                # Simple line-by-line comparison
                lines1 = first_output.splitlines()
                lines2 = output.splitlines()

                min_lines = min(len(lines1), len(lines2))
                for line_num in range(min_lines):
                    if lines1[line_num] != lines2[line_num]:
                        print(f"Line {line_num + 1}:")
                        print(f"  Run 1: {lines1[line_num]}")
                        print(f"  Run {i}: {lines2[line_num]}")
                        break

                if len(lines1) != len(lines2):
                    print(f"Different number of lines: {len(lines1)} vs {len(lines2)}")

        return False


def main():
    """Main function to run the determinism test."""
    print("Testing fuzzer determinism...")
    print("=" * 50)

    success = test_deterministic_output()

    if success:
        print("\n🎉 Test PASSED: Fuzzer is deterministic!")
        sys.exit(0)
    else:
        print("\n❌ Test FAILED: Fuzzer is not deterministic!")
        sys.exit(1)


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Test to verify fuzzer produces deterministic output with same seed."""import subprocessimport sysfrom pathlib import Pathdef run_fuzzer_with_seed(seed):

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `run_fuzzer_with_seed`, `test_deterministic_output`, `main`

**Key imports**: subprocess, sys, Path


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/experimental/torchfuzz`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `subprocess`
- `sys`
- `pathlib`: Path


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python tools/experimental/torchfuzz/test_determinism.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/experimental/torchfuzz`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`codegen.py_docs.md`](./codegen.py_docs.md)
- [`tensor_fuzzer.py_docs.md`](./tensor_fuzzer.py_docs.md)
- [`fuzzer.py_docs.md`](./fuzzer.py_docs.md)
- [`visualize_graph.py_docs.md`](./visualize_graph.py_docs.md)
- [`checks.py_docs.md`](./checks.py_docs.md)
- [`type_promotion.py_docs.md`](./type_promotion.py_docs.md)
- [`ops_fuzzer.py_docs.md`](./ops_fuzzer.py_docs.md)
- [`multi_process_fuzzer.py_docs.md`](./multi_process_fuzzer.py_docs.md)


## Cross-References

- **File Documentation**: `test_determinism.py_docs.md`
- **Keyword Index**: `test_determinism.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
