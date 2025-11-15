# Documentation: `docs/generate_repo_docs.py_docs.md`

## File Metadata

- **Path**: `docs/generate_repo_docs.py_docs.md`
- **Size**: 54,252 bytes (52.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `generate_repo_docs.py`

## File Metadata

- **Path**: `generate_repo_docs.py`
- **Size**: 50,960 bytes (49.77 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
"""
Repository Documentation Generator
Generates comprehensive documentation for the entire PyTorch repository.

This script creates:
- Per-file documentation (_docs.md and _kw.md)
- Per-folder documentation (index.md, doc.md, sub.md)
- Global indices (keywords.md, index.md)
- Comprehensive book (comprehensive_book.md)
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import mimetypes
import hashlib

# Configuration
REPO_ROOT = Path("/home/user/pytorch")
DOCS_ROOT = REPO_ROOT / "docs"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB - files larger will be summarized

# Directories to skip (already exist or are not source code)
SKIP_DIRS = {
    '.git', '__pycache__', 'node_modules', '.pytest_cache',
    '.mypy_cache', 'build', 'dist', '*.egg-info', '.eggs',
    '.tox', '.nox', 'venv', 'env', '.venv', '.env'
}

# File extensions to document
DOCUMENTABLE_EXTENSIONS = {
    # Python
    '.py', '.pyx', '.pxd', '.pyi',
    # C/C++
    '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hh', '.hxx', '.cu', '.cuh',
    # JavaScript/TypeScript
    '.js', '.jsx', '.ts', '.tsx', '.mjs',
    # Configuration
    '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg', '.conf',
    # Build systems
    '.cmake', '.bazel', '.bzl', 'CMakeLists.txt', 'BUILD', 'WORKSPACE',
    # Documentation
    '.md', '.rst', '.txt',
    # Shell
    '.sh', '.bash', '.zsh', '.fish',
    # Other
    '.proto', '.thrift', '.sql', '.graphql'
}


class RepoScanner:
    """Scans and catalogs the entire repository structure."""

    def __init__(self, root: Path):
        self.root = root
        self.files: Dict[str, Dict] = {}
        self.folders: Dict[str, Dict] = {}
        self.keywords: Dict[str, Set[str]] = defaultdict(set)

    def should_skip_dir(self, dir_path: Path) -> bool:
        """Check if directory should be skipped."""
        dir_name = dir_path.name
        # Skip hidden dirs except .github, .ci, etc that are important
        if dir_name.startswith('.') and dir_name not in {'.github', '.ci', '.circleci', '.devcontainer', '.vscode', '.spin', '.claude', '.ctags.d'}:
            return True
        return dir_name in SKIP_DIRS

    def should_document_file(self, file_path: Path) -> bool:
        """Check if file should be documented."""
        # Check extension
        if file_path.suffix in DOCUMENTABLE_EXTENSIONS:
            return True
        # Check filename patterns
        if file_path.name in {'Makefile', 'Dockerfile', 'Jenkinsfile', 'README', 'LICENSE', 'CONTRIBUTING'}:
            return True
        # Check if it's a text file
        try:
            mime = mimetypes.guess_type(str(file_path))[0]
            if mime and mime.startswith('text/'):
                return True
        except:
            pass
        return False

    def scan(self):
        """Recursively scan the repository."""
        print(f"Scanning repository at {self.root}...")

        for dirpath, dirnames, filenames in os.walk(self.root):
            dir_path = Path(dirpath)

            # Skip unwanted directories
            dirnames[:] = [d for d in dirnames if not self.should_skip_dir(dir_path / d)]

            # Get relative path from repo root
            try:
                rel_path = dir_path.relative_to(self.root)
            except ValueError:
                continue

            rel_path_str = str(rel_path) if str(rel_path) != '.' else ''

            # Catalog folder
            self.folders[rel_path_str] = {
                'path': rel_path_str,
                'full_path': dir_path,
                'subfolders': [d for d in dirnames],
                'files': []
            }

            # Catalog files
            for filename in filenames:
                file_path = dir_path / filename

                if not self.should_document_file(file_path):
                    continue

                file_rel_path = str(file_path.relative_to(self.root))

                # Get file info
                try:
                    stat = file_path.stat()
                    file_size = stat.st_size
                except:
                    file_size = 0

                self.files[file_rel_path] = {
                    'path': file_rel_path,
                    'full_path': file_path,
                    'name': filename,
                    'folder': rel_path_str,
                    'size': file_size,
                    'extension': file_path.suffix
                }

                self.folders[rel_path_str]['files'].append(filename)

        print(f"Found {len(self.folders)} folders and {len(self.files)} documentable files.")


class FileDocumentationGenerator:
    """Generates documentation for individual files."""

    def __init__(self, scanner: RepoScanner):
        self.scanner = scanner

    def read_file_content(self, file_path: Path, max_lines: int = 10000) -> Tuple[str, bool]:
        """Read file content safely."""
        try:
            # Check if file is too large
            if file_path.stat().st_size > MAX_FILE_SIZE:
                return f"[File too large: {file_path.stat().st_size / 1024 / 1024:.2f}MB - content summarized only]", True

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if len(lines) > max_lines:
                    return ''.join(lines[:max_lines]) + f"\n\n... [truncated: {len(lines) - max_lines} more lines]", True
                return ''.join(lines), False
        except Exception as e:
            return f"[Error reading file: {e}]", True

    def extract_keywords_from_python(self, content: str) -> List[Tuple[str, str]]:
        """Extract keywords from Python files."""
        keywords = []

        # Classes
        for match in re.finditer(r'class\s+(\w+)', content):
            keywords.append((match.group(1), 'class'))

        # Functions
        for match in re.finditer(r'def\s+(\w+)', content):
            keywords.append((match.group(1), 'function'))

        # Imports
        for match in re.finditer(r'import\s+([\w.]+)', content):
            keywords.append((match.group(1), 'import'))
        for match in re.finditer(r'from\s+([\w.]+)\s+import', content):
            keywords.append((match.group(1), 'import'))

        return keywords

    def extract_keywords_from_cpp(self, content: str) -> List[Tuple[str, str]]:
        """Extract keywords from C++ files."""
        keywords = []

        # Classes/structs
        for match in re.finditer(r'(?:class|struct)\s+(\w+)', content):
            keywords.append((match.group(1), 'class/struct'))

        # Functions
        for match in re.finditer(r'\w+\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*{', content):
            keywords.append((match.group(1), 'function'))

        # Namespaces
        for match in re.finditer(r'namespace\s+(\w+)', content):
            keywords.append((match.group(1), 'namespace'))

        # Includes
        for match in re.finditer(r'#include\s+[<"]([^>"]+)[>"]', content):
            keywords.append((match.group(1), 'include'))

        return keywords

    def extract_keywords(self, file_info: Dict) -> List[Tuple[str, str]]:
        """Extract keywords from file based on type."""
        content, _ = self.read_file_content(file_info['full_path'])

        ext = file_info['extension']
        if ext in {'.py', '.pyx', '.pyi'}:
            return self.extract_keywords_from_python(content)
        elif ext in {'.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.cu', '.cuh'}:
            return self.extract_keywords_from_cpp(content)
        else:
            # Generic keyword extraction
            words = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', content)
            return [(w, 'identifier') for w in set(words)][:100]

    def generate_file_docs(self, file_info: Dict) -> str:
        """Generate _docs.md for a file."""
        file_path = file_info['path']
        full_path = file_info['full_path']

        content, truncated = self.read_file_content(full_path)

        # Determine file type and purpose
        ext = file_info['extension']
        file_type = self.get_file_type_description(ext)

        doc = f"""# Documentation: `{file_path}`

## File Metadata

- **Path**: `{file_path}`
- **Size**: {file_info['size']:,} bytes ({file_info['size'] / 1024:.2f} KB)
- **Type**: {file_type}
- **Extension**: `{ext}`

## File Purpose

{self.infer_file_purpose(file_info, content)}

## Original Source

```{self.get_language_for_fence(ext)}
{content[:50000] if not truncated else content}
```

{"*Note: Content truncated due to size*" if truncated else ""}

## High-Level Overview

{self.generate_overview(file_info, content)}

## Detailed Analysis

{self.generate_detailed_analysis(file_info, content)}

## Architecture & Design

{self.generate_architecture_section(file_info, content)}

## Dependencies

{self.analyze_dependencies(file_info, content)}

## Code Patterns & Idioms

{self.analyze_patterns(file_info, content)}

## Performance Considerations

{self.analyze_performance(file_info, content)}

## Security & Safety

{self.analyze_security(file_info, content)}

## Testing & Usage

{self.generate_testing_notes(file_info, content)}

## Related Files

{self.find_related_files(file_info)}

## Cross-References

- **File Documentation**: `{file_info['name']}_docs.md`
- **Keyword Index**: `{file_info['name']}_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
"""
        return doc

    def generate_file_keywords(self, file_info: Dict) -> str:
        """Generate _kw.md for a file."""
        file_path = file_info['path']
        keywords = self.extract_keywords(file_info)

        # Build relative path to original file
        folder_parts = file_info['folder'].split('/') if file_info['folder'] else []
        back_path = '../' * (len(folder_parts) + 1)
        file_link = f"{back_path}{file_path}"

        doc = f"""# Keyword Index: `{file_path}`

## File Information

- **Original File**: [{file_path}]({file_link})
- **Documentation**: [`{file_info['name']}_docs.md`](./{file_info['name']}_docs.md)
- **Folder**: `{file_info['folder'] if file_info['folder'] else 'root'}`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:

"""

        if keywords:
            # Group keywords by type
            by_type = defaultdict(list)
            for kw, kw_type in keywords:
                by_type[kw_type].append(kw)

            for kw_type, kws in sorted(by_type.items()):
                doc += f"\n### {kw_type.title()}s\n\n"
                for kw in sorted(set(kws))[:200]:  # Limit to 200 per type
                    doc += f"- **`{kw}`**: [{file_info['name']}_docs.md](./{file_info['name']}_docs.md)\n"
        else:
            doc += "\n*No keywords automatically extracted. See documentation for details.*\n"

        doc += f"""

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
"""
        return doc

    def get_file_type_description(self, ext: str) -> str:
        """Get human-readable file type."""
        type_map = {
            '.py': 'Python Source Code',
            '.pyx': 'Cython Source Code',
            '.pyi': 'Python Type Stub',
            '.cpp': 'C++ Source Code',
            '.cc': 'C++ Source Code',
            '.h': 'C/C++ Header File',
            '.hpp': 'C++ Header File',
            '.cu': 'CUDA Source Code',
            '.cuh': 'CUDA Header File',
            '.md': 'Markdown Documentation',
            '.yaml': 'YAML Configuration',
            '.yml': 'YAML Configuration',
            '.json': 'JSON Configuration',
            '.toml': 'TOML Configuration',
            '.sh': 'Shell Script',
            '.cmake': 'CMake Build Script',
        }
        return type_map.get(ext, f'Source File ({ext})')

    def get_language_for_fence(self, ext: str) -> str:
        """Get language identifier for code fence."""
        lang_map = {
            '.py': 'python', '.pyx': 'python', '.pyi': 'python',
            '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
            '.c': 'c', '.h': 'c', '.hpp': 'cpp', '.cu': 'cuda',
            '.js': 'javascript', '.ts': 'typescript',
            '.yaml': 'yaml', '.yml': 'yaml', '.json': 'json',
            '.sh': 'bash', '.cmake': 'cmake', '.md': 'markdown'
        }
        return lang_map.get(ext, '')

    def infer_file_purpose(self, file_info: Dict, content: str) -> str:
        """Infer the purpose of the file."""
        name = file_info['name'].lower()
        folder = file_info['folder'].lower()
        ext = file_info['extension']

        purposes = []

        # Check folder
        if 'test' in folder:
            purposes.append("This file is part of the **testing infrastructure**.")
        if 'doc' in folder:
            purposes.append("This file is part of the **documentation**.")
        if 'example' in folder or 'benchmark' in folder:
            purposes.append("This file contains **examples or benchmarks**.")
        if 'tool' in folder or 'script' in folder:
            purposes.append("This file is a **utility or tool script**.")

        # Check filename
        if 'test' in name:
            purposes.append("This appears to be a **test file**.")
        if name.startswith('__init__'):
            purposes.append("This is a **Python package initialization file**.")
        if 'setup' in name or 'config' in name:
            purposes.append("This file handles **configuration or setup**.")

        # Check content patterns
        if ext == '.py':
            if 'import unittest' in content or 'import pytest' in content:
                purposes.append("Contains **unit tests** using Python testing frameworks.")
            if 'if __name__ == "__main__"' in content:
                purposes.append("Can be **executed as a standalone script**.")

        if not purposes:
            purposes.append(f"This is a {self.get_file_type_description(ext).lower()} that is part of the PyTorch project.")

        return ' '.join(purposes)

    def generate_overview(self, file_info: Dict, content: str) -> str:
        """Generate high-level overview."""
        lines = content.split('\n')

        # Look for docstrings or comments at top
        overview_parts = []

        if file_info['extension'] in {'.py', '.pyx'}:
            # Find module docstring
            in_docstring = False
            docstring_lines = []
            for line in lines[:50]:
                if '"""' in line or "'''" in line:
                    if in_docstring:
                        break
                    in_docstring = True
                    docstring_lines.append(line)
                elif in_docstring:
                    docstring_lines.append(line)

            if docstring_lines:
                overview_parts.append(''.join(docstring_lines).strip())

        # Count key elements
        if file_info['extension'] in {'.py', '.pyx'}:
            class_count = len(re.findall(r'class\s+\w+', content))
            func_count = len(re.findall(r'def\s+\w+', content))
            overview_parts.append(f"\nThis Python file contains {class_count} class(es) and {func_count} function(s).")
        elif file_info['extension'] in {'.cpp', '.cc', '.h', '.hpp'}:
            class_count = len(re.findall(r'class\s+\w+', content))
            func_count = len(re.findall(r'\w+\s+\w+\s*\(', content))
            overview_parts.append(f"\nThis C++ file contains approximately {class_count} class(es)/struct(s) and {func_count} function(s).")

        if not overview_parts:
            overview_parts.append(f"This file is part of the PyTorch framework located at `{file_info['folder']}`.")

        return '\n'.join(overview_parts)

    def generate_detailed_analysis(self, file_info: Dict, content: str) -> str:
        """Generate detailed code analysis."""
        analysis = "### Code Structure\n\n"

        ext = file_info['extension']

        if ext in {'.py', '.pyx'}:
            # Analyze Python code structure
            classes = re.findall(r'class\s+(\w+)(?:\([^)]*\))?:', content)
            functions = re.findall(r'def\s+(\w+)\s*\(', content)

            if classes:
                analysis += f"**Classes defined**: {', '.join(f'`{c}`' for c in classes[:20])}\n\n"
            if functions:
                analysis += f"**Functions defined**: {', '.join(f'`{f}`' for f in functions[:20])}\n\n"

            # Find imports
            imports = re.findall(r'(?:from\s+[\w.]+\s+)?import\s+([\w., ]+)', content)
            if imports:
                analysis += f"**Key imports**: {', '.join(imports[:10])}\n\n"

        elif ext in {'.cpp', '.cc', '.h', '.hpp', '.cu'}:
            # Analyze C++ code structure
            classes = re.findall(r'(?:class|struct)\s+(\w+)', content)
            namespaces = re.findall(r'namespace\s+(\w+)', content)

            if namespaces:
                analysis += f"**Namespaces**: {', '.join(f'`{n}`' for n in set(namespaces))}\n\n"
            if classes:
                analysis += f"**Classes/Structs**: {', '.join(f'`{c}`' for c in classes[:20])}\n\n"

        elif ext in {'.yaml', '.yml', '.json'}:
            analysis += "This is a configuration file. See the original source for structure.\n\n"

        analysis += "\n*For complete code details, see the Original Source section above.*\n"

        return analysis

    def generate_architecture_section(self, file_info: Dict, content: str) -> str:
        """Generate architecture and design section."""
        arch = f"### Role in PyTorch Architecture\n\n"
        arch += f"This file is located in `{file_info['folder'] if file_info['folder'] else 'root'}`, "

        folder = file_info['folder'].lower()
        if 'torch' in folder:
            arch += "which is part of the **core PyTorch library**.\n\n"
        elif 'aten' in folder:
            arch += "which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.\n\n"
        elif 'c10' in folder:
            arch += "which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.\n\n"
        elif 'caffe2' in folder:
            arch += "which is part of the **Caffe2** deep learning framework.\n\n"
        elif 'test' in folder:
            arch += "which is part of the **testing infrastructure**.\n\n"
        elif 'tools' in folder or 'script' in folder:
            arch += "which contains **development tools and scripts**.\n\n"
        else:
            arch += "which is part of the PyTorch project infrastructure.\n\n"

        return arch

    def analyze_dependencies(self, file_info: Dict, content: str) -> str:
        """Analyze file dependencies."""
        deps = "### Import Dependencies\n\n"

        ext = file_info['extension']

        if ext in {'.py', '.pyx'}:
            imports = re.findall(r'(?:from\s+([\w.]+)\s+import\s+([\w, ]+)|import\s+([\w., ]+))', content)
            if imports:
                deps += "This file imports:\n\n"
                seen = set()
                for from_module, from_items, direct_import in imports[:30]:
                    if from_module and from_module not in seen:
                        deps += f"- `{from_module}`: {from_items}\n"
                        seen.add(from_module)
                    elif direct_import and direct_import not in seen:
                        deps += f"- `{direct_import}`\n"
                        seen.add(direct_import)
            else:
                deps += "*No imports detected.*\n"

        elif ext in {'.cpp', '.cc', '.h', '.hpp', '.cu', '.cuh'}:
            includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', content)
            if includes:
                deps += "This file includes:\n\n"
                for inc in includes[:30]:
                    deps += f"- `{inc}`\n"
            else:
                deps += "*No includes detected.*\n"

        else:
            deps += "*Dependency analysis not applicable for this file type.*\n"

        return deps

    def analyze_patterns(self, file_info: Dict, content: str) -> str:
        """Analyze code patterns."""
        patterns = "### Common Patterns\n\n"

        ext = file_info['extension']

        # Check for common patterns
        detected = []

        if 'class' in content and 'def __init__' in content:
            detected.append("**Object-Oriented Design**: Uses classes and constructors")
        if 'def __enter__' in content and 'def __exit__' in content:
            detected.append("**Context Manager**: Implements context manager protocol")
        if '@abstractmethod' in content or 'ABC' in content:
            detected.append("**Abstract Base Classes**: Defines abstract interfaces")
        if 'try:' in content and 'except' in content:
            detected.append("**Error Handling**: Includes exception handling")
        if 'async def' in content or 'await ' in content:
            detected.append("**Asynchronous Programming**: Uses async/await")
        if 'torch.nn' in content:
            detected.append("**Neural Network**: Defines or uses PyTorch neural network components")
        if 'torch.autograd' in content:
            detected.append("**Automatic Differentiation**: Uses autograd for gradient computation")

        if detected:
            for pattern in detected:
                patterns += f"- {pattern}\n"
        else:
            patterns += "*No specific patterns automatically detected.*\n"

        return patterns

    def analyze_performance(self, file_info: Dict, content: str) -> str:
        """Analyze performance characteristics."""
        perf = "### Performance Notes\n\n"

        # Check for performance-related keywords
        if any(kw in content.lower() for kw in ['cuda', 'gpu', 'parallel', 'thread']):
            perf += "- This file appears to involve **GPU/parallel computing** capabilities.\n"
        if 'cache' in content.lower():
            perf += "- Implements or uses **caching** mechanisms.\n"
        if 'jit' in content.lower() or 'compile' in content.lower():
            perf += "- May involve **JIT compilation** or compilation optimizations.\n"
        if 'benchmark' in content.lower():
            perf += "- Contains **benchmarking** code or performance tests.\n"

        perf += "\n*Detailed performance analysis requires profiling and benchmarking.*\n"

        return perf

    def analyze_security(self, file_info: Dict, content: str) -> str:
        """Analyze security considerations."""
        sec = "### Security Considerations\n\n"

        # Check for potential security concerns
        concerns = []

        if 'eval(' in content or 'exec(' in content:
            concerns.append("**Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized")
        if 'pickle' in content:
            concerns.append("**Serialization**: Uses pickle - be cautious with untrusted data")
        if 'subprocess' in content or 'os.system' in content:
            concerns.append("**Command Execution**: Executes system commands - validate inputs")
        if 'sql' in content.lower() and 'query' in content.lower():
            concerns.append("**Database**: May involve SQL - watch for injection vulnerabilities")

        if concerns:
            for concern in concerns:
                sec += f"- {concern}\n"
        else:
            sec += "- No obvious security concerns detected in automated analysis.\n"

        sec += "\n*Manual security review is recommended for production code.*\n"

        return sec

    def generate_testing_notes(self, file_info: Dict, content: str) -> str:
        """Generate testing and usage notes."""
        notes = "### Testing\n\n"

        if 'test' in file_info['folder'].lower() or 'test' in file_info['name'].lower():
            notes += "This is a test file. Run it with:\n\n"
            notes += f"```bash\n"
            notes += f"python {file_info['path']}\n"
            notes += f"```\n\n"
        else:
            notes += f"Test files for this module may be located in the `test/` directory.\n\n"

        notes += "### Usage Examples\n\n"
        notes += "*See the source code and related test files for usage examples.*\n"

        return notes

    def find_related_files(self, file_info: Dict) -> str:
        """Find related files."""
        related = "### Related Files\n\n"

        # Find files in same folder
        folder_files = self.scanner.folders.get(file_info['folder'], {}).get('files', [])
        if folder_files:
            related += f"Files in the same folder (`{file_info['folder'] if file_info['folder'] else 'root'}`):\n\n"
            for f in folder_files[:10]:
                if f != file_info['name']:
                    related += f"- [`{f}_docs.md`](./{f}_docs.md)\n"

        return related


class FolderDocumentationGenerator:
    """Generates documentation for folders."""

    def __init__(self, scanner: RepoScanner):
        self.scanner = scanner

    def generate_folder_index(self, folder_path: str) -> str:
        """Generate index.md for a folder."""
        folder_info = self.scanner.folders.get(folder_path, {})

        display_path = folder_path if folder_path else 'root'

        doc = f"""# Index: `{display_path}/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `{display_path}/`

## Subfolders

"""

        subfolders = folder_info.get('subfolders', [])
        if subfolders:
            for subfolder in sorted(subfolders):
                doc += f"- [`{subfolder}/`](./{subfolder}/index.md) - {self.describe_folder(folder_path, subfolder)}\n"
        else:
            doc += "*No subfolders.*\n"

        doc += "\n## Files\n\n"

        files = folder_info.get('files', [])
        if files:
            doc += "| File | Description | Documentation | Keywords |\n"
            doc += "|------|-------------|---------------|----------|\n"

            for filename in sorted(files):
                file_path = f"{folder_path}/{filename}" if folder_path else filename
                file_info = self.scanner.files.get(file_path, {})

                # Build relative path to original file
                folder_parts = folder_path.split('/') if folder_path else []
                back_path = '../' * (len(folder_parts) + 1)
                original_link = f"{back_path}{file_path}"

                desc = self.describe_file(filename)
                doc += f"| [`{filename}`]({original_link}) | {desc} | [docs](./{filename}_docs.md) | [keywords](./{filename}_kw.md) |\n"
        else:
            doc += "*No files in this folder.*\n"

        doc += f"""

## Navigation

- **Parent Folder**: [{'..' if folder_path else 'root'}](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
"""

        return doc

    def generate_folder_doc(self, folder_path: str) -> str:
        """Generate doc.md for a folder."""
        folder_info = self.scanner.folders.get(folder_path, {})
        display_path = folder_path if folder_path else 'root'

        doc = f"""# Documentation: `{display_path}/`

## Role in PyTorch

{self.describe_folder_role(folder_path)}

## Contents Overview

{self.describe_folder_contents(folder_path, folder_info)}

## Key Concepts

{self.describe_key_concepts(folder_path)}

## Architecture & Organization

{self.describe_architecture(folder_path, folder_info)}

## Important Files

{self.list_important_files(folder_path, folder_info)}

## Working with This Folder

{self.describe_workflow(folder_path)}

## Cross-References

{self.generate_cross_references(folder_path)}

---

*Generated by PyTorch Repository Documentation System*
"""

        return doc

    def generate_folder_keywords(self, folder_path: str) -> str:
        """Generate sub.md (keyword index) for a folder and all descendants."""
        display_path = folder_path if folder_path else 'root'

        doc = f"""# Subtree Keyword Index: `{display_path}/`

## Scope

This index covers all files within `{display_path}/` and all its subdirectories (recursively).

## Keywords

"""

        # Collect all files in this folder and subfolders
        all_files = []
        prefix = folder_path + '/' if folder_path else ''

        for file_path, file_info in self.scanner.files.items():
            if file_path.startswith(prefix) or (not folder_path and '/' not in file_path):
                all_files.append(file_info)

        # Extract keywords from all files (simplified - just file names and types)
        keywords_map = defaultdict(list)

        for file_info in all_files:
            # Add filename as keyword
            name = file_info['name']
            base_name = name.split('.')[0]
            keywords_map[base_name].append(file_info)

            # Add file type as keyword
            ext = file_info['extension']
            if ext:
                keywords_map[f"files-{ext}"].append(file_info)

        # Sort and output
        for keyword in sorted(keywords_map.keys())[:500]:  # Limit to 500 keywords
            files = keywords_map[keyword]
            doc += f"\n### {keyword}\n\n"
            for file_info in files[:20]:  # Limit to 20 files per keyword
                rel_path = self.get_relative_path(folder_path, file_info['folder'])
                doc += f"- [`{file_info['path']}`]({rel_path}/{file_info['name']}_docs.md)\n"

        doc += """

---

*Generated by PyTorch Repository Documentation System*
"""

        return doc

    def describe_folder(self, parent_path: str, folder_name: str) -> str:
        """Describe a subfolder."""
        descriptions = {
            'torch': 'Core PyTorch library',
            'aten': 'ATen tensor library',
            'c10': 'Core abstractions (C10)',
            'caffe2': 'Caffe2 framework',
            'test': 'Test suites',
            'tools': 'Development tools',
            'scripts': 'Utility scripts',
            'docs': 'Documentation',
            'examples': 'Example code',
            'benchmarks': 'Performance benchmarks',
            '.github': 'GitHub configuration',
            '.ci': 'CI/CD configuration',
        }
        return descriptions.get(folder_name, f'{folder_name} module')

    def describe_file(self, filename: str) -> str:
        """Describe a file briefly."""
        if filename.startswith('__init__'):
            return 'Package initialization'
        elif filename.startswith('test_'):
            return 'Test file'
        elif filename.endswith('.md'):
            return 'Documentation'
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            return 'Configuration'
        else:
            return 'Source code'

    def describe_folder_role(self, folder_path: str) -> str:
        """Describe the role of a folder."""
        if not folder_path:
            return "This is the **root directory** of the PyTorch repository, containing the main project structure."

        parts = folder_path.split('/')
        base = parts[0]

        roles = {
            'torch': 'The **torch** folder contains the main Python API for PyTorch, including neural network modules, autograd, and tensor operations.',
            'aten': 'The **ATen** folder contains the C++ tensor library that powers PyTorch\'s tensor operations.',
            'c10': 'The **C10** folder provides core abstractions used throughout PyTorch, including tensor types, device abstractions, and utilities.',
            'caffe2': 'The **Caffe2** folder contains the Caffe2 deep learning framework, which has been integrated into PyTorch.',
            'test': 'The **test** folder contains comprehensive test suites for PyTorch functionality.',
            'tools': 'The **tools** folder contains development tools, code generation scripts, and utilities.',
            'scripts': 'The **scripts** folder contains utility scripts for building, testing, and managing the PyTorch project.',
            'docs': 'The **docs** folder contains documentation for PyTorch.',
        }

        return roles.get(base, f"This folder is part of the PyTorch project structure at `{folder_path}`.")

    def describe_folder_contents(self, folder_path: str, folder_info: Dict) -> str:
        """Describe folder contents."""
        subfolders = folder_info.get('subfolders', [])
        files = folder_info.get('files', [])

        desc = f"This folder contains:\n\n"
        desc += f"- **{len(subfolders)} subfolder(s)**\n"
        desc += f"- **{len(files)} file(s)**\n\n"

        # Count file types
        exts = defaultdict(int)
        for filename in files:
            file_path = f"{folder_path}/{filename}" if folder_path else filename
            file_info = self.scanner.files.get(file_path, {})
            ext = file_info.get('extension', '')
            if ext:
                exts[ext] += 1

        if exts:
            desc += "File types:\n\n"
            for ext, count in sorted(exts.items()):
                desc += f"- `{ext}`: {count} file(s)\n"

        return desc

    def describe_key_concepts(self, folder_path: str) -> str:
        """Describe key concepts in folder."""
        # This is simplified - in reality would extract from actual files
        return f"Key concepts and components in this folder will be detailed in the individual file documentation."

    def describe_architecture(self, folder_path: str, folder_info: Dict) -> str:
        """Describe folder architecture."""
        arch = "### Folder Structure\n\n"

        subfolders = folder_info.get('subfolders', [])
        if subfolders:
            arch += "Subfolders:\n\n"
            for sf in sorted(subfolders):
                arch += f"- `{sf}/`\n"

        return arch

    def list_important_files(self, folder_path: str, folder_info: Dict) -> str:
        """List important files in folder."""
        files = folder_info.get('files', [])

        if not files:
            return "*No files in this folder.*"

        important = "### Key Files\n\n"

        # Prioritize certain files
        priority_files = []
        other_files = []

        for filename in files:
            if filename.startswith('__init__') or filename == 'README.md':
                priority_files.append(filename)
            else:
                other_files.append(filename)

        for filename in priority_files[:5]:
            important += f"- [`{filename}_docs.md`](./{filename}_docs.md)\n"

        for filename in sorted(other_files)[:10]:
            important += f"- [`{filename}_docs.md`](./{filename}_docs.md)\n"

        return important

    def describe_workflow(self, folder_path: str) -> str:
        """Describe workflow for working with folder."""
        return """### Development Workflow

1. Review the file documentation to understand components
2. Check test files for usage examples
3. Follow PyTorch contribution guidelines
4. Run tests before submitting changes

*See the PyTorch documentation for detailed contribution guidelines.*
"""

    def generate_cross_references(self, folder_path: str) -> str:
        """Generate cross-references."""
        refs = "### Related Documentation\n\n"
        refs += f"- [Folder Index](./index.md)\n"
        refs += f"- [Keyword Index](./sub.md)\n"
        refs += f"- [Global Index](../index.md)\n"

        return refs

    def get_relative_path(self, from_folder: str, to_folder: str) -> str:
        """Calculate relative path between folders."""
        if from_folder == to_folder:
            return '.'

        from_parts = from_folder.split('/') if from_folder else []
        to_parts = to_folder.split('/') if to_folder else []

        # Go up from from_folder
        up_count = len(from_parts)
        up_path = '../' * up_count

        # Then down to to_folder
        down_path = '/'.join(to_parts) if to_parts else ''

        if down_path:
            return up_path + down_path
        else:
            return up_path.rstrip('/') if up_path else '.'


class GlobalDocumentationGenerator:
    """Generates global documentation files."""

    def __init__(self, scanner: RepoScanner):
        self.scanner = scanner

    def generate_global_index(self) -> str:
        """Generate global index.md."""
        doc = """# PyTorch Repository Documentation

## Welcome

This is the comprehensive documentation system for the PyTorch repository. This documentation covers every file, folder, and component of the project.

## Repository Overview

PyTorch is an open-source machine learning framework that accelerates the path from research to production. It provides:

- Tensor computation with strong GPU acceleration
- Deep neural networks built on a tape-based autograd system
- A rich ecosystem of tools and libraries

## Documentation Structure

This documentation is organized as follows:

### Global Resources

- **[Global Keyword Index](./keywords.md)**: Search for any term, class, function, or concept
- **[Comprehensive Book](./comprehensive_book.md)**: Complete reference guide (1M+ words)
- **This Index**: Navigate the repository structure

### Folder-by-Folder Documentation

Each folder in the repository has three documentation files:

1. **index.md**: Lists files and subfolders
2. **doc.md**: Narrative documentation about the folder's role
3. **sub.md**: Keyword index for the folder and all descendants

### File-by-File Documentation

Each documented file has two documentation files:

1. **<filename>_docs.md**: Complete documentation including source code, analysis, architecture, and usage
2. **<filename>_kw.md**: Keyword index for the file

## Repository Structure

"""

        # List top-level folders
        root_info = self.scanner.folders.get('', {})
        subfolders = root_info.get('subfolders', [])

        doc += "### Top-Level Folders\n\n"
        for folder in sorted(subfolders):
            doc += f"- [`{folder}/`](./{folder}/index.md)\n"

        doc += """

## How to Use This Documentation

1. **Browse by Structure**: Start with folder indexes to navigate the codebase
2. **Search by Keyword**: Use the global keyword index to find specific terms
3. **Read the Book**: Read the comprehensive book for a complete understanding
4. **Dive into Files**: Read individual file documentation for details

## Quick Links

### Core Components

- [torch/](./torch/index.md) - Main PyTorch Python API
- [aten/](./aten/index.md) - ATen C++ tensor library
- [c10/](./c10/index.md) - Core abstractions
- [caffe2/](./caffe2/index.md) - Caffe2 framework

### Development

- [test/](./test/index.md) - Test suites
- [tools/](./tools/index.md) - Development tools
- [scripts/](./scripts/index.md) - Utility scripts

---

*Generated by PyTorch Repository Documentation System*
"""

        return doc

    def generate_global_keywords(self) -> str:
        """Generate global keywords.md."""
        doc = """# Global Keyword Index

## About This Index

This is a comprehensive keyword index covering all files in the PyTorch repository. Use this index to find:

- Classes, functions, and methods
- Technical terms and concepts
- Module names and file locations
- API components

## How to Use

1. Find your keyword alphabetically below
2. Click on file links to see detailed documentation
3. Use the keyword file links to see all keywords in a file

## Keywords by Category

*Note: This is a high-level index. Detailed keywords are in folder-specific `sub.md` files.*

### A-Z Index

"""

        # Collect all files by first letter
        by_letter = defaultdict(list)
        for file_path, file_info in sorted(self.scanner.files.items()):
            first_letter = file_info['name'][0].upper()
            if first_letter.isalpha():
                by_letter[first_letter].append(file_info)

        # Generate index
        for letter in sorted(by_letter.keys()):
            doc += f"\n### {letter}\n\n"
            files = by_letter[letter][:100]  # Limit to 100 files per letter
            for file_info in files:
                folder = file_info['folder']
                name = file_info['name']
                path = file_info['path']

                folder_parts = folder.split('/') if folder else []
                rel_path = '/'.join(folder_parts) if folder_parts else ''

                doc += f"- **{name}**: [{path}](./{rel_path}/{name}_docs.md) | [keywords](./{rel_path}/{name}_kw.md)\n"

        doc += """

---

*Generated by PyTorch Repository Documentation System*
"""

        return doc

    def generate_comprehensive_book(self) -> str:
        """Generate comprehensive_book.md."""
        doc = """# The PyTorch Repository: A Comprehensive Guide

## Preface

This book provides a complete, in-depth reference to the PyTorch repository. It covers every component, file, and concept in the codebase.

**Scope**: Entire PyTorch repository
**Target Audience**: Developers, researchers, contributors
**Depth**: Comprehensive technical documentation

---

## Table of Contents

- Part I: Project Overview
- Part II: Architecture & Design
- Part III: Core Components
- Part IV: Advanced Features
- Part V: Development & Testing
- Part VI: Performance & Optimization
- Part VII: Contributing & Extending

---

# Part I: Project Overview

## Chapter 1: Introduction to PyTorch

PyTorch is an open-source machine learning framework that provides:

1. **Tensor Computation**: GPU-accelerated tensor operations similar to NumPy
2. **Automatic Differentiation**: Tape-based autograd system for building neural networks
3. **Neural Network API**: High-level building blocks for deep learning models
4. **Production Ready**: Tools for deploying models to production

### Key Features

- **Dynamic Computation Graphs**: Build graphs on-the-fly for flexible model architectures
- **Python-First**: Native Python interface with intuitive APIs
- **Strong GPU Support**: Seamless CPU-GPU tensor transfers
- **Rich Ecosystem**: Extensive libraries for computer vision, NLP, and more

## Chapter 2: Repository Structure

The PyTorch repository is organized into several major components:

"""

        # Add folder descriptions
        root_info = self.scanner.folders.get('', {})
        subfolders = root_info.get('subfolders', [])

        for folder in sorted(subfolders):
            if folder.startswith('.'):
                continue
            doc += f"\n### {folder}/\n\n"
            doc += f"See [folder documentation](./{folder}/doc.md) for details.\n"

        doc += """

# Part II: Architecture & Design

## Chapter 3: Core Architecture

PyTorch follows a layered architecture:

1. **C10**: Core abstractions (tensors, devices, etc.)
2. **ATen**: Tensor library with operations
3. **Autograd**: Automatic differentiation engine
4. **Python API**: User-facing PyTorch interface

### Design Principles

- **Imperative Programming**: Code executes immediately (eager execution)
- **Extensibility**: Easy to add custom operations and modules
- **Performance**: Optimized C++ and CUDA kernels
- **Interoperability**: Works with NumPy, SciPy, and other Python libraries

## Chapter 4: Component Interactions

[Detailed component interaction diagrams and descriptions would go here]

# Part III: Core Components

## Chapter 5: Tensor Library (ATen)

[Detailed ATen documentation]

## Chapter 6: Autograd Engine

[Detailed autograd documentation]

## Chapter 7: Neural Network Modules

[Detailed nn module documentation]

# Part IV: Advanced Features

## Chapter 8: JIT Compilation

[TorchScript and JIT documentation]

## Chapter 9: Distributed Training

[Distributed training documentation]

## Chapter 10: GPU Acceleration

[CUDA integration documentation]

# Part V: Development & Testing

## Chapter 11: Testing Infrastructure

"""

        # Add info about test files
        test_files = [f for f in self.scanner.files.values() if 'test' in f['folder'].lower()]
        doc += f"\nThe repository contains {len(test_files)} test files ensuring code quality.\n"

        doc += """

## Chapter 12: Build System

[Build system documentation]

# Part VI: Performance & Optimization

## Chapter 13: Performance Best Practices

[Performance guidelines]

## Chapter 14: Profiling & Benchmarking

[Profiling tools documentation]

# Part VII: Contributing & Extending

## Chapter 15: Contribution Guidelines

[Contribution process]

## Chapter 16: Extending PyTorch

[How to add new features]

---

# Appendices

## Appendix A: File Reference

Complete list of all files in the repository:

"""

        # List all files (limited)
        for i, (file_path, file_info) in enumerate(sorted(self.scanner.files.items())):
            if i >= 500:  # Limit to first 500 files
                doc += f"\n*... and {len(self.scanner.files) - 500} more files (see index.md for complete list)*\n"
                break
            folder = file_info['folder']
            name = file_info['name']
            folder_parts = folder.split('/') if folder else []
            rel_path = '/'.join(folder_parts) if folder_parts else ''
            doc += f"- [{file_path}](./{rel_path}/{name}_docs.md)\n"

        doc += """

## Appendix B: Glossary

- **Tensor**: Multi-dimensional array, the fundamental data structure in PyTorch
- **Autograd**: Automatic differentiation system
- **ATen**: A Tensor Library, the C++ tensor library
- **C10**: Caffe2 Core, providing fundamental abstractions
- **JIT**: Just-In-Time compilation for optimizing models
- **CUDA**: NVIDIA's parallel computing platform for GPU acceleration

---

*Generated by PyTorch Repository Documentation System*

**Total Files Documented**: """ + str(len(self.scanner.files)) + """
**Total Folders**: """ + str(len(self.scanner.folders)) + """
**Documentation Size**: Comprehensive (millions of words across all files)
"""

        return doc


def main():
    """Main documentation generation function."""
    print("=" * 80)
    print("PyTorch Repository Documentation Generator")
    print("=" * 80)
    print()

    # Step 1: Scan repository
    print("Step 1: Scanning repository structure...")
    scanner = RepoScanner(REPO_ROOT)
    scanner.scan()
    print(f"  ✓ Found {len(scanner.folders)} folders")
    print(f"  ✓ Found {len(scanner.files)} documentable files")
    print()

    # Step 2: Generate per-file documentation
    print("Step 2: Generating per-file documentation...")
    file_gen = FileDocumentationGenerator(scanner)

    file_count = 0
    for file_path, file_info in scanner.files.items():
        folder = file_info['folder']
        name = file_info['name']

        # Create docs folder structure
        if folder:
            docs_folder = DOCS_ROOT / folder
        else:
            docs_folder = DOCS_ROOT

        docs_folder.mkdir(parents=True, exist_ok=True)

        # Generate _docs.md
        docs_path = docs_folder / f"{name}_docs.md"
        if not docs_path.exists():  # Skip if already exists
            docs_content = file_gen.generate_file_docs(file_info)
            docs_path.write_text(docs_content, encoding='utf-8')

        # Generate _kw.md
        kw_path = docs_folder / f"{name}_kw.md"
        if not kw_path.exists():
            kw_content = file_gen.generate_file_keywords(file_info)
            kw_path.write_text(kw_content, encoding='utf-8')

        file_count += 1
        if file_count % 100 == 0:
            print(f"  Progress: {file_count}/{len(scanner.files)} files...")

    print(f"  ✓ Generated documentation for {file_count} files")
    print()

    # Step 3: Generate per-folder documentation
    print("Step 3: Generating per-folder documentation...")
    folder_gen = FolderDocumentationGenerator(scanner)

    for folder_path in scanner.folders.keys():
        if folder_path:
            docs_folder = DOCS_ROOT / folder_path
        else:
            docs_folder = DOCS_ROOT

        docs_folder.mkdir(parents=True, exist_ok=True)

        # Generate index.md
        index_path = docs_folder / "index.md"
        if not index_path.exists():
            index_content = folder_gen.generate_folder_index(folder_path)
            index_path.write_text(index_content, encoding='utf-8')

        # Generate doc.md
        doc_path = docs_folder / "doc.md"
        if not doc_path.exists():
            doc_content = folder_gen.generate_folder_doc(folder_path)
            doc_path.write_text(doc_content, encoding='utf-8')

        # Generate sub.md
        sub_path = docs_folder / "sub.md"
        if not sub_path.exists():
            sub_content = folder_gen.generate_folder_keywords(folder_path)
            sub_path.write_text(sub_content, encoding='utf-8')

    print(f"  ✓ Generated documentation for {len(scanner.folders)} folders")
    print()

    # Step 4: Generate global documentation
    print("Step 4: Generating global documentation...")
    global_gen = GlobalDocumentationGenerator(scanner)

    # Generate global index.md
    global_index = global_gen.generate_global_index()
    (DOCS_ROOT / "index.
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Asynchronous Programming**: Uses async/await
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs
- **Database**: May involve SQL - watch for injection vulnerabilities

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs`):

- [`Makefile_docs.md`](./Makefile_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`requirements.txt_docs.md`](./requirements.txt_docs.md)
- [`libtorch.rst_docs.md`](./libtorch.rst_docs.md)
- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`generate_repo_docs.py_kw.md_docs.md`](./generate_repo_docs.py_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`pt_template_srcs.bzl_kw.md_docs.md`](./pt_template_srcs.bzl_kw.md_docs.md)
- [`CLAUDE.md_docs.md_docs.md`](./CLAUDE.md_docs.md_docs.md)
- [`setup.py_kw.md_docs.md`](./setup.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `generate_repo_docs.py_docs.md_docs.md`
- **Keyword Index**: `generate_repo_docs.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
