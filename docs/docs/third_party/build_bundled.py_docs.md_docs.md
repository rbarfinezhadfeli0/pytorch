# Documentation: `docs/third_party/build_bundled.py_docs.md`

## File Metadata

- **Path**: `docs/third_party/build_bundled.py_docs.md`
- **Size**: 11,210 bytes (10.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `third_party/build_bundled.py`

## File Metadata

- **Path**: `third_party/build_bundled.py`
- **Size**: 8,357 bytes (8.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
#!/usr/bin/env python3
import argparse
import os


mydir = os.path.dirname(__file__)
licenses = {'LICENSE', 'LICENSE.txt', 'LICENSE.rst', 'COPYING.BSD'}


def collect_license(current):
    collected = {}
    for root, dirs, files in os.walk(current):
        license = list(licenses & set(files))
        if license:
            name = root.split('/')[-1]
            license_file = os.path.join(root, license[0])
            try:
                ident = identify_license(license_file)
            except ValueError:
                raise ValueError('could not identify license file '
                                 f'for {root}') from None
            val = {
                'Name': name,
                'Files': [root],
                'License': ident,
                'License_file': [license_file],
            }
            if name in collected:
                # Only add it if the license is different
                if collected[name]['License'] == ident:
                    collected[name]['Files'].append(root)
                    collected[name]['License_file'].append(license_file)
                else:
                    collected[name + f' ({root})'] = val
            else:
                collected[name] = val
    return collected


def create_bundled(d, outstream, include_files=False):
    """Write the information to an open outstream"""
    collected = collect_license(d)
    sorted_keys = sorted(collected.keys())
    outstream.write('The PyTorch repository and source distributions bundle '
                    'several libraries that are \n')
    outstream.write('compatibly licensed.  We list these here.')
    files_to_include = []
    for k in sorted_keys:
        c = collected[k]
        files = ',\n     '.join(c['Files'])
        license_file = ',\n     '.join(c['License_file'])
        outstream.write('\n\n')
        outstream.write(f"Name: {c['Name']}\n")
        outstream.write(f"License: {c['License']}\n")
        outstream.write(f"Files: {files}\n")
        outstream.write('  For details, see')
        if include_files:
            outstream.write(' the files concatenated below: ')
            files_to_include += c['License_file']
        else:
            outstream.write(': ')
        outstream.write(license_file)
    for fname in files_to_include:
        outstream.write('\n\n')
        outstream.write(fname)
        outstream.write('\n' + '-' * len(fname) + '\n')
        with open(fname, 'r') as fid:
            outstream.write(fid.read())


def identify_license(f, exception=''):
    """
    Read f and try to identify the license type
    This is __very__ rough and probably not legally binding, it is specific for
    this repo.
    """
    def squeeze(t):
        """Remove 'n and ' ', normalize quotes
        """
        t = t.replace('\n', '').replace(' ', '')
        t = t.replace('``', '"').replace("''", '"')
        return t

    with open(f) as fid:
        txt = fid.read()
        if not exception and 'exception' in txt:
            license = identify_license(f, 'exception')
            return license + ' with exception'
        txt = squeeze(txt)
        if 'ApacheLicense' in txt:
            # Hmm, do we need to check the text?
            return 'Apache-2.0'
        elif 'MITLicense' in txt:
            # Hmm, do we need to check the text?
            return 'MIT'
        elif 'BSD-3-ClauseLicense' in txt:
            # Hmm, do we need to check the text?
            return 'BSD-3-Clause'
        elif 'BSD3-ClauseLicense' in txt:
            # Hmm, do we need to check the text?
            return 'BSD-3-Clause'
        elif 'BoostSoftwareLicense-Version1.0' in txt:
            # Hmm, do we need to check the text?
            return 'BSL-1.0'
        elif 'gettimeofday' in txt:
            # Used in opentelemetry-cpp/tools/vcpkg/ports/gettimeofday
            return 'Apache-2.0'
        elif 'libhungarian' in txt:
            # Used in opentelemetry-cpp/tools/vcpkg/ports/hungarian
            return 'Permissive (free to use)'
        elif 'PDCurses' in txt:
            # Used in opentelemetry-cpp/tools/vcpkg/ports/pdcurses
            return 'Public Domain for core'
        elif 'Copyright1999UniversityofNorthCarolina' in txt:
            # Used in opentelemetry-cpp/tools/vcpkg/ports/pqp
            return 'Apache-2.0'
        elif 'sigslot' in txt:
            # Used in opentelemetry-cpp/tools/vcpkg/ports/sigslot
            return 'Public Domain'
        elif squeeze("Clarified Artistic License") in txt:
            return 'Clarified Artistic License'
        elif all([squeeze(m) in txt.lower() for m in bsd3_txt]):
            return 'BSD-3-Clause'
        elif all([squeeze(m) in txt.lower() for m in bsd3_v1_txt]):
            return 'BSD-3-Clause'
        elif all([squeeze(m) in txt.lower() for m in bsd2_txt]):
            return 'BSD-2-Clause'
        elif all([squeeze(m) in txt.lower() for m in bsd3_src_txt]):
            return 'BSD-Source-Code'
        elif any([squeeze(m) in txt.lower() for m in mit_txt]):
            return 'MIT'
        else:
            raise ValueError('unknown license')

mit_txt = ['permission is hereby granted, free of charge, to any person ',
           'obtaining a copy of this software and associated documentation ',
           'files (the "software"), to deal in the software without ',
           'restriction, including without limitation the rights to use, copy, ',
           'modify, merge, publish, distribute, sublicense, and/or sell copies ',
           'of the software, and to permit persons to whom the software is ',
           'furnished to do so, subject to the following conditions:',

           'the above copyright notice and this permission notice shall be ',
           'included in all copies or substantial portions of the software.',

           'the software is provided "as is", without warranty of any kind, ',
           'express or implied, including but not limited to the warranties of ',
           'merchantability, fitness for a particular purpose and ',
           'noninfringement. in no event shall the authors or copyright holders ',
           'be liable for any claim, damages or other liability, whether in an ',
           'action of contract, tort or otherwise, arising from, out of or in ',
           'connection with the software or the use or other dealings in the ',
           'software.',
           ]

bsd3_txt = ['redistribution and use in source and binary forms, with or without '
            'modification, are permitted provided that the following conditions '
            'are met:',

            'redistributions of source code',

            'redistributions in binary form',

            'neither the name',

            'this software is provided by the copyright holders and '
            'contributors "as is" and any express or implied warranties, '
            'including, but not limited to, the implied warranties of '
            'merchantability and fitness for a particular purpose are disclaimed.',
            ]

# BSD2 is BSD3 without the "neither the name..." clause
bsd2_txt = bsd3_txt[:3] + bsd3_txt[4:]

# This BSD3 variant leaves "and contributors" out of the last clause of BSD-3,
# which is still valid BSD-3
v1 = bsd3_txt[4].replace('and contributors', '')
bsd3_v1_txt = bsd3_txt[:3] + [v1]

# This source variant of BSD-3 leaves the "redistributions in binary form" out
# which is https://spdx.org/licenses/BSD-Source-Code.html
bsd3_src_txt = bsd3_txt[:2] + bsd3_txt[4:]


if __name__ == '__main__':
    third_party = os.path.relpath(mydir)
    parser = argparse.ArgumentParser(
        description="Generate bundled licenses file",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=os.environ.get(
            "PYTORCH_THIRD_PARTY_BUNDLED_LICENSE_FILE",
            str(os.path.join(third_party, 'LICENSES_BUNDLED.txt'))
        ),
        help="location to output new bundled licenses file",
    )
    parser.add_argument(
        "--include-files",
        action="store_true",
        default=False,
        help="include actual license terms to the output",
    )
    args = parser.parse_args()
    fname = args.out_file
    print(f"+ Writing bundled licenses to {args.out_file}")
    with open(fname, 'w') as fid:
        create_bundled(third_party, fid, args.include_files)

```



## High-Level Overview

"""Write the information to an open outstream"""    collected = collect_license(d)    sorted_keys = sorted(collected.keys())    outstream.write('The PyTorch repository and source distributions bundle '                    'several libraries that are \n')    outstream.write('compatibly licensed.  We list these here.')    files_to_include = []    for k in sorted_keys:        c = collected[k]        files = ',\n     '.join(c['Files'])

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `collect_license`, `create_bundled`, `identify_license`, `squeeze`

**Key imports**: argparse, os


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `third_party`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `os`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`third_party`):

- [`glog.buck.bzl_docs.md`](./glog.buck.bzl_docs.md)
- [`generate-xnnpack-wrappers.py_docs.md`](./generate-xnnpack-wrappers.py_docs.md)
- [`generate-cpuinfo-wrappers.py_docs.md`](./generate-cpuinfo-wrappers.py_docs.md)
- [`xpu.txt_docs.md`](./xpu.txt_docs.md)
- [`kineto.buck.bzl_docs.md`](./kineto.buck.bzl_docs.md)
- [`xnnpack.buck.bzl_docs.md`](./xnnpack.buck.bzl_docs.md)
- [`xnnpack_wrapper_defs.bzl_docs.md`](./xnnpack_wrapper_defs.bzl_docs.md)
- [`eigen_pin.txt_docs.md`](./eigen_pin.txt_docs.md)
- [`LICENSES_BUNDLED.txt_docs.md`](./LICENSES_BUNDLED.txt_docs.md)
- [`sleef.bzl_docs.md`](./sleef.bzl_docs.md)


## Cross-References

- **File Documentation**: `build_bundled.py_docs.md`
- **Keyword Index**: `build_bundled.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/third_party`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/third_party`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/third_party`):

- [`substitution.bzl_kw.md_docs.md`](./substitution.bzl_kw.md_docs.md)
- [`xnnpack_buck_shim.bzl_kw.md_docs.md`](./xnnpack_buck_shim.bzl_kw.md_docs.md)
- [`LICENSES_BUNDLED.txt_kw.md_docs.md`](./LICENSES_BUNDLED.txt_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`kineto.buck.bzl_docs.md_docs.md`](./kineto.buck.bzl_docs.md_docs.md)
- [`generate-cpuinfo-wrappers.py_kw.md_docs.md`](./generate-cpuinfo-wrappers.py_kw.md_docs.md)
- [`xnnpack_buck_shim.bzl_docs.md_docs.md`](./xnnpack_buck_shim.bzl_docs.md_docs.md)
- [`eigen_pin.txt_docs.md_docs.md`](./eigen_pin.txt_docs.md_docs.md)
- [`build_bundled.py_kw.md_docs.md`](./build_bundled.py_kw.md_docs.md)
- [`generate-cpuinfo-wrappers.py_docs.md_docs.md`](./generate-cpuinfo-wrappers.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `build_bundled.py_docs.md_docs.md`
- **Keyword Index**: `build_bundled.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
