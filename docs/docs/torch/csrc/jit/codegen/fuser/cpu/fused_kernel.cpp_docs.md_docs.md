# Documentation: `docs/torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp_docs.md`
- **Size**: 12,998 bytes (12.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp`
- **Size**: 10,637 bytes (10.39 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/codegen/fuser/cpu/fused_kernel.h>

#include <ATen/DynamicLibrary.h>
#include <ATen/code_template.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <torch/csrc/jit/codegen/fuser/compiler.h>
#include <torch/csrc/jit/codegen/fuser/cpu/temp_file.h>
#include <optional>

#include <cstdlib>
#include <iostream>
#include <string>

namespace torch::jit::fuser::cpu {

#ifdef _MSC_VER
static const std::string getTempPath() {
  wchar_t lpTempPathBuffer[MAX_PATH];

  DWORD dwRetVal = GetTempPathW(
      MAX_PATH, // length of the buffer
      lpTempPathBuffer); // buffer for path

  TORCH_CHECK(dwRetVal < MAX_PATH && dwRetVal != 0, "GetTempPath failed.");

  return std::string(c10::u16u8(lpTempPathBuffer));
}
static const std::string temp_dir = getTempPath();
static const std::string so_template = temp_dir + "pytorch_fuserXXXXXX.dll";
static const std::string cpp_template = temp_dir + "pytorch_fuserXXXXXX.cpp";
static const std::string check_exists_string = "where ${program} > nul 2> nul";
static std::vector<std::wstring> env_list;
constexpr int so_suffix_len = 4;
constexpr int cpp_suffix_len = 4;
#else
static const std::string so_template = "/tmp/pytorch_fuserXXXXXX.so";
static const std::string cpp_template = "/tmp/pytorch_fuserXXXXXX.cpp";
static const std::string check_exists_string = "which ${program} > /dev/null";
constexpr int so_suffix_len = 3;
constexpr int cpp_suffix_len = 4;
#endif

#ifdef _MSC_VER
static std::optional<std::wstring> exec(const std::wstring& cmd) {
  std::array<wchar_t, 128> buffer;
  std::wstring result;
  std::unique_ptr<FILE, decltype(&_pclose)> pipe(
      _wpopen(cmd.c_str(), L"r"), _pclose);
  if (!pipe) {
    return std::nullopt;
  }
  while (fgetws(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) !=
         nullptr) {
    result += buffer.data();
  }
  return result;
}

inline std::wstring& rtrim(std::wstring& s, const wchar_t* t = L" \t\n\r\f\v") {
  s.erase(s.find_last_not_of(t) + 1);
  return s;
}

static void activate() {
  wchar_t* root = nullptr;
  std::wstring cmd;
  std::optional<std::wstring> exec_out;
  std::wstring path;
  std::wstring vcruntime_plat;
  std::wstring envvars;

  // Checking whether the environment is already activated
  if (_wgetenv(L"VSCMD_ARG_TGT_ARCH")) {
    return;
  }

  // Getting `ProgramFiles` through environment variable queries
  root = _wgetenv(L"ProgramFiles(x86)");
  if (!root) {
    root = _wgetenv(L"ProgramFiles");
  }
  if (!root) {
    return;
  }

  // Getting VS 2017 installation path through `vswhere`
  cmd = L"\"" + std::wstring(root) +
      L"\\Microsoft Visual Studio\\Installer\\vswhere.exe\""
      L" -latest -prerelease -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath";
  exec_out = exec(cmd);
  if (!exec_out) {
    return;
  }
  path = *exec_out;
  rtrim(path);

  // Checking whether the activation script `vcvarsall.bat` exists
  path += L"\\VC\\Auxiliary\\Build";
  struct _stati64 st;
  if (_wstati64(path.c_str(), &st) == -1 || !(st.st_mode & _S_IFDIR)) {
    return;
  }
  path += L"\\vcvarsall.bat";
  if (_waccess(path.c_str(), 0) == -1) {
    return;
  }

  // Determining current platform
  if (sizeof(void*) == 8) {
    vcruntime_plat = L"x64";
  } else {
    vcruntime_plat = L"x86";
  }

  // Getting environment variables after activating VS development shell
  cmd = L"\"" + path + L"\" " + vcruntime_plat + L">NUL && set";
  exec_out = exec(cmd);
  if (!exec_out) {
    return;
  }
  envvars = *exec_out;

  // Setting environment variables to the current environment
  std::wistringstream f(envvars);
  std::wstring envvar;
  while (getline(f, envvar, L'\n')) {
    env_list.push_back(envvar);
  }
}

static intptr_t run(const std::string& cmd) {
  // Getting the path of `cmd.exe`
  const wchar_t* comspec = _wgetenv(L"COMSPEC");
  if (!comspec) {
    comspec = L"C:\\Windows\\System32\\cmd.exe";
  }
  // Constructing the command line
  auto wCmd = c10::u8u16(cmd);
  const wchar_t* a[] = {L"/c", wCmd.c_str(), nullptr};
  // Constructing the env array
  // If `env_list` is not empty, then add char pointers ending with nullptr.
  // Otherwise, it will be nullptr, which implies the default env.
  std::vector<const wchar_t*> e;
  if (!env_list.empty()) {
    for (auto& s : env_list) {
      e.push_back(s.c_str());
    }
    e.push_back(nullptr);
  }
  // Running the command
  intptr_t r = _wspawnve(_P_WAIT, comspec, a, e.data());
  return r;
}
#endif

static bool programExists(const std::string& program) {
  std::stringstream ss;
  c10::printQuotedString(ss, program);
  at::jit::TemplateEnv env;
  env.s("program", ss.str());
  std::string cmd = format(check_exists_string, env);
#ifdef _MSC_VER
  return (run(cmd.c_str()) == 0);
#else
  return (system(cmd.c_str()) == 0);
#endif
}

// A single compiler config is accessed through getConfig() (below)
// Controls compilation options and may be updated based on the result
// of compilation attempts.
struct CompilerConfig {
  CompilerConfig() {
    const auto cxx_env = c10::utils::get_env("CXX");
    if (cxx_env) {
      cxx = cxx_env.value();
    }

#ifdef _MSC_VER
    activate();
#endif

    if (!programExists(cxx)) {
      TORCH_WARN("Compiler passed via CXX envvar does not exist!");
      cxx = "";
    }
  }

  ~CompilerConfig() = default;

#ifdef _MSC_VER
  std::string cxx = "cl";
  const std::string openmp_flags = "/openmp";
#elif defined(__clang__)
  std::string cxx = "clang++";
  const std::string openmp_flags = "-fopenmp";
#else
  std::string cxx = "g++";
  const std::string openmp_flags = "-fopenmp";
#endif
// Set openmp to true only if PyTorch is compiled with OpenMP support
// OpenMP is typically not available on MacOS platform
#if defined(_OPENMP)
  bool openmp = true;
#else
  bool openmp = false;
#endif
};

static CompilerConfig& getConfig() {
  static CompilerConfig config;
  return config;
}

// NB: -march=native not supported on PPC64 g++.  It's a bit annoying
// to do a configure-style test to decide whether or not the g++
// actually supports it or not, so we heuristically use the host
// compiler to predict if the runtime compiler supports the option we
// want.  This probably won't work if you're cross-compiling.
// NB: -march=native is disabled because it has caused problems where
// compiler and assembler do not agree on what native instruction they
// understand for AVX512. When we need better CPU performance this
// optimization can be re-enabled by tracking down the platforms where
// this error occurs and only selectively disabling it.
#if (defined(_MSC_VER) && !defined(_M_ARM64))
// According to https://stackoverflow.com/a/29178079, we are able to
// detect which arch level is supported by the vectorizer using
// the macro __isa_available. It is added during runtime.
// The result of __isa_available and the corresponding arch:
//  AVX       4
//  AVX2      5
//  AVX512    6
extern "C" int __isa_available;
static std::string getArchFlags() {
  if (__isa_available >= 6) {
    return "/arch:AVX512";
  } else if (__isa_available >= 5) {
    return "/arch:AVX2";
  } else if (__isa_available >= 4) {
    return "/arch:AVX";
  } else {
    return "";
  }
}
static const std::string arch_flags = getArchFlags();
static const std::string compile_string = "cd /D \"" + temp_dir +
    "\" && "
    "${cxx} /nologo /MD /O2 " +
    arch_flags +
    " /LD /EHsc "
    "${fopenmp} \"${cpp_file}\" /link /out:\"${so_file}\"";
#else
static const std::string compile_string =
    "\"${cxx}\" -O3 -g "
#ifndef __PPC64__
//  "-march=native "
#endif
    "-std=c++17 -fPIC ${fopenmp} -shared \"${cpp_file}\" -o \"${so_file}\" -lm";
#endif
static void runCompiler(
    const std::string& cpp_file,
    const std::string& so_file) {
  auto& config = getConfig();
  TORCH_CHECK(
      !config.cxx.empty(),
      "Failed to compile a fused CPU kernel: Compiler not found");
  at::jit::TemplateEnv env;
  env.s("cxx", config.cxx);
  env.s("fopenmp", config.openmp ? config.openmp_flags : "");
  env.s("cpp_file", cpp_file);
  env.s("so_file", so_file);
  std::string result = format(compile_string, env);
#ifdef _MSC_VER
  intptr_t r = run(result);
#else
  int r = system(result.c_str());
#endif
  if (config.openmp && r != 0) {
    std::cerr
        << "warning: pytorch jit fuser failed to compile with openmp, trying without it...\n";
    config.openmp = false; // disable for future compiles
    return runCompiler(cpp_file, so_file);
  }
  TORCH_CHECK(r == 0, "Failed to compile a fused CPU kernel");
}

#ifdef _MSC_VER
static const std::string disas_string =
    "dumpbin /DISASM:NOBYTES \"${so_file}\"";
#else
static const std::string disas_string = "objdump -M  intel -d \"${so_file}\"";
#endif
static void disas(const std::string& so_file) {
  at::jit::TemplateEnv env;
  env.s("so_file", so_file);
  std::string cmd = format(disas_string, env);
  int r = system(cmd.c_str());
  AT_ASSERT(r == 0);
}

FusedKernelCPU::FusedKernelCPU(
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random)
    : FusedKernel(
          std::move(name),
          std::move(code),
          std::move(input_desc),
          std::move(output_desc),
          std::move(chunk_desc),
          std::move(concat_desc),
          has_random) {
  TempFile so_file(so_template, so_suffix_len);
  TempFile cpp_file(cpp_template, cpp_suffix_len);
  cpp_file.write(code_);
  cpp_file.sync();
#ifdef _MSC_VER
  so_file.close();
  cpp_file.close();
#endif
  runCompiler(cpp_file.name(), so_file.name());
  if (debugFuser() >= 2)
    disas(so_file.name());
  so_lib = std::make_unique<at::DynamicLibrary>(so_file.name().c_str());
#pragma GCC diagnostic ignored "-Wpedantic"
  kernel =
      reinterpret_cast<void (*)(uint32_t, void**)>(so_lib->sym(name_.c_str()));
#pragma GCC diagnostic pop
}

static std::shared_ptr<FusedKernel> createFusionKernel(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random) {
  return std::make_shared<FusedKernelCPU>(
      std::move(name),
      std::move(code),
      std::move(input_desc),
      std::move(output_desc),
      std::move(chunk_desc),
      std::move(concat_desc),
      has_random);
}

static RegisterFusionBackend reg(DeviceType::CPU, createFusionKernel);
} // namespace torch::jit::fuser::cpu

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 26 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `_stati64`, `CompilerConfig`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/fuser/cpu`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/codegen/fuser/cpu/fused_kernel.h`
- `ATen/DynamicLibrary.h`
- `ATen/code_template.h`
- `c10/util/Exception.h`
- `c10/util/env.h`
- `torch/csrc/jit/codegen/fuser/compiler.h`
- `torch/csrc/jit/codegen/fuser/cpu/temp_file.h`
- `optional`
- `cstdlib`
- `iostream`
- `string`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/jit/codegen/fuser/cpu`):

- [`fused_kernel.h_docs.md`](./fused_kernel.h_docs.md)
- [`resource_strings.h_docs.md`](./resource_strings.h_docs.md)
- [`temp_file.h_docs.md`](./temp_file.h_docs.md)


## Cross-References

- **File Documentation**: `fused_kernel.cpp_docs.md`
- **Keyword Index**: `fused_kernel.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/codegen/fuser/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/codegen/fuser/cpu`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/codegen/fuser/cpu`):

- [`fused_kernel.h_docs.md_docs.md`](./fused_kernel.h_docs.md_docs.md)
- [`fused_kernel.cpp_kw.md_docs.md`](./fused_kernel.cpp_kw.md_docs.md)
- [`temp_file.h_docs.md_docs.md`](./temp_file.h_docs.md_docs.md)
- [`resource_strings.h_docs.md_docs.md`](./resource_strings.h_docs.md_docs.md)
- [`fused_kernel.h_kw.md_docs.md`](./fused_kernel.h_kw.md_docs.md)
- [`resource_strings.h_kw.md_docs.md`](./resource_strings.h_kw.md_docs.md)
- [`temp_file.h_kw.md_docs.md`](./temp_file.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `fused_kernel.cpp_docs.md_docs.md`
- **Keyword Index**: `fused_kernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
