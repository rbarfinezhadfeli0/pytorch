# Documentation: `docs/test/cpp/jit/test_lite_interpreter_direct.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_lite_interpreter_direct.cpp_docs.md`
- **Size**: 31,524 bytes (30.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_lite_interpreter_direct.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_lite_interpreter_direct.cpp`
- **Size**: 27,924 bytes (27.27 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <test/cpp/jit/test_utils.h>

#include <gtest/gtest.h>

#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/compatibility/backport.h>
#include <torch/csrc/jit/mobile/compatibility/backport_manager.h>
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <unordered_set>

// Tests go in torch::jit
namespace torch {
namespace jit {

TEST(LiteInterpreterDirectTest, UpsampleNearest2d) {
  Module m("m");
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  inputs.emplace_back(at::Scalar(2.0));
  auto ref = m.forward(inputs);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  res = bc.forward(inputs);

  auto resd = res.toTensor();
  auto refd = ref.toTensor();
  ASSERT_TRUE(resd.equal(refd));
}

TEST(LiteInterpreterDirectTest, CheckAttrAccess) {
  Module m("m");
  m.register_attribute("mobile_optimized", BoolType::get(), true);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  bool mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  AT_ASSERT(mobile_optimized);
  m.setattr("mobile_optimized", false);
  bc = jitModuleToMobile(m, options);
  mobile_optimized = bc.attr("mobile_optimized", false).toBool();
  AT_ASSERT(!mobile_optimized);
}

TEST(
    LiteInterpreterDirectTest,
    MethodInvocation) { // NOLINT (use =delete in gtest)
  const std::vector<std::string> test_programs{
      // test invoking a method with default parameter
      R"(
      def test_func(self, x, b : int = 4):
        return self.foo + x + b
      )",
      // inner method call with default parameter (gets inlined)
      R"(
      def add_with_default_arg(self, x, b : int = 4):
        return self.foo + x + b
      def test_func(self, x):
        return self.add_with_default_arg(x)  # invoke method w/ default arg
      )",
      // simple method call
      R"(
      def test_func(self, x):
        b = 4
        return self.foo + x + b
      )",
  };
  for (const auto& test_program : test_programs) {
    Module m("m");
    m.register_parameter("foo", torch::ones({}), false);
    m.define(test_program);

    const int fortyTwo = 42; // (keep linter happy)
    auto minput = fortyTwo * torch::ones({});
    auto ref = m.run_method("test_func", minput);

    CompilationOptions options;
    mobile::Module bc = jitModuleToMobile(m, options);
    const auto& test_func = bc.get_method("test_func");
    std::cerr << "hello " << std::endl;
    IValue res;
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }
    std::cerr << "hello 3" << std::endl;

    auto resd = res.toTensor().item<float>();
    auto refd = ref.toTensor().item<float>();
    AT_ASSERT(resd == refd);
  }
}

TEST(LiteInterpreterDirectTest, Conv) {
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  if (s && strcmp(s, "1") == 0)
    return;

  std::vector<torch::jit::IValue> inputs;

  Module m("m");
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  inputs.push_back(torch::ones({1, 1, 28, 28}));

  auto outputref = m.forward(inputs).toTensor();

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

TEST(LiteInterpreterDirectTest, Inline) {
  Module m("m");
  m.define(R"JIT(
  def foo1(self, x):
      return x + 1

  def foo2(self, x):
      return self.foo1(x) + 2

  def foo3(self, x):
      return self.foo2(x) + 3
  )JIT");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("foo3")(inputs);
  AT_ASSERT(output.toTensor().item<float>() == 7.0);
}

TEST(LiteInterpreterDirectTest, Tuple) {
  Module m("m");
  m.define(R"JIT(
  def foo(self, x):
      return (1, 2, x + 3)

  def forward(self, x):
      tuple = self.foo(x)
      return tuple
  )JIT");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("forward")(inputs);
  AT_ASSERT(output.toTupleRef().elements()[1].toInt() == 2);
}

TEST(LiteInterpreterDirectTest, Dict) {
  Module m("m");
  m.define(R"JIT(
  def foo(self, x):
      return {"result": x + 1}

  def forward(self, x):
      d = self.foo(x)
      return d
  )JIT");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("forward")(inputs);
  AT_ASSERT(output.toGenericDict().at("result").toTensor().item().toInt() == 2);
}

TEST(LiteInterpreterDirectTest, Prim) {
  Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return int(x)
  )JIT");

  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.run_method("forward", minput);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);

  IValue res;
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);
}

TEST(LiteInterpreterDirectTest, PrimScalar) {
  Module m("m");
  m.define(R"JIT(
        def forward(self, x):
            return int(x.item())
  )JIT");

  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.run_method("forward", minput);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);
}

TEST(LiteInterpreterDirectTest, WrongMethodName) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  ASSERT_THROWS_WITH_MESSAGE(
      bc.get_method("forward")(inputs), "is not defined");
}

TEST(LiteInterpreterDirectTest, SetState) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def __getstate__(self):
      return self.foo
    def __setstate__(self, a):
      self.foo = a
    def forward(self, x):
      b = 4
      return self.foo + x + b
  )");

  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);

  std::stringstream ms;
  m.save(ms);
  auto loaded_m = load(ms);
  auto ref = loaded_m.run_method("forward", minput);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

class TorchBindLiteInterpreterDirectTestStruct
    : public torch::jit::CustomClassHolder {
 public:
  std::string get(at::Tensor t) {
    std::stringstream ss;
    ss << "Hello! Your tensor has ";
    ss << t.numel();
    ss << " elements!";
    return ss.str();
  }
};

namespace {
struct ClassNamespaceValue : public SugaredValue {
  explicit ClassNamespaceValue(c10::QualifiedName name)
      : basename_(std::move(name)) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange&,
      GraphFunction&,
      const std::string& name) override {
    const auto fullName = c10::QualifiedName(basename_, name);

    // Check to see if it is a custom class.
    if (auto custom_class = getCustomClass(fullName.qualifiedName())) {
      return std::make_shared<ClassValue>(custom_class);
    }

    // If it's not a custom class, assume it's another namespace
    // NOLINTNEXTLINE(performance-move-const-arg)
    return std::make_shared<ClassNamespaceValue>(fullName);
  }

  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  c10::QualifiedName basename_;
};

struct TestModuleResolver : public Resolver {
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction&,
      const SourceRange&) override {
    if (name == "torch") {
      return std::make_shared<BuiltinModule>("aten");
    } else if (name == "__torch__") {
      return std::make_shared<ClassNamespaceValue>(c10::QualifiedName(name));
    }

    return nullptr;
  }

  TypePtr resolveType(const std::string&, const SourceRange&) override {
    return nullptr;
  }
};
} // namespace

TEST(LiteInterpreterDirectTest, BuiltinFunction) {
  script::Module m("m");
  auto custom_class_obj =
      make_custom_class<TorchBindLiteInterpreterDirectTestStruct>();
  m.register_attribute("my_obj", custom_class_obj.type(), custom_class_obj);
  m.define(R"(
    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )");

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  auto res =
      bc.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto str = res.toStringRef();
  std::string expected = "Hello! Your tensor has 12 elements!";
  AT_ASSERT(str == expected);
}

#if !defined FB_XPLAT_BUILD
TEST(LiteInterpreterDirectTest, GetRuntimeByteCodeVersion) {
  auto runtime_bytecode_version = _get_runtime_bytecode_version();
  AT_ASSERT(
      runtime_bytecode_version ==
      caffe2::serialize::kMaxSupportedBytecodeVersion);
}

TEST(LiteInterpreterDirectTest, GetRuntimeOperatorsVersion) {
  auto runtime_operators_version = _get_runtime_operators_min_max_versions();
  AT_ASSERT(
      runtime_operators_version.first ==
          caffe2::serialize::kMinSupportedFileFormatVersion &&
      runtime_operators_version.second ==
          caffe2::serialize::kMaxSupportedFileFormatVersion);
}

/**
 * The test below is disarmed for FB internal xplat builds since
 * BUCK requires us to pass in the script_module_v4.ptl file in
 * as a resource dependency of the build rule for this file, and
 * we would need to access it via the C++ Resources API instead
 * of directly reading from disk (which is what the open source
 * build/run does).
 */
TEST(LiteInterpreterDirectTest, GetByteCodeVersion) {
  std::string filePath(__FILE__);
  auto test_model_file_v4 =
      filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file_v4.append("script_module_v4.ptl");

  auto version_v4 = _get_model_bytecode_version(test_model_file_v4);
  AT_ASSERT(version_v4 == 4);
}

#endif // !defined(FB_XPLAT_BUILD)

TEST(LiteInterpreterDirectTest, GetRuntimeOpsAndInfo) {
  auto runtime_ops = _get_runtime_ops_and_info();
  // Ballpark estimate of the minimal number of ops; just used to
  // verify API returns a reasonably large number.
  AT_ASSERT(runtime_ops.size() > 2900);
}

TEST(LiteInterpreterDirectTest, Eval) {
  std::vector<torch::jit::IValue> inputs;

  Module m("m");
  m.define(R"(
    def __init__(self, x):
      self.training = True

    def forward(self, input):
      return torch.dropout(input, 1.0, self.training)
  )");

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  inputs.push_back(torch::ones({1, 1, 28, 28}));
  m.eval();
  auto outputref = m.forward(inputs).toTensor();

  // save m in training mode to make sure that mobile eval() will correctly
  // change back to eval mode
  m.train();
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  bc.eval();
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

TEST(LiteInterpreterDirectTest, FindWrongMethodName) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  ASSERT_TRUE(bc.find_method("forward") == std::nullopt);
}

TEST(LiteInterpreterDirectTest, FindAndRunMethod) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");

  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.get_method("add_it")(inputs);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    auto method = bc.find_method("add_it");
    AT_ASSERT(method != std::nullopt);
    res = (*method)(std::move(bcinputs));
  }

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

TEST(LiteInterpreterDirectTest, RunMethodVariadic) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add_three(self, x, y):
      return self.foo + x + y
  )");

  std::vector<IValue> inputs;
  auto inputx = 5 * torch::ones({});
  auto inputy = 4 * torch::ones({});
  auto ref = m.run_method("add_three", inputx, inputy);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res = bc.run_method("add_three", inputx, inputy);

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

TEST(LiteInterpreterDirectTest, DuplicateSetState) {
  Module m("M");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def __getstate__(self):
      return self.foo + self.foo
    def __setstate__(self, a):
      self.foo = a
    def forward(self, x):
      b = 4
      return self.foo + x + b
  )");

  Module b("B");
  b.register_module("M0", m);
  b.register_module("M1", m);
  b.define(R"(
    def forward(self, x):
      return self.M0.forward(x) + self.M1.forward(x)
  )");

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  const auto methods = bc.get_methods();
  const size_t expected_n = 3;
  ASSERT_EQ(methods.size(), expected_n);
}

TEST(LiteInterpreterDirectTest, OpNameExportFetchRootOperators) {
  torch::jit::Module m("m");
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  m.define(R"(
    def forward(self, input):
      x1 = torch.zeros(2, 2)
      x2 = torch.empty_like(torch.empty(2, 2))
      x3 = torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
      return (x1, x2, x3)
  )");
  m.eval();

  CompilationOptions options;
  mobile::Module ptl_model = jitModuleToMobile(m, options);
  std::set<std::string> operator_names =
      torch::jit::mobile::_export_operator_list(ptl_model);
  std::set<std::string> expected_operator_names = {
      "aten::_convolution",
      "aten::empty.memory_format",
      "aten::empty_like",
      "aten::zeros",
  };
  EXPECT_EQ(operator_names, expected_operator_names)
      << "Expected the root operator lists to be the same";
}

TEST(LiteInterpreterDirectTest, DefaultArgsConv) {
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  if (s && strcmp(s, "1") == 0)
    return;

  std::vector<torch::jit::IValue> inputs;

  Module m("m");
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  m.define(R"(
    def forward(self, input):
      return torch.conv2d(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], 1)
  )");

  inputs.emplace_back(torch::ones({1, 1, 28, 28}));

  auto outputref = m.forward(inputs).toTensor();

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 1; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(output.equal(outputref));
}

namespace {
void testLiteModuleCompareResultTensors(
    Module& m,
    const std::vector<torch::jit::IValue>& inputs,
    const std::string& method_name = "forward") {
  auto outputref = m.get_method(method_name)(inputs).toTensor();

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method(method_name)(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(output.equal(outputref));
}

void testDefaultArgsPinv2(int num_args) {
  Module m("m");
  if (num_args == 1) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input)
    )");
  } else if (num_args == 2) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5)
    )");
  } else if (num_args == 3) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, True)
    )");
  }

  std::vector<torch::jit::IValue> inputs;
  const int N = 28;
  auto input = torch::range(1, N * N, 1);
  input[0] = 1; // a more stable matrix
  input = input.view({N, N});
  inputs.emplace_back(input);
  testLiteModuleCompareResultTensors(m, inputs);
}
} // namespace

#if !defined FB_XPLAT_BUILD
TEST(LiteInterpreterDirectTest, DefaultArgsPinv) {
  // Test with different number of specified arguments.
  // Arguments not specified take default value.
  for (int num_args = 1; num_args <= 3; ++num_args) {
    testDefaultArgsPinv2(num_args);
  }

  //  bytecode with one specified argument:
  //  (6,
  //      ('__torch__.m.forward',
  //          (('instructions',
  //              (('STOREN', 1, 2),
  //                  ('DROPR', 1, 0),
  //                  ('MOVE', 2, 0),
  //                  ('OP', 0, 0),
  //                  ('RET', 0, 0))),
  //              ('operators', (('aten::linalg_pinv', '', 1),)),
  //              ('constants', (False, 1e-15)), # default constants are not
  //              used
  //              ('types', ()),
  //              ('register_size', 2)),
  //          (('arguments',
  //              ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
  //              None)),
  //                  (('name', 'input'), ('type', 'Tensor'), ('default_value',
  //                  None)))),
  //              ('returns',
  //                  ((('name', ''), ('type', 'Tensor'), ('default_value',
  //                  None)),)))))

  //  bytecode with 2 specified argument:
  //  (6,
  //      ('__torch__.m.forward',
  //          (('instructions',
  //              (('STOREN', 1, 2),
  //                  ('DROPR', 1, 0),
  //                  ('MOVE', 2, 0),
  //                  ('LOADC', 1, 0), # added LOADC for specified argument
  //                  ('OP', 0, 0),
  //                  ('RET', 0, 0))),
  //              ('operators', (('aten::linalg_pinv', '', 2),)),
  //              ('constants', (False, 1e-05)), # updated constant table
  //              ('types', ()),
  //              ('register_size', 2)),
  //          (('arguments',
  //              ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
  //              None)),
  //                  (('name', 'input'), ('type', 'Tensor'), ('default_value',
  //                  None)))),
  //              ('returns',
  //                  ((('name', ''), ('type', 'Tensor'), ('default_value',
  //                  None)),)))))

  //  bytecode with 3 specified arguments:
  //  (6,
  //      ('__torch__.m.forward',
  //          (('instructions',
  //              (('STOREN', 1, 2),
  //                  ('DROPR', 1, 0),
  //                  ('MOVE', 2, 0),
  //                  ('LOADC', 1, 0),
  //                  ('LOADC', 0, 0),
  //                  ('OP', 0, 0),
  //                  ('RET', 0, 0))),
  //              ('operators', (('aten::linalg_pinv', '', 3),)),
  //              ('constants', (True, 1e-05)),
  //              ('types', ()),
  //              ('register_size', 2)),
  //          (('arguments',
  //              ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
  //              None)),
  //                  (('name', 'input'), ('type', 'Tensor'), ('default_value',
  //                  None)))),
  //              ('returns',
  //                  ((('name', ''), ('type', 'Tensor'), ('default_value',
  //                  None)),)))))
}

TEST(LiteInterpreterDirectTest, DefaultArgsTensorinvSpecifyDefault) {
  // The second argument is specified, but the value is the same as the default
  // value. It's treated as "not specified" since the value can be fetched from
  // schema.
  Module m("m");
  m.define(R"(
    def forward(self, input):
      return torch.linalg_tensorinv(input, 2)
  )");
  torch::jit::MobileCode code(m.get_method("forward").graph(), "forward");
  auto arg_nums = code.op_to_num_specified_args();
  ASSERT_EQ(arg_nums.size(), 1);
  ASSERT_EQ(arg_nums["aten::linalg_tensorinv"], 1);
  std::vector<torch::jit::IValue> inputs;
  const int N = 4;
  auto input = torch::rand({N, N, N, N});
  inputs.emplace_back(input);
  testLiteModuleCompareResultTensors(m, inputs);
}

void testDefaultArgsPinvWithOutArg2(int num_args) {
  Module m("m");
  if (num_args == 1) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, out=input)
    )");
  } else if (num_args == 2) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, out=input)
    )");
  } else if (num_args == 3) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, True, out=input)
    )");
  }

  const int N = 28;
  auto input = torch::range(1, N * N, 1);
  input[0] = 10000; // a more stable matrix
  input = input.view({N, N});
  auto ref = m.run_method("forward", input);
  TORCH_CHECK(!input.equal(torch::range(1, N * N, 1)));
  TORCH_CHECK(input.equal(ref.toTensor()));
}

TEST(LiteInterpreterDirectTest, DefaultArgsPinvWithOutArg) {
  // Test with different number of specified arguments + out arg.
  // Arguments not specified take default value.
  for (int num_args = 1; num_args <= 3; ++num_args) {
    testDefaultArgsPinvWithOutArg2(num_args);
  }
}

TEST(LiteInterpreterDirectTest, DefaultArgsWithOutArg) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
      torch.add(x, h, out=x)
  )");

  std::vector<IValue> inputs;
  auto input_x = 2 * torch::ones({});
  auto input_h = torch::ones({});
  auto ref = m.run_method("forward", input_x, input_h);

  CompilationOptions options;
  mobile::Module bc = jitModuleToMobile(m, options);
  bc.run_method("forward", input_x, input_h);
  AT_ASSERT(input_x.equal(4 * torch::ones({})));
}

TEST(LiteInterpreterDirectTest, TestExceptionStackWithTwoLevelModuleHierarchy) {
  Module a("A");
  a.define(R"(
    def bar(self, x, y):
      return x + y
  )");
  Module b("B");
  b.register_module("A0", a);
  b.define(R"(
    def foo(self, x, y):
      return self.A0.bar(x, y) + 2
  )");
  Module c("C");
  c.register_module("B0", b);
  c.define(R"(
    def forward(self, x, y):
      return self.B0.foo(x, y) + 3
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  CompilationOptions options;
  auto lite_m = jitModuleToMobile(c, options);
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.B0(B)::foo.A0(A)::bar.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.B0.foo(x, y) + 3
             ~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in foo

    def foo(self, x, y):
      return self.A0.bar(x, y) + 2
             ~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in bar

    def bar(self, x, y):
      return x + y
             ~~~~~ <--- HERE
  )";
  ASSERT_THROWS_WITH_MESSAGE(lite_m.forward(inputs), error_pattern);
}
#endif // !defined(FB_XPLAT_BUILD)

namespace {
static auto reg =
    torch::class_<TorchBindLiteInterpreterDirectTestStruct>(
        "_TorchScriptTesting",
        "_LiteInterpreterDirectTest")
        .def(torch::init<>())
        .def("get", &TorchBindLiteInterpreterDirectTestStruct::get)
        .def_pickle(
            // __getattr__
            [](const c10::intrusive_ptr<
                TorchBindLiteInterpreterDirectTestStruct>&) -> int64_t {
              return 0;
            },
            // __setattr__
            [](int64_t) {
              return c10::make_intrusive<
                  TorchBindLiteInterpreterDirectTestStruct>();
            });

} // namespace

TEST(LiteInterpreterDirectTest, OperatorCacheDifferentiatesDefaultArgs) {
  // Create 3 methods:
  //
  // 1. forward() returns a tensor with dtype=torch.int64 (4)
  // 2. forward2() returns a tensor with dtype=torch.float32 (6)
  // 3. forward3() returns a tensor with dtype=torch.float32 but
  //    the dtype is inferred by the input tensor's dtype
  //
  // If caching works correctly, then the result from the full-jit
  // module and the lite module will be the same. Otherwise, it
  // will be different if we don't correctly ignore the cache
  // entry for an operator that has a different number of
  // arguments.
  Module m("m");
  m.define(R"(
    def forward(self):
      ret1 = torch.new_empty(torch.zeros(10), [10], dtype=4)
      return ret1.fill_(25)
  )");
  m.define(R"(
    def forward2(self):
      ret1 = torch.new_empty(torch.zeros(10), [10], dtype=6)
      return ret1.fill_(32.0)
  )");
  m.define(R"(
    def forward3(self):
      ret1 = torch.new_empty(torch.zeros(10), [10])
      return ret1.fill_(12.0)
  )");

  std::vector<torch::jit::IValue> inputs;
  testLiteModuleCompareResultTensors(m, inputs, "forward");
  testLiteModuleCompareResultTensors(m, inputs, "forward2");
  testLiteModuleCompareResultTensors(m, inputs, "forward3");
}

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 100 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`, `TEST`

**Classes/Structs**: `TorchBindLiteInterpreterDirectTestStruct`, `ClassNamespaceValue`, `TestModuleResolver`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `test/cpp/jit/test_utils.h`
- `gtest/gtest.h`
- `c10/core/TensorOptions.h`
- `torch/csrc/autograd/generated/variable_factories.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/frontend/resolver.h`
- `torch/csrc/jit/mobile/compatibility/backport.h`
- `torch/csrc/jit/mobile/compatibility/backport_manager.h`
- `torch/csrc/jit/mobile/compatibility/model_compatibility.h`
- `torch/csrc/jit/mobile/compatibility/runtime_compatibility.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/mobile/interpreter.h`
- `torch/csrc/jit/mobile/module.h`
- `torch/csrc/jit/mobile/parse_bytecode.h`
- `torch/csrc/jit/mobile/parse_operators.h`
- `torch/csrc/jit/serialization/export.h`
- `torch/csrc/jit/serialization/export_bytecode.h`
- `torch/csrc/jit/serialization/import.h`
- `torch/custom_class.h`
- `torch/torch.h`
- `unordered_set`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/jit/test_lite_interpreter_direct.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/jit`):

- [`test_code_template.cpp_docs.md`](./test_code_template.cpp_docs.md)
- [`test_memory_dag.cpp_docs.md`](./test_memory_dag.cpp_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_cleanup_passes.cpp_docs.md`](./test_cleanup_passes.cpp_docs.md)
- [`test_union.cpp_docs.md`](./test_union.cpp_docs.md)
- [`test_subgraph_rewriter.cpp_docs.md`](./test_subgraph_rewriter.cpp_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md)
- [`test_save_load.cpp_docs.md`](./test_save_load.cpp_docs.md)
- [`test_module_api.cpp_docs.md`](./test_module_api.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_lite_interpreter_direct.cpp_docs.md`
- **Keyword Index**: `test_lite_interpreter_direct.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/jit/test_lite_interpreter_direct.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/jit`):

- [`test_graph_iterator.cpp_kw.md_docs.md`](./test_graph_iterator.cpp_kw.md_docs.md)
- [`test_qualified_name.cpp_docs.md_docs.md`](./test_qualified_name.cpp_docs.md_docs.md)
- [`test_fuser.cpp_kw.md_docs.md`](./test_fuser.cpp_kw.md_docs.md)
- [`test_utils.cpp_docs.md_docs.md`](./test_utils.cpp_docs.md_docs.md)
- [`test_custom_class_registrations.h_docs.md_docs.md`](./test_custom_class_registrations.h_docs.md_docs.md)
- [`tests_setup.py_docs.md_docs.md`](./tests_setup.py_docs.md_docs.md)
- [`test_exception.cpp_kw.md_docs.md`](./test_exception.cpp_kw.md_docs.md)
- [`test_cs_debug_info_serialization.cpp_docs.md_docs.md`](./test_cs_debug_info_serialization.cpp_docs.md_docs.md)
- [`torch_python_test.cpp_docs.md_docs.md`](./torch_python_test.cpp_docs.md_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_lite_interpreter_direct.cpp_docs.md_docs.md`
- **Keyword Index**: `test_lite_interpreter_direct.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
