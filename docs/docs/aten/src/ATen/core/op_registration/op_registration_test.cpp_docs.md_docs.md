# Documentation: `docs/aten/src/ATen/core/op_registration/op_registration_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/op_registration/op_registration_test.cpp_docs.md`
- **Size**: 52,934 bytes (51.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/op_registration/op_registration_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/core/op_registration/op_registration_test.cpp`
- **Size**: 100,715 bytes (98.35 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This appears to be a **test file**.

## Original Source

```cpp
/**
 * This file contains some general registration test cases.
 * More detailed test cases containing different APIs for registering kernels
 * are found in other files in this directory.
 */

#include <gtest/gtest.h>

// This file intentionally tests some deprecated APIs
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <ATen/core/boxing/impl/test_helpers.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <functional>

#include <ATen/core/LegacyTypeDispatch.h>

#include <algorithm>

using c10::RegisterOperators;
using c10::OperatorKernel;
using c10::OperatorHandle;
using c10::Dispatcher;
using c10::IValue;
using c10::DispatchKey;

using torch::Library;
using torch::CppFunction;

using at::Tensor;

namespace {

struct DummyKernel final : OperatorKernel {
  void operator()(Tensor) {}
};

struct MockKernel final : OperatorKernel {
  MockKernel(bool* called): called_(called) {}

  void operator()(Tensor) {
    *called_ = true;
  }
private:
  bool* called_;
};

TEST(OperatorRegistrationTest, whenRegisteringWithSchemaBeforeKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().schema("_test::dummy(Tensor dummy) -> ()").catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringWithSchemaAfterKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called).schema("_test::dummy(Tensor dummy) -> ()"));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringWithNameBeforeKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().schema("_test::dummy").catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringWithNameAfterKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called).schema("_test::dummy"));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringWithoutSchema_thenFails) {
  expectThrows<c10::Error>([] {
    c10::RegisterOperators().op(c10::RegisterOperators::options().catchAllKernel<DummyKernel>());
  }, "In operator registration: Tried to register an operator without specifying a schema or operator name.");
}

TEST(OperatorRegistrationTest, whenCallingOpWithWrongDispatchKey_thenFails) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<DummyKernel>(c10::DispatchKey::CPU));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
  " backend.");
}

TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenCallingOp_thenCallsCatchallKernel) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

// TODO Rewrite (since this is now allowed) and reenable
// TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenRegisteringDispatchedKernel_thenFails) {
//   bool called = false;
//   auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));
//   expectThrows<c10::Error>([&] {
//     c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));
//   }, "for an operator which already has a catch-all kernel registered");
// }

// TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenRegisteringDispatchedKernelInSameOpCall_thenFails) {
//   bool called = false;
//   expectThrows<c10::Error>([&] {
//     auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
//       .catchAllKernel<MockKernel>(&called)
//       .kernel<MockKernel>(c10::DispatchKey::CPU, &called));
//   }, "for an operator which already has a catch-all kernel registered");
// }

TEST(OperatorRegistrationTest, givenOpWithDispatchedKernelOutOfScope_whenRegisteringCatchallKernelAndCallingOp_thenCallsCatchallKernel) {
  bool called = false;
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));
  }

  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

// TODO Rewrite (since this is now allowed) and reenable
// TEST(OperatorRegistrationTest, givenOpWithDispatchedKernel_whenRegisteringCatchallKernel_thenFails) {
//   bool called = false;
//   auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));
//   expectThrows<c10::Error>([&] {
//     c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));
//   }, "Tried to register a catch-all kernel for an operator which already has kernels for dispatch keys CPU. An operator can only have either a catch-all kernel or kernels with dispatch keys. The operator schema is _test::dummy");
// }
//
// TEST(OperatorRegistrationTest, givenOpWithDispatchedKernel_whenRegisteringCatchallKernelInSameOpCall_thenFails) {
//   bool called = false;
//   expectThrows<c10::Error>([&] {
//     auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
//       .kernel<MockKernel>(c10::DispatchKey::CPU, &called)
//       .catchAllKernel<MockKernel>(&called));
//   }, "Tried to register a catch-all kernel for an operator which already has kernels for dispatch keys CPU. An operator can only have either a catch-all kernel or kernels with dispatch keys. The operator schema is _test::dummy");
// }

TEST(OperatorRegistrationTest, givenOpWithCatchallKernelOutOfScope_whenRegisteringDispatchedKernelAndCallingOp_thenCallsCatchallKernel) {
  bool called = false;
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));
  }

  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringWithSchema_thenOnlyRegistersSchema) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CPU'"
  " backend.");
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringWithoutSchema_thenFails) {
  expectThrows<c10::Error>([&] {
    c10::RegisterOperators().op("_test::dummy");
  }, "Cannot infer operator schema in registration of operator _test::dummy because there is no kernel specified.");
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRunningOutOfScope_thenSchemaIsGone) {
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");
  }

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  EXPECT_FALSE(op.has_value());
}

TEST(OperatorRegistrationTest, givenOpWithoutKernelsWithoutTensorInputs_whenRegistering_thenRegisters) {
  // as long as we don't register non-catchall kernels, ops without tensor arguments are fine
  auto registrar = c10::RegisterOperators().op("_test::dummy() -> ()");

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered
}

TEST(OperatorRegistrationTest, givenMultipleKernelsWithSameDispatchKey_whenRegisteringInSameOpCall_thenFails) {
  expectThrows<c10::Error>([&] {
    auto registrar = c10::RegisterOperators()
        .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
            .kernel<DummyKernel>(c10::DispatchKey::CPU)
            .kernel<DummyKernel>(c10::DispatchKey::CPU));
  }, "In operator registration: Tried to register multiple kernels with same dispatch key CPU for operator schema _test::dummy");
}

TEST(OperatorRegistrationTest, givenMultipleCatchallKernels_whenRegisteringInSameOpCall_thenFails) {
  expectThrows<c10::Error>([&] {
    auto registrar = c10::RegisterOperators()
        .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
            .catchAllKernel<DummyKernel>()
            .catchAllKernel<DummyKernel>());
  }, "Tried to register multiple catch-all kernels for operator schema _test::dummy");
}

TEST(OperatorRegistrationTest, whenRegisteringCPUTensorType_thenCanOnlyCallUnboxedWithCPUDispatchKey) {
  bool called_kernel_cpu = false;
  auto registrar= c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<MockKernel>(c10::DispatchKey::CPU, &called_kernel_cpu));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  // Ensure that dispatcher doesn't take the dispatch key from the tensor but from the direct argument instead.
  called_kernel_cpu = false;
  callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, c10::DispatchKeySet(c10::DispatchKey::CPU), dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called_kernel_cpu);

  // Ensure that dispatch key from tensor is not used here.
  called_kernel_cpu = false;
  expectThrows<c10::Error>([&] {
    callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, c10::DispatchKeySet(c10::DispatchKey::CUDA), dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
  " backend.");
}

std::string expectedMessageForBackend(DispatchKey key) {
  std::string key_str(c10::toString(key));
  return "Could not run '_test::dummy' with arguments from the '" + key_str + "' backend";
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsInSameOpCallAndCalling_thenCallsCorrectKernel) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar0 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<MockKernel>(c10::DispatchKey::CPU, &called_kernel1)
    .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel2));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  called_kernel1 = called_kernel2 = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called_kernel1);
  EXPECT_FALSE(called_kernel2);

  called_kernel1 = called_kernel2 = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_FALSE(called_kernel1);
  EXPECT_TRUE(called_kernel2);

  // Test for out of tree lazy backends- ::Lazy key is now registered to TS backend in tree
  for (c10::DispatchKey key : {c10::DispatchKey::XLA}) {
    std::string expectMessage = expectedMessageForBackend(key);
    expectThrows<c10::Error>([&] {
      callOp(*op, dummyTensor(key));
    }, expectMessage.c_str());

    // also assert that the error message contains the available tensor type ids, but don't assert their order
    expectThrows<c10::Error>([&] {
      callOp(*op, dummyTensor(key));
    }, "CPU");
    expectThrows<c10::Error>([&] {
      callOp(*op, dummyTensor(key));
    }, "CUDA");
  }
}

bool called_stackbased_kernel = false;
void stackBasedKernel(const OperatorHandle&, c10::Stack* stack) {
  called_stackbased_kernel = true;
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsByNameAndNoneCanInferSchema_thenFails) {
  expectThrows<c10::Error>([&] {
    auto registrar1 = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
      .kernel<&stackBasedKernel>(c10::DispatchKey::CUDA)
      .kernel<&stackBasedKernel>(c10::DispatchKey::XLA)
      .kernel<&stackBasedKernel>(c10::DispatchKey::Lazy));
  }, "Cannot infer operator schema for this kind of kernel in registration of operator _test::dummy");
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsBySchemaAndNoneCanInferSchema_thenSucceeds) {
  bool called_kernel = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
    .kernel<&stackBasedKernel>(c10::DispatchKey::CUDA)
    .kernel<&stackBasedKernel>(c10::DispatchKey::XLA)
    .kernel<&stackBasedKernel>(c10::DispatchKey::Lazy));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    called_kernel = called_stackbased_kernel = false;
    callOp(*op, dummyTensor(key));
    EXPECT_TRUE(called_stackbased_kernel);
    EXPECT_FALSE(called_kernel);
  }
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsByNameAndOnlyOneCanInferSchema_thenSucceeds) {
  bool called_kernel = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
    .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
    .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel)
    .kernel<&stackBasedKernel>(c10::DispatchKey::XLA)
    .kernel<&stackBasedKernel>(c10::DispatchKey::Lazy));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_FALSE(called_stackbased_kernel);
  EXPECT_TRUE(called_kernel);

  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    called_kernel = called_stackbased_kernel = false;
    callOp(*op, dummyTensor(key));
    EXPECT_TRUE(called_stackbased_kernel);
    EXPECT_FALSE(called_kernel);
  }
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsBySchemaAndOnlyOneCanInferSchema_thenSucceeds) {
  bool called_kernel = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
    .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel)
    .kernel<&stackBasedKernel>(c10::DispatchKey::XLA)
    .kernel<&stackBasedKernel>(c10::DispatchKey::Lazy));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  called_kernel = called_stackbased_kernel = false;
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_FALSE(called_stackbased_kernel);
  EXPECT_TRUE(called_kernel);

  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    called_kernel = called_stackbased_kernel = false;
    callOp(*op, dummyTensor(key));
    EXPECT_TRUE(called_stackbased_kernel);
    EXPECT_FALSE(called_kernel);
  }
}

struct DummyKernelWithIntParam final : OperatorKernel {
  void operator()(Tensor, int64_t) {}
};

TEST(OperatorRegistrationTest, whenRegisteringMismatchingKernelsInSameOpCall_thenFails) {
  bool called_kernel = false;
  expectThrows<c10::Error>([&] {
    auto registrar1 = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel<DummyKernelWithIntParam>(c10::DispatchKey::CPU)
      .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel));
  }, "Mismatch in kernel C++ signatures");
}

void backend_fallback_kernel(const c10::OperatorHandle& op, c10::Stack* stack) {
  (*stack)[1] = (*stack)[1].toStringRef() + op.schema().name();
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernel_thenCanBeCalled) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  EXPECT_EQ("hello _test::dummy", stack[1].toStringRef());
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelForWrongBackend_thenCannotBeCalled) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CUDA, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  expectThrows<c10::Error>([&] {
    auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  }, "Could not run '_test::dummy' with arguments from the 'CPU' backend.");
}

bool called = false;

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenRegularKernelCanBeCalled) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", c10::RegisterOperators::options()
      .kernel(c10::DispatchKey::CUDA, [] (Tensor, std::string) {
        called = true;
      }));
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CUDA), "hello ");
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenFallbackKernelCanBeCalled) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", c10::RegisterOperators::options()
      .kernel(c10::DispatchKey::CUDA, [] (Tensor, std::string) {
        called = true;
      }));
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  EXPECT_FALSE(called);
  EXPECT_EQ("hello _test::dummy", stack[1].toStringRef());
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndRegularKernelForSameBackend_thenCallsRegularKernel) {
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", c10::RegisterOperators::options()
      .kernel(c10::DispatchKey::CPU, [] (Tensor, std::string) {
        called = true;
      }));
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  EXPECT_TRUE(called);
}

bool called_autograd = false;
bool called_nonautograd = false;

void nonautograd_kernel(Tensor a) {
  called_nonautograd = true;
}

void autograd_kernel(Tensor a) {
  called_autograd = true;
}

TEST(OperatorRegistrationTest, whenRegisteringAutogradKernel_thenCanCallAutogradKernel) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called_autograd = false;
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CPU'"
  " backend.");

  op->typed<void(Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_autograd);
}

TEST(OperatorRegistrationTest, whenRegisteringAutogradKernelWithRegularKernel_thenCanCallAutogradKernel) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(nonautograd_kernel), nonautograd_kernel>(DispatchKey::CPU)
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_FALSE(called_nonautograd);
  EXPECT_TRUE(called_autograd);
}

TEST(
    OperatorRegistrationTest,
    whenRegisteringAutogradKernelWithCatchAllKernel_thenCanCallCatchallKernel) {
  auto registrar = c10::RegisterOperators().op(
      "_test::dummy(Tensor dummy) -> ()",
      c10::RegisterOperators::options()
          .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>()
          .kernel<decltype(autograd_kernel), &autograd_kernel>(
              DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  // catchAll now maps to CompositeImplicitAutograd which has
  // higher precedence than Autograd
  called_nonautograd = called_autograd = false;
  op->typed<void(Tensor)>().call(
      dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);

  called_nonautograd = called_autograd = false;
  op->typed<void(Tensor)>().call(dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);
}

TEST(OperatorRegistrationTest, AutogradBackendOverridesAutogradKernel) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(nonautograd_kernel), &nonautograd_kernel>(DispatchKey::AutogradCPU)
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CPU'"
  " backend.");

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
  " backend.");

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CUDA, /*requires_grad=*/true));
  EXPECT_TRUE(called_autograd);
  EXPECT_FALSE(called_nonautograd);
}

void LazyBackendsAutogradOverridesAutogradKernel(DispatchKey key) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(nonautograd_kernel), &nonautograd_kernel>(c10::getAutogradKeyFromBackend(toBackendComponent(key)))
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  std::string expectedMessage = expectedMessageForBackend(key);
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(key));
  }, expectedMessage.c_str());

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(key, /*requires_grad=*/true));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);

  called_nonautograd = called_autograd = false;
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_autograd);
  EXPECT_FALSE(called_nonautograd);
}

// no longer test ::Lazy key here
// since it is now registered to TS backend in-tree and thus behaves differently,
// does not throw the expected 'could not run..' messages
TEST(OperatorRegistrationTest, AutogradXLAOverridesAutogradKernel) {
  LazyBackendsAutogradOverridesAutogradKernel(DispatchKey::XLA);
}

void whenRegisterWithLazyBackendsAndCatchAll_AutogradLazyBackendsIsNotFilled(DispatchKey key) {
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
      .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>());

    auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
    ASSERT_TRUE(op.has_value());

    called_nonautograd = called_autograd = false;
    op->typed<void (Tensor)>().call(dummyTensor(key, /*requires_grad=*/true));
    EXPECT_TRUE(called_nonautograd);
    EXPECT_FALSE(called_autograd);

    called_nonautograd = called_autograd = false;
    op->typed<void (Tensor)>().call(dummyTensor(key));
    EXPECT_FALSE(called_autograd);
    EXPECT_TRUE(called_nonautograd);
  }
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
      .kernel<decltype(autograd_kernel), &autograd_kernel>(key)
      .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>());

    auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
    ASSERT_TRUE(op.has_value());

    // When there's direct registration to XLA / Lazy backend, Autograd{XLA, Lazy} doesn't pick up catchAll
    // kernel in precompute but just keep fallthrough kernel from backend fallback.
    // Thus it falls through Autograd{XLA, Lazy} and reaches the kernel at XLA / Lazy key.
    called_nonautograd = called_autograd = false;
    op->typed<void (Tensor)>().call(dummyTensor(key, /*requires_grad=*/true));
    EXPECT_FALSE(called_nonautograd);
    EXPECT_TRUE(called_autograd);

    called_nonautograd = called_autograd = false;
    op->typed<void (Tensor)>().call(dummyTensor(key));
    EXPECT_TRUE(called_autograd);
    EXPECT_FALSE(called_nonautograd);
  }
}

TEST(OperatorRegistrationTest, whenRegisterWithXLAKernelAndCatchAll_AutogradXLAIsNotFilled) {
  whenRegisterWithLazyBackendsAndCatchAll_AutogradLazyBackendsIsNotFilled(DispatchKey::XLA);
}

TEST(OperatorRegistrationTest, whenRegisterWithLazyKernelAndCatchAll_AutogradLazyIsNotFilled) {
  whenRegisterWithLazyBackendsAndCatchAll_AutogradLazyBackendsIsNotFilled(DispatchKey::Lazy);
}

TEST(OperatorRegistrationTest, whenregisteringwithinvalidoverloadname) {
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy.default", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {}));
  }, "default is not a legal overload name for aten operators");
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy.__name__", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {}));
  }, "__name__ is not a legal overload name for aten operators");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenRegisteringWithMismatchingCppSignatures_thenFails) {
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {})
      .kernel(DispatchKey::CUDA, [] (int64_t) {}));
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenRegisteringCatchAllAndBackendWithMismatchingCppSignatures_thenFails) {
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {})
      .catchAllKernel([] (int64_t) {}));
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenRegisteringBackendAndCatchAllWithMismatchingCppSignatures_thenFails) {
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .catchAllKernel([] (const int64_t&) {})
      .kernel(DispatchKey::CPU, [] (int64_t) {}));
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenAccessingWithMismatchingCppSignatures_thenFails) {
  auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
    .kernel(DispatchKey::CPU, [] (int64_t) {}));
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int _0) -> ()");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails) {
  auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
    .catchAllKernel([] (int64_t) {}));
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int _0) -> ()");
}

TEST(OperatorRegistrationTest, givenTorchLibrary_whenRegisteringWithMismatchingCppSignatures_thenFails) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("dummy(int a) -> ()");
  m.impl("dummy", DispatchKey::CPU, [] (int64_t) {});
  expectThrows<c10::Error>([&] {
    m.impl("dummy", DispatchKey::CUDA, [] (const int64_t&) {});
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenTorchLibrary_whenAccessingWithMismatchingCppSignatures_thenFails) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("dummy(int a) -> ()");
  m.impl("dummy", DispatchKey::CPU, [] (int64_t) {});
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int a) -> ()");
}

TEST(OperatorRegistrationTest, givenTorchLibrary_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("dummy(int a) -> ()", [] (int64_t) {});
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int a) -> ()");
}

/**
 * This is used to check that a given type works correctly when passed as input
 * to or as output from a kernel.
 *
 * Call ArgTypeTestKernel<Input, Output>::test(input, inputExpectation, output, outputExpectation, schema)
 * to test that a kernel with `Input` as input type and `Output` as output types,
 * when called with `input` fulfills `inputExpectation` inside the kernel, then
 * returns `output` and the returned value fulfills `outputExpectation`.
 *
 * `inputExpectation` and `outputExpectation` should be lambdas that run
 * googletest expect macros (or use other ways to assert the expectation is met).
 *
 * Optionally, you can specify the argument list part of a function schema
 * (e.g. "(Tensor a) -> Tensor") as an additional argument to use when
 * registering the kernel. In this case, the operator registration logic will
 * check that the kernel function signature matches the one you specified.
 */
struct TestModernAPI final {};
struct TestLegacyAPI final {};
struct TestModernAndLegacyAPI final {};

template<class InputType, class OutputType = InputType>
struct ArgTypeTestKernel final : OperatorKernel {
  explicit ArgTypeTestKernel(InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output)
  : input_(std::move(input)), inputExpectation_(std::move(inputExpectation)), output_(std::move(output)) {}

  OutputType operator()(InputType input) const {
    inputExpectation_(std::move(input));
    return output_;
  }

  static void test(TestModernAndLegacyAPI, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
    test(TestModernAPI(), input, inputExpectation, output, outputExpectation, schema);
    test(TestLegacyAPI(), input, inputExpectation, output, outputExpectation, schema);
  }

  static void test(TestModernAPI, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
    return test_([&] {
      return c10::RegisterOperators().op("_test::my_op" + schema, c10::RegisterOperators::options().catchAllKernel<ArgTypeTestKernel>(input, inputExpectation, output));
    }, input, inputExpectation, output, outputExpectation, schema);
  }

  static void test(TestLegacyAPI, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
    return test_([&] {
      return c10::RegisterOperators().op("_test::my_op" + schema, [=] (InputType input) -> OutputType {
        inputExpectation(std::move(input));
        return output;
      });
    }, input, inputExpectation, output, outputExpectation, schema);
  }

private:
  static void test_(std::function<c10::RegisterOperators()> registration, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
    auto registry = registration();
    auto op = Dispatcher::singleton().findSchema({"_test::my_op", ""});
    ASSERT_TRUE(op.has_value()); // assert schema is registered
    auto actualOutput = callOp(*op, input);
    outputExpectation(actualOutput);
  }

  InputType input_;
  std::function<void(const InputType&)> inputExpectation_;
  OutputType output_;
  std::string schema_;
};

template<class InputType, class OutputType = InputType>
struct testArgTypes final {
  template<class APIType = TestModernAndLegacyAPI>
  static void test(InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const IValue&)> outputExpectation, const std::string& schema) {
    // Test with explicitly specified schema
    ArgTypeTestKernel<InputType, OutputType>::test(
      APIType(), input, inputExpectation, output, [&] (const c10::Stack& output) {
        EXPECT_EQ(1, output.size());
        outputExpectation(output[0]);
      }, schema
    );

    // Test with inferred schema
    ArgTypeTestKernel<InputType, OutputType>::test(
      APIType(), input, inputExpectation, output, [&] (const c10::Stack& output) {
        EXPECT_EQ(1, output.size());
        outputExpectation(output[0]);
      }, ""
    );

    // Test taking argument and returning nothing
    ArgTypeTestKernel<InputType, std::tuple<>>::test(
      APIType(), input, inputExpectation, {}, [] (const c10::Stack&) {}, ""
    );

    // Test taking argument and returning multiple outputs
    ArgTypeTestKernel<InputType, std::tuple<int64_t, OutputType>>::test(
      APIType(), input, inputExpectation, std::tuple<int64_t, OutputType>{3, output}, [&] (const c10::Stack& output) {
        EXPECT_EQ(2, output.size());
        EXPECT_EQ(3, output[0].toInt());
        outputExpectation(output[1]);
      }, ""
    );
  }
};

TEST(OperatorRegistrationTest, testAvailableArgTypes) {
  // TODO Test Scalar

  // primitive types
  testArgTypes<double>::test(
    1.5, [] (const double& v) {EXPECT_EQ(1.5, v);},
    2.5, [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());},
    "(float a) -> float");
  testArgTypes<int64_t>::test(
    1, [] (const int64_t& v) {EXPECT_EQ(1, v);},
    2, [] (const IValue& v) {EXPECT_EQ(2, v.toInt());},
    "(int a) -> int");
  testArgTypes<bool>::test(
    true, [] (const bool& v) {EXPECT_EQ(true, v);},
    false, [] (const IValue& v) {EXPECT_EQ(false, v.toBool());},
    "(bool a) -> bool");
  testArgTypes<bool>::test(
    false, [] (const bool& v) {EXPECT_EQ(false, v);},
    true, [] (const IValue& v) {EXPECT_EQ(true, v.toBool());},
    "(bool a) -> bool");
  testArgTypes<std::string>::test(
    "string1", [] (const std::string& v) {EXPECT_EQ("string1", v);},
    "string2", [] (const IValue& v) {EXPECT_EQ("string2", v.toStringRef());},
    "(str a) -> str");
  testArgTypes<Tensor>::test(
    dummyTensor(c10::DispatchKey::CPU), [] (const Tensor& v) {EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v));},
    dummyTensor(c10::DispatchKey::CUDA), [] (const IValue& v) {EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.toTensor()));},
    "(Tensor a) -> Tensor");


  // optional types (with has_value() == true)
  testArgTypes<std::optional<double>>::test(
    std::optional<double>(1.5), [] (const std::optional<double>& v) {EXPECT_EQ(1.5, v.value());},
    std::optional<double>(2.5), [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());},
    "(float? a) -> float?");
  testArgTypes<std::optional<int64_t>>::test(
    std::optional<int64_t>(1), [] (const std::optional<int64_t>& v) {EXPECT_EQ(1, v.value());},
    std::optional<int64_t>(2), [] (const IValue& v) {EXPECT_EQ(2, v.toInt());},
    "(int? a) -> int?");
  testArgTypes<std::optional<bool>>::test(
    std::optional<bool>(true), [] (const std::optional<bool>& v) {EXPECT_EQ(true, v.value());},
    std::optional<bool>(false), [] (const IValue& v) {EXPECT_EQ(false, v.toBool());},
    "(bool? a) -> bool?");
  testArgTypes<std::optional<bool>>::test(
    std::optional<bool>(false), [] (const std::optional<bool>& v) {EXPECT_EQ(false, v.value());},
    std::optional<bool>(true), [] (const IValue& v) {EXPECT_EQ(true, v.toBool());},
    "(bool? a) -> bool?");
  testArgTypes<std::optional<std::string>>::test(
    std::optional<std::string>("string1"), [] (const std::optional<std::string>& v) {EXPECT_EQ("string1", v.value());},
    std::optional<std::string>("string2"), [] (const IValue& v) {EXPECT_EQ("string2", v.toStringRef());},
    "(str? a) -> str?");
  testArgTypes<std::optional<Tensor>>::test(
    std::optional<Tensor>(dummyTensor(c10::DispatchKey::CPU)), [] (const std::optional<Tensor>& v) {EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.value()));},
    std::optional<Tensor>(dummyTensor(c10::DispatchKey::CUDA)), [] (const IValue& v) {EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.toTensor()));},
    "(Tensor? a) -> Tensor?");


  // optional types (with has_value() == false)
  testArgTypes<std::optional<double>>::test(
    std::optional<double>(std::nullopt), [] (const std::optional<double>& v) {EXPECT_FALSE(v.has_value());},
    std::optional<double>(std::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(float? a) -> float?");
  testArgTypes<std::optional<int64_t>>::test(
    std::optional<int64_t>(std::nullopt), [] (const std::optional<int64_t>& v) {EXPECT_FALSE(v.has_value());},
    std::optional<int64_t>(std::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(int? a) -> int?");
  testArgTypes<std::optional<bool>>::test(
    std::optional<bool>(std::nullopt), [] (const std::optional<bool>& v) {EXPECT_FALSE(v.has_value());},
    std::optional<bool>(std::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(bool? a) -> bool?");
  testArgTypes<std::optional<bool>>::test(
    std::optional<bool>(std::nullopt), [] (const std::optional<bool>& v) {EXPECT_FALSE(v.has_value());},
    std::optional<bool>(std::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(bool? a) -> bool?");
  testArgTypes<std::optional<std::string>>::test(
    std::optional<std::string>(std::nullopt), [] (const std::optional<std::string>& v) {EXPECT_FALSE(v.has_value());},
    std::optional<std::string>(std::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(str? a) -> str?");
  testArgTypes<std::optional<Tensor>>::test(
    std::optional<Tensor>(std::nullopt), [] (const std::optional<Tensor>& v) {EXPECT_FALSE(v.has_value());},
    std::optional<Tensor>(std::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(Tensor? a) -> Tensor?");


  // list types (with empty list)
  testArgTypes<c10::List<double>>::test(
    c10::List<double>(), [] (const c10::List<double>& v) {EXPECT_EQ(0, v.size());},
    c10::List<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<double>>().size());},
    "(float[] a) -> float[]");
  testArgTypes<c10::List<int64_t>>::test(
    c10::List<int64_t>(), [] (const c10::List<int64_t>& v) {EXPECT_EQ(0, v.size());},
    c10::List<int64_t>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
    "(int[] a) -> int[]");
  testArgTypes<c10::List<bool>>::test(
    c10::List<bool>(), [] (const c10::List<bool>& v) {EXPECT_EQ(0, v.size());},
    c10::List<bool>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<bool>>().size());},
    "(bool[] a) -> bool[]");
  testArgTypes<c10::List<std::string>>::test(
    c10::List<std::string>(), [] (const c10::List<std::string>& v) {EXPECT_EQ(0, v.size());},
    c10::List<std::string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    "(str[] a) -> str[]");


  // list types (with non-empty list)
  testArgTypes<c10::List<double>>::test(
    c10::List<double>({1.5, 2.5}), [] (const c10::List<double>& v) {expectListEquals({1.5, 2.5}, v);},
    c10::List<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<c10::List<double>>());},
    "(float[] a) -> float[]");
  testArgTypes<c10::List<int64_t>>::test(
    c10::List<int64_t>({1, 2}), [] (const c10::List<int64_t>& v) {expectListEquals({1, 2}, v);},
    c10::List<int64_t>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
    "(int[] a) -> int[]");
  testArgTypes<c10::List<bool>>::test(
    c10::List<bool>({true, false}), [] (const c10::List<bool>& v) {expectListEquals({true, false}, v);},
    c10::List<bool>({true, false}), [] (const IValue& v) {expectListEquals({true, false}, v.to<c10::List<bool>>());},
    "(bool[] a) -> bool[]");
  testArgTypes<c10::List<std::string>>::test(
    c10::List<std::string>({"first", "second"}), [] (const c10::List<std::string>& v) {expectListEquals({"first", "second"}, v);},
    c10::List<std::string>({"first", "second"}), [] (const IValue& v) {
      EXPECT_EQ(2, v.toListRef().size());
      EXPECT_EQ("first", v.toListRef()[0].toStringRef());
      EXPECT_EQ("second", v.toListRef()[1].toStringRef());
    },
    "(str[] a) -> str[]");
  testArgTypes<c10::List<Tensor>>::test(
    c10::List<Tensor>({dummyTensor(c10::DispatchKey::CPU), dummyTensor(c10::DispatchKey::CUDA)}), [] (const c10::List<Tensor>& v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.get(0)));
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.get(1)));
    },
    c10::List<Tensor>({dummyTensor(c10::DispatchKey::CUDA), dummyTensor(c10::DispatchKey::CPU)}), [] (const IValue& v) {
      EXPECT_EQ(2, v.to<c10::List<at::Tensor>>().size());
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(0)));
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(1)));
    },
    "(Tensor[] a) -> Tensor[]");

  // ArrayRef list types (with empty list)
  testArgTypes<c10::ArrayRef<double>, c10::List<double>>::test(
    c10::ArrayRef<double>(), [] (c10::ArrayRef<double> v) {EXPECT_EQ(0, v.size());},
    c10::List<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<double>>().size());},
    "(float[] a) -> float[]");
  testArgTypes<c10::ArrayRef<int64_t>, c10::List<int64_t>>::test(
    c10::ArrayRef<int64_t>(), [] (c10::ArrayRef<int64_t> v) {EXPECT_EQ(0, v.size());},
    c10::List<int64_t>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
    "(int[] a) -> int[]");
  testArgTypes<c10::ArrayRef<std::string>, c10::List<std::string>>::test(
    c10::ArrayRef<std::string>(), [] (c10::ArrayRef<std::string> v) {EXPECT_EQ(0, v.size());},
    c10::List<std::string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    "(str[] a) -> str[]");


  // list types (with non-empty list)
  testArgTypes<c10::ArrayRef<double>, c10::List<double>>::test(
    c10::ArrayRef<double>({1.5, 2.5}), [] (c10::ArrayRef<double> v) {expectListEquals({1.5, 2.5}, v);},
    c10::List<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<c10::List<double>>());},
    "(float[] a) -> float[]");
  testArgTypes<c10::ArrayRef<int64_t>, c10::List<int64_t>>::test(
    c10::ArrayRef<int64_t>({1, 2}), [] (c10::ArrayRef<int64_t> v) {expectListEquals({1, 2}, v);},
    c10::List<int64_t>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
    "(int[] a) -> int[]");
  testArgTypes<c10::ArrayRef<std::string>, c10::List<std::string>>::test(
    c10::ArrayRef<std::string>({"first", "second"}), [] (c10::ArrayRef<std::string> v) {expectListEquals({"first", "second"}, v);},
    c10::List<std::string>({"first", "second"}), [] (const IValue& v) {
      EXPECT_EQ(2, v.toListRef().size());
      EXPECT_EQ("first", v.toListRef()[0].toStringRef());
      EXPECT_EQ("second", v.toListRef()[1].toStringRef());
    },
    "(str[] a) -> str[]");
  testArgTypes<c10::ArrayRef<Tensor>, c10::List<Tensor>>::test(
    c10::ArrayRef<Tensor>({dummyTensor(c10::DispatchKey::CPUTensorId), dummyTensor(c10::DispatchKey::CUDATensorId)}), [] (c10::ArrayRef<Tensor> v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(c10::DispatchKey::CPUTensorId, extractDispatchKey(v[0]));
      EXPECT_EQ(c10::DispatchKey::CUDATensorId, extractDispatchKey(v[1]));
    },
    c10::List<Tensor>({dummyTensor(c10::DispatchKey::CUDATensorId), dummyTensor(c10::DispatchKey::CPUTensorId)}), [] (const IValue& v) {
      EXPECT_EQ(2, v.to<c10::List<at::Tensor>>().size());
      EXPECT_EQ(c10::DispatchKey::CUDATensorId, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(0)));
      EXPECT_EQ(c10::DispatchKey::CPUTensorId, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(1)));
    },
    "(Tensor[] a) -> Tensor[]");


  // std::array list types (with empty list)
  testArgTypes<std::array<double, 0>>::test(
    std::array<double, 0>(), [] (std::array<double, 0> v) {},
    std::array<double, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<c10::List<double>>().size()));},
    "(float[0] a) -> float[0]");
  testArgTypes<std:
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/core/op_registration`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/core/op_registration`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/aten/src/ATen/core/op_registration/op_registration_test.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/core/op_registration`):

- [`op_registration.cpp_kw.md_docs.md`](./op_registration.cpp_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`op_registration.h_kw.md_docs.md`](./op_registration.h_kw.md_docs.md)
- [`adaption.h_kw.md_docs.md`](./adaption.h_kw.md_docs.md)
- [`infer_schema.h_kw.md_docs.md`](./infer_schema.h_kw.md_docs.md)
- [`op_allowlist_test.cpp_docs.md_docs.md`](./op_allowlist_test.cpp_docs.md_docs.md)
- [`op_registration.h_docs.md_docs.md`](./op_registration.h_docs.md_docs.md)
- [`infer_schema.cpp_docs.md_docs.md`](./infer_schema.cpp_docs.md_docs.md)
- [`op_registration_test.cpp_kw.md_docs.md`](./op_registration_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `op_registration_test.cpp_docs.md_docs.md`
- **Keyword Index**: `op_registration_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
