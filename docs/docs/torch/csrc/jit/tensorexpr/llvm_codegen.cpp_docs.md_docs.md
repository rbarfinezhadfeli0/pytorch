# Documentation: `docs/torch/csrc/jit/tensorexpr/llvm_codegen.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/llvm_codegen.cpp_docs.md`
- **Size**: 53,592 bytes (52.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/llvm_codegen.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/llvm_codegen.cpp`
- **Size**: 92,591 bytes (90.42 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef TORCH_ENABLE_LLVM

#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>

#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/llvm_jit.h>

// Note [llvm::SCEVPredicate non-virtual destructor]
// llvm::SCEVPredicate has virtual function but non-virtual destructor
// https://github.com/llvm/llvm-project/blob/c1a0a213378a458fbea1a5c77b315c7dce08fd05/llvm/include/llvm/Analysis/ScalarEvolution.h#L198
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <llvm/Analysis/TargetTransformInfo.h>
#pragma GCC diagnostic pop

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
// Fixes compilation warnings when gcc-11 is used
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wmismatched-new-delete")
#include <llvm/IR/IRBuilder.h>
C10_DIAGNOSTIC_POP()
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Pass.h>

// see Note [llvm::SCEVPredicate non-virtual destructor]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <llvm/Passes/PassBuilder.h>
#pragma GCC diagnostic pop

#if LLVM_VERSION_MAJOR >= 18
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/Scalar/DCE.h>
#include <llvm/Transforms/Vectorize/LoopVectorize.h>
#include <llvm/Transforms/Vectorize/SLPVectorizer.h>

#if LLVM_VERSION_MAJOR >= 10
#include <llvm/Support/CodeGen.h>
#else
#include <llvm/Target/TargetMachine.h>
#endif

#if LLVM_VERSION_MAJOR >= 11
#include <llvm/Support/TypeSize.h>
#endif

#if LLVM_VERSION_MAJOR < 15
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#endif

#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/Scalar.h>

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/half_support.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <torch/csrc/jit/jit_log.h>

#include <memory>

using namespace torch::jit::tensorexpr;

C10_DEFINE_bool(
    torch_jit_llvm_use_fast_intrinsics,
    false,
    "Use fast (but slightly less accurate) implementations of tanh and sigmoid")

namespace torch::jit::tensorexpr {

std::optional<std::string>& LLVMTargetTriple() {
  static std::optional<std::string> triple = std::nullopt;
  return triple;
}
std::optional<std::string>& LLVMTargetCPU() {
  static std::optional<std::string> cpu = std::nullopt;
  return cpu;
}
std::optional<std::string>& LLVMTargetAttrs() {
  static std::optional<std::string> attrs = std::nullopt;
  return attrs;
}
bool& LLVMAOTWorkflow() {
  static bool aot_workflow = false;
  return aot_workflow;
}

namespace {

#if LLVM_VERSION_MAJOR >= 15
// Address and type pair to assist in handling of opaque pointers.
struct TypedPointer {
  TypedPointer() = default;
  TypedPointer(llvm::Type* t, llvm::Value* a) : type(t), addr(a) {}
  llvm::Type* type = nullptr;
  llvm::Value* addr = nullptr;
};
#endif

llvm::CmpInst::Predicate llvm_comparison_predicate(
    CompareSelectOperation compare_op,
    const ScalarType& type) {
  switch (compare_op) {
    case CompareSelectOperation::kEQ:
      return llvm::ICmpInst::ICMP_EQ;
    case CompareSelectOperation::kNE:
      return llvm::ICmpInst::ICMP_NE;
    case CompareSelectOperation::kGT:
      return c10::isSignedType(type) ? llvm::ICmpInst::ICMP_SGT
                                     : llvm::ICmpInst::ICMP_UGT;
    case CompareSelectOperation::kGE:
      return c10::isSignedType(type) ? llvm::ICmpInst::ICMP_SGE
                                     : llvm::ICmpInst::ICMP_UGE;
    case CompareSelectOperation::kLT:
      return c10::isSignedType(type) ? llvm::ICmpInst::ICMP_SLT
                                     : llvm::ICmpInst::ICMP_ULT;
    case CompareSelectOperation::kLE:
      return c10::isSignedType(type) ? llvm::ICmpInst::ICMP_SLE
                                     : llvm::ICmpInst::ICMP_ULE;
    default:
      // TODO: change to a proper error report
      throw std::runtime_error("invalid operator type");
  }
}

llvm::CmpInst::Predicate llvm_fp_comparison_predicate(
    CompareSelectOperation compare_op) {
  switch (compare_op) {
    case CompareSelectOperation::kEQ:
      return llvm::FCmpInst::FCMP_OEQ;
    case CompareSelectOperation::kNE:
      return llvm::FCmpInst::FCMP_ONE;
    case CompareSelectOperation::kGT:
      return llvm::FCmpInst::FCMP_OGT;
    case CompareSelectOperation::kGE:
      return llvm::FCmpInst::FCMP_OGE;
    case CompareSelectOperation::kLT:
      return llvm::FCmpInst::FCMP_OLT;
    case CompareSelectOperation::kLE:
      return llvm::FCmpInst::FCMP_OLE;
    default:
      // TODO: change to a proper error report
      throw std::runtime_error("invalid operator type");
  }
}

#if LLVM_VERSION_MAJOR <= 9
int ElementCount(int lanes) {
  return lanes;
}
#else
llvm::ElementCount ElementCount(int lanes) {
#if LLVM_VERSION_MAJOR <= 11
  return llvm::ElementCount(static_cast<unsigned>(lanes), false);
#elif LLVM_VERSION_MAJOR >= 12
  return llvm::ElementCount::getFixed(lanes);
#else
#error Only LLVM versions 8 and above are supported.
#endif
}
#endif

#if LLVM_VERSION_MAJOR >= 9

using FunctionCallee = llvm::FunctionCallee;

#elif LLVM_VERSION_MAJOR == 8 && LLVM_VERSION_PATCH == 20181009

struct FunctionCallee {
  FunctionCallee() {}

  FunctionCallee(llvm::Constant* fn)
      : v_(fn), ft_(cast<llvm::Function>(v_)->getFunctionType()) {}

  llvm::FunctionType* getFunctionType() {
    return ft_;
  }

  llvm::Value* getCallee() {
    return v_;
  }

 private:
  llvm::Value* v_{nullptr};
  llvm::FunctionType* ft_{nullptr};
};

#else
#error Only LLVM versions 8 and above are supported.
#endif
} // namespace

class LLVMCodeGenCallee {
 public:
  LLVMCodeGenCallee(
      std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit,
      void* kernelAddress)
      : jit_(std::move(jit)), kernelAddress_(kernelAddress) {}

  llvm::orc::PytorchLLVMJIT* getJIT() {
    return jit_.get();
  }

  void* getKernelAddress() {
    return kernelAddress_;
  }

  void setKernelAddress(void* kernelAddress) {
    kernelAddress_ = kernelAddress;
  }

 private:
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit_;
  void* kernelAddress_;
};

class LLVMCodeGenImpl : public IRVisitor {
 private:
  std::unique_ptr<llvm::LLVMContext> context_;
  llvm::IRBuilder<> irb_;
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit_;
  std::unique_ptr<llvm::Module> module_;
  llvm::Function* fn_;
  llvm::BasicBlock* bb_;
  llvm::Value* value_{nullptr};
  llvm::JITTargetAddress kernelAddress_;
  std::string kernel_func_name_;

#define LLVM_TYPE_DECLARE(_1, Name) llvm::Type* Name##Ty_;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, LLVM_TYPE_DECLARE)
#undef LLVM_TYPE_DECLARE

#if LLVM_VERSION_MAJOR >= 15
  llvm::Type* OpqPtrTy_;
#else
  llvm::Type* Int8PtrTy_;
#endif
  llvm::Type* VoidTy_;
  std::unordered_map<VarPtr, int> varToArg_;
  std::unordered_map<VarPtr, llvm::Value*> varToVal_;
  std::unordered_set<BufPtr> bufsExtAlloc_;
  std::unordered_map<VarPtr, llvm::Value*> bufsExtToFreeVal_;
  std::unordered_multimap<BufPtr, BufPtr> bufsExtAllocReuse_;
  std::unordered_map<BlockPtr, std::vector<VarPtr>> scopeToVar_;
  BlockPtr scope_;

  std::string llvmCode_;
  std::string asmCode_;

 private:
  llvm::LLVMContext& getContext();
  llvm::Type* dtypeToLLVM(Dtype dtype);
  llvm::Type* dtypeToLLVMPtr(Dtype dtype);
  void emitWrapper(const std::vector<llvm::Type*>& params);
  void emitKernel(StmtPtr stmt, const std::vector<llvm::Type*>& params);
  llvm::Value* toVec(llvm::Value* v, int lanes);

  enum Arity {
    Unary,
    Binary,
  };

  using SimdCallee = std::tuple<llvm::FunctionType*, llvm::Value*, bool>;
  SimdCallee getSimdFunction(
      const std::string& name,
      llvm::Type* type,
      Arity arity,
      int lanes);

  llvm::Value* varToValue(VarPtr var);
  void replaceVarMapping(
      const std::vector<VarPtr>& vars,
      const std::vector<llvm::Value*>& vals);

#if LLVM_VERSION_MAJOR >= 15
  TypedPointer packFuncArgs(const std::vector<llvm::Value*>& func_args);
  std::vector<llvm::Value*> unpackFuncArgs(TypedPointer packed, int arg_count);
#else
  llvm::Value* packFuncArgs(const std::vector<llvm::Value*>& func_args);
  std::vector<llvm::Value*> unpackFuncArgs(llvm::Value* packed, int arg_count);
#endif

  void processParallelFor(ForPtr v);
  void handleBufReuse(BufPtr buf, BufPtr buf_to_reuse);

 public:
  LLVMCodeGenImpl(
      StmtPtr stmt,
      const std::vector<CodeGen::BufferArg>& args,
      at::Device device,
      Dtype dtype,
      std::string kernel_func_name,
      std::optional<std::string> triple,
      std::optional<std::string> cpu,
      std::optional<std::string> attrs);
  ~LLVMCodeGenImpl() override = default;

  llvm::JITTargetAddress getKernelAddress() const;
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> releaseJIT();

  void visit(const AddPtr& v) override;
  void visit(const SubPtr& v) override;
  void visit(const MulPtr& v) override;
  void visit(const DivPtr& v) override;
  void visit(const ModPtr& v) override;
  void visit(const MaxPtr& v) override;
  void visit(const MinPtr& v) override;
  void visit(const AndPtr& v) override;
  void visit(const OrPtr& v) override;
  void visit(const XorPtr& v) override;
  void visit(const LshiftPtr& v) override;
  void visit(const RshiftPtr& v) override;
  void visit(const CompareSelectPtr& v) override;

#define IMM_VISIT_DECLARE(_1, Name) void visit(const Name##ImmPtr& v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_VISIT_DECLARE)
#undef IMM_VISIT_DECLARE

  void visit(const CastPtr& v) override;
  void visit(const BitCastPtr& v) override;
  void visit(const VarPtr& v) override;
  void visit(const RampPtr& v) override;
  void visit(const LoadPtr& v) override;
  void visit(const ForPtr& v) override;
  void visit(const BlockPtr& v) override;
  void visit(const StorePtr& v) override;
  void visit(const BroadcastPtr& v) override;
  void visit(const IfThenElsePtr& v) override;
  void visit(const IntrinsicsPtr& v) override;
  void visit(const AllocatePtr& v) override;
  void visit(const FreePtr& v) override;
  void visit(const FreeExtPtr& v) override;
  void visit(const PlacementAllocatePtr& v) override;
  void visit(const LetPtr& v) override;
  void visit(const CondPtr& v) override;
  void visit(const ExternalCallPtr& v) override;
  void visit(const ExternalCallWithAllocPtr& v) override;

  void emitIsNan(IntrinsicsPtr v);

  llvm::Value* emitUnmaskedLoad(
      llvm::Type* ty,
      llvm::Value* addr,
      llvm::Value* idx);
  llvm::Value* emitMaskedLoad(
      llvm::Type* ty,
      llvm::Value* addr,
      llvm::Value* idx,
      llvm::Value* mask);
  void emitUnmaskedStore(
      llvm::Type* ty,
      llvm::Value* base,
      llvm::Value* idx,
      llvm::Value* val);
  void emitMaskedStore(
      llvm::Type* ty,
      llvm::Value* base,
      llvm::Value* idx,
      llvm::Value* mask,
      llvm::Value* val);

  void optimize(llvm::Module& M);
  std::string getLLVMCodeText() {
    return llvmCode_;
  }
  std::string getASMCodeText() {
    return asmCode_;
  }
};

} // namespace torch::jit::tensorexpr

LLVMCodeGen::~LLVMCodeGen() = default;

LLVMCodeGen::LLVMCodeGen(StmtPtr stmt)
    : LLVMCodeGen(stmt, std::vector<CodeGen::BufferArg>()) {}

LLVMCodeGen::LLVMCodeGen(
    StmtPtr stmt,
    const std::vector<BufferArg>& args,
    at::Device device,
    const std::string& kernel_func_name,
    Dtype dtype,
    std::optional<std::string> triple,
    std::optional<std::string> cpu,
    std::optional<std::string> attrs)
    : CodeGen(stmt, args, device, kernel_func_name) {
  impl_ = std::make_unique<LLVMCodeGenImpl>(
      this->stmt(),
      args,
      device,
      dtype,
      this->kernel_func_name(),
      triple,
      cpu,
      attrs);
  callee_ = std::make_unique<LLVMCodeGenCallee>(
      impl_->releaseJIT(), (void*)impl_->getKernelAddress());
}

void LLVMCodeGen::cleanup_memory() {
  impl_.reset(nullptr);
}

void LLVMCodeGen::call_raw(const std::vector<void*>& args) {
  value<float>(const_cast<void**>(args.data()));
}

void LLVMCodeGen::call_with_numel(void** args, int64_t /* numel */) {
  value<float>(const_cast<void**>(args));
}

void LLVMCodeGen::call(const std::vector<CallArg>& args) {
  auto& buf_args = buffer_args();
  if (args.size() != buf_args.size()) {
    throw malformed_input("wrong number of args in call");
  }

  constexpr unsigned nargs = 8;
  c10::SmallVector<void*, nargs> argv;
  argv.resize(buf_args.size());
  for (size_t i = 0, e = buf_args.size(); i < e; i++) {
    auto const& bufferArg = buf_args[i];
    auto const& callArg = args[i];
    argv[i] = argToPtr(bufferArg, callArg);
  }
  value<float>(argv.data());
}

at::Tensor LLVMCodeGen::empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  return at::native::empty_strided_cpu(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

void* LLVMCodeGen::getKernelAddress(LLVMCodeGenCallee* callee) {
  return (void*)callee->getKernelAddress();
}

std::string LLVMCodeGen::getCodeText(const std::string& attr /*=""*/) {
  TORCH_INTERNAL_ASSERT(
      impl_.get(),
      "LLVMCodeGen memory has been cleaned up. So, code text is not available at this point");
  if (attr == "asm") {
    return impl_->getASMCodeText();
  } else {
    return impl_->getLLVMCodeText();
  }
}

llvm::JITTargetAddress LLVMCodeGenImpl::getKernelAddress() const {
  return kernelAddress_;
}

std::unique_ptr<llvm::orc::PytorchLLVMJIT> LLVMCodeGenImpl::releaseJIT() {
  return std::move(jit_);
}

namespace {
// Global mutex to protect LLVM initialization.  TargetRegistry::lookupTarget
// in particular is not thread-safe.
static std::mutex llvmInitMutex;
} // namespace

LLVMCodeGenImpl::LLVMCodeGenImpl(
    StmtPtr stmt,
    const std::vector<CodeGen::BufferArg>& args,
    at::Device device,
    Dtype dtype,
    std::string kernel_func_name,
    std::optional<std::string> triple,
    std::optional<std::string> cpu,
    std::optional<std::string> attrs)
    : context_(std::make_unique<llvm::LLVMContext>()),
      irb_(getContext()),
      kernel_func_name_(std::move(kernel_func_name)),
      bufsExtAlloc_(ExternalAllocBufFinder::find(stmt)) {
  if (!triple) {
    triple = LLVMTargetTriple();
  }
  if (!cpu) {
    cpu = LLVMTargetCPU();
  }
  if (!attrs) {
    attrs = LLVMTargetAttrs();
  }
  // Manually map types to LLVM types.
  ByteTy_ = llvm::Type::getInt8Ty(getContext());
  CharTy_ = llvm::Type::getInt8Ty(getContext());
  ShortTy_ = llvm::Type::getInt16Ty(getContext());
  IntTy_ = llvm::Type::getInt32Ty(getContext());
  LongTy_ = llvm::Type::getInt64Ty(getContext());
  HalfTy_ = llvm::Type::getHalfTy(getContext());
  FloatTy_ = llvm::Type::getFloatTy(getContext());
  DoubleTy_ = llvm::Type::getDoubleTy(getContext());
  VoidTy_ = llvm::Type::getVoidTy(getContext());
  BoolTy_ = ByteTy_;
#if LLVM_VERSION_MAJOR >= 15
  OpqPtrTy_ = llvm::PointerType::getUnqual(getContext());
#else
  Int8PtrTy_ = llvm::Type::getInt8PtrTy(getContext());
#endif

  {
    std::lock_guard<std::mutex> g(llvmInitMutex);
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    jit_ = std::make_unique<llvm::orc::PytorchLLVMJIT>(triple, cpu, attrs);
  }

  module_ = std::make_unique<llvm::Module>("pytorch", getContext());
  module_->setDataLayout(jit_->getDataLayout());
  module_->setTargetTriple(
#if LLVM_VERSION_MAJOR >= 21
      llvm::Triple(jit_->getTargetMachine().getTargetTriple())
#else
      jit_->getTargetMachine().getTargetTriple().str()
#endif
  );

  // We support float16 ops by casting expr inputs to float32
  // and then casting the result back to float16

  GRAPH_DEBUG("Before HalfRewriter ", *stmt);
  HalfRewriter hsFix;
  stmt = stmt->accept_mutator(&hsFix);
  GRAPH_DEBUG("After HalfRewriter ", *stmt);

  // Emit prototype and bind argument Vars to parameter indices.
  llvm::Type* retTy = dtypeToLLVM(dtype);
  std::vector<llvm::Type*> params;
  for (const auto i : c10::irange(args.size())) {
    auto const& arg = args[i];
    if (arg.isVar()) {
      params.push_back(dtypeToLLVM(arg.dtype()));
    } else {
      params.push_back(dtypeToLLVMPtr(arg.dtype()));
    }
    varToArg_[arg.var()] = i;
  }
  llvm::FunctionType* fntype = llvm::FunctionType::get(retTy, params, false);
  fn_ = llvm::Function::Create(
      fntype, llvm::Function::PrivateLinkage, "pytorch", module_.get());
  fn_->addFnAttr(llvm::Attribute::AlwaysInline);
  for (const auto i : c10::irange(args.size())) {
    if (!args[i].isVar()) {
      fn_->addParamAttr(i, llvm::Attribute::NoAlias);
    }
  }

  emitWrapper(params);
  emitKernel(stmt, params);

  jit_->addModule(std::move(module_), std::move(context_));
  if (!LLVMAOTWorkflow()) {
    auto sym = jit_->findSymbol(kernel_func_name_);
    kernelAddress_ = assertSuccess(sym.getAddress());
  }
}

llvm::LLVMContext& LLVMCodeGenImpl::getContext() {
  return *context_;
}

llvm::Type* LLVMCodeGenImpl::dtypeToLLVM(Dtype dtype) {
  switch (dtype.scalar_type()) {
#define TYPE_CASE(_1, n) \
  case ScalarType::n:    \
    return n##Ty_;       \
    break;

    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::QInt8:
      return CharTy_;
      break;

    case ScalarType::QUInt8:
      return ByteTy_;
      break;

    case ScalarType::BFloat16:
      return ShortTy_;
      break;

    default:
      throw unsupported_dtype();
  }
  return nullptr;
}

llvm::Type* LLVMCodeGenImpl::dtypeToLLVMPtr(Dtype dtype) {
  return dtypeToLLVM(dtype)->getPointerTo();
}

void LLVMCodeGenImpl::emitWrapper(const std::vector<llvm::Type*>& params) {
#if LLVM_VERSION_MAJOR >= 15
  auto wrapper = llvm::Function::Create(
      llvm::FunctionType::get(IntTy_, {OpqPtrTy_}, false),
      llvm::Function::ExternalLinkage,
      kernel_func_name_,
      module_.get());
#else
  auto voidPtrTy = llvm::Type::getInt8PtrTy(getContext());
  auto voidPtrPtrTy = voidPtrTy->getPointerTo();
  auto wrapper = llvm::Function::Create(
      llvm::FunctionType::get(IntTy_, {voidPtrPtrTy}, false),
      llvm::Function::ExternalLinkage,
      kernel_func_name_,
      module_.get());
#endif

  {
    // Work around UBSAN crashes which reads 8 byte in front of every function.
    // Otherwise, if the function was placed at the beginning of a page, reading
    // 8B before the page could trigger a wild-addr-read ASAN failure if the
    // page before this function was not mapped.
    // - https://reviews.llvm.org/D148665
    // - https://github.com/llvm/llvm-project/issues/65253
    // Place the variable just before the function,
    // the optimizer might otherwise disable this workaround.
    // https://llvm.org/docs/LangRef.html#prefix-data
    wrapper->setPrefixData(llvm::Constant::getNullValue(
        llvm::ArrayType::get(llvm::Type::getInt8Ty(getContext()), 8)));
  }

  auto wrapBB = llvm::BasicBlock::Create(getContext(), "wrapBB", wrapper);
  irb_.SetInsertPoint(wrapBB);
  llvm::SmallVector<llvm::Value*, 6> wrappedArgs;
  for (const auto i : c10::irange(params.size())) {
#if LLVM_VERSION_MAJOR >= 15
    auto argp = irb_.CreateGEP(
        OpqPtrTy_,
        wrapper->arg_begin(),
        llvm::ConstantInt::getSigned(IntTy_, i));
    if (params[i]->isPointerTy()) {
      auto arg =
          irb_.CreatePointerCast(irb_.CreateLoad(OpqPtrTy_, argp), params[i]);
      wrappedArgs.push_back(arg);
    } else {
      auto p =
          irb_.CreatePointerCast(irb_.CreateLoad(OpqPtrTy_, argp), OpqPtrTy_);
      auto arg = irb_.CreateLoad(params[i], p);
      wrappedArgs.push_back(arg);
    }
#else
    auto argp = irb_.CreateGEP(
        voidPtrTy,
        wrapper->arg_begin(),
        llvm::ConstantInt::getSigned(IntTy_, i));
    if (params[i]->isPointerTy()) {
      auto arg = irb_.CreatePointerCast(
          irb_.CreateLoad(argp->getType()->getPointerElementType(), argp),
          params[i]);
      wrappedArgs.push_back(arg);
    } else {
      auto p = irb_.CreatePointerCast(
          irb_.CreateLoad(argp->getType()->getPointerElementType(), argp),
          params[i]->getPointerTo());
      auto arg = irb_.CreateLoad(p->getType()->getPointerElementType(), p);
      wrappedArgs.push_back(arg);
    }
#endif
  }
  auto cc = irb_.CreateCall(fn_, wrappedArgs);
  irb_.CreateRet(cc);
}

class LLVMIntrinsicsExpander : public GenericIntrinsicsExpander {
 private:
  ExprPtr mutate(const IntrinsicsPtr& v) override {
    if (v->op_type() == kTanh) {
      ScalarType stype = v->dtype().scalar_type();
      if (stype == ScalarType::Float) {
        return fast_tanh(ExprHandle(v->param(0)->accept_mutator(this))).node();
      }
    } else if (v->op_type() == kSigmoid) {
      ScalarType stype = v->dtype().scalar_type();
      if (stype == ScalarType::Float) {
        return fast_sigmoid(ExprHandle(v->param(0)->accept_mutator(this)))
            .node();
      }
    }
    // TODO: fast exp
    // TODO: fast erf
    // TODO: fast sigmoid
    return GenericIntrinsicsExpander::mutate(v);
  }
};

void LLVMCodeGenImpl::emitKernel(
    StmtPtr stmt,
    const std::vector<llvm::Type*>& params) {
  // Set insert point to the real function.
  bb_ = llvm::BasicBlock::Create(getContext(), "entry", fn_);
  irb_.SetInsertPoint(bb_);

  // Maybe expand some of the intrinsics.
  if (FLAGS_torch_jit_llvm_use_fast_intrinsics) {
    LLVMIntrinsicsExpander intrinsics_expander;
    stmt = stmt->accept_mutator(&intrinsics_expander);
  } else {
    GenericIntrinsicsExpander intrinsics_expander;
    stmt = stmt->accept_mutator(&intrinsics_expander);
  }

  // Compile the kernel.
  stmt->accept(this);

  // If the kernel is empty, set a default return value.
  if (value_ == nullptr) {
    value_ = llvm::ConstantInt::get(IntTy_, 0);
  }

  irb_.CreateRet(value_);

  // print graph debug info before optimization
  llvm::SmallVector<char, 0> asmBuffer;
  llvm::raw_svector_ostream asmStream(asmBuffer);
  if (GRAPH_DEBUG_ENABLED) {
    module_->print(asmStream, nullptr);
  }
  GRAPH_DEBUG(
      "\nLLVM module before optimizations\n\n", asmStream.str().str(), "\n");

  if (llvm::verifyFunction(*fn_, &llvm::outs())) {
    throw std::runtime_error("Function verification failed");
  }

  optimize(*module_);

  asmBuffer.clear();
  module_->print(asmStream, nullptr);
  llvmCode_ = asmStream.str().str();
  GRAPH_DEBUG(
      "\nLLVM module after optimizations\n\n", asmStream.str().str(), "\n");

  // print graph debug info after optimization
  asmBuffer.clear();
  llvm::legacy::PassManager PM;
  jit_->getTargetMachine().addPassesToEmitFile(
      PM,
      asmStream,
      nullptr,
#if LLVM_VERSION_MAJOR >= 18
      llvm::CodeGenFileType::AssemblyFile);
#elif LLVM_VERSION_MAJOR >= 10
      llvm::CodeGenFileType::CGFT_AssemblyFile);
#else
      llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
#endif
  PM.run(*module_);
  asmCode_ = asmStream.str().str();

  GRAPH_DEBUG("\nLLVM generated assembly code\n\n", asmCode_, "\n");
}

// TODO: The binary ops are copypaste.

void LLVMCodeGenImpl::visit(const AddPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFAdd(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateAdd(lhs, rhs);
  } else {
    throw malformed_input("llvm_codegen: bad type in Add", v);
  }
}

void LLVMCodeGenImpl::visit(const SubPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFSub(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateSub(lhs, rhs);
  } else {
    throw malformed_input("llvm_codegen: bad type in Sub", v);
  }
}

void LLVMCodeGenImpl::visit(const MulPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFMul(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateMul(lhs, rhs);
  } else {
    throw malformed_input("llvm_codegen: bad type in Mul", v);
  }
}

void LLVMCodeGenImpl::visit(const DivPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  // TODO: Handle arg promotion.
  if (lfp && rfp) {
    value_ = irb_.CreateFDiv(lhs, rhs);
  } else if (!lfp && !rfp) {
    value_ = irb_.CreateSDiv(lhs, rhs);
  } else {
    throw malformed_input("llvm_codegen: bad type in Div", v);
  }
}

void LLVMCodeGenImpl::visit(const AndPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateAnd(lhs, rhs);
  } else {
    throw malformed_input("llvm_codegen: bad type in And", v);
  }
}

void LLVMCodeGenImpl::visit(const OrPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateOr(lhs, rhs); // codespell:ignore
  } else {
    throw malformed_input("llvm_codegen: bad type in Or", v);
  }
}

void LLVMCodeGenImpl::visit(const XorPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateXor(lhs, rhs);
  } else {
    throw malformed_input("llvm_codegen: bad type in Xor", v);
  }
}

void LLVMCodeGenImpl::visit(const LshiftPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateShl(lhs, rhs);
  } else {
    throw malformed_input("llvm_codegen: bad type in Lshift", v);
  }
}

void LLVMCodeGenImpl::visit(const RshiftPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    if (v->lhs()->dtype().is_signed()) {
      value_ = irb_.CreateAShr(lhs, rhs);
    } else {
      value_ = irb_.CreateLShr(lhs, rhs);
    }
  } else {
    throw malformed_input("llvm_codegen: bad type in Rshift", v);
  }
}

void LLVMCodeGenImpl::visit(const ModPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  bool lfp = lhs->getType()->isFPOrFPVectorTy();
  v->rhs()->accept(this);
  auto rhs = this->value_;
  bool rfp = rhs->getType()->isFPOrFPVectorTy();

  if (!lfp && !rfp) {
    value_ = irb_.CreateSRem(lhs, rhs);
  } else {
    throw malformed_input("llvm_codegen: bad type in Mod", v);
  }
}

void LLVMCodeGenImpl::visit(const MaxPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  v->rhs()->accept(this);
  auto rhs = this->value_;

  if (v->dtype().is_integral()) {
    auto icmp = v->dtype().is_signed() ? irb_.CreateICmpSGT(lhs, rhs)
                                       : irb_.CreateICmpUGT(lhs, rhs);
    value_ = irb_.CreateSelect(icmp, lhs, rhs);
    return;
  }

  value_ = irb_.CreateSelect(
      irb_.CreateFCmp(
          llvm::FCmpInst::FCMP_UNO,
          lhs,
          llvm::ConstantFP::get(lhs->getType(), 0.0)),
      lhs,
      irb_.CreateSelect(
          irb_.CreateFCmp(llvm::FCmpInst::FCMP_OGT, lhs, rhs), lhs, rhs));
}

void LLVMCodeGenImpl::visit(const MinPtr& v) {
  v->lhs()->accept(this);
  auto lhs = this->value_;
  v->rhs()->accept(this);
  auto rhs = this->value_;
  if (v->dtype().is_integral()) {
    auto icmp = v->dtype().is_signed() ? irb_.CreateICmpSLT(lhs, rhs)
                                       : irb_.CreateICmpULT(lhs, rhs);
    value_ = irb_.CreateSelect(icmp, lhs, rhs);
    return;
  }

  value_ = irb_.CreateSelect(
      irb_.CreateFCmp(
          llvm::FCmpInst::FCMP_UNO,
          lhs,
          llvm::ConstantFP::get(lhs->getType(), 0.0)),
      lhs,
      irb_.CreateSelect(
          irb_.CreateFCmp(llvm::FCmpInst::FCMP_OLT, lhs, rhs), lhs, rhs));
}

void LLVMCodeGenImpl::visit(const CompareSelectPtr& v) {
  auto genUnbiased = [this, v]() -> llvm::Value* {
    v->lhs()->accept(this);
    auto lhs = this->value_;
    v->rhs()->accept(this);
    auto rhs = this->value_;
    v->ret_val1()->accept(this);
    auto retval1 = this->value_;
    v->ret_val2()->accept(this);
    auto retval2 = this->value_;

    auto type_used = v->lhs()->dtype().scalar_type();

    llvm::Value* cmp_;
    CompareSelectOperation cmp_op_ = v->compare_select_op();

    if (c10::isIntegralType(type_used, true)) {
      cmp_ = irb_.CreateICmp(
          llvm_comparison_predicate(cmp_op_, type_used), lhs, rhs);
    } else if (c10::isFloatingType(type_used)) {
      cmp_ = irb_.CreateFCmp(llvm_fp_comparison_predicate(cmp_op_), lhs, rhs);
    } else {
      throw std::runtime_error("invalid type for CompareSelect");
    }

    return irb_.CreateSelect(cmp_, retval1, retval2);
  };

  auto genBiased = [this, v]() -> llvm::Value* {
    v->lhs()->accept(this);
    auto lhs = this->value_;
    v->rhs()->accept(this);
    auto rhs = this->value_;

    auto cmp_type = v->lhs()->dtype().scalar_type();
    auto cmp_op = v->compare_select_op();
    llvm::Value* cmp;

    if (c10::isIntegralType(cmp_type, true)) {
      cmp = irb_.CreateICmp(
          llvm_comparison_predicate(cmp_op, cmp_type), lhs, rhs);
    } else if (c10::isFloatingType(cmp_type)) {
      cmp = irb_.CreateFCmp(llvm_fp_comparison_predicate(cmp_op), lhs, rhs);
    } else {
      throw std::runtime_error("invalid type for CompareSelect");
    }

    auto lanes = v->lhs()->dtype().lanes();
    if (lanes > 1) {
      auto maskType = llvm::Type::getIntNTy(getContext(), lanes);
      auto zero = llvm::ConstantInt::get(maskType, 0);
      auto mask = irb_.CreateBitOrPointerCast(cmp, maskType);
      cmp = irb_.CreateICmpNE(mask, zero);
    }

    auto then_block = llvm::BasicBlock::Create(getContext(), "then", fn_);
    auto else_block = llvm::BasicBlock::Create(getContext(), "else", fn_);
    auto end_block = llvm::BasicBlock::Create(getContext(), "block", fn_);
    constexpr int32_t total_weight = 100000;
    auto true_weight = v->bias() == kLikely ? total_weight : 0;
    auto false_weight = total_weight - true_weight;
    irb_.CreateCondBr(
        cmp,
        then_block,
        else_block,
        llvm::MDBuilder(getContext())
            .createBranchWeights(true_weight, false_weight));

    irb_.SetInsertPoint(then_block);
    v->ret_val1()->accept(this);
    llvm::Value* then_val = value_;
    then_block = irb_.GetInsertBlock();
    irb_.CreateBr(end_block);

    irb_.SetInsertPoint(else_block);
    v->ret_val2()->accept(this);
    llvm::Value* else_val = value_;
    else_block = irb_.GetInsertBlock();
    irb_.CreateBr(end_block);

    irb_.SetInsertPoint(end_block);
    llvm::PHINode* phi = irb_.CreatePHI(then_val->getType(), 2);
    phi->addIncoming(then_val, then_block);
    phi->addIncoming(else_val, else_block);
    return phi;
  };

  value_ = v->bias() == kUnbiased ? genUnbiased() : genBiased();
}

template <typename T>
std::enable_if_t<std::is_integral_v<T>, llvm::Value*> getFromType(
    llvm::Type* type,
    T value) {
  return llvm::ConstantInt::get(type, value, std::is_signed_v<T>);
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, llvm::Value*> getFromType(
    llvm::Type* type,
    T value) {
  return llvm::ConstantFP::get(type, value);
}

#define IMM_VISIT_DECLARE(Type, Name)                  \
  void LLVMCodeGenImpl::visit(const Name##ImmPtr& v) { \
    value_ = getFromType<Type>(Name##Ty_, v->value()); \
  }
AT_FORALL_SCALAR_TYPES(IMM_VISIT_DECLARE)
#undef IMM_VISIT_DECLARE

void LLVMCodeGenImpl::visit(const HalfImmPtr& v) {
  value_ = llvm::ConstantFP::get(HalfTy_, v->value());
}

void LLVMCodeGenImpl::visit(const BFloat16ImmPtr& v) {
  value_ = llvm::ConstantInt::get(ShortTy_, v->value().x);
}

void LLVMCodeGenImpl::visit(const BoolImmPtr& v) {
  value_ = llvm::ConstantInt::get(BoolTy_, v->value());
}

static llvm::Type* llvmTypeToVec(llvm::Type* type, int lanes) {
  if (lanes > 1) {
    return llvm::VectorType::get(type, ElementCount(lanes));
  } else {
    return type;
  }
}

void LLVMCodeGenImpl::visit(const CastPtr& v) {
  v->src_value()->accept(this);

  auto dst_type = v->dtype().scalar_type();
  auto src_type = v->src_value()->dtype().scalar_type();
  bool is_to_bf16 = (dst_type == c10::kBFloat16);
  bool is_to_float = (dst_type == c10::kFloat);
  bool is_from_bf16 = (src_type == c10::kBFloat16);
  bool is_from_float = (src_type == c10::kFloat);

  bool cast_from_bf16_to_fp32 = is_from_bf16 && is_to_float;
  bool cast_from_fp32_to_bf16 = is_from_float && is_to_bf16;
  bool non_bf16_cast = (!is_to_bf16) && (!is_from_bf16);
  bool valid_bf16_cast = cast_from_bf16_to_fp32 || cast_from_fp32_to_bf16;
  TORCH_CHECK(
      valid_bf16_cast || non_bf16_cast,
      "Cast is not implemented for the conversion between ",
      src_type,
      " and ",
      dst_type,
      ".");

  llvm::Type* dstType =
      llvmTypeToVec(dtypeToLLVM(v->dtype()), v->dtype().lanes());
  llvm::Type* srcType = dtypeToLLVM(v->src_value()->dtype());

  if (srcType == dstType) {
    // do nothing.
    return;
  }

  bool destUnsigned = v->dtype().scalar_type() == ScalarType::Byte ||
      v->dtype().scalar_type() == ScalarType::QUInt8 ||
      v->dtype().scalar_type() == ScalarType::Bool;
  bool srcUnsigned =
      v->src_value()->dtype().scalar_type() == ScalarType::Byte ||
      v->src_value()->dtype().scalar_type() == ScalarType::QUInt8 ||
      v->src_value()->dtype().scalar_type() == ScalarType::Bool;

  // Scalar casts
  if (is_from_bf16) {
    // Shift the BF16 value left by 16bits and then bit cast the shifted value
    // to FP32.
    //   FP32_VAL = BF16_VAL << 16
    auto lans = v->dtype().lanes();
    value_ = irb_.CreateZExt(value_, llvmTypeToVec(IntTy_, lans));
    auto vec_shl_val = toVec(llvm::ConstantInt::get(IntTy_, 16), lans);
    value_ = irb_.CreateShl(value_, vec_shl_val);
    value_ = irb_.CreateBitOrPointerCast(value_, llvmTypeToVec(FloatTy_, lans));
    return;
  }

  if (is_to_bf16) {
    // Convert the FP32 value by RNE(Rounding to Nearest Even). Algorithm is as
    // follows:
    //   STEP1: U32_VAL = BITCAST(F32_VAL)
    //   STEP2: U32_VAL_TMP = U32_VAL >> 16
    //   STEP3: U32_VAL_TMP = U32_VAL_TMP & 1
    //   STEP4: ROUNDING_BIAS = U32_VAL_TMP + UINT32(0x7FFF)
    //   STEP5: U32_VAL_TMP = U32_VAL + ROUNDING_BIAS
    //   STEP6: BF16_VAL = static_cast<UINT16>(U32_VAL_TMP >> 16)
    auto lans = v->src_value()->dtype().lanes();
    auto shift_len = llvm::ConstantInt::get(IntTy_, 16);
    auto one = llvm::ConstantInt::get(ShortTy_, 1);
    auto rounding_bias = llvm::ConstantInt::get(ShortTy_, 0x7FFF);
    auto bf16_nan = llvm::ConstantInt::get(ShortTy_, 0xFFFF);

    auto mask = irb_.CreateFCmpOEQ(value_, value_);
    // STEP1: U32_VAL = BITCAST(F32_VAL)
    auto fp32_i32_value =
        irb_.CreateBitOrPointerCast(value_, llvmTypeToVec(IntTy_, lans));
    // STEP2: U32_VAL_TMP = (U32_VAL >> 16)
    value_ = irb_.CreateLShr(fp32_i32_value, toVec(shift_len, lans));
    value_ = irb_.CreateTrunc(value_, llvmTypeToVec(ShortTy_, lans));
    // STEP3: U32_VAL_TMP = U32_VAL_TMP & 1
    value_ = irb_.CreateAnd(value_, toVec(one, lans));
    // STEP4: ROUNDING_BIAS = U32_VAL_TMP + UINT32(0x7FFF)
    value_ = irb_.CreateAdd(value_, toVec(rounding_bias, lans));
    value_ = irb_.CreateZExt(value_, llvmTypeToVec(IntTy_, lans));
    // STEP5: U32_VAL_TMP = U32_VAL + ROUNDING_BIAS
    value_ = irb_.CreateAdd(value_, fp32_i32_value);
    // STEP6: BF16_VAL = static_cast<UINT16>(U32_VAL_TMP >> 16)
    value_ = irb_.CreateLShr(value_, toVec(shift_len, lans));
    value_ = irb_.CreateTrunc(value_, llvmTypeToVec(ShortTy_, lans));
    value_ = irb_.CreateBitOrPointerCast(value_, llvmTypeToVec(ShortTy_, lans));
    // If the value is NaN, return BF16 NaN.
    value_ = irb_.CreateSelect(mask, value_, toVec(bf16_nan, lans));
    return;
  }

  if (srcType->isFPOrFPVectorTy()) {
    if (dstType->isFPOrFPVectorTy()) {
      // as with eager, convert from Double -> Half by Converting to Float then
      // Half. TODO: __truncdfhf2
      if (v->dtype().scalar_type() == ScalarType::Half &&
          v->src_value()->dtype().scalar_type() == ScalarType::Double) {
        value_ = irb_.CreateFPCast(
            value_, llvmTypeToVec(FloatTy_, v->dtype().lanes()));
      }
      value_ = irb_.CreateFPCast(value_, dstType);
    } else if (dstType->isIntOrIntVectorTy()) {
      // Strictly casting from Float -> i8 doesn't give correct results
      // set one bit true if the input float is not 0
      if (v->dtype().scalar_type() == ScalarType::Bool) {
        llvm::Value* zero =
            toVec(llvm::ConstantFP::get(srcType, 0.), v->dtype().lanes());
        value_ = irb_.CreateFCmp(llvm::FCmpInst::FCMP_UNE, value_, zero);
        value_ = irb_.CreateICmpEQ(
            value_, llvm::ConstantInt::get(value_->getType(), 1));
        value_ = irb_.CreateIntCast(value_, dstType, !destUnsigned);
        return;
      }

      if (destUnsigned) {
        value_ = irb_.CreateFPToUI(value_, dstType);
      } else {
        value_ = irb_.CreateFPToSI(value_, dstType);
      }
    } else {
      throw unimplemented_lowering(v);
    }
    return;
  }

  if (!srcType->isIntOrIntVectorTy()) {
    throw unimplemented_lowering(v);
  }
  if (dstType->isFPOrFPVectorTy()) {
    if (srcUnsigned) {
      value_ = irb_.CreateUIToFP(value_, dstType);
    } else {
      value_ = irb_.CreateSIToFP(value_, dstType);
    }
  } else if (dstType->isIntOrIntVectorTy()) {
    // Ensure bool true value is exactly one, since we convert to int
    // from bool by zero extending the int8
    if (v->dtype().scalar_type() == ScalarType::Bool) {
      llvm::Value* zero =
          toVec(llvm::ConstantInt::get(srcType, 0), v->dtype().lanes());
      value_ = irb_.CreateICmpNE(value_, zero);
    }
    value_ = irb_.CreateIntCast(value_, dstType, !destUnsigned);
  } else {
    throw unimplemented_lowering(v);
  }
}

void LLVMCodeGenImpl::visit(const BitCastPtr& v) {
  v->src_value()->accept(this);

  llvm::Type* dstType = dtypeToLLVM(v->dtype());
  if (v->dtype().lanes() > 1) {
    dstType = llvm::VectorType::get(dstType, ElementCount(v->dtype().lanes()));
  }
  llvm::Type* srcType = dtypeToLLVM(v->src_value()->dtype());

  if (srcType == dstType) {
    // do nothing.
    return;
  }

  TORCH_CHECK(llvm::CastInst::isBitCastable(
      srcType->getScalarType(), dstType->getScalarType()));
  value_ = irb_.CreateBitOrPointerCast(value_, dstType);
}

void LLVMCodeGenImpl::visit(const VarPtr& v) {
  value_ = varToValue(v);
}

llvm::Value* LLVMCodeGenImpl::varToValue(VarPtr v) {
  // It is possible for v to be in both varToVal_ and varToArgs.
  // In that case, varToVal_ takes precedence.
  if (varToVal_.count(v)) {
    return varToVal_.at(v);
  } else if (varToArg_.count(v)) {
    auto idx = varToArg_.at(v);
    auto arg = fn_->arg_begin() + idx;
    return arg;
  }
  return nullptr;
}

void LLVMCodeGenImpl::replaceVarMapping(
    const std::vector<VarPtr>& vars,
    const std::vector<llvm::Value*>& vals) {
  TORCH_CHECK(vars.size() == vals.size());
  for (const auto i : c10::irange(vars.size())) {
    VarPtr var = vars[i];
    llvm::Value* val = vals[i];
    if (val) {
      varToVal_[var] = val;
    } else {
      varToVal_.erase(var);
    }
  }
}

void LLVMCodeGenImpl::visit(const RampPtr& v) {
  v->base()->accept(this);
  auto base = this->value_;
  v->stride()->accept(this);
  auto stride = this->value_;
  int lanes = v->lanes();

  if (llvm::ConstantInt* const_stride =
          llvm::dyn_cast<llvm::ConstantInt>(stride)) {
    std::vector<llvm::Constant*> vals = {
        llvm::ConstantInt::get(base->getType(), 0)};
    for (int i = 1; i < lanes; ++i) {
      vals.push_back(llvm::ConstantExpr::getAdd(vals.back(), const_stride));
    }

    llvm::Value* offsets = llvm::ConstantVector::get(vals);
    llvm::Value* splat = irb_.CreateVectorSplat(lanes, base);
    value_ = irb_.CreateAdd(splat, offsets);
    return;
  }

  llvm::Type* vecType = nullptr;
  auto element_count = ElementCount(lanes);
  switch (v->dtype().scalar_type()) {
#define TYPE_CASE(_1, Name)                                    \
  case ScalarType::Name:                                       \
    vecType = llvm::VectorType::get(Name##Ty_, element_count); \
    break;
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::QInt8:
      vecType = llvm::VectorType::get(CharTy_, element_count);
      break;
    case ScalarType::QUInt8:
      vecType = llvm::VectorType::get(ByteTy_, element_count);
      break;
    case ScalarType::BFloat16:
      vecType = llvm::VectorType::get(ShortTy_, element_count);
      break;
    default:
      throw std::runtime_error("invalid dtype in Ramp");
  }

  value_ = llvm::UndefValue::get(vecType);
  for (int i = 0; i < lanes; ++i) {
    value_ = irb_.CreateInsertElement(value_, base, i);
    base = irb_.CreateAdd(base, stride);
  }
}
llvm::Value* LLVMCodeGenImpl::emitUnmaskedLoad(
    llvm::Type* ty,
    llvm::Value* base,
    llvm::Value* idx) {
#if LLVM_VERSION_MAJOR >= 15
  auto addr = irb_.CreateGEP(ty, base, idx);
  return irb_.CreateLoad(ty, addr);
#else
  auto addr = irb_.CreateGEP(
      base->getType()->getScalarType()->getPointerElementType(), base, idx);
  return irb_.CreateLoad(addr->getType()->getPointerElementType(), addr);
#endif
}

llvm::Value* LLVMCodeGenImpl::emitMaskedLoad(
    llvm::Type* ty,
    llvm::Value* base,
    llvm::Value* idx,
    llvm::Value* mask) {
  // Create block structure for the masked load.
  auto preheader = irb_.GetInsertBlock();
  auto condblock = llvm::BasicBlock::Create(getContext(), "cond", fn_);
  auto tailblock = llvm::BasicBlock::Create(getContext(), "tail", fn_);

  // Test the mask
  auto cond = irb_.CreateICmpEQ(mask, llvm::ConstantInt::get(IntTy_, 1));
  irb_.CreateCondBr(cond, condblock, tailblock);

  // Do the load
  irb_.SetInsertPoint(condblock);

#if LLVM_VERSION_MAJOR >= 15
  auto addr = irb_.CreateGEP(ty, base, idx);
  auto load = irb_.CreateLoad(ty, addr);
#else
  auto addr = irb_.CreateGEP(
      base->getType()->getScalarType()->getPointerElementType(), base, idx);
  auto load = irb_.CreateLoad(addr->getType()->getPointerElementType(), addr);
#endif

  irb_.CreateBr(tailblock);

  // Merge the masked and unmasked CFG edges
  irb_.SetInsertPoint(tailblock);
  auto phi = irb_.CreatePHI(load->getType(), 2);
  phi->addIncoming(llvm::UndefValue::get(load->getType()), preheader);
  phi->addIncoming(load, condblock);

  return phi;
}

void LLVMCodeGenImpl::visit(const LoadPtr& v) {
  if (v->dtype().lanes() == 1) {
    v->base_handle()->accept(this);
    auto base = this->value_;
    v->flat_index()->accept(this);
    auto idx = this->value_;
    value_ = emitUnmaskedLoad(dtypeToLLVM(v->dtype()), base, idx);
    return;
  }

  llvm::Type* loadType = nullptr;

  auto element_count = ElementCount(v->dtype().lanes());
  switch (v->dtype().scalar_type()) {
#define TYPE_CASE(_1, Name)                                     \
  case ScalarType::Name:                                        \
    loadType = llvm::VectorType::get(Name##Ty_, element_count); \
    break;
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::QInt8:
      loadType = llvm::VectorType::get(CharTy_, element_count);
      break;
    case ScalarType::QUInt8:
      loadType = llvm::VectorType::get(ByteTy_, element_count);
      break;
    case ScalarType::BFloat16:
      loadType = llvm::VectorType::get(ShortTy_, element_count);
      break;
    default:
      throw std::runtime_error("invalid dtype in Load");
  }

  // Handle the case where the load is contiguous and unmasked efficiently
  auto idx_ramp = to<Ramp>(v->flat_index());
  if (idx_ramp) {
    auto stride_imm = intValue(idx_ramp->stride());
    if (stride_imm && *stride_imm == 1) {
      v->base_handle()->accept(this);
      auto base = this->value_;
      idx_ramp->base()->accept(this);
      auto first_idx = this->value_;

#if LLVM_VERSION_MAJOR >= 15
      auto addr = irb_.CreateGEP(dtypeToLLVM(v->dtype()), base, first_idx);
#else
      auto addr = irb_.CreateGEP(
          base->getType()->getScalarType()->getPointerElementType(),
          base,
          first_idx);
#endif

      auto vaddr = irb_.CreateBitOrPointerCast(
          addr, llvm::PointerType::get(loadType, 0));
#if LLVM_VERSION_MAJOR >= 12
      value_ = irb_.CreateAlignedLoad(loadType, vaddr, llvm::MaybeAlign(4));
#else
      value_ = irb_.CreateAlignedLoad(loadType, vaddr, 4);
#endif
      return;
    }
  }

  // Fallback to a scalar implementation
  v->base_handle()->accept(this);
  auto base = this->value_;
  v->flat_index()->accept(this);
  auto idx = this->value_;

  llvm::Value* load = llvm::UndefValue::get(loadType);
  for (int i = 0; i < v->dtype().lanes(); ++i) {
    auto sub_idx = irb_.CreateExtractElement(idx, i);
    llvm::Value* sub_load = nullptr;
    sub_load = emitUnmaskedLoad(dtypeToLLVM(v->dtype()), base, sub_idx);
    load = irb_.CreateInsertElement(load, sub_load, i);
  }

  value_ = load;
}

#if LLVM_VERSION_MAJOR >= 15
// Pack the arguments into an aggregate struct for forwarding.
TypedPointer LLVMCodeGenImpl::packFuncArgs(
    const std::vector<llvm::Value*>& func_args) {
  if (func_args.empty()) {
    llvm::PointerType* VoidPtrType = llvm::PointerType::getUnqual(getContext());
    return TypedPointer(
        VoidPtrType, llvm::ConstantPointerNull::get(VoidPtrType));
  }
  std::vector<llvm::Type*> arg_types(func_args.size());
  for (const auto i : c10::irange(func_args.size())) {
    arg_types[i] = func_args[i]->getType();
  }
  llvm::StructType* packed_type = llvm::StructType::create(arg_types);
  llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
  llvm::Value* one = llvm::ConstantInt::get(IntTy_, 1);
  llvm::Value* packed = irb_.CreateAlloca(packed_type, one);
  for (const auto i : c10::irange(func_args.size())) {
    llvm::Value* dst_ptr = irb_.CreateInBoundsGEP(
        packed_type, packed, {zero, llvm::ConstantInt::get(IntTy_, i)});
    irb_.CreateStore(func_args[i], dst_ptr);
  }
  return TypedPointer(packed_type, packed);
}

// Unpack the aggregate struct into individual arguments.
std::vector<llvm::Value*> LLVMCodeGenImpl::unpackFuncArgs(
    TypedPointer packed,
    int arg_count) {
  // TODO: extract arg_count from packed.
  std::vector<llvm::Value*> func_args(arg_count);
  llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
  for (const auto i : c10::irange(arg_count)) {
    llvm::Type* feild_type = packed.type->getStructElementType(i);
    llvm::Value* feild_addr = irb_.CreateInBoundsGEP(
        packed.type, packed.addr, {zero, llvm::ConstantInt::get(IntTy_, i)});
    func_args[i] = irb_.CreateLoad(feild_type, feild_addr);
  }
  return func_args;
}
#else
// Pack the arguments into an aggregate struct for forwarding.
llvm::Value* LLVMCodeGenImpl::packFuncArgs(
    const std::vector<llvm::Value*>& func_args) {
  if (func_args.empty()) {
    llvm::PointerType* VoidPtrType = llvm::Type::getInt8PtrTy(getContext());
    llvm::Constant* NullPtr = llvm::ConstantPointerNull::get(VoidPtrType);
    return NullPtr;
  }
  std::vector<llvm::Type*> arg_types(func_args.size());
  for (const auto i : c10::irange(func_args.size())) {
    arg_types[i] = func_args[i]->getType();
  }
  llvm::StructType* packed_type = llvm::StructType::create(arg_types);
  llvm::Value* zero = llvm::ConstantInt::get(IntTy_, 0);
  llvm::Value* one = llvm::ConstantInt::get(IntTy_, 1);
  llvm::Value* packed = irb_.CreateAlloca(packed_type, one);
  for (const auto i : c10::irange(func_args.size())) {
    llvm::Value* dst_ptr = irb_.CreateInBoundsGEP(
        packed_type, packed, {zero, llvm::ConstantInt::get(IntTy_, i)});
    irb_.CreateStore(func_args[i], dst_ptr);
  }
  return packed;
}

// Unpack the aggregate struct into individual arguments.
std::vector<llvm::Value*> LLVMCodeGenImpl::unpackFuncAr
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/tensorexpr`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/tensorexpr`):

- [`loopnest.h_kw.md_docs.md`](./loopnest.h_kw.md_docs.md)
- [`expr.h_docs.md_docs.md`](./expr.h_docs.md_docs.md)
- [`block_codegen.h_kw.md_docs.md`](./block_codegen.h_kw.md_docs.md)
- [`ir_cloner.cpp_kw.md_docs.md`](./ir_cloner.cpp_kw.md_docs.md)
- [`types.cpp_docs.md_docs.md`](./types.cpp_docs.md_docs.md)
- [`tensorexpr_init.h_docs.md_docs.md`](./tensorexpr_init.h_docs.md_docs.md)
- [`lowerings.cpp_kw.md_docs.md`](./lowerings.cpp_kw.md_docs.md)
- [`graph_opt.h_kw.md_docs.md`](./graph_opt.h_kw.md_docs.md)
- [`eval.h_kw.md_docs.md`](./eval.h_kw.md_docs.md)
- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `llvm_codegen.cpp_docs.md_docs.md`
- **Keyword Index**: `llvm_codegen.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
