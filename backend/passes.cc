#include <fstream>
#include <filesystem>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/CommandFlags.h"
#define DEBUG_TYPE "backend-passes"

#include "passes.h"
#include "utils.h"
#include "cuda_utils.h"

namespace fs = std::filesystem;

using namespace mlir;

namespace mlir::toy {

#define GEN_PASS_DEF_LOWERMEMREFTOLLVMPASS
#define GEN_PASS_DEF_INJECTRUNTIMECTXPASS
#define GEN_PASS_DEF_LOWERLLVMTOOBJECTPASS
#include "backend/passes.h.inc"

} // namespace mlir::toy

namespace {

struct LowerMemRefToLLVMFuncOpPattern : public mlir::OpConversionPattern<func::FuncOp> {  
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp oldFunc, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    auto typeConverter = this->getTypeConverter();
    //===============================================================================================
    // 1. Create newFunc
    //===============================================================================================
    auto oldFuncType = oldFunc.getFunctionType();

    // Convert input types
    SmallVector<Type> newInputTypes;
    if (failed(typeConverter->convertTypes(oldFuncType.getInputs(), newInputTypes))) {
      return failure();
    }

    // Convert result types
    SmallVector<Type> newResultTypes;
    if (failed(typeConverter->convertTypes(oldFuncType.getResults(), newResultTypes))) {
      return failure();
    }

    if (newResultTypes.size() > 1) {
      llvm::errs() << "The function " << oldFunc.getName() << " returns more than 1 value\n";
      return failure();
    }

    auto resType = newResultTypes.size() == 1 ? newResultTypes[0] : LLVM::LLVMVoidType::get(oldFunc.getContext());

    auto newFuncType = LLVM::LLVMFunctionType::get(resType, newInputTypes);
    auto newFunc = rewriter.create<LLVM::LLVMFuncOp>(oldFunc.getLoc(), oldFunc.getName(), newFuncType);

    // Copy attrs except the type
    for (auto attr : oldFunc->getAttrs()) {
      if (attr.getName() != newFunc.getFunctionTypeAttrName()) {
        newFunc->setAttr(attr.getName(), attr.getValue());
      }
    }

    newFunc.setVisibility(oldFunc.getVisibility());

    // Move the body.
    rewriter.inlineRegionBefore(oldFunc.getFunctionBody(), newFunc.getBody(), newFunc.end());
    
    Block& entryBlock = newFunc.front();
    auto sig = typeConverter->convertBlockSignature(&entryBlock);
    if (!sig.has_value()) {
      llvm::errs() << "Failed to convert entry block signature\n";
      return failure();
    }

    rewriter.applySignatureConversion(&entryBlock, sig.value(), typeConverter);

    //===============================================================================================
    // 3. Replace oldFunc with newFunc
    //===============================================================================================
    rewriter.replaceOp(oldFunc, newFunc);
    LLVM_DEBUG(llvm::dbgs() << "\nAfter FuncOpConverter:\n" << *newFunc->getParentOp() << "\n\n");

    return success();
  }
};

struct LowerMemRefToLLVMReturnOpPattern : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp oldReturnOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto newReturnOp = rewriter.create<LLVM::ReturnOp>(oldReturnOp.getLoc(), oldReturnOp.getOperands());
    rewriter.replaceOp(oldReturnOp, newReturnOp);

    return success();
  }
};


struct LowerMemRefToLLVMLoadOpPattern : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto elemType = loadOp.getMemRefType().getElementType();
    auto resType = LLVM::LLVMPointerType::get(getContext());
    auto operands = adaptor.getOperands();

    // Type resultType, Type elementType, Value basePtr, ValueRange indices
    auto loadPtr = rewriter.create<LLVM::GEPOp>(loadOp.getLoc(), resType, elemType, operands[0], operands[1]);
    auto newOp = rewriter.create<LLVM::LoadOp>(loadOp.getLoc(), elemType, loadPtr);
    rewriter.replaceOp(loadOp, newOp);

    return success();
  }
};

struct LowerMemRefToLLVMStoreOpPattern : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto elemType = storeOp.getMemRefType().getElementType();
    auto resType = LLVM::LLVMPointerType::get(getContext());
    auto operands = adaptor.getOperands();

    // Type resultType, Type elementType, Value basePtr, ValueRange indices
    auto storePtr = rewriter.create<LLVM::GEPOp>(storeOp.getLoc(), resType, elemType, operands[1], operands[2]);
    auto newOp = rewriter.create<LLVM::StoreOp>(storeOp.getLoc(), operands[0], storePtr);
    rewriter.replaceOp(storeOp, newOp);

    return success();
  }
};

struct LowerMemRefToLLVMTypeConverter : public TypeConverter {
  LowerMemRefToLLVMTypeConverter() {
    addConversion([](Type srcType) -> Type {
      if (auto memrefType = dyn_cast<MemRefType>(srcType)) {
        return LLVM::LLVMPointerType::get(srcType.getContext());
      }

      if (auto indexType = dyn_cast<IndexType>(srcType)) {
        return IntegerType::get(srcType.getContext(), 64);
      }

      return srcType;
    });

    // Register source materialization: memref<?xf32> -> tensor<?xf32>
    addSourceMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange convertedValues, mlir::Location loc) -> mlir::Value {
          assert(convertedValues.size() == 1 && "convertedValues must have size = 1");
          auto srcValue = builder.create<arith::IndexCastOp>(loc, resultType, convertedValues[0]);
          return srcValue;
        });

    // Register target materialization: tensor<?xf32> -> memref<?xf32>
    addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::TypeRange resultTypes,
           mlir::ValueRange srcValues, mlir::Location loc) -> SmallVector<Value> {

          SmallVector<Value> tgtValues;
          for (const auto& [t, v] : llvm::zip(resultTypes, srcValues)) {
            auto ret = builder.create<arith::IndexCastOp>(loc, t, v);
            tgtValues.push_back(ret);
          }

          return tgtValues;
        });
  }
};

class LowerMemRefToLLVMPass : public mlir::toy::impl::LowerMemRefToLLVMPassBase<LowerMemRefToLLVMPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Define type converter
    LowerMemRefToLLVMTypeConverter typeConverter;

    // Define conversion target
    ConversionTarget target(getContext());

    //===========================================================================
    // 1. Convert function signature
    //===========================================================================
    target.addLegalDialect<LLVM::LLVMDialect>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto isLegal = typeConverter.isSignatureLegal(op.getFunctionType());
      if (!isLegal) {
        LLVM_DEBUG(llvm::dbgs() << "\nFuncOp " << op.getOperation() << " is not legal ❌\n");
        return false;
      }

      LLVM_DEBUG(llvm::dbgs() << "\nFuncOp is legal ✅\n");
      return true;
    });

    // Populate conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<
      LowerMemRefToLLVMFuncOpPattern,
      LowerMemRefToLLVMLoadOpPattern,
      LowerMemRefToLLVMStoreOpPattern,
      LowerMemRefToLLVMReturnOpPattern
    >(typeConverter, &getContext());

    // Apply partial convertion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      LLVM_DEBUG(llvm::dbgs() << "\nRestored module:\n");
      LLVM_DEBUG(llvm::dbgs() << module << "\n");
    }
  }
};

static constexpr char kRuntimeCtx[] = "RuntimeCtx";
struct InjectRuntimeCtxLLVMFuncOpPattern : public mlir::OpConversionPattern<LLVM::LLVMFuncOp> {
  using OpConversionPattern<LLVM::LLVMFuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp oldFunc, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    auto typeConverter = this->getTypeConverter();
    //===============================================================================================
    // 1. Create newFunc
    //===============================================================================================
    auto oldFuncType = oldFunc.getFunctionType();
    auto oldInputTypes = oldFuncType.getParams();

    // Inject RuntimeCtx
    SmallVector<Type> newInputTypes;
    newInputTypes.push_back(LLVM::LLVMPointerType::get(oldFunc.getContext()));
    newInputTypes.append(oldInputTypes.begin(), oldInputTypes.end());

    auto resType = oldFuncType.getReturnType();

    auto newFuncType = LLVM::LLVMFunctionType::get(resType, newInputTypes);
    auto newFunc = rewriter.create<LLVM::LLVMFuncOp>(oldFunc.getLoc(), oldFunc.getName(), newFuncType);

    // Copy attrs except the type
    for (auto attr : oldFunc->getAttrs()) {
      if (attr.getName() != newFunc.getFunctionTypeAttrName()) {
        newFunc->setAttr(attr.getName(), attr.getValue());
      }
    }

    newFunc.setVisibility(oldFunc.getVisibility());

    // Move the body.
    rewriter.inlineRegionBefore(oldFunc.getFunctionBody(), newFunc.getBody(), newFunc.end());
    
    // Recreate entry block to conform with newFuncType 
    Block& entryBlock = newFunc.front();
    SmallVector<mlir::Location> argLocs;
    argLocs.push_back(newFunc.getLoc());
    for (auto& arg : entryBlock.getArguments()) {
      argLocs.push_back(arg.getLoc());
    }
    auto newEntryBlock = rewriter.createBlock(&newFunc.front(), newInputTypes, argLocs);
    auto newArgValues = newEntryBlock->getArguments();
    SmallVector<Value> argValues(newArgValues.begin() + 1, newArgValues.end());
    rewriter.inlineBlockBefore(&entryBlock, newEntryBlock, newEntryBlock->end(), argValues);

    //===============================================================================================
    // 3. Replace oldFunc with newFunc
    //===============================================================================================
    newFunc->setAttr(kRuntimeCtx, rewriter.getBoolAttr(true));
    rewriter.replaceOp(oldFunc, newFunc);
    LLVM_DEBUG(llvm::dbgs() << "\nAfter InjectRuntimeCtxLLVMFuncOpPattern:\n" << *newFunc->getParentOp() << "\n\n");

    return success();
  }
};

struct InjectRuntimeCtxLaunchFuncOpPattern : public OpConversionPattern<gpu::LaunchFuncOp> {
  using OpConversionPattern<gpu::LaunchFuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(funcOp && "The parent of gpu::LaunchOp must be LLVM::LLVMFuncOp");
    auto runtimeCtx = cast<Value>(funcOp.getArgument(0));

    // Extract stream
    rewriter.setInsertionPointToStart(&funcOp.front());
    auto elemType = LLVM::LLVMPointerType::get(getContext());
    auto resType = LLVM::LLVMPointerType::get(getContext());
    Value c0 = rewriter.create<LLVM::ConstantOp>(funcOp.getLoc(), rewriter.getI32Type(), 0);

    // Type resultType, Type elementType, Value basePtr, ValueRange indices
    auto loadPtr = rewriter.create<LLVM::GEPOp>(funcOp.getLoc(), resType, elemType, runtimeCtx, ValueRange{c0});
    auto stream = rewriter.create<LLVM::LoadOp>(funcOp.getLoc(), elemType, loadPtr);
    rewriter.startOpModification(op);
    op.getAsyncObjectMutable().assign(stream);
    op->setAttr(kRuntimeCtx, rewriter.getBoolAttr(true));
    rewriter.finalizeOpModification(op);

    return success();
  }
};

class InjectRuntimeCtxPass : public mlir::toy::impl::InjectRuntimeCtxPassBase<InjectRuntimeCtxPass> {
  void runOnOperation() override {
    auto module = getOperation();

    // Apply partial convertion
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp funcOp) {
      if (!funcOp->getAttr(kRuntimeCtx)) {
        return false;
      }

      return true;
    });

    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>([](gpu::LaunchFuncOp launchOp) {
      if (!launchOp->getAttr(kRuntimeCtx)) {
        return false;
      }

      return true;
    });

    // Populate conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<
      InjectRuntimeCtxLLVMFuncOpPattern,
      InjectRuntimeCtxLaunchFuncOpPattern
    >(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};


class HostModuleToObject {
 public:
  HostModuleToObject(ModuleOp& moduleOp,
                     StringRef triple,
                     StringRef chip,
                     StringRef features = {}, 
                     int optLevel = 3) 
    : moduleOp(moduleOp),
      triple(triple),
      chip(chip),
      features(features),
      optLevel(optLevel) {
  }

  LogicalResult emitObjectFile(const std::string& objectFilePath) {
    // Translate the module to LLVM IR.
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(moduleOp, llvmContext);
    if (!llvmModule) {
      moduleOp.emitError() << "Failed creating the llvm::Module.";
      return failure();
    }

    auto moduleName = moduleOp.getName();
    if (moduleName.has_value()) {
      llvmModule->setModuleIdentifier(moduleName.value().str());
    }

    if (::toy::utils::getBoolEnv("TOY_DUMP_LLVMIR")) {
      ::toy::utils::dumpLLVMIR(*llvmModule);
    }

    setDataLayoutAndTriple(*llvmModule);

    // Optimize the module.
    if (failed(optimizeModule(*llvmModule, optLevel))) {
      return failure();
    }

    std::string objectStr;
    llvm::raw_string_ostream stream(objectStr);
    auto& targetMachine = *getOrCreateTargetMachine().value();

    { // Drop pstream after this to prevent the ISA from being stuck buffering
      llvm::buffer_ostream pstream(stream);
      llvm::legacy::PassManager codegenPasses;

      if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                            llvm::CodeGenFileType::ObjectFile))
        return failure();

      if (!codegenPasses.run(*llvmModule)) {
        return failure();
      }
    }

    std::ofstream ofs(objectFilePath, std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
      llvm::errs() << "Failed to open objectFilePath " << objectFilePath << "\n";
      return failure();
    }

    ofs.write(objectStr.c_str(), objectStr.size());
    ofs.flush();
    ofs.close();

    return success();
  }

 private:
  std::optional<llvm::TargetMachine *>
  getOrCreateTargetMachine() {
    if (targetMachine)
      return targetMachine.get();
    // Load the target.
    std::string error;
    llvm::Triple parsedTriple(triple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(parsedTriple, error);
    if (!target) {
      moduleOp.emitError()
          << "Failed to lookup target for triple '" << triple << "' " << error;
      return std::nullopt;
    }

    // Create the target machine using the target.
    llvm::TargetOptions targetOptions =
        llvm::codegen::InitTargetOptionsFromCodeGenFlags(parsedTriple);
    targetMachine.reset(
        target->createTargetMachine(parsedTriple,
                                    chip,
                                    features,
                                    targetOptions,
                                    llvm::Reloc::Model::PIC_));
    if (!targetMachine)
      return std::nullopt;
    return targetMachine.get();
  }

  void setDataLayoutAndTriple(llvm::Module &module) {
    // Create the target machine.
    std::optional<llvm::TargetMachine *> targetMachine =
        getOrCreateTargetMachine();
    if (targetMachine) {
      // Set the data layout and target triple of the module.
      module.setDataLayout((*targetMachine)->createDataLayout());
      module.setTargetTriple((*targetMachine)->getTargetTriple());
    }
  }

  LogicalResult optimizeModule(llvm::Module &module, int optLevel) {
    if (optLevel < 0 || optLevel > 3)
      return moduleOp.emitError()
            << "Invalid optimization level: " << optLevel << ".";

    std::optional<llvm::TargetMachine *> targetMachine =
        getOrCreateTargetMachine();
    if (!targetMachine)
      return moduleOp.emitError()
            << "Target Machine unavailable for triple " << triple
            << ", can't optimize with LLVM\n";
    (*targetMachine)->setOptLevel(static_cast<llvm::CodeGenOptLevel>(optLevel));

    auto transformer =
        makeOptimizingTransformer(optLevel, /*sizeLevel=*/0, *targetMachine);
    auto error = transformer(&module);
    if (error) {
      InFlightDiagnostic mlirError = moduleOp.emitError();
      llvm::handleAllErrors(
          std::move(error), [&mlirError](const llvm::ErrorInfoBase &ei) {
            mlirError << "Could not optimize LLVM IR: " << ei.message() << "\n";
          });
      return mlirError;
    }
    return success();
  }

 private:
  ModuleOp& moduleOp;
  StringRef triple;
  StringRef chip;
  StringRef features;
  int optLevel;
  std::unique_ptr<llvm::TargetMachine> targetMachine;
};

class LowerLLVMToObjectPass : public mlir::toy::impl::LowerLLVMToObjectPassBase<LowerLLVMToObjectPass> {
 public:
  LowerLLVMToObjectPass() = default;

  LowerLLVMToObjectPass(const mlir::toy::LowerLLVMToObjectPassOptions& options)
    : LowerLLVMToObjectPassBase(options) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();
    std::string triple = llvm::sys::getDefaultTargetTriple();
    std::string chip = "generic";
    HostModuleToObject trantor(moduleOp, triple, chip);
    if (failed(trantor.emitObjectFile(objectFilePath))) {
      signalPassFailure();
    }
  }
};

} // namespace
