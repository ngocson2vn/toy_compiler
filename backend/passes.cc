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

#include "lld/Common/Driver.h"
LLD_HAS_DRIVER(elf);

#define DEBUG_TYPE "backend-passes"

#include "passes.h"
#include "utils.h"
#include "cuda_utils.h"

#include "frontend/toy_dialect.h"

namespace fs = std::filesystem;

using namespace mlir;

namespace mlir::toy {

#define GEN_PASS_DEF_LOWERMEMREFTOLLVMPASS
#define GEN_PASS_DEF_DUMPGPUMODULEPASS
#define GEN_PASS_DEF_INJECTRUNTIMECTXPASS
#define GEN_PASS_DEF_LOWERLLVMTOOBJECTPASS
#define GEN_PASS_DEF_LOWEROBJECTTOBINARYPASS
#include "backend/passes.h.inc"

} // namespace mlir::toy

namespace toy {
namespace compiler {
namespace backend {

namespace {

static constexpr char kMainFuncName[] = "main";
static bool hasMain = false;

struct LowerMemRefToLLVMFuncOpPattern : public mlir::OpConversionPattern<func::FuncOp> {  
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp oldFunc, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    auto typeConverter = this->getTypeConverter();
    //===============================================================================================
    // 1. Create newFunc
    //===============================================================================================
    auto oldFuncType = oldFunc.getFunctionType();

    // Convert input types
    SmallVector<mlir::Type> newInputTypes;
    if (failed(typeConverter->convertTypes(oldFuncType.getInputs(), newInputTypes))) {
      return failure();
    }

    // Convert result types
    SmallVector<mlir::Type> newResultTypes;
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

    // Replace func::ReturnOp
    auto returnOp = cast<func::ReturnOp>(newFunc.front().getTerminator());
    if (returnOp.getNumOperands() == 0) {
      rewriter.setInsertionPointAfter(returnOp);
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp, returnOp.getOperands());
    }

    //===============================================================================================
    // 2. Replace oldFunc with newFunc
    //===============================================================================================
    rewriter.replaceOp(oldFunc, newFunc);
    LLVM_DEBUG(llvm::dbgs() << "\nAfter FuncOpConverter:\n" << *newFunc->getParentOp() << "\n\n");

    return success();
  }
};

struct LowerMemRefToLLVMReturnOpPattern : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp oldReturnOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(oldReturnOp, adaptor.getOperands());

    return success();
  }
};


struct LowerMemRefToLLVMLoadOpPattern : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto memref = loadOp.getMemRef();
    auto tieOp = memref.getDefiningOp<mlir::toy::TieDimsOp>();
    if (!tieOp) {
      llvm::errs() << "ERROR Invalid argument at " << loc << "\n";
      return failure();
    }

    auto rdims = llvm::reverse(tieOp.getDims());
    //  dims = [d0, d1, d2] -> strides = [d1 * d2, d2, 1]
    // rdims = [d2, d1, d0] -> strides = [d1 * d2, d2, 1]

    SmallVector<Value> strides;
    auto c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);

    Value s = c1;
    strides.insert(strides.begin(), s);
    for (auto it = rdims.begin(); it != (rdims.end() - 1); it++) {
      Value rhs = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), *it);
      s = rewriter.create<LLVM::MulOp>(loc, s, rhs);
      strides.insert(strides.begin(), s);
    }

    auto elemType = loadOp.getMemRefType().getElementType();
    auto resType = LLVM::LLVMPointerType::get(getContext());

    Value linearIdx = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
    for (const auto& [i, s] : llvm::zip(adaptor.getIndices(), strides)) {
      Value tmpVal = rewriter.create<LLVM::MulOp>(loc, i, s);
      linearIdx = rewriter.create<LLVM::AddOp>(loc, tmpVal, linearIdx);
    }

    // Already converted to LLVM::LLVMPointerType
    auto basePtr = adaptor.getMemref();

    // Type resultType, Type elementType, Value basePtr, ValueRange indices
    auto loadPtr = rewriter.create<LLVM::GEPOp>(loc, resType, elemType, basePtr, ValueRange{linearIdx});
    auto newOp = rewriter.create<LLVM::LoadOp>(loc, elemType, loadPtr);
    rewriter.replaceOp(loadOp, newOp);

    return success();
  }
};

struct LowerMemRefToLLVMStoreOpPattern : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    auto memref = storeOp.getMemRef();
    auto tieOp = memref.getDefiningOp<mlir::toy::TieDimsOp>();
    if (!tieOp) {
      llvm::errs() << "ERROR Invalid argument at " << loc << "\n";
      return failure();
    }

    auto rdims = llvm::reverse(tieOp.getDims());
    //  dims = [d0, d1, d2] -> strides = [d1 * d2, d2, 1]
    // rdims = [d2, d1, d0] -> strides = [d1 * d2, d2, 1]

    SmallVector<Value> strides;
    auto c1 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);

    Value s = c1;
    strides.insert(strides.begin(), s);
    for (auto it = rdims.begin(); it != (rdims.end() - 1); it++) {
      Value rhs = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), *it);
      s = rewriter.create<LLVM::MulOp>(loc, s, rhs);
      strides.insert(strides.begin(), s);
    }

    auto elemType = storeOp.getMemRefType().getElementType();
    auto resType = LLVM::LLVMPointerType::get(getContext());

    Value linearIdx = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
    for (const auto& [i, s] : llvm::zip(adaptor.getIndices(), strides)) {
      Value tmpVal = rewriter.create<LLVM::MulOp>(loc, i, s);
      linearIdx = rewriter.create<LLVM::AddOp>(loc, tmpVal, linearIdx);
    }

    // Already converted to LLVM::LLVMPointerType
    auto basePtr = adaptor.getMemref();

    // Type resultType, Type elementType, Value basePtr, ValueRange indices
    auto storePtr = rewriter.create<LLVM::GEPOp>(loc, resType, elemType, basePtr, ValueRange{linearIdx});
    auto newOp = rewriter.create<LLVM::StoreOp>(loc, adaptor.getValue(), storePtr);
    rewriter.replaceOp(storeOp, newOp);

    return success();
  }
};

struct LowerToyToLLVMCallOpPattern : public OpConversionPattern<mlir::toy::CallOp> {
  using OpConversionPattern<mlir::toy::CallOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::toy::CallOp callOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type, 1> resultTypes;
    auto status = getTypeConverter()->convertTypes(callOp.getResults(), resultTypes);
    if (failed(status)) {
      return failure();
    }

    auto calleeAttr = rewriter.getStringAttr(callOp.getCallee());
    auto newOp = rewriter.create<LLVM::CallOp>(
                  callOp.getLoc(), resultTypes, calleeAttr, adaptor.getArgs());
    rewriter.replaceOp(callOp, newOp);

    return success();
  }
};

struct LowerToyToLLVMConstantOpPattern : public OpConversionPattern<mlir::toy::ConstantOp> {
  using OpConversionPattern<mlir::toy::ConstantOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::toy::ConstantOp constOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<LLVM::ConstantOp>(constOp.getLoc(), adaptor.getValue());
    rewriter.replaceOp(constOp, newOp);

    return success();
  }
};

struct LowerToyToLLVMAllocOpPattern : public OpConversionPattern<mlir::toy::AllocOp> {
  using OpConversionPattern<mlir::toy::AllocOp>::OpConversionPattern;
  const llvm::StringRef kAllocFuncName = "toy_alloc";
  LogicalResult matchAndRewrite(mlir::toy::AllocOp allocOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto context = getContext();

    SmallVector<Value> args;

    auto loc = allocOp.getLoc();
    auto dims = allocOp.getDims();
    assert(dims.size() > 0 && "dims is empty!");
    Value numElems = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), dims.front());
    for (auto it = dims.begin() + 1; it != dims.end(); it++) {
      Value rhs = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), *it);
      numElems = rewriter.create<LLVM::MulOp>(loc, numElems, rhs);
    }
    args.push_back(numElems);

    auto memrefType = cast<MemRefType>(allocOp.getType());
    auto encodedElemType = getEncodedType(memrefType.getElementType());
    auto encodedElemTypeVal = rewriter.create<LLVM::ConstantOp>(
                                loc,
                                rewriter.getI32Type(),
                                static_cast<int64_t>(encodedElemType)
                              );
    args.push_back(encodedElemTypeVal);

    auto moduleOp = allocOp->getParentOfType<mlir::ModuleOp>();
    LLVM::LLVMFuncOp allocFuncOp = dyn_cast_or_null<LLVM::LLVMFuncOp>(moduleOp.lookupSymbol(kAllocFuncName));
    if (!allocFuncOp) {
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto i32Type = IntegerType::get(context, 32);
      auto ptrType = LLVM::LLVMPointerType::get(context);
      auto allocFuncType = LLVM::LLVMFunctionType::get(ptrType, llvm::ArrayRef<mlir::Type>{i32Type, i32Type});
      allocFuncOp = rewriter.create<LLVM::LLVMFuncOp>(loc, kAllocFuncName, allocFuncType, LLVM::Linkage::External);
    }

    rewriter.setInsertionPoint(allocOp);
    auto newOp = rewriter.create<LLVM::CallOp>(loc, allocFuncOp, args);
    rewriter.replaceOp(allocOp, newOp);

    return success();
  }
};

struct LowerToyToLLVMRandomOpPattern : public OpConversionPattern<mlir::toy::RandomOp> {
  using OpConversionPattern<mlir::toy::RandomOp>::OpConversionPattern;
  const llvm::StringRef kRandomFuncName = "toy_random";
  LogicalResult matchAndRewrite(mlir::toy::RandomOp randomOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto arg = randomOp.getArg();
    auto allocOp = arg.getDefiningOp<mlir::toy::AllocOp>();
    if (!allocOp) {
      return failure();
    }

    auto dims = allocOp.getDims();
    assert(dims.size() > 0 && "dims is empty!");

    SmallVector<Value> args;
    args.push_back(adaptor.getArg());
    auto loc = randomOp.getLoc();
    auto context = getContext();

    Value numElems = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), dims.front());
    for (auto it = dims.begin() + 1; it != dims.end(); it++) {
      Value rhs = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), *it);
      numElems = rewriter.create<LLVM::MulOp>(loc, numElems, rhs);
    }
    args.push_back(numElems);

    auto memrefType = cast<MemRefType>(randomOp.getType());
    auto encodedElemType = getEncodedType(memrefType.getElementType());
    auto encodedElemTypeVal = rewriter.create<LLVM::ConstantOp>(
                                loc,
                                rewriter.getI32Type(),
                                static_cast<int64_t>(encodedElemType)
                              );
    args.push_back(encodedElemTypeVal);

    auto moduleOp = randomOp->getParentOfType<mlir::ModuleOp>();
    LLVM::LLVMFuncOp randomFuncOp = dyn_cast_or_null<LLVM::LLVMFuncOp>(moduleOp.lookupSymbol(kRandomFuncName));
    if (!randomFuncOp) {
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto i32Type = IntegerType::get(context, 32);
      auto ptrType = LLVM::LLVMPointerType::get(context);
      auto randomFuncType = LLVM::LLVMFunctionType::get(ptrType, llvm::ArrayRef<mlir::Type>{ptrType, i32Type, i32Type});
      randomFuncOp = rewriter.create<LLVM::LLVMFuncOp>(loc, kRandomFuncName, randomFuncType, LLVM::Linkage::External);
    }

    rewriter.setInsertionPoint(randomOp);
    auto newOp = rewriter.create<LLVM::CallOp>(loc, randomFuncOp, args);
    rewriter.replaceOp(randomOp, newOp);

    return success();
  }
};

struct LowerToyToLLVMPrintOpPattern : public OpConversionPattern<mlir::toy::PrintOp> {
  using OpConversionPattern<mlir::toy::PrintOp>::OpConversionPattern;
  const llvm::StringRef kPrintFuncName = "toy_print";
  LogicalResult matchAndRewrite(mlir::toy::PrintOp printOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto arg = printOp.getArg();
    auto allocOp = arg.getDefiningOp<mlir::toy::AllocOp>();
    if (!allocOp) {
      return failure();
    }

    auto dims = allocOp.getDims();
    assert(dims.size() > 0 && "dims is empty!");

    SmallVector<Value> args;
    args.push_back(adaptor.getArg());
    auto loc = printOp.getLoc();
    auto context = getContext();

    Value numElems = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), dims.front());
    for (auto it = dims.begin() + 1; it != dims.end(); it++) {
      Value rhs = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), *it);
      numElems = rewriter.create<LLVM::MulOp>(loc, numElems, rhs);
    }
    args.push_back(numElems);

    auto memrefType = cast<MemRefType>(printOp.getArg().getType());
    auto encodedElemType = getEncodedType(memrefType.getElementType());
    auto encodedElemTypeVal = rewriter.create<LLVM::ConstantOp>(
                                loc,
                                rewriter.getI32Type(),
                                static_cast<int64_t>(encodedElemType)
                              );
    args.push_back(encodedElemTypeVal);

    // argName
    int argNameLen = 1;
    auto argName = dyn_cast_or_null<StringAttr>(printOp->getAttr("argName"));
    std::string argNameStr = "";
    if (!!argName) {
      argNameLen = argName.size() + 1; // +1 for '\0' character
      argNameStr = argName.str();
    }
    argNameStr.push_back('\0');

    auto i8Type = IntegerType::get(context, 8);
    auto i32Type = IntegerType::get(context, 32);
    auto i64Type = IntegerType::get(context, 64);
    auto ptrType = LLVM::LLVMPointerType::get(context);
    auto argNameType = LLVM::LLVMArrayType::get(i8Type, argNameLen);
    auto argNameVal = rewriter.create<LLVM::ConstantOp>(loc, argNameType, rewriter.getStringAttr(argNameStr));
    auto c1 = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 1);
    auto argNamePtr  = rewriter.create<LLVM::AllocaOp>(loc, ptrType, argNameType, c1);
    rewriter.create<LLVM::StoreOp>(loc, argNameVal, argNamePtr);

    // auto c0 = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 0);
    // auto argNamePtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, i8Type, argNameVal, ValueRange{c0});
    args.push_back(argNamePtr);

    auto moduleOp = printOp->getParentOfType<mlir::ModuleOp>();
    LLVM::LLVMFuncOp printFuncOp = dyn_cast_or_null<LLVM::LLVMFuncOp>(moduleOp.lookupSymbol(kPrintFuncName));
    if (!printFuncOp) {
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto printFuncType = LLVM::LLVMFunctionType::get(i64Type, llvm::ArrayRef<mlir::Type>{ptrType, i32Type, i32Type, ptrType});
      printFuncOp = rewriter.create<LLVM::LLVMFuncOp>(loc, kPrintFuncName, printFuncType, LLVM::Linkage::External);
    }

    rewriter.setInsertionPoint(printOp);
    auto newOp = rewriter.create<LLVM::CallOp>(loc, printFuncOp, args);
    rewriter.replaceOp(printOp, newOp);

    return success();
  }
};

struct LowerMemRefToLLVMTypeConverter : public TypeConverter {
  LowerMemRefToLLVMTypeConverter() {
    addConversion([](mlir::Type srcType) -> mlir::Type {
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
          auto castOp = builder.create<UnrealizedConversionCastOp>(loc, resultType, convertedValues[0]);
          return castOp.getResult(0);
        });

    // Register target materialization: tensor<?xf32> -> memref<?xf32>
    addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::TypeRange resultTypes,
           mlir::ValueRange srcValues, mlir::Location loc) -> SmallVector<Value> {

          SmallVector<Value> tgtValues;
          for (const auto& [t, v] : llvm::zip(resultTypes, srcValues)) {
            auto castOp = builder.create<UnrealizedConversionCastOp>(loc, t, v);
            tgtValues.push_back(castOp.getResult(0));
          }

          return tgtValues;
        });
  }
};

struct LowerToyToLLVMTieDimsOpConverter : public OpConversionPattern<mlir::toy::TieDimsOp> {
  using OpConversionPattern<mlir::toy::TieDimsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::toy::TieDimsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto arg = adaptor.getArg();
    if (isa<BlockArgument>(arg)) {
      rewriter.replaceOp(op, arg);
    } else if (op.getResult().getNumUses() == 0) {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

class LowerMemRefToLLVMPass : public mlir::toy::impl::LowerMemRefToLLVMPassBase<LowerMemRefToLLVMPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    if (!!module.lookupSymbol(kMainFuncName)) {
      hasMain = true;
    }

    // Define type converter
    LowerMemRefToLLVMTypeConverter typeConverter;

    // Define conversion target
    ConversionTarget target(getContext());

    //===========================================================================
    // 1. Convert function signature
    //===========================================================================
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    // target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    //   auto isLegal = typeConverter.isSignatureLegal(op.getFunctionType());
    //   if (!isLegal) {
    //     LLVM_DEBUG(llvm::dbgs() << "\nFuncOp " << op.getOperation() << " is not legal ❌\n");
    //     return false;
    //   }

    //   LLVM_DEBUG(llvm::dbgs() << "\nFuncOp is legal ✅\n");
    //   return true;
    // });
  
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<mlir::toy::ToyDialect>();

    // Populate conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<
      LowerMemRefToLLVMFuncOpPattern,
      LowerMemRefToLLVMLoadOpPattern,
      LowerMemRefToLLVMStoreOpPattern,
      LowerMemRefToLLVMReturnOpPattern,
      LowerToyToLLVMConstantOpPattern,
      LowerToyToLLVMAllocOpPattern,
      LowerToyToLLVMRandomOpPattern,
      LowerToyToLLVMCallOpPattern,
      LowerToyToLLVMPrintOpPattern,
      LowerToyToLLVMTieDimsOpConverter
    >(typeConverter, &getContext());

    // Apply partial convertion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      LLVM_DEBUG(llvm::dbgs() << "\nRestored module:\n");
      LLVM_DEBUG(llvm::dbgs() << module << "\n");
    }
  }
};

class DumpGPUModulePass : public mlir::toy::impl::DumpGPUModulePassBase<DumpGPUModulePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    if (!::toy::utils::getBoolEnv("TOY_DUMP_LLVMIR")) {
      return;
    }

    auto module = getOperation();
    for (auto gpuMod : module.getOps<gpu::GPUModuleOp>()) {
      ::toy::utils::dumpGPUModule(gpuMod);
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
    SmallVector<mlir::Type> newInputTypes;
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

    if (!!module.lookupSymbol(kMainFuncName)) {
      return;
    }

    // Apply partial convertion
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp funcOp) {
      if (funcOp.getSymName() == kMainFuncName) {
        // main function is always legal
        return true;
      } else if (funcOp.getBlocks().size() == 0) {
        // external function is always legal
        return true;
      } else if (!!funcOp->getAttr(kRuntimeCtx)) {
        return true;
      }

      return false;
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

class LowerObjectToBinaryPass : public mlir::toy::impl::LowerObjectToBinaryPassBase<LowerObjectToBinaryPass> {
 public:
  LowerObjectToBinaryPass() = default;

  LowerObjectToBinaryPass(const mlir::toy::LowerObjectToBinaryPassOptions& options)
    : LowerObjectToBinaryPassBase(options) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto rpath = std::filesystem::path(
                      libRuntimePath.getValue()).parent_path().string();
    std::string outputFile = hasMain ? "./main" : "./libkernel.so";
    std::string libUnwindPath = rpath + "/libunwind.a";
    std::string libClangRtPath = rpath + "/libclang_rt.builtins.a";
    std::vector<const char *> args;
    if (hasMain) {
      args = {
        "ld.lld",
        "-pie",
        "--dynamic-linker=/lib64/ld-linux-x86-64.so.2",
        "--no-dependent-libraries",
        "-rpath", rpath.c_str(),
        "-o", outputFile.c_str(),
        "/usr/lib/x86_64-linux-gnu/Scrt1.o",      // dynamic-PIE startup (not rcrt1.o)
        "/usr/lib/x86_64-linux-gnu/crti.o",       // first
        objectFilePath.getValue().c_str(),
        libRuntimePath.getValue().c_str(),
        libUnwindPath.c_str(),
        libClangRtPath.c_str(),
        "-L/lib/x86_64-linux-gnu",
        "-lc",
        "-lm",
        "-lcuda",
        "/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib/x86_64-linux-gnu/crtn.o"
      };
    } else {
      args = {
        "ld.lld",
        "-shared",
        "-o", outputFile.c_str(),
        objectFilePath.getValue().c_str()
      };
    }

    auto res = lld::lldMain(args, llvm::outs(), llvm::errs(), 
                              {{lld::Gnu, &lld::elf::link}});
    if (res.retCode == 0) {
      llvm::outs() << "Output: " << outputFile << "\n";
    }
  }
};

} // namespace

} // namespace backend
} // namespace compiler
} // namespace toy
