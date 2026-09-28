#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "middleend-passes"

#include "frontend/toy_dialect.h"

#define TOY_ARG_ATTR_NAME "toy.dynamic_dim_arg_idx"

using namespace mlir;

namespace mlir::toy {

#define GEN_PASS_DEF_CONVERTTOYTOSTDPASS
#define GEN_PASS_DEF_CONVERTTOYTOLINALGPASS
#define GEN_PASS_DEF_CONVERTTENSORTOMEMREFPASS
#define GEN_PASS_DEF_FOLDMEMREFDIMOPPASS
#include "middleend/passes.h.inc"

} // namespace mlir::toy

namespace {

using namespace mlir;

// Conversion pattern for toy.func to func.func
struct ToyFuncOpConverter : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(toy::FuncOp oldFunc, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto funcType = oldFunc.getFunctionType();
    auto newFunc = rewriter.create<func::FuncOp>(oldFunc.getLoc(), oldFunc.getName(), funcType);
    rewriter.inlineRegionBefore(oldFunc.getBody(), newFunc.getBody(), newFunc.end());
    rewriter.replaceOp(oldFunc, newFunc);

    return success();
  }
};

// Conversion pattern for toy.return to func.return
struct ToyReturnOpConverter : public OpConversionPattern<toy::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(toy::ReturnOp oldReturnOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto newReturnOp = rewriter.create<func::ReturnOp>(oldReturnOp.getLoc(), oldReturnOp.getOperands());
    rewriter.replaceOp(oldReturnOp, newReturnOp);

    return success();
  }
};

// Pass to lower toy dialect to arith dialect
struct ConvertToyToStdPass : public toy::impl::ConvertToyToStdPassBase<ConvertToyToStdPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    // Define the conversion target (arith dialect is legal)
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<toy::ToyDialect>();

    target.addIllegalOp<toy::FuncOp>();
    target.addIllegalOp<toy::ReturnOp>();

    // Define the conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<ToyFuncOpConverter, ToyReturnOpConverter>(&getContext());

    // Apply the conversion
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};


//==============================================================================
// Toy to Linalg
//==============================================================================
struct ToyAddOpConverter : public OpConversionPattern<toy::AddOp> {
  using OpConversionPattern<toy::AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      toy::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto parentOp = op->getParentOp();
    LLVM_DEBUG(llvm::dbgs() << "\nBegin ToyAddOpConverter:\n" << *parentOp << "\n\n");

    auto x = op.getLhs();
    auto y = op.getRhs();
    auto size = op.getSize();

    auto xArgIdx = cast<BlockArgument>(x).getArgNumber();
    auto yArgIdx = cast<BlockArgument>(y).getArgNumber();
    auto sizeArgIdx = cast<BlockArgument>(size).getArgNumber();
    
    auto funcOp = op->getParentOfType<func::FuncOp>();
    funcOp.setArgAttr(xArgIdx, TOY_ARG_ATTR_NAME, rewriter.getI32IntegerAttr(sizeArgIdx));
    funcOp.setArgAttr(yArgIdx, TOY_ARG_ATTR_NAME, rewriter.getI32IntegerAttr(sizeArgIdx));

    auto out = op.getResult();
    auto resultType = cast<RankedTensorType>(out.getType());

    // Create output tensor
    Value dynamicSize = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getIndexType(), size);
    Value outTensor = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType, dynamicSize);

    int64_t rank = resultType.getRank();

    // Indexing maps: identity for x and output, empty or identity for y.
    SmallVector<AffineExpr> exprs;
    for (int64_t i = 0; i < rank; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    auto xIndexMap = AffineMap::get(rank, 0, exprs, rewriter.getContext());
    auto yIndexMap = xIndexMap;
    auto outputIndexMap = xIndexMap;
    SmallVector<AffineMap> indexingMaps = {xIndexMap, yIndexMap, outputIndexMap};

    // Set iterator types: all parallel for element-wise operation.
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Create linalg.generic operation with memref semantics.
    auto linalgOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(),
      /*resultTypes=*/TypeRange{resultType},
      /*inputs=*/ValueRange{x, y},
      /*outputs=*/ValueRange{outTensor},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange args) {
        Value xVal = args[0];
        Value yVal = args[1];
        Value result = nestedBuilder.create<arith::AddFOp>(loc, xVal, yVal);
        // nestedBuilder.create<linalg::YieldOp>(loc);
        nestedBuilder.create<linalg::YieldOp>(loc, result);
      }
    );

    rewriter.replaceOp(op, linalgOp);

    LLVM_DEBUG(llvm::dbgs() << "\nAfter ToyAddOpConverter:\n" << *parentOp << "\n");

    return success();
  }
};

struct ToyStoreOpConverter : public OpConversionPattern<toy::StoreOp> {
  using OpConversionPattern<toy::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      toy::StoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto dst = op.getDst();


    SmallVector<Value, 1> newOuts = {dst};
    auto genericOp = src.getDefiningOp<linalg::GenericOp>();
    if (!!genericOp && genericOp.getNumDpsInits() == 1) {
      auto newGenericOp = rewriter.clone(*genericOp);
      newGenericOp->setOperands(
        genericOp.getNumDpsInputs(),
        genericOp.getNumDpsInits(),
        newOuts
      );

      rewriter.replaceOp(op, newGenericOp);
      return success();
    }

    LLVM_DEBUG(llvm::dbgs() << "src op: " << *src.getDefiningOp() << "\n");

    // 
    // If `src` is not a `linalg::GenericOp`, then
    // lower `toy::StoreOp` to a `linalg::GenericOp`
    //
    auto resultType = cast<RankedTensorType>(dst.getType());
    int64_t rank = resultType.getRank();

    // Indexing maps: identity for x and output, empty or identity for y.
    SmallVector<AffineExpr> exprs;
    for (int64_t i = 0; i < rank; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    auto srcIndexMap = AffineMap::get(rank, 0, exprs, rewriter.getContext());
    auto dstIndexMap = srcIndexMap;
    SmallVector<AffineMap> indexingMaps = {srcIndexMap, dstIndexMap};

    // Set iterator types: all parallel for element-wise operation.
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Create linalg.generic operation with memref semantics.
    auto linalgOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(),
      /*resultTypes=*/TypeRange{resultType},
      /*inputs=*/ValueRange{src},
      /*outputs=*/ValueRange{dst},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange args) {
        // nestedBuilder.create<linalg::YieldOp>(loc);
        nestedBuilder.create<linalg::YieldOp>(loc, args[0]);
      }
    );

    // Erase toy::StoreOp
    rewriter.replaceOp(op, linalgOp);

    return success();
  }
};

struct ConvertToyToLinalgPass : public toy::impl::ConvertToyToLinalgPassBase<ConvertToyToLinalgPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    // Define the conversion target (arith dialect is legal)
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<toy::ToyDialect>();

    target.addIllegalOp<toy::AddOp, toy::StoreOp>();

    // Define the conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<ToyAddOpConverter, ToyStoreOpConverter>(&getContext());

    // Apply the conversion
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};


//==============================================================================
// Tensor to MemRef
//==============================================================================
// Step 1: Define the TypeConverter
struct TensorToMemRefConverter : public TypeConverter {
  TensorToMemRefConverter() {
    addConversion([](Type srcType) -> Type {
      if (auto rankedTensor = dyn_cast<RankedTensorType>(srcType)) {
        return MemRefType::get(rankedTensor.getShape(), rankedTensor.getElementType());
      }

      return srcType;
    });

    // Register source materialization: memref<?xf64> -> tensor<?xf64>
    addSourceMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange convertedValues, mlir::Location loc) -> mlir::Value {
          assert(convertedValues.size() == 1 && "convertedValues must have size = 1");
          auto srcValue = builder.create<UnrealizedConversionCastOp>(loc, resultType, convertedValues[0]);
          // return mlir::Value();
          return srcValue.getOutputs()[0];
        });

    // Register target materialization: tensor<?xf64> -> memref<?xf64>
    addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::TypeRange resultTypes,
           mlir::ValueRange srcValues, mlir::Location loc) -> SmallVector<Value> {

          SmallVector<Value> tgtValues;
          for (const auto& [t, v] : llvm::zip(resultTypes, srcValues)) {
            auto ret = builder.create<UnrealizedConversionCastOp>(loc, t, v);
            llvm::outs() << "\nMaterialized " << ret << "\n";
            tgtValues.push_back(ret.getOutputs()[0]);
          }

          return tgtValues;
        });
  }
};


// Step 2: Function Conversion Pattern
struct FuncOpConverter : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp oldFunc, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    LLVM_DEBUG(llvm::dbgs() << "\nBegin FuncOpConverter:\n" << *oldFunc->getParentOp() << "\n\n");

    //===============================================================================================
    // 1. Create newFunc
    //===============================================================================================
    auto typeConverter = getTypeConverter();
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

    auto newFuncType = FunctionType::get(getContext(), newInputTypes, newResultTypes);
    SmallVector<Type> newArgTypes(newFuncType.getInputs().begin(), newFuncType.getInputs().end());
    auto newFunc = rewriter.create<func::FuncOp>(oldFunc.getLoc(), oldFunc.getName(), newFuncType);

    // Copy attrs except the type
    for (auto attr : oldFunc->getAttrs()) {
      if (attr.getName() != oldFunc.getFunctionTypeAttrName()) {
        newFunc->setAttr(attr.getName(), attr.getValue());
      }
    }

    newFunc.setVisibility(oldFunc.getVisibility());

    // Move the body.
    rewriter.inlineRegionBefore(oldFunc.getBody(), newFunc.getBody(), newFunc.end());
    
    Block& entryBlock = newFunc.front();
    auto sig = typeConverter->convertBlockSignature(&entryBlock);
    if (!sig.has_value()) {
      llvm::errs() << "Failed to convert entry block signature\n";
      return failure();
    }
    rewriter.applySignatureConversion(&entryBlock, sig.value(), typeConverter);

    //===============================================================================================
    // 2. Update func::ReturnOp operands
    //===============================================================================================
    auto retOp = cast<func::ReturnOp>(newFunc.front().getTerminator());
    rewriter.setInsertionPoint(retOp);
    auto newRetOperands = typeConverter->materializeTargetConversion(rewriter, retOp.getLoc(), newResultTypes, retOp.getOperands());
    assert(newRetOperands.size() == retOp.getNumOperands() && "Failed to convert func::ReturnOp operands");
    for (int i = 0; i < retOp.getNumOperands(); i++) {
      retOp.setOperand(i, newRetOperands[i]);
    }

    //===============================================================================================
    // 3. Replace oldFunc with newFunc
    //===============================================================================================
    rewriter.replaceOp(oldFunc, newFunc);
    LLVM_DEBUG(llvm::dbgs() << "\nAfter FuncOpConverter:\n" << *newFunc->getParentOp() << "\n\n");

    return success();
  }
};


// Step 3: Op Conversion Patterns
struct LinalgGenericOpConverter : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    // Create linalg.generic operation with memref semantics.
    auto newOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(),
      /*resultTypes=*/TypeRange{},
      /*inputs=*/adaptor.getInputs(),
      /*outputs=*/adaptor.getOutputs(),
      /*indexingMaps=*/op.getIndexingMapsAttr(),
      /*iteratorTypes=*/op.getIteratorTypesAttr(),
      op.getDocAttr(),
      op.getLibraryCallAttr()
    );

    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().begin());
    rewriter.replaceOp(op, adaptor.getOutputs());

    return success();
  }
};


// Step 4: Define the Pass
struct ConvertTensorToMemRefPass : public toy::impl::ConvertTensorToMemRefPassBase<ConvertTensorToMemRefPass> {
  static bool checkOpLegality(Operation* op) {
    if (op->getDialect()->getNamespace() == tensor::TensorDialect::getDialectNamespace()) {
      return false;
    }

    if (op->getNumOperands() > 0) {
      for (const auto& opr : op->getOperands()) {
        if (isa<TensorType>(opr.getType())) {
          LLVM_DEBUG(llvm::dbgs() << "\nOperation (" << op << ") " << op->getName().getStringRef() << " is not legal ❌\n");
          return false;
        }
      }
    } else {
      for (const auto& resType : op->getResultTypes()) {
        if (isa<TensorType>(resType)) {
          LLVM_DEBUG(llvm::dbgs() << "\nOperation (" << op << ") " << op->getName().getStringRef() << " is not legal ❌\n");
          return false;
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "\nOperation (" << op << ") " << op->getName().getStringRef() << " is legal ✅\n");

    return true;
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Define type converter
    TensorToMemRefConverter typeConverter;

    // Define conversion target
    ConversionTarget target(getContext());

    //===========================================================================
    // 1. Convert function signature
    //===========================================================================
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<toy::ToyDialect>();

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
    patterns.add<FuncOpConverter>(typeConverter, &getContext());

    // Apply partial convertion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      LLVM_DEBUG(llvm::dbgs() << "\nRestored module:\n");
      LLVM_DEBUG(llvm::dbgs() << module << "\n");
      return;
    }


    //===========================================================================
    // 2. Convert other ops
    //===========================================================================
    ConversionTarget otherTarget(getContext());
    otherTarget.addLegalDialect<BuiltinDialect>();
    otherTarget.addLegalDialect<func::FuncDialect>();
    otherTarget.addLegalDialect<arith::ArithDialect>();
    otherTarget.addLegalDialect<linalg::LinalgDialect>();
    otherTarget.addLegalDialect<memref::MemRefDialect>();

    for (auto& op : module.getBody()->getOperations()) {
      if (auto funcOp = dyn_cast<func::FuncOp>(&op)) {
        for (auto& innerOp : funcOp.getBody().front().getOperations()) {
          if (!isa<UnrealizedConversionCastOp>(innerOp)) {
            otherTarget.addDynamicallyLegalOp(innerOp.getName(), checkOpLegality);
          }
        }
      }
    }

    patterns.clear();
    patterns.add<LinalgGenericOpConverter>(typeConverter, &getContext());
    if (failed(applyPartialConversion(module, otherTarget, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct MemRefDimOpConverter : public OpConversionPattern<memref::DimOp> {
  using OpConversionPattern<memref::DimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::DimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcArgIdx = cast<BlockArgument>(src).getArgNumber();

    auto funcOp = op->getParentOfType<func::FuncOp>();
    auto attr = cast<mlir::IntegerAttr>(funcOp.getArgAttr(srcArgIdx, TOY_ARG_ATTR_NAME));
    Value dimIntVal = funcOp.getArgument(attr.getInt());
    Value dimIdxVal = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getIndexType(), dimIntVal);
    rewriter.replaceOp(op, dimIdxVal);
    funcOp.removeArgAttrsAttr();

    return success();
  }
};

struct FoldMemRefDimOpPass : public toy::impl::FoldMemRefDimOpPassBase<FoldMemRefDimOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    // Define the conversion target (arith dialect is legal)
    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    target.addIllegalOp<memref::DimOp>();

    // Define the conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<MemRefDimOpConverter>(&getContext());

    // Apply the conversion
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}