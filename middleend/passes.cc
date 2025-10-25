#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "middleend-passes"

#include "frontend/toy_dialect.h"

using namespace mlir;

namespace mlir::toy {

#define GEN_PASS_DEF_CONVERTTOYTOSTDPASS
#define GEN_PASS_DEF_CONVERTTENSORTOMEMREFPASS
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
    // 3. Replace oldFunc with newFunc
    //===============================================================================================
    rewriter.replaceOp(oldFunc, newFunc);
    LLVM_DEBUG(llvm::dbgs() << "\nAfter FuncOpConverter:\n" << *newFunc->getParentOp() << "\n\n");

    return success();
  }
};


// Step 3: AddOpConverter Conversion Pattern
struct AddOpConverter : public OpConversionPattern<toy::AddOp> {
  using OpConversionPattern<toy::AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      toy::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto parentOp = op->getParentOp();
    LLVM_DEBUG(llvm::dbgs() << "\nBegin AddOpConverter:\n" << *parentOp << "\n\n");

    auto xMemref = adaptor.getOperands()[0];
    auto yMemref = adaptor.getOperands()[1];
    auto size = adaptor.getOperands()[2];

    auto userRange = op->getUsers();
    SmallVector<mlir::Operation*> userOps(userRange.begin(), userRange.end());
    if (userOps.size() != 1) {
      llvm::errs() << "Failed to convert '" << op.getOperationName() << "' "
                   << "because IR seems invalid!\n";
      return failure();
    }

    Value outputMemref;
    SmallVector<mlir::Operation*> rmOps;
    if (auto storeOp = mlir::dyn_cast<mlir::toy::StoreOp>(*userOps.begin())) {
      auto dstOperand = storeOp.getOperand(1);
      if (auto castOp = dstOperand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (mlir::dyn_cast<mlir::MemRefType>(castOp.getOperand(0).getType())) {
          outputMemref = castOp.getOperand(0);
          rmOps.push_back(storeOp);
        }
      }
    }

    if (!outputMemref) {
      llvm::errs() << "Failed to convert '" << op.getOperationName() << "' "
                   << "because IR seems invalid!\n";
      return failure();
    }

    auto outputMemrefType = cast<MemRefType>(getTypeConverter()->convertType(op.getResult().getType()));
    int64_t rank = outputMemrefType.getRank();

    // // Indexing maps: identity for x and output, empty or identity for y.
    // SmallVector<AffineExpr> exprs;
    // for (int64_t i = 0; i < rank; ++i) {
    //   exprs.push_back(rewriter.getAffineDimExpr(i));
    // }
    // auto xIndexMap = AffineMap::get(rank, 0, exprs, rewriter.getContext());
    // auto yIndexMap = xIndexMap;
    // auto outputIndexMap = xIndexMap;
    // SmallVector<AffineMap> indexingMaps = {xIndexMap, yIndexMap, outputIndexMap};

    // // Set iterator types: all parallel for element-wise operation.
    // SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // // Create linalg.generic operation with memref semantics.
    // auto linalgOp = rewriter.create<linalg::GenericOp>(
    //   op.getLoc(),
    //   /*resultTypes=*/TypeRange{}, // No tensor results; output is written to memref
    //   /*inputs=*/ValueRange{xMemref, yMemref},
    //   /*outputs=*/ValueRange{outputMemref},
    //   /*indexingMaps=*/indexingMaps,
    //   /*iteratorTypes=*/iteratorTypes,
    //   [&](OpBuilder &nestedBuilder, Location loc, ValueRange args) {
    //     Value xVal = args[0];
    //     Value yVal = args[1];
    //     Value result = nestedBuilder.create<arith::AddFOp>(loc, xVal, yVal);
    //     nestedBuilder.create<linalg::YieldOp>(loc, result);
    //   }
    // );

    // Using SCF dialect
    auto c0 = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
    auto c1 = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(1));
    auto ub = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getIndexType(), size);
    rewriter.create<scf::ParallelOp>(op.getLoc(), ValueRange{c0}, ValueRange{ub}, ValueRange{c1}, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs, ValueRange values) {
        Value idx = ivs[0];
        Value xVal = nestedBuilder.create<memref::LoadOp>(loc, xMemref, ValueRange{idx});
        Value yVal = nestedBuilder.create<memref::LoadOp>(loc, yMemref, ValueRange{idx});
        Value result = nestedBuilder.create<arith::AddFOp>(loc, xVal, yVal);
        nestedBuilder.create<memref::StoreOp>(loc, result, outputMemref, ValueRange{idx});
      }
    );

    rewriter.replaceOp(op, outputMemref);
    for (auto op : rmOps) {
      rewriter.eraseOp(op);
    }

    LLVM_DEBUG(llvm::dbgs() << "\nAfter AddOpConverter:\n" << *parentOp << "\n");

    return success();
  }
};


// Step 4: Define the Pass
struct ConvertTensorToMemRefPass : public toy::impl::ConvertTensorToMemRefPassBase<ConvertTensorToMemRefPass> {
  static bool checkOpLegality(Operation* op) {
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
    // 2. Convert toy ops
    //===========================================================================
    // target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<toy::ToyDialect>();
    for (auto& op : module.getBody()->getOperations()) {
      if (auto funcOp = dyn_cast<func::FuncOp>(&op)) {
        for (auto& innerOp : funcOp.getBody().front().getOperations()) {
          target.addDynamicallyLegalOp(innerOp.getName(), checkOpLegality);
        }
      }
    }

    patterns.clear();
    patterns.add<AddOpConverter>(typeConverter, &getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}