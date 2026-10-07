#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "toy_dialect.h"
#include "frontend/toy_dialect.cpp.inc"

using namespace mlir;
using namespace mlir::bufferization;

// Register the dialect with the MLIR context
void mlir::toy::ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "frontend/toy_ops.cpp.inc"
    >();
}

namespace {

struct TensorDimOpPattern : OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::DimOp op, PatternRewriter& rewriter) const override {
    auto tie = op.getSource().getDefiningOp<toy::TieDimsOp>();
    auto idx = op.getConstantIndex();
    if (!tie || !idx) {
      return failure();
    }

    auto tensorType = cast<RankedTensorType>(tie.getType());
    if (!tensorType.isDynamicDim(idx.value())) {
      return failure();
    }

    auto dimVal = tie.getDims()[idx.value()];
    rewriter.replaceOp(op, dimVal);

    return success();
  }
};

struct MemRefDimOpPattern : OpRewritePattern<memref::DimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::DimOp op, PatternRewriter& rewriter) const override {
    auto tie = op.getSource().getDefiningOp<toy::TieDimsOp>();
    auto idx = op.getConstantIndex();
    if (!tie || !idx) {
      return failure();
    }

    auto memrefType = cast<MemRefType>(tie.getType());
    if (!memrefType.isDynamicDim(idx.value())) {
      return failure();
    }

    auto dimVal = tie.getDims()[idx.value()];
    rewriter.replaceOp(op, dimVal);

    return success();
  }
};

struct TieDimsBufferizableModel
  : public bufferization::BufferizableOpInterface::ExternalModel<TieDimsBufferizableModel, toy::TieDimsOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  bool isWritable(Operation *op, Value value,
                  const bufferization::AnalysisState &state) const {
    return isa<OpResult>(value);  // the result is a view of a writable buffer
  }
};

}

namespace mlir {
namespace toy {

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

void TieDimsOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* ctx) {
  patterns.add<TensorDimOpPattern, MemRefDimOpPattern>(ctx);
}

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ToyDialect *dialect) {
    TieDimsOp::attachInterface<TieDimsBufferizableModel>(*ctx);
  });
}

} // namespace toy
} // namespace mlir

#define GET_OP_CLASSES
#include "frontend/toy_ops.cpp.inc"
