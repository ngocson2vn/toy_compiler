// MLIR Built-in IRs
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

// MLIR Dialects
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"

// MLIR Passes
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// MLIR Support
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

// Toy Compiler headers
#include "frontend/toy_dialect.h"
#include "passes.h"
#include "middleend.h"

using namespace mlir;

namespace toy {
namespace compiler {
namespace middleend {

LogicalResult lower(mlir::ModuleOp& module) {
  auto& context = *module.getContext();

  DialectRegistry registry;
  registry.insert<mlir::toy::ToyDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<bufferization::BufferizationDialect>();

  mlir::toy::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Set up the pass manager
  context.disableMultithreading();
  PassManager pm(&context);
  std::string errorMessage;

  // lowering
  auto output = mlir::openOutputFile("lowering_middleend.mlir", &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  mlir::OpPrintingFlags printFlag{};
  pm.enableIRPrinting(
    /*shouldPrintBeforePass=*/[](mlir::Pass* p, mlir::Operation* op) {
      return false;
    },
    /*shouldPrintAfterPass=*/[](mlir::Pass* p, mlir::Operation * op) {
      return true;
    },
    /*printModuleScope=*/true, 
    /*printAfterOnlyOnChange=*/true,
    /*printAfterOnlyOnFailure=*/false, 
    output->os(), printFlag
  );
  output->keep();

  // Toy to Std
  pm.addPass(mlir::toy::createConvertToyToStdPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Toy to Linalg
  pm.addPass(mlir::toy::createConvertToyToLinalgPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Producer-Consumer Fusion
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Fold empty tensors
  pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
  pm.addPass(mlir::toy::createFoldMaterializeInDestinationOpPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // To MemRef
  pm.addPass(mlir::toy::createConvertTensorToMemRefPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // To Loops
  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Fold toy::TieDimsOp ops
  pm.addPass(mlir::toy::createFoldTieDimsOpPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Apply the pass
  if (failed(pm.run(module))) {
    llvm::errs() << "[middleend] passes execution failed\n";
    return failure();
  }

  return success();
}

} // namespace middleend
} // namespace compiler
} // namespace toy
