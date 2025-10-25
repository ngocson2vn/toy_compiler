#include "mlir/InitAllExtensions.h"

// MLIR IRs
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

// MLIR Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"

// MLIR Passes
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/CodeGen/CommandFlags.h"

#include "passes.h"
#include "backend.h"
#include "cuda_utils.h"


using namespace mlir;

namespace toy {
namespace compiler {
namespace backend {

// Initialize
bool __initialized = []() -> bool {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });

  static llvm::codegen::RegisterCodeGenFlags CGF;

  return true;
}();

llvm::LogicalResult lower(ModuleOp& module) {
  auto& context = *module.getContext();

  // Load necessary dialects

  // Set up the pass manager
  context.disableMultithreading();
  PassManager pm(&context);
  std::string errorMessage;

  // lowering
  auto output = mlir::openOutputFile("lowering_backend.mlir", &errorMessage);
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


  //============================================================================
  // Lower Loops to GPU
  //============================================================================
  mlir::SmallVector<int64_t> tileSizes{128};
  pm.addNestedPass<func::FuncOp>(::mlir::toy::createTileLoopsPass(tileSizes));
  pm.addNestedPass<func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createConvertParallelLoopToGpuPass());
  
  // memref parameters -> llvm pointers
  pm.addPass(mlir::toy::createLowerMemRefToLLVMPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::createGpuLaunchSinkIndexComputationsPass());

  const std::string dataLayoutStr = "#dlti.dl_spec<#dlti.dl_entry<index, 64 : i64>>";
  mlir::GpuKernelOutliningPassOptions outliningOptions{dataLayoutStr};
  pm.addPass(mlir::createGpuKernelOutliningPass(outliningOptions));
  
  pm.addPass(mlir::createLowerAffinePass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());


  //============================================================================
  // Lower GPU to LLVM
  //============================================================================
  // Register LLVM translation interfaces
  DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);

  // Lower SCF -> CF
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Lower GPUModuleOp to LLVM
  auto& kernelPm = pm.nest<gpu::GPUModuleOp>();
  kernelPm.addPass(mlir::createConvertGpuOpsToNVVMOps());
  kernelPm.addPass(mlir::createConvertNVVMToLLVMPass());
  kernelPm.addPass(mlir::LLVM::createNVVMOptimizeForTargetPass());

  // Re-create gpu.launch_func to conform with changed kernel function signature
  pm.addPass(mlir::createGpuToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  //============================================================================
  // Lower GPUModuleOp to CUBIN
  //============================================================================
  mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
  nvvmTargetOptions.chip = cuda::getArch();
  nvvmTargetOptions.features = cuda::getFeatures();
  pm.addPass(mlir::createGpuNVVMAttachTarget(nvvmTargetOptions));

  mlir::GpuModuleToBinaryPassOptions binPassOptions;
  binPassOptions.compilationTarget = "bin";
  pm.addPass(mlir::createGpuModuleToBinaryPass(binPassOptions));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  //============================================================================
  // Lower host code
  //============================================================================
  pm.addPass(mlir::toy::createInjectRuntimeCtxPass());

  mlir::toy::LowerLLVMToObjectPassOptions objectPassOptions;
  pm.addPass(mlir::toy::createLowerLLVMToObjectPass(objectPassOptions));

  // Apply passes
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed\n";
    return failure();
  }

  module.print(output->os(), printFlag);

  return success();
}

} // namespace backend
} // namespace compiler
} // namespace toy
