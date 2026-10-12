# Lower GPUModule

### 1. Create GpuModuleToBinaryPass
[backend/backend.cc](backend/backend.cc)
```cpp
mlir::GpuModuleToBinaryPassOptions binPassOptions;
binPassOptions.compilationTarget = "bin";
pm.addPass(mlir::createGpuModuleToBinaryPass(binPassOptions));
```
<br/>

### 2. Lookup NVVMTargetAttrImpl
[llvm-project/mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp](llvm-project/mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp)
```cpp
-> transformGpuModulesToBinaries()
   -> moduleSerializer()

LogicalResult moduleSerializer(GPUModuleOp op,
                               OffloadingLLVMTranslationAttrInterface handler,
                               const TargetOptions &targetOptions) {
  // 
  // Omit for brevity
  // 

  for (auto targetAttr : op.getTargetsAttr()) {
    assert(targetAttr && "Target attribute cannot be null.");
    auto target = dyn_cast<gpu::TargetAttrInterface>(targetAttr);
    assert(target &&
           "Target attribute doesn't implements `TargetAttrInterface`.");
    std::optional<SmallVector<char, 0>> serializedModule =
        target.serializeToObject(op, targetOptions);
    if (!serializedModule) {
      op.emitError("An error happened while serializing the module.");
      return failure();
    }

    Attribute object =
        target.createObject(op, *serializedModule, targetOptions);
    if (!object) {
      op.emitError("An error happened while creating the object.");
      return failure();
    }
    objects.push_back(object);
  }

  // 
  // Omit for brevity
  // 
}
```
<br/>

### 3. Create NVPTXSerializer
[llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp](llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp)
```cpp
std::optional<SmallVector<char, 0>>
NVVMTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                      const gpu::TargetOptions &options) const {
  // 
  // Omit for brevity
  // 

  NVPTXSerializer serializer(*module, cast<NVVMTargetAttr>(attribute), options);
  serializer.init();
  std::optional<SmallVector<char, 0>> result = serializer.run();

  // 
  // Omit for brevity
  // 
}
```
<br/>

### 4. LLVM + NVVM dialects to LLVM IR
[llvm-project/mlir/lib/Target/LLVM/ModuleToObject.cpp](llvm-project/mlir/lib/Target/LLVM/ModuleToObject.cpp)
```cpp
std::optional<SmallVector<char, 0>> ModuleToObject::run() {
  // 
  // Omit for brevity
  // 

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateToLLVMIR(llvmContext);

  // 
  // Omit for brevity
  // 

  return moduleToObject(*llvmModule);
}
```

The **nvvm dialect** operations translate directly into **LLVM NVPTX intrinsics**. For example,<br/>
`nvvm.read.ptx.sreg.ctaid.x` is translated to `@llvm.nvvm.read.ptx.sreg.ctaid.x()`.

Check [LLVMDialectModule_kernel.llir](LLVMDialectModule_kernel.llir)
<br/>

### 5. LLVM IR to PTX
[llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp](llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp)
```cpp
std::optional<SmallVector<char, 0>>
NVPTXSerializer::moduleToObject(llvm::Module &llvmModule) {
  // 
  // Omit for brevity
  // 

  // Emit PTX code.
  std::optional<llvm::TargetMachine *> targetMachine =
      getOrCreateTargetMachine();
  if (!targetMachine) {
    getOperation().emitError() << "Target Machine unavailable for triple "
                               << triple << ", can't optimize with LLVM\n";
    return std::nullopt;
  }
  moduleToObjectTimer.startTimer();
  std::optional<std::string> serializedISA =
      translateToISA(llvmModule, **targetMachine);

  // 
  // Omit for brevity
  // 
}
```
Check [LLVMDialectModule_kernel.ptx](LLVMDialectModule_kernel.ptx)
<br/>

# Appendix
### Dump llvm Module and PTX

[llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp](llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp)
```cpp
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

std::optional<SmallVector<char, 0>>
NVPTXSerializer::moduleToObject(llvm::Module &llvmModule) {
  // Dump llvmModule before PTX
  {
    std::string errorMessage;
    auto fileName = llvmModule.getName().str() + "_kernel.llir";
    auto output = mlir::openOutputFile(fileName, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
    } else {
      output->keep();
      output->os() << llvmModule;
    }
  }

  // 
  // Omit for brevity
  // 

  // Dump PTX
  {
    std::string errorMessage;
    auto fileName = llvmModule.getName().str() + "_kernel.ptx";
    auto output = mlir::openOutputFile(fileName, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
    } else {
      output->keep();
      output->os() << serializedISA.value();
    }
  }

  // 
  // Omit for brevity
  // 
}
```
<br/>

### LLVM IR -> NVPTX Backend -> PTX
Call stack:
```cpp
llvm::NVPTXDAGToDAGISel::Select(llvm::SDNode*) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp:169)
llvm::SelectionDAGISel::DoInstructionSelection() (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp:1365)
llvm::SelectionDAGISel::CodeGenAndEmitDAG() (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp:1134)
llvm::SelectionDAGISel::SelectBasicBlock(llvm::ilist_iterator_w_bits<llvm::ilist_detail::node_options<llvm::Instruction, true, false, void, true, llvm::BasicBlock>, false, true>, llvm::ilist_iterator_w_bits<llvm::ilist_detail::node_options<llvm::Instruction, true, false, void, true, llvm::BasicBlock>, false, true>, bool&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp:889)
llvm::SelectionDAGISel::SelectAllBasicBlocks(llvm::Function const&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp:1917)
llvm::SelectionDAGISel::runOnMachineFunction(llvm::MachineFunction&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp:627)
llvm::NVPTXDAGToDAGISel::runOnMachineFunction(llvm::MachineFunction&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp:71)
llvm::SelectionDAGISelLegacy::runOnMachineFunction(llvm::MachineFunction&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp:390)
llvm::MachineFunctionPass::runOnFunction(llvm::Function&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/CodeGen/MachineFunctionPass.cpp:108)
llvm::FPPassManager::runOnFunction(llvm::Function&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/IR/LegacyPassManager.cpp:1398)
llvm::FPPassManager::runOnModule(llvm::Module&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/IR/LegacyPassManager.cpp:1444)
(anonymous namespace)::MPPassManager::runOnModule(llvm::Module&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/IR/LegacyPassManager.cpp:1513)
llvm::legacy::PassManagerImpl::run(llvm::Module&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/IR/LegacyPassManager.cpp:531)
llvm::legacy::PassManager::run(llvm::Module&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/lib/IR/LegacyPassManager.cpp:1640)
mlir::LLVM::ModuleToObject::translateToISA[abi:cxx11](llvm::Module&, llvm::TargetMachine&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Target/LLVM/ModuleToObject.cpp:222)
(anonymous namespace)::NVPTXSerializer::moduleToObject(llvm::Module&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp:715)
mlir::LLVM::ModuleToObject::run() (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Target/LLVM/ModuleToObject.cpp:283)
(anonymous namespace)::NVVMTargetAttrImpl::serializeToObject(mlir::Attribute, mlir::Operation*, mlir::gpu::TargetOptions const&) const (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp:776)
mlir::gpu::detail::TargetAttrInterfaceInterfaceTraits::FallbackModel<(anonymous namespace)::NVVMTargetAttrImpl>::serializeToObject(mlir::gpu::detail::TargetAttrInterfaceInterfaceTraits::Concept const*, mlir::Attribute, mlir::Operation*, mlir::gpu::TargetOptions const&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/build/tools/mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.h.inc:255)
mlir::gpu::TargetAttrInterface::serializeToObject(mlir::Operation*, mlir::gpu::TargetOptions const&) const (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/build/tools/mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.cpp.inc:19)
(anonymous namespace)::moduleSerializer(mlir::gpu::GPUModuleOp, mlir::gpu::OffloadingLLVMTranslationAttrInterface, mlir::gpu::TargetOptions const&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp:91)
mlir::gpu::transformGpuModulesToBinaries(mlir::Operation*, mlir::gpu::OffloadingLLVMTranslationAttrInterface, mlir::gpu::TargetOptions const&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp:125)
(anonymous namespace)::GpuModuleToBinaryPass::runOnOperation() (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp:69)
mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_44::operator()() const (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Pass/Pass.cpp:609)
void llvm::function_ref<void ()>::callback_fn<mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_44>(long) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:46)
llvm::function_ref<void ()>::operator()() const (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:69)
void mlir::MLIRContext::executeAction<mlir::PassExecutionAction, mlir::Pass&>(llvm::function_ref<void ()>, llvm::ArrayRef<mlir::IRUnit>, mlir::Pass&) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/include/mlir/IR/MLIRContext.h:290)
mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Pass/Pass.cpp:603)
mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Pass/Pass.cpp:682)
mlir::PassManager::runPasses(mlir::Operation*, mlir::AnalysisManager) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Pass/Pass.cpp:1117)
mlir::PassManager::run(mlir::Operation*) (/data02/home/son.nguyen/workspace/toy_compiler/llvm-project/mlir/lib/Pass/Pass.cpp:1091)
toy::compiler::backend::lower(mlir::ModuleOp&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&) (/data02/home/son.nguyen/workspace/toy_compiler/backend/backend.cc:184)
main (/data02/home/son.nguyen/workspace/toy_compiler/main.cpp:50)
__libc_start_call_main (libc_start_call_main.h:58)
__libc_start_main_impl (libc-start.c:360)
_start (:12)
```