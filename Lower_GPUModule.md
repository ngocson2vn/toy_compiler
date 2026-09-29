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

Check [LLVMDialectModule_kernel.mlir](LLVMDialectModule_kernel.mlir)
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
Dump llvm Module and PTX:

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