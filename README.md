# Toy ML Compiler
This toy compiler is for learning MLIR and how to build an ML compiler from scratch. <br/>
Currently, what it can do is compile a simple toy function, written in a toy DSL, into a real CUDA kernel that can be launched on an NVIDIA GPU.
<br/>

**Example:**<br/>
[add_two_vectors.toy](add_two_vectors.toy)<br/>
```python
# This function will be compiled into a CUDA kernel
def add_two_vectors(a: tensor<d0xd1xi32>, b: tensor<d0xd1xi32>, c: tensor<d0xd1xi32>, d0: int, d1: int) {
  c = add(a, b);
}

def main() {
  var d0 : int = 4;
  var d1 : int = 8;
  var a : tensor<d0xd1xi32>;
  var b : tensor<d0xd1xi32>;
  var c : tensor<d0xd1xi32>;

  # Generate random tensors a and b
  random(a);
  random(b);

  # Launch add_two_vectors kernel
  add_two_vectors(a, b, c, d0, d1);

  print(a);
  print(b);
  print(c);
}
```

Compile [add_two_vectors.toy](add_two_vectors.toy) to `./main` binary and run it:
```bash
# Required
export CUDA_ROOT=/usr/local/cuda-12.4
./output/bin/compiler ./add_two_vectors.toy

./main 
INFO Successfully initialized CUDA!
INFO Device Compute Capability: 120
INFO [toy_alloc] numElems=32 elemType=i32
INFO [toy_alloc] numElems=32 elemType=i32
INFO [toy_alloc] numElems=32 elemType=i32
a tensor: [90, 65, 20, 91, 48, 34, 58, 51, 18, 70, 91, 19, 5, 41, 74, 70, 31, 79, 90, 95, 86, 24, 10, 47, 56, 55, 72, 22, 41, 29, 94, 93]
b tensor: [65, 44, 58, 21, 58, 81, 19, 81, 79, 60, 5, 57, 91, 91, 77, 68, 85, 34, 11, 49, 58, 78, 80, 11, 52, 50, 64, 63, 69, 15, 38, 74]
c tensor: [155, 109, 78, 112, 106, 115, 77, 132, 97, 130, 96, 76, 96, 132, 151, 138, 116, 113, 101, 144, 144, 102, 90, 58, 108, 105, 136, 85, 110, 44, 132, 167]
```

<br/>

**Lowering Pipeline:**<br/>
<img src="./docs/pipeline.png" width="100%"/>
<br/>

**Runtime:**<br/>
<img src="./docs/runtime.png" width="100%"/>

## How to build
**Prerequisites**<br/>
```
1. CUDA Toolkit version >= 12.4
2. CMake version >= 3.31.6
3. Clang version >= 17
```
<br/>

**Build commands**
```Bash
git submodule update --init --recursive
export CUDA_ROOT=/usr/local/cuda-12.4
make
```
Output: ./build/compiler

## Compile add_two_tensors.toy
```Bash
./compile.sh
```
Output: ./libkernel.so

## Launch generated CUDA kernel
```Bash
./test.sh
```

## Discussion Channel
If you are interested in this toy project, feel free to reach out to me on LinkedIn.<br/>
My LinkedIn: https://www.linkedin.com/in/ngocson2vn/
<br/>

# Development Notes
## Lowering GPU module to CUBIN
Read more [Lower_GPUModule.md](Lower_GPUModule.md)

The backend passes generate a GPU module which is represented in both **LLVM Dialect** and **NVVM Dialect** as follows:
```MLIR
  gpu.module @add_two_vectors_kernel [#nvvm.target<chip = "sm_86", features = "+ptx84">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = llvm.mul %3, %1 overflow<nsw> : i64
      %7 = llvm.mul %3, %0 overflow<nsw> : i64
      %8 = llvm.add %7, %arg0 : i64
      %9 = llvm.intr.smin(%8, %1) : (i64, i64) -> i64
      %10 = llvm.icmp "slt" %5, %9 : i64
      llvm.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %11 = llvm.add %5, %6 : i64
      %12 = llvm.getelementptr %arg1[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %13 = llvm.load %12 : !llvm.ptr -> f32
      %14 = llvm.getelementptr %arg2[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %15 = llvm.load %14 : !llvm.ptr -> f32
      %16 = llvm.fadd %13, %15 : f32
      %17 = llvm.getelementptr %arg3[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %16, %17 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
```
<br/>

Lowering pipeline: GPU module (**LLVM Dialect** and **NVVM Dialect**) -> LLVM IR -> PTX ISA -> CUBIN. <br/>

The following backend pass is responsible for lowering the GPU module to CUBIN:
```C++
mlir::GpuModuleToBinaryPassOptions binPassOptions;
binPassOptions.compilationTarget = "bin";
pm.addPass(mlir::createGpuModuleToBinaryPass(binPassOptions));

// llvm-project/build/tools/mlir/include/mlir/Dialect/GPU/Transforms/Passes.h.inc
template <typename DerivedT>
class GpuModuleToBinaryPassBase : public ::mlir::OperationPass<> {
private:
  friend std::unique_ptr<::mlir::Pass> createGpuModuleToBinaryPass(GpuModuleToBinaryPassOptions options) {
    return std::make_unique<DerivedT>(std::move(options));
  }

// llvm-project/mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp
class GpuModuleToBinaryPass
    : public impl::GpuModuleToBinaryPassBase<GpuModuleToBinaryPass> {
public:
  using Base::Base;
  void runOnOperation() final;
};

// Call Stack:
GpuModuleToBinaryPass::runOnOperation()
  -> mlir::gpu::transformGpuModulesToBinaries()
    -> moduleSerializer()
      -> NVVMTargetAttrImpl::serializeToObject()
        -> mlir::LLVM::ModuleToObject::run()
          -> NVPTXSerializer::moduleToObject()

// Key steps:
// 1. translateToLLVMIR() translates GPU module to LLVM IR
// llvm-project/mlir/lib/Target/LLVM/ModuleToObject.cpp
// std::unique_ptr<llvm::Module> llvmModule = translateToLLVMIR(llvmContext);
// 
// 2. translateToISA() translates LLVM IR to PTX ISA
// llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp
// std::optional<std::string> serializedISA = translateToISA(llvmModule, **targetMachine);
// 
// 3. compileToBinary() compiles PTX to CUBIN
// llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp
// result = compileToBinary(*serializedISA);
```

Read more [Lower_GPUModule.md](Lower_GPUModule.md)
