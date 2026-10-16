#!/bin/bash

set -e

ROOT_DIR=$(pwd)
echo "==================================================="
echo "1. Update git submodules"
echo "==================================================="
if [[ ! -f ./.git_submodule_updated ]]; then
  git submodule update --init --recursive
  touch ./.git_submodule_updated
fi
echo "DONE"

echo
echo "==================================================="
echo "2. Build llvm-project/llvm"
echo "==================================================="
LLVM_BUILD_DIR=${ROOT_DIR}/llvm-project/build

if [ ! -f ./.llvm.build.done ]; then
  mkdir -p ${LLVM_BUILD_DIR}

  # -DCMAKE_BUILD_TYPE=Debug | Release \
  cmake -G Ninja -S llvm-project/llvm -B ${LLVM_BUILD_DIR} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt;libunwind" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;X86;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=OFF \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_CCACHE_BUILD=ON \
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DCOMPILER_RT_BUILD_SANITIZERS=ON

  cmake --build ${LLVM_BUILD_DIR}
  touch ./.llvm.build.done
  echo "DONE"
fi
echo "DONE"

echo
echo "==================================================="
echo "3. Build toy compiler"
echo "==================================================="
mkdir -p ${ROOT_DIR}/build

cmake -G Ninja -S . -B build \
  -DCUDA_ROOT=${CUDA_ROOT} \
  -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir \
  -DLLD_DIR=${LLVM_BUILD_DIR}/lib/cmake/lld

# cmake --build build/ -v
cmake --build build/
echo "DONE"

echo
echo "==================================================="
echo "4. Copy binaries to output/bin"
echo "==================================================="
cd ${ROOT_DIR}/
mkdir -p output/bin/
rsync -avP build/compiler output/bin/
rsync -avP build/add_two_vectors output/bin/
mkdir -p output/lib/
rsync -avP build/runtime/libruntime.a output/lib/
rsync -avP llvm-project/build/lib/clang/22/lib/x86_64-unknown-linux-gnu/libclang_rt.builtins.a output/lib/
rsync -avP llvm-project/build/lib/x86_64-unknown-linux-gnu/libunwind.a output/lib/
find output/
