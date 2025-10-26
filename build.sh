#!/bin/bash

set -e

ROOT_DIR=$(pwd)
echo "ROOT_DIR=${ROOT_DIR}"
mkdir -p ${ROOT_DIR}/build

pre_hash=""
if [ -f ./.cmake.sha256 ]; then
  pre_hash=$(cat ./.cmake.sha256)
fi
now_hash=$(sha256sum ./CMakeLists.txt | awk '{print $1}')

if [ "${now_hash}" != "${pre_hash}" ]; then
  echo "${now_hash} != ${pre_hash}"
  echo
  echo "==================================================="
  echo "Generate ninja build file"
  echo "==================================================="
  cd ${ROOT_DIR}/build

  # -DCMAKE_BUILD_TYPE=Debug | Release \

  cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS="mlir;compiler-rt" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;X86;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DCOMPILER_RT_BUILD_SANITIZERS=ON \
    -DCUDA_ROOT=${CUDA_ROOT}

  cmake --build .

  cd ${ROOT_DIR}/
  yes | echo ${now_hash} > ./.cmake.sha256
fi

echo
echo "==================================================="
echo "Run ninja build"
echo "==================================================="
cd ${ROOT_DIR}/build
cmake --build . -v

echo
echo "==================================================="
echo "Copy binaries to output/bin"
echo "==================================================="
cd ${ROOT_DIR}/
rsync -avP build/compiler output/bin/
rsync -avP build/add_two_vectors output/bin/
find output/
