#include "builtin_ops.h"

#include <cstdio>
#include <random>
#include <cassert>
#include <functional>

#include "common/common.h"
#include "cuda_utils.h"

#include <cuda_fp16.h>

using namespace toy::compiler::common;
using float16 = __half;
using float32 = float;

namespace runtime {

namespace {

template <typename ElemType>
void random(void* devicePtr, int numElems) {
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int32_t> idist(0.0, 100);
  std::uniform_real_distribution<float32> fdist(0.0, 0.1);
  std::vector<ElemType> hostVec(numElems, 0);

  if constexpr(std::is_same<ElemType, int>()) {
    for (int i = 0; i < numElems; i++) {
      hostVec[i] = idist(gen);
    }
  } else if constexpr(std::is_same<ElemType, float32>()) {
    for (int i = 0; i < numElems; i++) {
      hostVec[i] = fdist(gen);
    }
  } else if constexpr(std::is_same<ElemType, float16>()) {
    for (int i = 0; i < numElems; i++) {
      hostVec[i] = float16(fdist(gen));
    }
  } else {
    assert(false && "The ElemType is unsupported");
  }

  auto numBytes = numElems * sizeof(ElemType);
  cuda::copyH2D(devicePtr, hostVec.data(), numBytes);
}

template <typename ElemType>
int64_t print(void* devicePtr, int numElems) {
  if (numElems < 1) {
    printf("tensor data: []\n");
    return 0;
  }

  std::vector<ElemType> hostVec(numElems, 0);
  auto numBytes = sizeof(ElemType) * numElems;
  cuda::copyD2H(hostVec.data(), devicePtr, numBytes);

  printf("tensor data: [");
  if constexpr(std::is_same<ElemType, int32_t>()) {
    printf("%d", hostVec[0]);
    for (int i = 1; i < numElems; i++) {
      printf(", %d", hostVec[i]);
    }
    printf("]\n");
  } else if constexpr(std::is_same<ElemType, float32>()) {
    printf("%.2f", hostVec[0]);
    for (int i = 1; i < numElems; i++) {
      printf(", %.2f", hostVec[i]);
    }
    printf("]\n");
  } else if constexpr(std::is_same<ElemType, float16>()) {
    printf("%.2f", __half2float(hostVec[0]));
    for (int i = 1; i < numElems; i++) {
      printf(", %.2f", __half2float(hostVec[i]));
    }
    printf("]\n");
  } else {
    assert(false && "The ElemType is unsupported");
  }

  return numBytes;
}

} // namespace

extern "C" {
  void* toy_alloc(int numElems, int elemType) {
    auto type = static_cast<Type>(elemType);
    auto typeName = getTypeName(type);
    printf("INFO [toy_alloc] numElems=%d elemType=%s\n", numElems, typeName.c_str());
    int numBytes = numElems * getTypeBytes(type);
    CUdeviceptr devPtr = 0;
    CUDA_CHECK_RET_ZERO(cuMemAlloc(&devPtr, numBytes));
    return reinterpret_cast<void*>(devPtr);
  }

  void* toy_random(void* devicePtr, int numElems, int elemType) {
    auto type = static_cast<Type>(elemType);
    switch (type) {
      case Type::I32: {
        random<int32_t>(devicePtr, numElems);
        break;
      }
      case Type::F32: {
        random<float32>(devicePtr, numElems);
        break;
      }
      case Type::F16: {
        random<float16>(devicePtr, numElems);
        break;
      }
      default: {
        fprintf(stderr, "ERROR The elemType = %d is unsupported\n", elemType);
        assert(false && "The elemType is unsupported");
      }
    }

    return devicePtr;
  }

  int64_t toy_print(void* devicePtr, int numElems, int elemType) {
    auto type = static_cast<Type>(elemType);
    switch (type) {
      case Type::I32: {
        return print<int32_t>(devicePtr, numElems);
      }
      case Type::F32: {
        return print<float32>(devicePtr, numElems);
      }
      case Type::F16: {
        return print<float16>(devicePtr, numElems);
      }
      default:
        assert(false && "The type is unsupported");
    }
  }
}

} // namespace runtime