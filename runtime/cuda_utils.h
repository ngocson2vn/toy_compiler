#pragma once

#include <cstdio>
#include <stdexcept>

#include <cuda.h>

#define STRINGIFY(x) #x
#define TO_STR(x) STRINGIFY(x)

// Error checking macro
#define CUDA_CHECK(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return EXIT_FAILURE; \
  } \
} while (0)

#define CUDA_CHECK_RET_VOID(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return; \
  } \
} while (0)

#define CUDA_CHECK_RET_FALSE(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return false; \
  } \
} while (0)

#define CUDA_CHECK_RET_NULL(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return nullptr; \
  } \
} while (0)

#define CUDA_CHECK_RET_ZERO(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return 0; \
  } \
} while (0)

#define CUDA_CHECK_ABORT(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    std::abort(); \
  } \
} while (0)

namespace runtime::cuda {

class DevicePtr {
 public:
  // Allow custom constructor
  DevicePtr(CUdeviceptr ptr) : ptr_(ptr) {
    fprintf(stdout, "Allocated device ptr_ %llu\n", ptr_);
  }

  // Allow move constructor
  DevicePtr(DevicePtr&& rhs) noexcept {
    // By default, ptr_ is initialized to 0
    std::swap(ptr_, rhs.ptr_);
    fprintf(stdout, "DevicePtr move ctor ptr_ %llu rhs.ptr_ %llu\n", ptr_, rhs.ptr_);
  }

  // Forbid default constructor
  DevicePtr() = delete;

  // Forbid copy constructor
  DevicePtr(const DevicePtr& rhs) = delete;

  static DevicePtr alloc(std::size_t numBytes) {
    CUdeviceptr ptr = 0;
    CUDA_CHECK_RET_ZERO(cuMemAlloc(&ptr, numBytes));
    CUDA_CHECK_ABORT(cuMemsetD8(ptr, 0.0f, numBytes));

    return ptr;
  }

  // Forbid copy assignment operator
  DevicePtr& operator=(const DevicePtr& rhs) = delete;

  // Forbid move assignment operator
  DevicePtr& operator=(DevicePtr&& rhs) = delete;

  ~DevicePtr() {
    if (ptr_) {
      CUresult res = cuMemFree(ptr_);
      if (res != CUDA_SUCCESS) {
        const char* errMsg;
        cuGetErrorString(res, &errMsg);
        fprintf(stderr, "Failed to free device ptr %llu, error: %s\n", ptr_, errMsg);
      } else {
        fprintf(stdout, "Successfully free device ptr %llu\n", ptr_);
      }
    }
  }

  CUdeviceptr& get() {
    return ptr_;
  }

 private:
  CUdeviceptr ptr_ = 0;
};

DevicePtr copyH2D(void* hostPtr, std::size_t numBytes);
DevicePtr copyH2DAsync(void* hostPtr, std::size_t numBytes, CUstream hStream);

} // namespace runtime::cuda
