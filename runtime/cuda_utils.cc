#include <cstdio>
#include <stdexcept>
#include "cuda_utils.h"

namespace runtime::cuda {

static bool __initialized = []() {
  CUDA_CHECK_ABORT(cuInit(0));

  CUdevice device;
  CUDA_CHECK_ABORT(cuDeviceGet(&device, 0));

  static CUcontext context;
  CUDA_CHECK_ABORT(cuDevicePrimaryCtxRetain(&context, device));
  CUDA_CHECK_ABORT(cuCtxPushCurrent(context));

  fprintf(stdout, "INFO Successfully initialized CUDA\n");
  return true;
}();

DevicePtr copyH2D(void* hostPtr, std::size_t numBytes) {
  CUdeviceptr devPtr = 0;
  CUDA_CHECK_RET_ZERO(cuMemAlloc(&devPtr, numBytes));
  CUDA_CHECK_RET_ZERO(cuMemcpyHtoD(devPtr, hostPtr, numBytes));

  return devPtr;
}

DevicePtr copyH2DAsync(void* hostPtr, std::size_t numBytes, CUstream hStream) {
  CUdeviceptr devPtr = 0;
  CUDA_CHECK_RET_ZERO(cuMemAlloc(&devPtr, numBytes));
  CUDA_CHECK_RET_ZERO(cuMemcpyHtoDAsync(devPtr, hostPtr, numBytes, hStream));

  return devPtr;
}

} // namespace runtime::cuda
