#include "cuda_utils.h"

namespace runtime::cuda {

void copyD2H(void* hostPtr, void* devPtr, std::size_t numBytes) {
  auto res = cuMemcpyDtoH(hostPtr, reinterpret_cast<CUdeviceptr>(devPtr), numBytes);
  if (res != CUDA_SUCCESS) {
    const char* errMsg;
    cuGetErrorString(res, &errMsg);
    fprintf(stderr, "CUDA Error: %s\n", errMsg);
    std::abort();
  }
}

void copyH2D(void* devPtr, const void* hostPtr, std::size_t numBytes) {
  auto res = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(devPtr), hostPtr, numBytes);
  if (res != CUDA_SUCCESS) {
    const char* errMsg;
    cuGetErrorString(res, &errMsg);
    fprintf(stderr, "CUDA Error: %s\n", errMsg);
    std::abort();
  }
}

DevicePtr copyH2D(const void* hostPtr, std::size_t numBytes) {
  CUdeviceptr devPtr = 0;
  CUDA_CHECK_RET_ZERO(cuMemAlloc(&devPtr, numBytes));
  CUDA_CHECK_RET_ZERO(cuMemcpyHtoD(devPtr, hostPtr, numBytes));

  return devPtr;
}

DevicePtr copyH2DAsync(const void* hostPtr, std::size_t numBytes, CUstream hStream) {
  CUdeviceptr devPtr = 0;
  CUDA_CHECK_RET_ZERO(cuMemAlloc(&devPtr, numBytes));
  CUDA_CHECK_RET_ZERO(cuMemcpyHtoDAsync(devPtr, hostPtr, numBytes, hStream));

  return devPtr;
}

static bool __initialized = []() {
  CUDA_CHECK_ABORT(cuInit(0));

  CUdevice device;
  CUDA_CHECK_ABORT(cuDeviceGet(&device, 0));

  static CUcontext context;
  CUDA_CHECK_ABORT(cuDevicePrimaryCtxRetain(&context, device));
  CUDA_CHECK_ABORT(cuCtxPushCurrent(context));

  int major = 0;
  int minor = 0;
  CUDA_CHECK_ABORT(
    cuDeviceGetAttribute(
      &major, 
      CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      device
    ));
  CUDA_CHECK_ABORT(
    cuDeviceGetAttribute(
      &minor, 
      CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      device
    ));

  int cc = major * 10 + minor;

  fprintf(stdout, "INFO Successfully initialized CUDA!\n");
  fprintf(stdout, "INFO Device Compute Capability: %d\n", cc);

  return true;
}();

} // namespace runtime::cuda
