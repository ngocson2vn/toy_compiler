#include <string>
#include <iostream>
#include <filesystem>

#include <cuda_runtime.h>
#include "utils.h"
#include "cuda_utils.h"

namespace fs = std::filesystem;

namespace cuda {

status::Result<cudaDeviceProp> getDeviceProperties() {
  int dev = 0;
  std::string defaultErrMsg = "Failed to get Device Properties";
  cudaError_t err = cudaSetDevice(dev);
  if (err != cudaError::cudaSuccess) {
    std::string errMsg = cudaGetErrorString(err);
    return {
      {},
      errMsg.empty() ? defaultErrMsg : errMsg
    };
  }

  cudaDeviceProp deviceProp;
  err = cudaGetDeviceProperties(&deviceProp, dev);
  if (err != cudaError::cudaSuccess) {
    std::string errMsg = cudaGetErrorString(err);
    return {
      {},
      errMsg.empty() ? defaultErrMsg : errMsg
    };
  }

  return std::move(deviceProp);
}

status::Result<std::string> getCudaRoot() {
  fs::path cudaRoot = toy::utils::getStrEnv("CUDA_ROOT");
  if (cudaRoot.empty()) {
    return {
      "",
      "${CUDA_ROOT} is unset!"
    };
  }

  if (!fs::exists(cudaRoot)) {
    return {
      "",
      std::string("${CUDA_ROOT} ").append(cudaRoot.string())
                                 .append(" does not exist!")
    };
  }

  return cudaRoot.string();
}

int getComputeCapability() {
  auto devPropRes = getDeviceProperties();
  if (!devPropRes.ok()) {
    std::cerr << devPropRes.error_message() << "\n";
    std::abort();
  }

  auto devProp = devPropRes.value();
  auto cc = devProp.major * 10 + devProp.minor;
  if (cc < 75) {
    std::cerr << "Unsupported GPU with Compute Capability " << cc << "\n";
    std::abort();
  }
  std::cout << "GPU Compute Capability: " << cc << "\n";

  return devProp.major * 10 + devProp.minor;
}

static const int kComputeCapability = getComputeCapability();

std::string getArch() {
  return std::string("sm_").append(std::to_string(kComputeCapability));
}

std::string getSupportedPtxVersion() {
  int cudaRuntimeVersion;
  cudaError_t err = cudaRuntimeGetVersion(&cudaRuntimeVersion);
  if (err != cudaError::cudaSuccess) {
    return std::string("");
  }

  int major = cudaRuntimeVersion / 1000;
  int minor = (cudaRuntimeVersion % 1000) / 10;

  if (major >= 11) {
    major -= 4;
  }

  std::string ptxVersion = std::to_string(major).append(std::to_string(minor));
  std::cout << "ptxVersion: " << ptxVersion << std::endl;

  return ptxVersion;
}

std::string getFeatures() {
  std::string ptxVersion = getSupportedPtxVersion();
  
  return std::string("+ptx").append(ptxVersion);
}

std::string getPtxasPath() {
  auto cudaRoot = getCudaRoot();
  return cudaRoot.value() + "/bin/ptxas";
}

std::string getLibdevice() {
  auto cudaRoot = getCudaRoot();
  return std::string(cudaRoot.value()).append("/nvvm/libdevice/libdevice.10.bc");
}

status::Result<int> ParseCudaArch(const std::string& arch_str) {
  auto prefixPos = arch_str.find("sm_");
  if (prefixPos == std::string::npos) {
    return {
      0,
      "Could not parse cuda architecture prefix (expected sm_)"
    };
  }

  return std::stoi(arch_str.substr(3, arch_str.size() - 3));
}

// Ensure that CUDA_ROOT, ptxas, and libdevice exist
static bool __cuda_check = []() -> bool {
  auto cudaRootRes = getCudaRoot();
  if (!cudaRootRes.ok()) {
    std::cerr << cudaRootRes.error_message() << "\n";
    std::abort();
  }

  auto ptxasPath = getPtxasPath();
  if (!fs::exists(ptxasPath)) {
    std::cerr << ptxasPath << " does not exists!\n";
    std::abort();
  }

  std::string libdevice = getLibdevice();
  if (!fs::exists(libdevice)) {
    std::cerr << libdevice << " does not exists!\n";
    std::abort();
  }

  return true;
}();

} // namespace cuda
