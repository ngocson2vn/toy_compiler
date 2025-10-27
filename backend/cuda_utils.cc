#include <string>
#include <iostream>
#include <filesystem>

#include <cuda_runtime.h>
#include "utils.h"
#include "cuda_utils.h"

namespace fs = std::filesystem;

namespace cuda {

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
  std::cout << "Supported PTX version: " << ptxVersion << std::endl;

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
