#pragma once

#include <string>

#include <driver_types.h>

#include "common/status.h"

namespace cuda {

// CUDA compute capability, as reported by the device description.
struct CudaComputeCapability {
  int major = 0;
  int minor = 0;

  // MSVC does not like "PASCAL" symbol.
  enum CudaComputeCapabilities { PASCAL_ = 6, VOLTA = 7, AMPERE = 8 };

  CudaComputeCapability() {}
  CudaComputeCapability(int major, int minor) {
    this->major = major;
    this->minor = minor;
  }

  bool IsAtLeast(int other_major, int other_minor = 0) const {
    return !(*this < CudaComputeCapability{other_major, other_minor});
  }

  bool operator<(const CudaComputeCapability &other) const {
    return ToPair() < other.ToPair();
  }

  bool operator==(const CudaComputeCapability &other) const {
    return ToPair() == other.ToPair();
  }

  bool operator!=(const CudaComputeCapability &other) const {
    return !(*this == other);
  }

  // Maximum resident blocks per multiprocessor, values taken from
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities.
  int GetMaxResidentBlocksPerSM() const {
    if (IsAtLeast(8, 6)) {
      return 16;
    } else if (IsAtLeast(8)) {
      return 32;
    } else if (IsAtLeast(7, 5)) {
      return 16;
    }
    return 32;
  }

  // Maximum resident warps per multiprocessor, values taken from
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities.
  int GetMaxResidentWarpsPerSM() const {
    if (IsAtLeast(8, 6)) {
      return 48;
    } else if (IsAtLeast(8)) {
      return 64;
    } else if (IsAtLeast(7, 5)) {
      return 32;
    }
    return 64;
  }

  std::string ToString() const { 
    return std::to_string(major).append(".").append(std::to_string(minor)); 
  }

  std::pair<int, int> ToPair() const { return std::make_pair(major, minor); }
};

status::Result<std::string> getCudaRoot();

status::Result<int> ParseCudaArch(const std::string& arch_str);

status::Result<cudaDeviceProp> getDeviceProperties();

int getComputeCapability();

std::string getArch();

std::string getFeatures();

std::string getPtxasPath();

std::string getSupportedPtxVersion();

std::string getLibdevice();

}