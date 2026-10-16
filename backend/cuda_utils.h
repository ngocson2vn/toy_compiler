#pragma once

#include <string>

#include <driver_types.h>

#include "common/common.h"
#include "common/status.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda {

status::Result<std::string> getCudaRoot();

std::string getComputeCapability();

std::string getFeatures();

std::string getPtxasPath();

std::string getSupportedPtxVersion();

std::string getLibdevice();

}