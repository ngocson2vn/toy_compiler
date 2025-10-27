#pragma once

#include <string>

#include <driver_types.h>

#include "common/status.h"

namespace cuda {

status::Result<std::string> getCudaRoot();

std::string getFeatures();

std::string getPtxasPath();

std::string getSupportedPtxVersion();

std::string getLibdevice();

}