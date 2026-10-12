#pragma once

#include <string>
#include <vector>

#include "common/common.h"
#include "common/status.h"

#include "mlir/IR/Types.h"

namespace mlir {
namespace gpu {
class GPUModuleOp;
}
}

namespace llvm {
class Module;
}

namespace toy {
namespace compiler {

using namespace common;
namespace backend {

Type getEncodedType(mlir::Type ty);

} // namespace backend
} // namespace compiler
} // namespace toy

namespace toy::utils {

bool getBoolEnv(const std::string &env);

std::string getStrEnv(const std::string &env);

std::vector<std::string> splitStringBySpace(const std::string& str);

std::string genTempFile();
std::string readBinFile(const std::string& filePath);
bool writeBinFile(const std::string& data, const std::string& filePath);

status::Result<bool> runCommand(const std::string& cmd, std::string& stdoutOutput, std::string& stderrOutput);

void dumpGPUModule(mlir::gpu::GPUModuleOp& gpuMod);

void dumpLLVMIR(llvm::Module& llvmMod);

} // namespace toy::utils
