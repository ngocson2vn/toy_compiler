#pragma once

#include <string>
#include <vector>

#include "common/status.h"

namespace llvm {
class Module;
}

namespace toy::utils {

bool getBoolEnv(const std::string &env);

std::string getStrEnv(const std::string &env);

std::vector<std::string> splitStringBySpace(const std::string& str);

std::string genTempFile();
std::string readBinFile(const std::string& filePath);
bool writeBinFile(const std::string& data, const std::string& filePath);

status::Result<bool> runCommand(const std::string& cmd, std::string& stdoutOutput, std::string& stderrOutput);

void dumpLLVMIR(llvm::Module& llvmMod);

} // namespace toy::utils