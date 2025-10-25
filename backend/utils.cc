// C headers
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>

#include <array>
#include <mutex>
#include <fstream>
#include <sstream>
#include <memory>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include "mlir/Support/FileUtilities.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"

#include "utils.h"

namespace toy::utils {

static std::mutex getenv_mutex;

// return value of a cache-invalidating boolean environment variable
bool getBoolEnv(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}

std::string getStrEnv(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return "";
  std::string result(cstr);
  return result;
}

std::vector<std::string> splitStringBySpace(const std::string& str) {
  std::vector<std::string> tokens;
  std::stringstream ss(str); // Initialize stringstream with the input string
  std::string token;

  // Extract words (tokens) from the stringstream until no more words are found
  while (ss >> token) { 
    tokens.push_back(token); // Add the extracted token to the vector
  }
  return tokens;
}

std::string readBinFile(const std::string& filePath) {
  std::string tmpOut;
  std::ifstream ifs(filePath, std::ios::in | std::ios::binary);
  ifs.seekg(0, ifs.end);
  int fileSize = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  tmpOut.resize(fileSize);
  ifs.read(tmpOut.data(), fileSize);

  return tmpOut;
}

bool writeBinFile(const std::string& data, const std::string& filePath) {
  std::ofstream ofs(filePath, std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    return false;
  }

  ofs.write(data.data(), data.size());
  ofs.flush();
  ofs.close();

  return true;
}

status::Result<bool> runCommand(const std::string& cmd, std::string& stdoutOutput, std::string& stderrOutput) {
  std::vector<std::string> parts = splitStringBySpace(cmd);
  std::string prog = parts[0];
  int numArgs = parts.size();
  std::unique_ptr<char*> args_ptr(new char*[numArgs]);
  char** argv = args_ptr.get();
  for (int i = 0; i < numArgs - 1; i++) {
    argv[i] = parts[i + 1].data();
  }
  argv[numArgs - 1] = nullptr;

  std::cout << "prog: " << prog << std::endl;
  for (int i = 0; i < numArgs - 1; i++) {
    std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
  }

  int stdoutPipe[2], stderrPipe[2];
  if (pipe(stdoutPipe) == -1 || pipe(stderrPipe) == -1) {
    return {
      false,
      "[runCommand] Failed to open stdout/stderr pipe"
    };
  }

  pid_t pid = fork();
  if (pid == -1) {
    return {
      false,
      "[runCommand] Failed to fork this program"
    };
  }

  // Child process
  if (pid == 0) {
    // Close read ends of pipes
    close(stdoutPipe[0]);
    close(stderrPipe[0]);

    // Redirect stdout and stderr to respective pipes
    dup2(stdoutPipe[1], STDOUT_FILENO);
    dup2(stderrPipe[1], STDERR_FILENO);
    close(stdoutPipe[1]);
    close(stderrPipe[1]);

    // Execute command
    execvp(prog.c_str(), argv);

    // execvp only returns if an error occurred
    fprintf(stderr, "[runCommand] execvp %s: %s\n", prog.c_str(), strerror(errno));
    exit(EXIT_FAILURE); // Child process exits
  }

  // Parent process
  else {
    // Close write ends of pipes
    close(stdoutPipe[1]);
    close(stderrPipe[1]);

    // Read from pipes
    std::array<char, 128> buffer;
    ssize_t bytesRead;

    // Read stdout
    while ((bytesRead = read(stdoutPipe[0], buffer.data(), buffer.size() - 1)) > 0) {
      buffer[bytesRead] = '\0';
      stdoutOutput += buffer.data();
    }
    close(stdoutPipe[0]);

    // Read stderr
    while ((bytesRead = read(stderrPipe[0], buffer.data(), buffer.size() - 1)) > 0) {
      buffer[bytesRead] = '\0';
      stderrOutput += buffer.data();
    }
    close(stderrPipe[0]);

    // Wait for child to finish
    int ret;
    waitpid(pid, &ret, 0);
    bool status = (WEXITSTATUS(ret) == 0);
  
    return {
      status ? true : false,
      status ? "" : "[runCommand] Failed to execute command"
    };
  }
}

std::string genTempFile() {
  // Get current time as a time_point
  auto now = std::chrono::system_clock::now();
  
  // Convert to epoch time in seconds
  auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  std::string tempName = std::string("triton_temp.").append(std::to_string(epoch_time));
  std::filesystem::path tempPath = std::filesystem::temp_directory_path() / tempName;
  return tempPath.string();
}

void dumpLLVMIR(llvm::Module& llvmMod) {
  std::string errorMessage;
  auto fileName = llvmMod.getName().str() + ".mlir";
  auto output = mlir::openOutputFile(fileName, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return;
  }
  output->keep();

  llvmMod.print(output->os(), nullptr);
}

} // namespace toy::utils
