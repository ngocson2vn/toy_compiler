#pragma once

#include <vector>

#include "cuda_utils.h"

namespace runtime {

struct RuntimeCtx {
  CUstream stream;
};

struct InputType {
  void* ptr;
  std::size_t bytes;
};

struct OutputType {
  cuda::DevicePtr devPtr;
  std::size_t bytes;
};

using FuncType = void (*)(const void** args);

class ModuleMgr {
 public:
  ModuleMgr(const char* modulePath);

  ~ModuleMgr();

  bool ok();

  template <typename FuncType, typename... Arg>
  bool call(const char* funcName, Arg... arg) {
    void* funcHandle = getFunc(funcName);
    if (!funcHandle) {
      return false;
    }

    auto funcPtr = reinterpret_cast<FuncType>(funcHandle);
    (funcPtr)(arg...);

    return true;
  }

 private:
  void* getFunc(const char* name);

 private:
  const char* modulePath_;
  void* moduleHandle_;
  bool loaded_ = false;
};

} // namespace runtime