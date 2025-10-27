#pragma once

#include <vector>

#include "common/common.h"
#include "cuda_utils.h"

namespace runtime {

struct RuntimeCtx {
  CUstream stream;
};


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
  void* moduleHandle_ = nullptr;
  bool loaded_ = false;
};

} // namespace runtime

using namespace runtime;