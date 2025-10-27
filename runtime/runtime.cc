#include <string>
#include <dlfcn.h>

#include "common/common.h"
#include "cuda_utils.h"
#include "runtime.h"

namespace runtime {

ModuleMgr::ModuleMgr(const char* modulePath) : modulePath_(modulePath) {
  void* handle = dlopen(modulePath_, RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    LOG_ERROR("Failed to load %s, error: %s", modulePath_, dlerror());
    return;
  }

  moduleHandle_ = handle;
  loaded_ = true;
  LOG_INFO("Successfully loaded %s", modulePath_);
}

bool ModuleMgr::ok() {
  return loaded_;
}

ModuleMgr::~ModuleMgr() {
  if (moduleHandle_) {
    dlclose(moduleHandle_);
    LOG_INFO("Successfully unloaded %s", modulePath_);
  }
}

void* ModuleMgr::getFunc(const char* name) {
  if (!moduleHandle_) {
    LOG_ERROR("%s hasn't been loaded yet", modulePath_);
    return nullptr;
  }

  void* func = dlsym(moduleHandle_, name);
  if (!func) {
    LOG_ERROR("Couldn't find function %s in %s", name, modulePath_);
    return nullptr;
  }

  return func;
}

} // namespace runtime
