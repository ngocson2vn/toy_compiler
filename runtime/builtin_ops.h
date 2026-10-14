#pragma once
#include <cstdint>

namespace runtime {

extern "C" {
  void* toy_alloc(int numElems, int elemType);
  void* toy_random(void* devicePtr, int numElems, int elemType);
  int64_t toy_print(void* devicePtr, int numElems, int elemType);
}

} // namespace runtime