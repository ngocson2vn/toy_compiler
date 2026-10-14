#include "common.h"
#include <cassert>

#include <cuda_fp16.h>

using float16 = __half;
using float32 = float;

namespace toy {
namespace compiler {
namespace common {

std::string getTypeName(Type type) {
  switch(type) {
    case Type::I32:
      return type::name::I32;
    case Type::F32:
      return type::name::F32;
    case Type::F16:
      return type::name::F16;
    default:
      return type::name::INVALID;
  }
}

int getTypeBytes(Type type) {
  switch (type) {
    case Type::I32:
    case Type::F32:
      return sizeof(float32);
    case Type::F16:
      return sizeof(float16);
    default:
      assert(false && "The type is unsupported");
  }
}

} // namespace common
} // namespace compiler
} // namespace toy
