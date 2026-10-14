#pragma once

#include <cstdio>
#include <string>

#define LOG_ERROR(...) \
  fprintf(stderr, "ERROR %s:%d: ", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); \
  fprintf(stderr, "\n")

#define LOG_INFO(...) \
  fprintf(stderr, "INFO "); \
  fprintf(stdout, __VA_ARGS__); \
  fprintf(stdout, "\n")

namespace toy {
namespace compiler {
namespace common {

enum class Type {
  TENSOR = 1,
  I32 = 2,
  F32 = 3,
  F16 = 4,
  INVALID = 99
};

namespace type::name {
  constexpr char F32[] = "f32";
  constexpr char F16[] = "f16";
  constexpr char I32[] = "i32";
  constexpr char TENSOR[] = "tensor";
  constexpr char INVALID[] = "invalid";
}

std::string getTypeName(Type type);
int getTypeBytes(Type type);

}
}
}