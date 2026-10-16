#pragma once

#include <cstdio>
#include <string>

#define STRINGIFY(x) #x
#define TO_STR(x) STRINGIFY(x)

#define LOG_ERROR(...) \
  fprintf(stderr, "ERROR %s:%d: ", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); \
  fprintf(stderr, "\n")

#define LOG_INFO(...) \
  fprintf(stderr, "INFO "); \
  fprintf(stdout, __VA_ARGS__); \
  fprintf(stdout, "\n")

// Error checking macro
#define CUDA_CHECK(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return EXIT_FAILURE; \
  } \
} while (0)

#define CUDA_CHECK_RET_VOID(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return; \
  } \
} while (0)

#define CUDA_CHECK_RET_FALSE(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return false; \
  } \
} while (0)

#define CUDA_CHECK_RET_NULL(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return nullptr; \
  } \
} while (0)

#define CUDA_CHECK_RET_ZERO(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return 0; \
  } \
} while (0)

#define CUDA_CHECK_ABORT(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    std::abort(); \
  } \
} while (0)

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