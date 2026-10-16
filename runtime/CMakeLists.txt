add_library(runtime
  CudaRuntimeWrappers.cpp
  cuda_utils.cc
  builtin_ops.cc
  runtime.cc

  $<TARGET_OBJECTS:common>
)
