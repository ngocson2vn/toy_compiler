#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "runtime/runtime.h"


using DataType = float;

// Kernel function name defined in the add_two_vectors.toy source file.
static constexpr char kAddTwoVectors[] = "add_two_vectors";

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s /path/to/libkernel.so numElements\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char* libkernelPath = argv[1];
  
  int numElements = -1;
  try {
    numElements = std::stoi(argv[2]);
    if (numElements < 1) {
      LOG_ERROR("numElements should be a positive integer number.");
      return EXIT_FAILURE;
    }
  } catch (std::exception& ex) {
    LOG_ERROR("Failed to parse BLOCK_SIZE, error: %s\n", ex.what());
    return EXIT_FAILURE;
  }

  //
  // Client code
  //

  // Load shared object file
  ModuleMgr modMgr(libkernelPath);
  if (!modMgr.ok()) {
    return EXIT_FAILURE;
  }

  // Allocate host input
  std::vector<DataType> hInputVec1(numElements, 0.0f);
  std::vector<DataType> hInputVec2(numElements, 0.0f);

  // Allocate host output
  std::vector<DataType> hOutputVec(numElements, 0.0f);

  // Initialize input data
  for (int i = 0; i < numElements; i++) {
    hInputVec1[i] = static_cast<DataType>(i+1);
    hInputVec2[i] = static_cast<DataType>(i+1);
  }

  printf("\n");
  printf("hInputVec1: ");
  for (int i = 0; i < numElements; i++) {
    printf("%.3f ", hInputVec1[i]);
  }
  printf("\n\n");

  printf("hInputVec2: ");
  for (int i = 0; i < numElements; i++) {
    printf("%.3f ", hInputVec2[i]);
  }
  printf("\n\n");

  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CUstream_flags_enum::CU_STREAM_NON_BLOCKING));
  std::unique_ptr<RuntimeCtx> runtimeCtx(new RuntimeCtx());
  runtimeCtx->stream = stream;

  std::size_t numBytes = numElements * sizeof(DataType);

  // Inputs
  auto devPtr1 = cuda::copyH2DAsync(hInputVec1.data(), numBytes, stream);
  if (!devPtr1.get()) {
    return EXIT_FAILURE;
  }

  auto devPtr2 = cuda::copyH2DAsync(hInputVec2.data(), numBytes, stream);
  if (!devPtr2.get()) {
    return EXIT_FAILURE;
  }

  // Outputs
  cuda::DevicePtr devOutput = cuda::DevicePtr::alloc(numBytes);

  printf("\n");
  LOG_INFO("Call function %s", kAddTwoVectors);

  // Function arguments:
  //   - arg0: An additional pointer to RuntimeCtx object which holds a CUstream pointer
  // The arg1 ~ arg4 are matched with the original function parameters
  //   - arg1: The pointer to the first device vector
  //   - arg2: The pointer to the second device vector
  //   - arg3: The pointer to the result device vector
  //   - arg4: The number of elements
  bool ok = modMgr.call(kAddTwoVectors,
                        /*arg0*/reinterpret_cast<void*>(runtimeCtx.get()),
                        /*arg1*/reinterpret_cast<void*>(devPtr1.get()),
                        /*arg2*/reinterpret_cast<void*>(devPtr2.get()),
                        /*arg3*/reinterpret_cast<void*>(devOutput.get()),
                        /*arg4*/numElements);

  if (!ok) {
    return EXIT_FAILURE;
  }

  CUDA_CHECK(cuMemcpyDtoHAsync(hOutputVec.data(), devOutput.get(), numBytes, stream));
  CUDA_CHECK(cuStreamSynchronize(stream));

  // Print output
  printf("\n");
  printf("hOutputVec: ");
  for (int i = 0; i < numElements; i++) {
    printf("%.3f ", hOutputVec[i]);
  }
  printf("\n\n");


  return 0;
}
