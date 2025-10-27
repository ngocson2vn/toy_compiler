#!/bin/bash

set -e

export TOY_DUMP_LLVMIR=true

# Required
export CUDA_ROOT=/usr/local/cuda-12.4

OBJECT_FILE=./output.o
SHARED_OBJECT_FILE=./libkernel.so

# Remove old object file if it exists
rm -fv ${OBJECT_FILE}

# Compile source file to an object file with embedded CUBIN
# - arg0: the path to a .toy source file
# - arg1: the sm architecture
./output/bin/compiler ./add_two_vectors.toy sm_86

# Generate a shared object file from the generated object file
if [ -f ${OBJECT_FILE} ]; then
  g++ -shared -fPIC ${OBJECT_FILE} -o ${SHARED_OBJECT_FILE}
  echo
  echo Output: ${SHARED_OBJECT_FILE}
fi
