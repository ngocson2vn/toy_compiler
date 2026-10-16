#!/bin/bash

set -e

export TOY_DUMP_LLVMIR=true

# Required
export CUDA_ROOT=/usr/local/cuda-12.4

# Remove old object file if it exists
rm -fv ${OBJECT_FILE}

# Compile source file to an object file with embedded CUBIN
# - arg0: the path to a .toy source file
# - arg1: the sm architecture
# ./output/bin/compiler -debug ./add_two_vectors.toy sm_86
# ./output/bin/compiler ./add_two_vectors.toy sm_86

./output/bin/compiler ./add_max_vectors.toy sm_86 false false #-debug

OBJECT_FILE=./output.o
OUTPUT_FILE=./libkernel.so
if readelf -s ./output.o | grep FUNC | grep -E ' main$' 2>&1>/dev/null; then
  OUTPUT_FILE=./main
fi

# Generate a shared object file from the generated object file
if [ -f ${OBJECT_FILE} ]; then
  g++ -L/lib/x86_64-linux-gnu ${OBJECT_FILE} -o ${OUTPUT_FILE} build/runtime/libruntime.a build/common/libcommon.a -lcuda
  echo
  echo Output: ${OUTPUT_FILE}
fi
