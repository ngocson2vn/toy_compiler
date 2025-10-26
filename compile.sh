#!/bin/bash

set -e

export TOY_DUMP_LLVMIR=true
export CUDA_ROOT=/usr/local/cuda-12.4

OBJECT_FILE=./output.o
SHARED_OBJECT_FILE=./libkernel.so

rm -fv ${OBJECT_FILE}
# output/bin/compiler ./add_two_tensors.toy -debug
./output/bin/compiler ./add_two_tensors.toy

if [ -f ${OBJECT_FILE} ]; then
  g++ -shared -fPIC ${OBJECT_FILE} -o ${SHARED_OBJECT_FILE}
  echo
  echo Output: ${SHARED_OBJECT_FILE}
fi
