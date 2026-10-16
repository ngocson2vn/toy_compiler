#!/bin/bash

set -e

export TOY_DUMP_LLVMIR=true

# Required
export CUDA_ROOT=/usr/local/cuda-12.4

OBJECT_FILE=./output.o

# Remove old object file if it exists
rm -fv ${OBJECT_FILE}

# Compile source file to an object file with embedded CUBIN
# - arg0: the path to a .toy source file
# - arg1: the sm architecture
# ./output/bin/compiler -debug ./add_two_vectors.toy sm_86
# ./output/bin/compiler ./add_two_vectors.toy sm_86
./output/bin/compiler ./add_two_vectors.toy #--verbose
