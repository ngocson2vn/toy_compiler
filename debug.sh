#!/bin/bash

set -e

# Required
export CUDA_ROOT=/usr/local/cuda-12.4

# Compile source file to an object file with embedded CUBIN
# - arg0: the path to a .toy source file
# - arg1: the sm architecture
# ./output/bin/compiler -debug ./add_two_vectors.toy sm_86
# ./output/bin/compiler ./add_two_vectors.toy sm_86

./output/bin/compiler ./pointwise_ops.toy #--debug
