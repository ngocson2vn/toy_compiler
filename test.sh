#!/bin/bash

set -e

# build/bin/test cubinFile kernelName numElements blockSize
# Note: blockSize MUST be equal to the value that was set at compile time
./output/bin/add_two_vectors ./libkernel.so 256
