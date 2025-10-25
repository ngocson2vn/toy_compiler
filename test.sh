#!/bin/bash

set -e

# build/bin/test cubinFile kernelName numElements blockSize
# Note: blockSize MUST be equal to the value that was set at compile time
build/test ./libkernel.so 256
