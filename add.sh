#!/bin/bash

set -e

# Adds two 256-element vectors.
# Note: ./libkernel.so was generated from the source file add_two_vectors.toy
./output/bin/add_two_vectors ./libkernel.so 256
