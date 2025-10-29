# Toy ML Compiler
What it can do is compile a simple [add_two_vectors](./add_two_vectors.toy) function, written in a toy DSL, into a real CUDA kernel that can be launched on an NVIDIA GPU.
<br/>

Lowering Pipeline:<br/>
<img src="./docs/pipeline.png" width="100%"/>
<br/>

Runtime:<br/>
<img src="./docs/runtime.png" width="100%"/>

## How to build
**Prerequisites**<br/>
```
1. CUDA Toolkit version >= 12.4
2. CMake version >= 3.31.6
3. Clang version >= 17
```
<br/>

**Build commands**
```Bash
git submodule update --init --recursive
export CUDA_ROOT=/usr/local/cuda-12.4
make
```
Output: ./build/compiler

## Compile add_two_tensors.toy
```Bash
./compile.sh
```
Output: ./libkernel.so

## Launch generated CUDA kernel
```Bash
./test.sh
```

## Discussion Channel
If you are interested in this toy project, feel free to reach out to me on LinkedIn.<br/>
My LinkedIn: https://www.linkedin.com/in/ngocson2vn/
