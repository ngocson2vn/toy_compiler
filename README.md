# Toy ML Compiler
Lowering Pipeline:<br/>
<img src="./docs/pipeline.png" width="100%"/>
<br/>

Runtime:<br/>
<img src="./docs/runtime.png" width="100%"/>

## How to build
```Bash
git submodule update --init --recursive
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
