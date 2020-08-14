# ComScribe

<p align="center">
  <img src="/transformer.PNG" width="320">
</p>


***ComScribe*** is a tool that identifies communication among all GPU-GPU and CPU-GPU pairs in a single-node multi-GPU system.

- [Usage](##Usage) 
- [Installation](##Installation)
- [Benchmarks](##Benchmarks)
- [Publication](##Publication)
   
   
## Usage

to do

## Installation

You will need the following programs:

- [Python](https://www.python.org/): ComScribe is a Python script. It uses several packages listed in [`requirements.txt`](/requirements.txt), which you can install via the command:

`pip install requirements.txt`

- [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview): ComScribe parses the outputs of NVIDIA's profiler *nvprof*, which is a light-weight command-line profiler available since CUDA 5.

No further installation is required.

## Benchmarks
We have used our tool with the following benchmarks:
* NVIDIA Monte Carlo Simluation of 2D Ising-GPU | [GitHub](https://github.com/NVIDIA/ising-gpu/tree/master/optimized)
* NVIDIA Multi-GPU Jacobi Solver | [GitHub](https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/single_threaded_copy/jacobi.cu)
* Comm|Scope | [Paper](https://dl.acm.org/doi/10.1145/3297663.3310299) | [GitHub](https://github.com/c3sr/comm_scope)
    * Full-Duplex | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/cudaMemcpyAsync-duplex/gpu_gpu_peer.cpp)
    * Full-Duplex with Unified Memory | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/demand-duplex/gpu_gpu.cu)
    * Half-Duplex with peer access | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/cudaMemcpyPeerAsync/gpu_to_gpu_peer.cpp)
    * Half-Duplex without peer access | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/cudaMemcpyPeerAsync/gpu_to_gpu.cpp)
    * Zero-copy Memory (both Read and Write benchmarks) | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/zerocopy/gpu_to_gpu.cu)
* MGBench | [Github](https://github.com/tbennun/mgbench)
    * Full-Duplex | [GitHub](https://github.com/tbennun/mgbench/blob/master/src/L1/fullduplex.cpp)
    * Scatter-Gather | [GitHub](https://github.com/tbennun/mgbench/blob/master/src/L1/scatter.cpp)
    * Game Of Life | [GitHub](https://github.com/tbennun/mgbench/blob/master/src/L2/gol/main.cpp)
* Eidetic 3D LSTM | [Paper](https://openreview.net/forum?id=B1lKS2AqtX) | [GitHub](https://github.com/google/e3d_lstm )
* Transformer | [Paper](http://arxiv.org/abs/1706.03762) | [GitHub](https://github.com/tensorflow/tensor2tensor/)


## Publication:
To be published as: _Akthar, P., Tezcan, E., Qararyah, F.Q., and Unat, D. "ComScribe: Identifying Intra-node GPU Communication"_
