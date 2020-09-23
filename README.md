# ComScribe

<p align="center">
  <img src="/example_outputs/transformer.PNG" width="320">
</p>


***ComScribe*** is a tool that identifies communication among all GPU-GPU and CPU-GPU pairs in a single-node multi-GPU system.

- [Installation](#installation)
- [Usage](#usage) 
- [Benchmarks](#benchmarks)
- [Publication](#publication)
   
## Installation

You will need the following programs:

- [Python](https://www.python.org/): ComScribe is a Python script. It uses several packages listed in [`requirements.txt`](/requirements.txt), which you can install via the command:

`pip3 install -r requirements.txt`

- [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview): ComScribe parses the outputs of NVIDIA's profiler *nvprof*, which is a light-weight command-line profiler available since CUDA 5.

No further installation is required.

## Usage

To obtain the communication matrices of your application (`app`):

1. Put `comscribe.py` in the same directory with `app`
2. `python3 comscribe.py -g <num_gpus> -s log|linear -i <cmd_to_run>`
    1. `-g` lets our tool know how many GPUs will be used, however note that if the application to be run requires such a parameter too, it must be explicitly specified (see `-i` below).
    2. `-s` can be `log` for log scale or `linear` for linear scale for the output figures.
    3. `-i` takes the input command as a string such as: `-i './app --foo 20 --bar 5'`
    
3. The communication matrix for a communication type is only generated if it is detected, e.g. if there are no Unified Memory transfers then there will not be any output regarding Unified Memory transfers. For the types of communication detected, the generated figures are saved as PDF files in the directory of the script.

## Benchmarks
We have used our tool in an NVIDIA V100 DGX2 system with up to 16 GPUs using CUDA v10.0.130 for the following benchmarks:
* NVIDIA Monte Carlo Simluation of 2D Ising-GPU | [GitHub](https://github.com/NVIDIA/ising-gpu/tree/master/optimized)
* NVIDIA Multi-GPU Jacobi Solver | [GitHub](https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/single_threaded_copy/jacobi.cu)
* Comm|Scope | [Paper](https://dl.acm.org/doi/10.1145/3297663.3310299) | [GitHub](https://github.com/c3sr/comm_scope)
    * Full-Duplex | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/cudaMemcpyAsync-duplex/gpu_gpu_peer.cpp)
    * Full-Duplex with Unified Memory | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/demand-duplex/gpu_gpu.cu)
    * Half-Duplex with peer access | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/cudaMemcpyPeerAsync/gpu_to_gpu_peer.cpp)
    * Half-Duplex without peer access | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/cudaMemcpyPeerAsync/gpu_to_gpu.cpp)
    * Zero-copy Memory (both Read and Write benchmarks) | [GitHub](https://github.com/c3sr/comm_scope/blob/master/src/zerocopy/gpu_to_gpu.cu)
    
    Note: In order to run a Comm|Scope benchmark with fixed iterations e.g. 100, in the source code of benchmark, replace it's registration with:
    ```
    benchmark::RegisterBenchmark(...)->SMALL_ARGS()->Iterations(100);
    ```
* MGBench | [Github](https://github.com/tbennun/mgbench)
    * Full-Duplex | [GitHub](https://github.com/tbennun/mgbench/blob/master/src/L1/fullduplex.cpp)
    * Scatter-Gather | [GitHub](https://github.com/tbennun/mgbench/blob/master/src/L1/scatter.cpp)
    * Game Of Life | [GitHub](https://github.com/tbennun/mgbench/blob/master/src/L2/gol/main.cpp)
* Eidetic 3D LSTM | [Paper](https://openreview.net/forum?id=B1lKS2AqtX) | [GitHub](https://github.com/google/e3d_lstm )
* Transformer | [Paper](http://arxiv.org/abs/1706.03762) | [GitHub](https://github.com/tensorflow/tensor2tensor/)

### Example: _Comm|Scope Zero-copy Memory Read Half-Duplex Micro-benchmark_
`python3 comscribe.py -g 4 -i './scope --benchmark_filter="Comm_ZeroCopy_GPUToGPU_Read.*18.*" -n 0' -s log`

Gives the bar-chart for Zero-copy memory transfers:
<p align="center">
  <img src="/example_outputs/commscope_zcm_read.png" width="320">
</p>

### Example: _Comm|Scope Unified Memory Full Duplex Micro-benchmark_
`python3 comscribe.py -g 4 -i './scope --benchmark_filter="Comm_Demand_Duplex_GPUGPU.*18.*"' -s linear`

Gives two matrices, bytes transferred (left) and number of transfers made (right):
<p align="center" float="left">
  <img src="/example_outputs/um_num_bytes.png" width="320" />
  <img src="/example_outputs/um_num_times.png" width="320" />
</p>

### Example: _MGBench Full Duplex Micro-benchmark_
`python3 comscribe.py -g 4 -i './fullduplex' -s linear`

Gives two matrices, bytes transferred (left) and number of transfers made (right):
<p align="center" float="left">
  <img src="/example_outputs/mgbench_full_duplex_bytes.png" width="320" />
  <img src="/example_outputs/mgbench_full_duplex_times.png" width="320" />
</p>

### Example: _NVIDIA Monte Carlo Simluation of 2D Ising-GPU_
`python3 comscribe.py -g 4 -i './cuIsing -y 32768 -x 65536 -n 128 -p 16 -d 4 -t 1.5' -s log`

Gives two matrices, bytes transferred (left) and number of transfers made (right):
<p align="center" float="left">
  <img src="/example_outputs/ising_gpu_num_bytes.png" width="320" />
  <img src="/example_outputs/ising_gpu_num_times.png" width="320" />
</p>


## Publication:
To be published as: _Akthar, P., Tezcan, E., Qararyah, F.M., and Unat, D. "ComScribe: Identifying Intra-node GPU Communication"_
