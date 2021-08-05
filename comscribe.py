import math
import subprocess
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from utils import remove_existing_files, parse_args, check_nccl
from generators import *

NCCL_SHARED_LIBRARY_PATH = "./nccl/build/lib/libnccl.so"

def plot_comm_matrix(comm_matrix, num_devices, matrix_type, scale='linear'):
    colormap = plt.cm.ocean_r
    plt.figure(figsize=(9, 7))
    # # For Linear Scale
    if scale == 'linear':
        formatter = tkr.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax = sns.heatmap(data=comm_matrix, cmap=colormap, linewidths=.5, cbar_kws={'format': formatter})
    else:
    # # For Log scale
        data = np.array(comm_matrix)
        log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        cbar_ticks = [math.pow(10, i) for i in range(math.floor(0), 1+math.ceil(math.log10(data.max().max())))]
        ax = sns.heatmap(data=data, cmap=colormap, linewidths=.5, norm=log_norm, cbar_kws={"ticks": cbar_ticks}, vmin = 0.1)

    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=12)
    cax.yaxis.offsetText.set(size=12)
    plt.gca().invert_yaxis()
    x1,x2,y1,y2 = plt.axis()
    labels = ['H',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    if matrix_type == "nccl_num_bytes_comm_matrix" or matrix_type == "nccl_num_times_comm_matrix":
        plt.xticks(np.arange(0.5, num_devices, 1), labels=labels[1:])
        plt.yticks(np.arange(0.5, num_devices, 1), labels=labels[1:])
    else:
        plt.xticks(np.arange(0.5, num_devices + 1, 1), labels=labels)
        plt.yticks(np.arange(0.5, num_devices + 1, 1), labels=labels)
    plt.xlabel("GPU IDs", size=24)
    plt.ylabel("GPU IDs", size=24)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig('{}.pdf'.format(matrix_type))

def plot_bar_chart(matrix, n_groups):
    plt.rcParams["figure.figsize"] = (8,6)
    zcm_read = []
    sysmem_write = []
    zcm_write = []
    sysmem_read = []
    for i in range(len(matrix)):
        zcm_read.append(matrix[i][0])
        zcm_write.append(matrix[i][1])
        sysmem_read.append(matrix[i][2])
        sysmem_write.append(matrix[i][3])

    fig, ax = plt.subplots(figsize=(8,6))
    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.8

    rects1 = plt.bar(index, zcm_read, bar_width,
    alpha=opacity,
    color='b',
    label='Zero-copy memory Read')

    rects2 = plt.bar(index + bar_width, zcm_write, bar_width,
    alpha=opacity,
    color='g',
    label='Zero-copy memory Write')

    rects3 = plt.bar(index+bar_width+bar_width, sysmem_read, bar_width,
    alpha=opacity,
    color='y',
    label='System Memory Read')

    rects4 = plt.bar(index + bar_width+bar_width+bar_width, sysmem_write, bar_width,
    alpha=opacity,
    color='r',
    label='System Memory Write')

    y_ticks = tuple(list(range(n_groups)))

    plt.legend(prop={"size":20}, loc=1)
    plt.yscale('log')
    plt.xticks(index + bar_width, y_ticks, fontsize=24)
    plt.xlabel('GPU IDs', size=24)
    fig.text(0.015, 0.5, 'Number of bytes Per GPU-Pair', va='center', ha='center', rotation='vertical', fontsize=21)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    plt.tight_layout()
    plt.savefig('zcm_bar_chart.pdf')

def merge_matrices(h2d_comm_matrix, p2p_comm_matrix):
    for x in range(0, len(p2p_comm_matrix)):
        for y in range(0, len(p2p_comm_matrix)):
            h2d_comm_matrix[x + 1][y + 1] = p2p_comm_matrix[x][y]

    return h2d_comm_matrix

def main(argv):
    args = parse_args()

    # # Run app with NCCL
    if(args.nccl):
        check_nccl(NCCL_SHARED_LIBRARY_PATH)
        preload = f'LD_PRELOAD={NCCL_SHARED_LIBRARY_PATH}'

        file_regex = f'comscribe_{args.coll_type}_*.csv'
        file_paths = glob.glob(file_regex)
        remove_existing_files(file_paths)
        nccl_cmd = f'{preload} {args.ifile}'
        subprocess.run([nccl_cmd], shell=True)

        nccl_comm = NcclCommMatrixGenerator(args.num_gpus)
        nccl_num_bytes_comm_matrix, nccl_num_times_comm_matrix = nccl_comm.generate_comm_matrix(filepath_prefix=file_regex)

        print("Nccl Memory Bytes: \n", nccl_num_bytes_comm_matrix)
        print("Nccl Memory Transfers: \n", nccl_num_times_comm_matrix)

        outputfile_nccl_num_bytes_comm_matrix = "nccl_num_bytes_comm_matrix"
        outputfile_nccl_num_times_comm_matrix = "nccl_num_times_comm_matrix"

        plot_comm_matrix(nccl_num_bytes_comm_matrix, args.num_gpus, outputfile_nccl_num_bytes_comm_matrix, args.scale)
        plot_comm_matrix(nccl_num_times_comm_matrix, args.num_gpus, outputfile_nccl_num_times_comm_matrix, args.scale)

    # # Run app with GPU-Trace
    gpu_trace_cmd = "nvprof --print-gpu-trace --csv --log-file gpu_trace.csv {}".format(args.ifile)
    subprocess.run([gpu_trace_cmd], shell=True)
    gpu_trace_file = "gpu_trace.csv"

    # # Run app with Metric Trace
    metric_trace_cmd = "nvprof --print-gpu-trace --metrics nvlink_user_data_received,nvlink_user_data_transmitted,sysmem_read_bytes,sysmem_write_bytes --csv --log-file metric_trace.csv {}".format(args.ifile)
    subprocess.run([metric_trace_cmd], shell=True)

    metric_trace_file = "metric_trace.csv"

    # # Unified Memory
    h2d_um_memcpy_comm = H2DUnifiedMemoryCommMatrixGenerator(args.num_gpus)
    h2d_um_num_bytes_comm_matrix, h2d_um_num_times_comm_matrix = h2d_um_memcpy_comm.generate_comm_matrix(gpu_trace_file)
    p2p_um_memcpy_comm = P2PUnifiedMemoryCommMatrixGenerator(args.num_gpus)
    p2p_um_num_bytes_comm_matrix, p2p_um_num_times_comm_matrix = p2p_um_memcpy_comm.generate_comm_matrix(gpu_trace_file)

    all_um_num_bytes_comm_matrix = merge_matrices(h2d_um_num_bytes_comm_matrix, p2p_um_num_bytes_comm_matrix)
    all_um_num_times_comm_matrix = merge_matrices(h2d_um_num_times_comm_matrix, p2p_um_num_times_comm_matrix)

    if  max(map(max, all_um_num_bytes_comm_matrix)) != 0 and max(map(max, all_um_num_times_comm_matrix)) !=0:
        print("Unified Memory Bytes: \n", all_um_num_bytes_comm_matrix)
        print("Unified Memory Transfers: \n", all_um_num_times_comm_matrix)

        outputfile_um_num_bytes_comm_matrix = "um_num_bytes_comm_matrix"
        outputfile_um_num_times_comm_matrix = "um_num_times_comm_matrix"

        plot_comm_matrix(all_um_num_bytes_comm_matrix, args.num_gpus, outputfile_um_num_bytes_comm_matrix, args.scale)
        plot_comm_matrix(all_um_num_times_comm_matrix, args.num_gpus, outputfile_um_num_times_comm_matrix, args.scale)

    # # Explicit Transfers
    h2d_et_memcpy_comm = H2DCudaMemcpyCommMatrixGenerator(args.num_gpus)
    h2d_et_num_bytes_comm_matrix, h2d_et_num_times_comm_matrix = h2d_et_memcpy_comm.generate_comm_matrix(gpu_trace_file)
    p2p_et_memcpy_comm = P2PCudaMemcpyCommMatrixGenerator(args.num_gpus)
    p2p_et_num_bytes_comm_matrix, p2p_et_num_times_comm_matrix = p2p_et_memcpy_comm.generate_comm_matrix(gpu_trace_file)

    all_et_num_bytes_comm_matrix = merge_matrices(h2d_et_num_bytes_comm_matrix, p2p_et_num_bytes_comm_matrix)
    all_et_num_times_comm_matrix = merge_matrices(h2d_et_num_times_comm_matrix, p2p_et_num_times_comm_matrix)

    if max(map(max, all_et_num_bytes_comm_matrix)) != 0 and max(map(max, all_et_num_times_comm_matrix)) !=0:
        print("Explicit Transfers Bytes: \n", all_et_num_bytes_comm_matrix)
        print("Explicit Transfers Transfers: \n", all_et_num_times_comm_matrix)

        outputfile_et_num_bytes_comm_matrix = "et_num_bytes_comm_matrix"
        outputfile_et_num_times_comm_matrix = "et_num_times_comm_matrix"

        plot_comm_matrix(all_et_num_bytes_comm_matrix, args.num_gpus, outputfile_et_num_bytes_comm_matrix, args.scale)
        plot_comm_matrix(all_et_num_times_comm_matrix, args.num_gpus, outputfile_et_num_times_comm_matrix, args.scale)

    # # Zero-Copy Memory Transfers
    all_zc_comm = ZeroCopyInfoGenerator(args.num_gpus)
    all_zc_num_bytes_comm_matrix, all_zc_num_times_comm_matrix = all_zc_comm.generate_comm_matrix(metric_trace_file)

    if max(map(max, all_zc_num_bytes_comm_matrix)) != 0 and max(map(max, all_zc_num_times_comm_matrix)) !=0:
        print("ZeroCopy Memory Bytes: \n", all_zc_num_bytes_comm_matrix)
        print("ZeroCopy Memory Transfers: \n", all_zc_num_times_comm_matrix)
        plot_bar_chart(all_zc_num_bytes_comm_matrix, args.num_gpus)
        plot_bar_chart(all_zc_num_times_comm_matrix, args.num_gpus)

        # # Intra-node Memory Transfers
        outputfile_intra_node_num_bytes_comm_matrix = "intra_node_num_bytes_comm_matrix"
        outputfile_intra_node_num_times_comm_matrix = "intra_node_num_times_comm_matrix"
        all_intra_node_num_bytes_comm_matrix = merge_matrices_for_intranode(all_et_num_bytes_comm_matrix, nccl_num_bytes_comm_matrix)
        all_intra_node_num_transfers_comm_matrix = merge_matrices_for_intranode(all_et_num_times_comm_matrix, nccl_num_times_comm_matrix)
        if max(map(max, all_intra_node_num_bytes_comm_matrix)) != 0 and max(map(max, all_intra_node_num_transfers_comm_matrix)) !=0:
            print("Intra-node Memory Bytes: \n", all_intra_node_num_bytes_comm_matrix)
            print("Intra-node Memory Transfers: \n", all_intra_node_num_transfers_comm_matrix)
            plot_comm_matrix(all_intra_node_num_bytes_comm_matrix, num_devices, outputfile_intra_node_num_bytes_comm_matrix, scale)
            plot_comm_matrix(all_intra_node_num_transfers_comm_matrix, num_devices, outputfile_intra_node_num_times_comm_matrix, scale)


if __name__ == "__main__":
    main(sys.argv[1:])
