import csv
import getopt
import math
import re
import subprocess
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm


class P2PCudaMemcpyCommMatrixGenerator():

    def __init__(self, num_devices):
        # Needed headers for P2P memcpy
        self.headers = ['SrcMemType', 'DstMemType', 'Size', 'Src Dev', 'Dst Dev', 'Name', 'Device']
        self.num_bytes_comm_matrix = [[0] * num_devices for _ in range(num_devices)]
        self.num_times_comm_matrix = [[0] * num_devices for _ in range(num_devices)]
    
    def has_all_headers(self, line):
        for header in self.headers:
            if not re.search(header, line):
                return False
        return True

    def get_indices_of_headers(self, line):
        name_to_index = {}
        for header in self.headers:
            name_to_index[header] = line.index(header)
        return name_to_index

    def get_size_and_gpu_ids(self, splitted_line, name_to_index, num_of_elems):
        size, src_index, dst_index = None, None, None
        if len(splitted_line) == num_of_elems:
            src_mem_type = splitted_line[name_to_index['SrcMemType']]
            dst_mem_type = splitted_line[name_to_index['DstMemType']]
            name = splitted_line[name_to_index['Name']]
            
            if src_mem_type == "Device" and dst_mem_type == "Device":
                size = splitted_line[name_to_index['Size']]
                src_index = splitted_line[name_to_index['Src Dev']]
                dst_index = splitted_line[name_to_index['Dst Dev']]
            
            if "[CUDA memcpy DtoD]" in name:
                size = splitted_line[name_to_index['Size']]
                src_index = splitted_line[name_to_index['Device']]
                dst_index = splitted_line[name_to_index['Device']]

        return size, src_index, dst_index

    def get_num_of_elems(self, splitted_line):
        return len(splitted_line)

    def get_size_type(self, line, name_to_index):
        splitted_line = self._clean_and_split_line(line)
        size_type = splitted_line[name_to_index['Size']]
        return size_type
    
    def _clean_and_split_line(self, line):
        clean_line = line.replace('"', '')
        splitted_line = clean_line.split(',')
        return splitted_line

    def generate_comm_matrix(self, filepath):
        multiply_by = 1
        find_headers = True
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                stripped_line = line.strip()
                if find_headers:
                    if self.has_all_headers(stripped_line):
                        splitted_line = self._clean_and_split_line(stripped_line)
                        name_to_index = self.get_indices_of_headers(splitted_line)
                        num_of_elems = self.get_num_of_elems(splitted_line)
                        line = fp.readline()
                        size_type = self.get_size_type(line, name_to_index)
                        if size_type == "KB":
                            multiply_by = 1024
                        elif size_type == "MB":
                            multiply_by = 1024 * 1024
                        elif size_type == "GB":
                            multiply_by = 1024 * 1024 * 1024
                        find_headers = False
                else:
                    splitted_line = self._clean_and_split_line(stripped_line)
                    comm_size, src_dev, dst_dev = self.get_size_and_gpu_ids(splitted_line, name_to_index, num_of_elems)
                    if comm_size and src_dev and dst_dev:
                        src_id = int(re.findall('\((.*?)\)', src_dev)[0])
                        dst_id = int(re.findall('\((.*?)\)', dst_dev)[0])
                        self.num_bytes_comm_matrix[dst_id][src_id] += float(comm_size) * multiply_by
                        self.num_times_comm_matrix[dst_id][src_id] += 1.0
        return self.num_bytes_comm_matrix, self.num_times_comm_matrix


class P2PUnifiedMemoryCommMatrixGenerator():

    def __init__(self, num_devices):
        # Needed headers for UM memcpy
        self.headers = ['Device', 'Unified Memory', 'Name']
        self.num_bytes_comm_matrix = [[0] * num_devices for _ in range(num_devices)]
        self.num_times_comm_matrix = [[0] * num_devices for _ in range(num_devices)]
    
    def has_all_headers(self, line):
        for header in self.headers:
            if not re.search(header, line):
                return False
        return True

    def get_indices_of_headers(self, line):
        name_to_index = {}
        for header in self.headers:
            name_to_index[header] = line.index(header)
        return name_to_index

    def get_size_and_gpu_ids(self, splitted_line, name_to_index, num_of_elems):
        size, src_index, dst_index = None, None, None
        if len(splitted_line) == num_of_elems + 1:
            mem_transfer_type = splitted_line[name_to_index['Name'] + 1]
            if mem_transfer_type == "[Unified Memory Memcpy DtoD]": 
                size = splitted_line[name_to_index['Unified Memory'] + 1]
                src_index = splitted_line[name_to_index['Device']]
                dst_index = splitted_line[name_to_index['Device'] + 1]
        return size, src_index, dst_index

    def get_num_of_elems(self, splitted_line):
        return len(splitted_line)

    def get_size_type(self, line, name_to_index):
        splitted_line = self._clean_and_split_line(line)
        size_type = splitted_line[name_to_index['Unified Memory']]
        return size_type
    
    def _clean_and_split_line(self, line):
        clean_line = line.replace('"', '')
        splitted_line = clean_line.split(',')
        return splitted_line

    def generate_comm_matrix(self, filepath):
        multiply_by = 1
        find_headers = True
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                stripped_line = line.strip()
                if find_headers:
                    if self.has_all_headers(stripped_line):
                        splitted_line = self._clean_and_split_line(stripped_line)
                        name_to_index = self.get_indices_of_headers(splitted_line)
                        num_of_elems = self.get_num_of_elems(splitted_line)
                        line = fp.readline()
                        size_type = self.get_size_type(line, name_to_index)
                        if size_type == "KB":
                            multiply_by = 1024
                        elif size_type == "MB":
                            multiply_by = 1024 * 1024
                        elif size_type == "GB":
                            multiply_by = 1024 * 1024 * 1024
                        find_headers = False
                else:
                    splitted_line = self._clean_and_split_line(stripped_line)
                    comm_size, src_dev, dst_dev = self.get_size_and_gpu_ids(splitted_line, name_to_index, num_of_elems)
                    if comm_size:
                        src_id = int(re.findall('\((.*?)\)', src_dev)[0])
                        dst_id = int(re.findall('\((.*?)\)', dst_dev)[0])
                        self.num_bytes_comm_matrix[dst_id][src_id] += float(comm_size) * multiply_by
                        self.num_times_comm_matrix[dst_id][src_id] += 1.0
        return self.num_bytes_comm_matrix, self.num_times_comm_matrix

class H2DUnifiedMemoryCommMatrixGenerator():

    def __init__(self, num_devices):
        # Needed headers for UM memcpy
        self.headers = ['Device', 'Unified Memory', 'Name']
        self.num_bytes_comm_matrix = [[0] * (num_devices + 1) for _ in range(num_devices + 1)]
        self.num_times_comm_matrix = [[0] * (num_devices + 1) for _ in range(num_devices + 1)]
    
    def has_all_headers(self, line):
        for header in self.headers:
            if not re.search(header, line):
                return False
        return True

    def get_indices_of_headers(self, line):
        name_to_index = {}
        for header in self.headers:
            name_to_index[header] = line.index(header)
        return name_to_index

    def get_size_and_gpu_ids(self, splitted_line, name_to_index, num_of_elems):
        size, src_index, dst_index = None, None, None
        if len(splitted_line) == num_of_elems:
            mem_transfer_type = splitted_line[name_to_index['Name']]
            if mem_transfer_type == "[Unified Memory Memcpy HtoD]": 
                size = splitted_line[name_to_index['Unified Memory']]
                dst_index = splitted_line[name_to_index['Device']]
            elif mem_transfer_type == "[Unified Memory Memcpy DtoH]": 
                size = splitted_line[name_to_index['Unified Memory']]
                src_index = splitted_line[name_to_index['Device']]
        return size, src_index, dst_index

    def get_num_of_elems(self, splitted_line):
        return len(splitted_line)

    def get_size_type(self, line, name_to_index):
        splitted_line = self._clean_and_split_line(line)
        size_type = splitted_line[name_to_index['Unified Memory']]
        return size_type
    
    def _clean_and_split_line(self, line):
        clean_line = line.replace('"', '')
        splitted_line = clean_line.split(',')
        return splitted_line

    def generate_comm_matrix(self, filepath):
        multiply_by = 1
        find_headers = True
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                stripped_line = line.strip()
                if find_headers:
                    if self.has_all_headers(stripped_line):
                        splitted_line = self._clean_and_split_line(stripped_line)
                        name_to_index = self.get_indices_of_headers(splitted_line)
                        num_of_elems = self.get_num_of_elems(splitted_line)
                        line = fp.readline()
                        size_type = self.get_size_type(line, name_to_index)
                        if size_type == "KB":
                            multiply_by = 1024
                        elif size_type == "MB":
                            multiply_by = 1024 * 1024
                        elif size_type == "GB":
                            multiply_by = 1024 * 1024 * 1024
                        find_headers = False
                else:
                    splitted_line = self._clean_and_split_line(stripped_line)
                    comm_size, src_dev, dst_dev = self.get_size_and_gpu_ids(splitted_line, name_to_index, num_of_elems)
                    if comm_size:
                        if not src_dev and dst_dev:
                            dst_id = int(re.findall('\((.*?)\)', dst_dev)[0])
                            self.num_bytes_comm_matrix[dst_id + 1][0] += float(comm_size) * multiply_by
                            self.num_times_comm_matrix[dst_id + 1][0] += 1.0
                        elif src_dev and not dst_dev:
                            src_id = int(re.findall('\((.*?)\)', src_dev)[0])
                            self.num_bytes_comm_matrix[0][src_id + 1] += float(comm_size) * multiply_by
                            self.num_times_comm_matrix[0][src_id + 1] += 1.0
        return self.num_bytes_comm_matrix, self.num_times_comm_matrix


class H2DCudaMemcpyCommMatrixGenerator():

    def __init__(self, num_devices):
        # Needed headers for H2D memcpy
        self.headers = ['Device', 'Size', 'Name']
        self.num_bytes_comm_matrix = [[0] * (num_devices + 1) for _ in range(num_devices + 1)]
        self.num_times_comm_matrix = [[0] * (num_devices + 1) for _ in range(num_devices + 1)]
    
    def has_all_headers(self, line):
        for header in self.headers:
            if not re.search(header, line):
                return False
        return True

    def get_indices_of_headers(self, line):
        name_to_index = {}
        for header in self.headers:
            name_to_index[header] = line.index(header)
        return name_to_index

    def get_size_and_gpu_ids(self, splitted_line, name_to_index, num_of_elems):
        size, src_index, dst_index = None, None, None
        if len(splitted_line) == num_of_elems:
            mem_transfer_type = splitted_line[name_to_index['Name']]
            if mem_transfer_type == "[CUDA memcpy HtoD]": 
                size = splitted_line[name_to_index['Size']]
                dst_index = splitted_line[name_to_index['Device']]
            elif mem_transfer_type == "[CUDA memcpy DtoH]": 
                size = splitted_line[name_to_index['Size']]
                src_index = splitted_line[name_to_index['Device']]
        return size, src_index, dst_index

    def get_num_of_elems(self, splitted_line):
        return len(splitted_line)

    def get_size_type(self, line, name_to_index):
        splitted_line = self._clean_and_split_line(line)
        size_type = splitted_line[name_to_index['Size']]
        return size_type
    
    def _clean_and_split_line(self, line):
        clean_line = line.replace('"', '')
        splitted_line = clean_line.split(',')
        return splitted_line

    def generate_comm_matrix(self, filepath):
        multiply_by = 1
        find_headers = True
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                stripped_line = line.strip()
                if find_headers:
                    if self.has_all_headers(stripped_line):
                        splitted_line = self._clean_and_split_line(stripped_line)
                        name_to_index = self.get_indices_of_headers(splitted_line)
                        num_of_elems = self.get_num_of_elems(splitted_line)
                        line = fp.readline()
                        size_type = self.get_size_type(line, name_to_index)
                        if size_type == "KB":
                            multiply_by = 1024
                        elif size_type == "MB":
                            multiply_by = 1024 * 1024
                        elif size_type == "GB":
                            multiply_by = 1024 * 1024 * 1024
                        find_headers = False
                else:
                    splitted_line = self._clean_and_split_line(stripped_line)
                    comm_size, src_dev, dst_dev = self.get_size_and_gpu_ids(splitted_line, name_to_index, num_of_elems)
                    if comm_size:
                        if not src_dev and dst_dev:
                            dst_id = int(re.findall('\((.*?)\)', dst_dev)[0])
                            self.num_bytes_comm_matrix[dst_id + 1][0] += float(comm_size) * multiply_by
                            self.num_times_comm_matrix[dst_id + 1][0] += 1.0
                        elif src_dev and not dst_dev:
                            src_id = int(re.findall('\((.*?)\)', src_dev)[0])
                            self.num_bytes_comm_matrix[0][src_id + 1] += float(comm_size) * multiply_by
                            self.num_times_comm_matrix[0][src_id + 1] += 1.0
        return self.num_bytes_comm_matrix, self.num_times_comm_matrix

class NcclCommMatrixGenerator:
    def __init__(self, num_devices):
        # Needed headers for H2D memcpy
        self.headers = ['src', 'dst', 'size']
        self.num_bytes_comm_matrix = [[0] * num_devices for _ in range(num_devices)]
        self.num_times_comm_matrix = [[0] * num_devices for _ in range(num_devices)]

    def generate_comm_matrix(self, filepath_prefix="comscribe_nccl_*.csv"):
        file_paths = glob.glob(filepath_prefix)
        
        for file_path in file_paths:
            with open(file_path) as fp:
                lines = fp.readlines()

                for line in lines:
                    src, dst, size = line.split(",")
                    self.num_bytes_comm_matrix[int(dst)][int(src)] += int(size)
                    self.num_times_comm_matrix[int(dst)][int(src)] += 1
        
        return self.num_bytes_comm_matrix, self.num_times_comm_matrix

class ZeroCopyInfoGenerator():
    def __init__(self, num_devices):
        # Needed headers for zerocopy memory
        self.headers = [
            'Device', 'nvlink_user_data_received',
            'nvlink_user_data_transmitted', 'sysmem_read_bytes', 
            'sysmem_write_bytes']

        self.num_bytes_comm_matrix = [[0] * (4) for _ in range(num_devices)]
        self.num_times_comm_matrix = [[0] * (4) for _ in range(num_devices)]

    def has_all_headers(self, line):
        for header in self.headers:
            if not re.search(header, line):
                return False
        return True

    def get_indices_of_headers(self, line):
        name_to_index = {}
        for header in self.headers:
            name_to_index[header] = line.index(header)
        return name_to_index

    def get_size_and_gpu_ids(self, splitted_line, name_to_index, num_of_elems):
        device_id,nvlink_user_data_received,nvlink_user_data_transmitted,sysmem_read_bytes,sysmem_write_bytes = None, None, None, None, None
        if len(splitted_line) == num_of_elems + 1:
            device_id = splitted_line[name_to_index['Device']]
            nvlink_user_data_received = splitted_line[name_to_index['nvlink_user_data_received'] + 1]
            nvlink_user_data_transmitted = splitted_line[name_to_index['nvlink_user_data_transmitted'] + 1]
            sysmem_read_bytes = splitted_line[name_to_index['sysmem_read_bytes'] + 1]
            sysmem_write_bytes = splitted_line[name_to_index['sysmem_write_bytes'] + 1]
        return (device_id,nvlink_user_data_received,nvlink_user_data_transmitted,sysmem_read_bytes,sysmem_write_bytes)

    def get_num_of_elems(self, splitted_line):
        return len(splitted_line)
    
    def _clean_and_split_line(self, line):
        clean_line = line.replace('"', '')
        splitted_line = clean_line.split(',')
        return splitted_line

    def generate_comm_matrix(self, filepath):
        multiply_by = 1
        find_headers = True
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                stripped_line = line.strip()
                if find_headers:
                    if self.has_all_headers(stripped_line):
                        splitted_line = self._clean_and_split_line(stripped_line)
                        name_to_index = self.get_indices_of_headers(splitted_line)
                        num_of_elems = self.get_num_of_elems(splitted_line)
                        line = fp.readline()
                        find_headers = False
                else:
                    splitted_line = self._clean_and_split_line(stripped_line)
                    zerocopy_memory_info = self.get_size_and_gpu_ids(splitted_line, name_to_index, num_of_elems)
                    if not None in zerocopy_memory_info:
                        device = zerocopy_memory_info[0]
                        device_id = int(re.findall('\((.*?)\)', device)[0])
                        self.num_bytes_comm_matrix[device_id][0] += int(zerocopy_memory_info[1])
                        self.num_bytes_comm_matrix[device_id][1] += int(zerocopy_memory_info[2])
                        self.num_bytes_comm_matrix[device_id][2] += int(zerocopy_memory_info[3])
                        self.num_bytes_comm_matrix[device_id][3] += int(zerocopy_memory_info[4])
                        if int(zerocopy_memory_info[1]):
                            self.num_times_comm_matrix[device_id][0] += 1.0
                        if int(zerocopy_memory_info[2]):
                            self.num_times_comm_matrix[device_id][1] += 1.0
                        if int(zerocopy_memory_info[3]):
                            self.num_times_comm_matrix[device_id][2] += 1.0
                        if int(zerocopy_memory_info[4]):
                            self.num_times_comm_matrix[device_id][3] += 1.0
        return self.num_bytes_comm_matrix, self.num_times_comm_matrix

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
    num_devices = 0
    scale = ''
    is_nccl = False
    try:
        opts, args = getopt.getopt(argv,"h:i:g:s:n",["app=, num_gpus=, scale=, nccl="])
        print(opts)
    except getopt.GetoptError:
        print("comscribe.py -g <num_gpus> -i <'./app parameters'> -s <plotting_scale>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("comscribe.py -g <num_gpus> -i <'./app parameters'> -s <plotting_scale>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            application = arg
        elif opt in ("-g", "--num_gpus"):
            num_devices = int(arg)
        elif opt in ("-s", "--scale"):
            scale = arg
        if opt in ("-n", "--nccl"):
            is_nccl = True
    
    # # Run app with NCCL
    if(is_nccl):
        nccl_cmd = "{}".format(application)
        subprocess.run([nccl_cmd], shell=True)

        nccl_comm = NcclCommMatrixGenerator(num_devices)
        nccl_num_bytes_comm_matrix, nccl_num_times_comm_matrix = nccl_comm.generate_comm_matrix()

        print("Nccl Memory Bytes: \n", nccl_num_bytes_comm_matrix)
        print("Nccl Memory Transfers: \n", nccl_num_times_comm_matrix)

        outputfile_nccl_num_bytes_comm_matrix = "nccl_num_bytes_comm_matrix"
        outputfile_nccl_num_times_comm_matrix = "nccl_num_times_comm_matrix"

        plot_comm_matrix(nccl_num_bytes_comm_matrix, num_devices, outputfile_nccl_num_bytes_comm_matrix, scale)
        plot_comm_matrix(nccl_num_times_comm_matrix, num_devices, outputfile_nccl_num_times_comm_matrix, scale)
    else:
        # # Run app with GPU-Trace
        gpu_trace_cmd = "nvprof --print-gpu-trace --csv --log-file gpu_trace.csv {}".format(application)
        subprocess.run([gpu_trace_cmd], shell=True)
        gpu_trace_file = "gpu_trace.csv"

        # # Run app with Metric Trace
        metric_trace_cmd = "nvprof --print-gpu-trace --metrics nvlink_user_data_received,nvlink_user_data_transmitted,sysmem_read_bytes,sysmem_write_bytes --csv --log-file metric_trace.csv {}".format(application)
        subprocess.run([metric_trace_cmd], shell=True)

        metric_trace_file = "metric_trace.csv"
        
        # # Unified Memory
        h2d_um_memcpy_comm = H2DUnifiedMemoryCommMatrixGenerator(num_devices)
        h2d_um_num_bytes_comm_matrix, h2d_um_num_times_comm_matrix = h2d_um_memcpy_comm.generate_comm_matrix(gpu_trace_file)
        p2p_um_memcpy_comm = P2PUnifiedMemoryCommMatrixGenerator(num_devices)
        p2p_um_num_bytes_comm_matrix, p2p_um_num_times_comm_matrix = p2p_um_memcpy_comm.generate_comm_matrix(gpu_trace_file)
        
        all_um_num_bytes_comm_matrix = merge_matrices(h2d_um_num_bytes_comm_matrix, p2p_um_num_bytes_comm_matrix)
        all_um_num_times_comm_matrix = merge_matrices(h2d_um_num_times_comm_matrix, p2p_um_num_times_comm_matrix)
        
        if  max(map(max, all_um_num_bytes_comm_matrix)) != 0 and max(map(max, all_um_num_times_comm_matrix)) !=0:
            print("Unified Memory Bytes: \n", all_um_num_bytes_comm_matrix)
            print("Unified Memory Transfers: \n", all_um_num_times_comm_matrix)

            outputfile_um_num_bytes_comm_matrix = "um_num_bytes_comm_matrix"
            outputfile_um_num_times_comm_matrix = "um_num_times_comm_matrix"

            plot_comm_matrix(all_um_num_bytes_comm_matrix, num_devices, outputfile_um_num_bytes_comm_matrix, scale)
            plot_comm_matrix(all_um_num_times_comm_matrix, num_devices, outputfile_um_num_times_comm_matrix, scale)

        # # Explicit Transfers
        h2d_et_memcpy_comm = H2DCudaMemcpyCommMatrixGenerator(num_devices)
        h2d_et_num_bytes_comm_matrix, h2d_et_num_times_comm_matrix = h2d_et_memcpy_comm.generate_comm_matrix(gpu_trace_file)
        p2p_et_memcpy_comm = P2PCudaMemcpyCommMatrixGenerator(num_devices)
        p2p_et_num_bytes_comm_matrix, p2p_et_num_times_comm_matrix = p2p_et_memcpy_comm.generate_comm_matrix(gpu_trace_file)
        
        all_et_num_bytes_comm_matrix = merge_matrices(h2d_et_num_bytes_comm_matrix, p2p_et_num_bytes_comm_matrix)
        all_et_num_times_comm_matrix = merge_matrices(h2d_et_num_times_comm_matrix, p2p_et_num_times_comm_matrix)

        if max(map(max, all_et_num_bytes_comm_matrix)) != 0 and max(map(max, all_et_num_times_comm_matrix)) !=0:
            print("Explicit Transfers Bytes: \n", all_et_num_bytes_comm_matrix)
            print("Explicit Transfers Transfers: \n", all_et_num_times_comm_matrix)
            
            outputfile_et_num_bytes_comm_matrix = "et_num_bytes_comm_matrix"
            outputfile_et_num_times_comm_matrix = "et_num_times_comm_matrix"
        
            plot_comm_matrix(all_et_num_bytes_comm_matrix, num_devices, outputfile_et_num_bytes_comm_matrix, scale)
            plot_comm_matrix(all_et_num_times_comm_matrix, num_devices, outputfile_et_num_times_comm_matrix, scale)

        # # Zero-Copy Memory Transfers
        all_zc_comm = ZeroCopyInfoGenerator(num_devices)
        all_zc_num_bytes_comm_matrix, all_zc_num_times_comm_matrix = all_zc_comm.generate_comm_matrix(metric_trace_file)

        if max(map(max, all_zc_num_bytes_comm_matrix)) != 0 and max(map(max, all_zc_num_times_comm_matrix)) !=0:
            print("ZeroCopy Memory Bytes: \n", all_zc_num_bytes_comm_matrix)
            print("ZeroCopy Memory Transfers: \n", all_zc_num_times_comm_matrix)
            plot_bar_chart(all_zc_num_bytes_comm_matrix, num_devices)
            plot_bar_chart(all_zc_num_times_comm_matrix, num_devices)


if __name__ == "__main__":
    main(sys.argv[1:])
