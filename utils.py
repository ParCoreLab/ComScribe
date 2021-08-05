import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ifile', help='input command as a string such as: -i "./app --foo 20 --bar 5"', required=True)
    parser.add_argument('-g', '--num-gpus', help='Number of gpus',type=int, required=True)
    parser.add_argument('-s', '--scale', help='"log" for log scale or "linear" for linear scale for the output figures', default='linear')
    parser.add_argument('-n', '--nccl', help='"log" for log scale or "linear" for linear scale for the output figures', action='store_true', default=False)
    parser.add_argument('-c', '--coll-type', help="Collective type that will be profiled. If not specified, all collectives will be profiled", default='*')
    return parser.parse_args()

def remove_existing_files(file_paths):
    if len(file_paths) > 0:
        for file in file_paths:
            os.remove(file)

def check_nccl(so_path):
    if not os.path.exists(so_path):
        sys.exit("ComScribe's modified NCCL shared library could not be found. Make sure to compile that first.")
