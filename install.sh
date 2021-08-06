#!/bin/bash

pip3 install -r requirements.txt
cd nccl && make -j src.build && cd ..
