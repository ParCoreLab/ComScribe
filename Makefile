CC=gcc
INC=/usr/local/cuda/include
OPT=-fPIC -shared

nccl_intercept: nccl_intercept.c
	$(CC) -I$(INC) $(OPT) $^ -o $@

.PHONY: clean
clean:
	rm nccl_intercept
