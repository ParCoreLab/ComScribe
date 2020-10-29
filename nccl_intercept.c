#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "nccl.h"

///////////////////////////////////////////////// REAL NCCL FUNCTIONS ///////////////////////////////////////////////// 
ncclResult_t realNcclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t,  ncclComm_t, cudaStream_t);
        fn = dlsym(RTLD_NEXT, "ncclAllReduce");
        return (*fn)(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t realNcclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
        fn = dlsym(RTLD_NEXT, "ncclBroadcast");
        return (*fn)(sendbuff, recvbuff, count, datatype, root, comm, stream);	
}

ncclResult_t realNcclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t);
	fn = dlsym(RTLD_NEXT, "ncclReduce");
	return (*fn)(sendbuff, recvbuff, count, datatype, op, root, comm, stream);	
}

ncclResult_t realNcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
        fn = dlsym(RTLD_NEXT, "ncclAllGather");
        return (*fn)(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t realNcclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
        fn = dlsym(RTLD_NEXT, "ncclReduceScatter");
        return (*fn)(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}


///////////////////////////////////////////////// INTERCEPT FUNCTIONS ///////////////////////////////////////////////// 
/*
 *                   .--0--.
 *                  /       \
 *                 3         1
 *                  \       /
 *                   *--2--*
 *
 *  Each node always sends to the next clockwise node in the ring, and receives
 *  from the previous one. There are two stages:
 *  	1. Reduce-Scatter: Each gpu sends (N/P) elements and it does it (P-1) times.
 *  	2. All-Gather: Each gpu sends (N/P) elements and it does it (P-1) times
 *
 *  Total: Each gpu sends 2*((N/P)×(P-1)) bytes to the next gpu.
 */
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t result = realNcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
	
	int device;
        ncclCommCuDevice(comm, &device);

	if(result == ncclSuccess) {
            char* filename = "comscribe_nccl_allreduce.csv";
            FILE *fptr = fopen(filename,"a");
            int P; // Number of gpus in a communicator
            ncclCommCount(comm, &P);
	    int N = (int)sizeof(datatype) * (int)count;
            
            fprintf(fptr,"%d,%d,%d\n",device, (device + 1) % P, (2 * N * (P - 1)) / P );

            fclose(fptr);
        }else {
	    printf("NCCL failure\n");
	}
        return result;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream){
	ncclResult_t result = realNcclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
	
	int device;
	ncclCommCuDevice(comm, &device);

        if(result == ncclSuccess && device == root) {
            char* filename = "comscribe_nccl_broadcast.csv";
            FILE *fptr = fopen(filename,"a");
            int P;
            ncclCommCount(comm, &P);
	    int N = (int)sizeof(datatype) * (int)count;
	    
	    for(int i = 0; i < P; i++) {
		if((device + 1) % P != root) {
                    fprintf(fptr,"%d -> %d: %d bytes\n", i, (i+1) % P, N);
		}
            }   
            fprintf(fptr,"\n");

            fclose(fptr);
        }
        return result;
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t result = realNcclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
	
	if(result == ncclSuccess) {
	    char* filename = "comscribe_nccl_reduce.csv";
	    FILE *fptr = fopen(filename,"a");
	    int P;
	    ncclCommCount(comm, &P);
	    int N = (int)sizeof(datatype) * (int)count;

	    for(int i = 0; i < P; i++) {
		fprintf(fptr,"%d -> %d: %d bytes\n",i, (i + 1) % P, N / P);
	    }	
	    fprintf(fptr,"\n");
	
	    fclose(fptr);
	}
	return result;
}

/*
 * Second step of AllReduce
 * Each gpu sends (N/P) elements and it does it (P-1) times
 */
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t result = realNcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
	
	int device;
	ncclCommCuDevice(comm, &device);

        if(result == ncclSuccess) {
            char* filename = "comscribe_nccl_allgather.csv";
            FILE *fptr = fopen(filename,"a");
            int P; // Number of gpus in a communicator
            ncclCommCount(comm, &P);
            int N = (int)sizeof(datatype) * (int)sendcount * P;
            
            fprintf(fptr,"%d,%d,%d\n",device, (device + 1) % P,  N * (P - 1) / P );

            fclose(fptr);
        }else {
	    printf("NCCL failure\n");
	}
        return result;
}
/*
 * First step of AllReduce
 * Each gpu sends (N/P) elements and it does it (P-1) times
 */
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t result = realNcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
	
	int device;
	ncclCommCuDevice(comm, &device);

        if(result == ncclSuccess) {
            char* filename = "comscribe_nccl_reducescatter.csv";
            FILE *fptr = fopen(filename,"a");
            int P; // Number of gpus in a communicator
            ncclCommCount(comm, &P);
            int N = (int)sizeof(datatype) * (int)recvcount * P;

            fprintf(fptr,"%d,%d,%d\n",device, (device + 1) % P,  N * (P - 1) / P );

            fclose(fptr);
        }else {
	    printf("NCCL failure\n");
	}
        return result;
}
