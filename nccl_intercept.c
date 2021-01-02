#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "nccl.h"

///////////////////////////////////////////////// REAL NCCL FUNCTIONS ///////////////////////////////////////////////// 
static ncclResult_t realNcclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t,  ncclComm_t, cudaStream_t);
	
	void *handle;
	char *error;
	char* NCCL_PATH = getenv("NCCL_PATH");
	
	handle = dlopen(NCCL_PATH, RTLD_LAZY | RTLD_GLOBAL);
        
	if (!handle) {
		printf("Cannot find the handle\n");
        }

        fn = dlsym(handle, "ncclAllReduce");
        if ((error = dlerror()) != NULL)  {
		printf("Cannot find the handle\n");
       	}
	
	return (*fn)(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

static ncclResult_t realNcclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
        
	void *handle;
	char *error;
	char* NCCL_PATH = getenv("NCCL_PATH");
	
	handle = dlopen(NCCL_PATH, RTLD_LAZY | RTLD_GLOBAL);
        
	if (!handle) {
		printf("Cannot find the handle\n");
        }

        fn = dlsym(handle, "ncclBroadcast");
        if ((error = dlerror()) != NULL)  {
		printf("Cannot find the handle\n");
       	}
	
        return (*fn)(sendbuff, recvbuff, count, datatype, root, comm, stream);	
}

static ncclResult_t realNcclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t);
	
	void *handle;
	char *error;
	char* NCCL_PATH = getenv("NCCL_PATH");
	
	handle = dlopen(NCCL_PATH, RTLD_LAZY | RTLD_GLOBAL);
        
	if (!handle) {
		printf("Cannot find the handle\n");
        }

        fn = dlsym(handle, "ncclReduce");
        if ((error = dlerror()) != NULL)  {
		printf("Cannot find the handle\n");
       	}

	return (*fn)(sendbuff, recvbuff, count, datatype, op, root, comm, stream);	
}

static ncclResult_t realNcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
        
	void *handle;
	char *error;
	char* NCCL_PATH = getenv("NCCL_PATH");
	
	handle = dlopen(NCCL_PATH, RTLD_LAZY | RTLD_GLOBAL);
        
	if (!handle) {
		printf("Cannot find the handle\n");
        }

        fn = dlsym(handle, "ncclAllGather");
        if ((error = dlerror()) != NULL)  {
		printf("Cannot find the handle\n");
       	}

        return (*fn)(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

static ncclResult_t realNcclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
	ncclResult_t (*fn)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
        
	void *handle;
	char *error;
	char* NCCL_PATH = getenv("NCCL_PATH");
	
	handle = dlopen(NCCL_PATH, RTLD_LAZY | RTLD_GLOBAL);
        
	if (!handle) {
		printf("Cannot find the handle\n");
        }

        fn = dlsym(handle, "ncclReduceScatter");
        if ((error = dlerror()) != NULL)  {
		printf("Cannot find the handle\n");
       	}
	
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
	struct timeval start, stop;
        gettimeofday(&start, NULL);
	ncclResult_t result = realNcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
	gettimeofday(&stop, NULL);

	int device;
        ncclCommCuDevice(comm, &device);

	if(result == ncclSuccess) {
	    if(device != 0) return result;
            
	    char* filename = "comscribe_nccl_allreduce.csv";
            FILE *fptr = fopen(filename,"a");
            int P; // Number of gpus in a communicator
            ncclCommCount(comm, &P);
	    int N = (int)sizeof(datatype) * (int)count;
            
	    for(int i = 0; i < P; i++) {
            	fprintf(fptr,"%d,%d,%d,%ld,%ld\n",i, (i + 1) % P, (2 * N * (P - 1)) / P, start.tv_usec, stop.tv_usec);
	    } 

            fclose(fptr);
        }else {
	    printf("NCCL failure\n");
	}
        return result;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream){
	struct timeval start, stop;
        gettimeofday(&start, NULL);
	ncclResult_t result = realNcclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
	gettimeofday(&stop, NULL);

	int device;
	ncclCommCuDevice(comm, &device);

        if(result == ncclSuccess) {
	    if(device != root) return result;

            char* filename = "comscribe_nccl_broadcast.csv";
            FILE *fptr = fopen(filename,"a");
            int P;
            ncclCommCount(comm, &P);
	    int N = (int)sizeof(datatype) * (int)count;
	    
	    for(int i = 0; i < P; i++) {
		if((i + 1) % P != root) {
                    fprintf(fptr,"%d,%d,%d,%ld,%ld\n", i, (i+1) % P, N, start.tv_usec, stop.tv_usec);
		}
            }   
            fclose(fptr);
        }else {
	    printf("NCCL failure\n");
	}
        return result;
}

/*
  if (prevRank == root) {
      prims.send(thisInput+offset, nelem);
  } else if (rank == root) {
      prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
  } else {
      prims.recvReduceSend(thisInput+offset, nelem);
  }
 */

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
	struct timeval start, stop;
        gettimeofday(&start, NULL);
	ncclResult_t result = realNcclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
	gettimeofday(&stop, NULL);

	int device;
	ncclCommCuDevice(comm, &device);
	
	if(result == ncclSuccess) {
	    if(device != root) return result;
	    char* filename = "comscribe_nccl_reduce.csv";
	    FILE *fptr = fopen(filename,"a");
	    int P;
	    ncclCommCount(comm, &P);
	    int N = (int)sizeof(datatype) * (int)count;

	    for(int i = 0; i < P; i++) {
		if(i != root) {
		    fprintf(fptr,"%d,%d,%d,%ld,%ld\n",i, (i + 1) % P, N / P, start.tv_usec, stop.tv_usec);
		}	
	    }	
	    fclose(fptr);
	}else {
	    printf("NCCL failure\n");
	}
	return result;
}

/*
 * Second step of AllReduce
 * Each gpu sends (N/P) elements and it does it (P-1) times
 */
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
	struct timeval start, stop;
        gettimeofday(&start, NULL);
	ncclResult_t result = realNcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
	gettimeofday(&stop, NULL);

	int device;
	ncclCommCuDevice(comm, &device);

        if(result == ncclSuccess) {
	    if(device != 0) return result;

            char* filename = "comscribe_nccl_allgather.csv";
            FILE *fptr = fopen(filename,"a");
            int P; // Number of gpus in a communicator
            ncclCommCount(comm, &P);
            int N = (int)sizeof(datatype) * (int)sendcount * P;
            
	    for(int i = 0; i < P; i++) {
            	fprintf(fptr,"%d,%d,%d,%ld,%ld\n",i, (i + 1) % P,  N * (P - 1) / P, start.tv_usec, stop.tv_usec);
	    }

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
	struct timeval start, stop;
        gettimeofday(&start, NULL);
	ncclResult_t result = realNcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
	gettimeofday(&stop, NULL);

	int device;
	ncclCommCuDevice(comm, &device);

        if(result == ncclSuccess) {
	    if(device != 0) return result;

            char* filename = "comscribe_nccl_reducescatter.csv";
            FILE *fptr = fopen(filename,"a");
            int P; // Number of gpus in a communicator
            ncclCommCount(comm, &P);
            int N = (int)sizeof(datatype) * (int)recvcount * P;

	    for(int i = 0; i < P; i++) {
            	fprintf(fptr,"%d,%d,%d,%ld,%ld\n",i, (i + 1) % P,  N * (P - 1) / P, start.tv_usec, stop.tv_usec);
	    }

            fclose(fptr);
        }else {
	    printf("NCCL failure\n");
	}
        return result;
}
