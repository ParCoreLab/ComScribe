#ifndef COMSCRIBE_LOG_H_
#define COMSCRIBE_LOG_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "info.h"
#include "comm.h"
#include "devcomm.h"

extern uint64_t comscribeCommId;

void logAllReduce(struct ncclInfo* info) {
  char hostname[HOST_NAME_MAX + 1];
  gethostname(hostname, HOST_NAME_MAX + 1);
  int rank = info->comm->rank; // my rank in the communicator
  int cudaDeviceIndex = info->comm->cudaDev;
  int nRanks = info->comm->nRanks;
  size_t N = info->nBytes;

  char filename[32];
  snprintf(filename, 32, "comscribe_allreduce_%d.csv", rank);
  FILE *fptr = fopen(filename,"a");
  
  if(info->algorithm == 0) { // Tree Algorithm
    int up = info->comm->channels[0].tree.up;
    
    if(up != -1) {
      fprintf(fptr,"%s,%lx,%d,%d,%d,%ld,Tree\n", hostname, comscribeCommId, cudaDeviceIndex, rank, up, N);
    }

    for(int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
      int down = info->comm->channels[0].tree.down[i];
      if(down != -1) {
        fprintf(fptr,"%s,%lx,%d,%d,%d,%ld,Tree\n", hostname, comscribeCommId, cudaDeviceIndex, rank, down, N);
      }
    }
  }else if(info->algorithm == 1) { // Ring Algorithm
    int next = info->comm->channels[0].ring.next;
    // Hostname, commId, cudaDeviceIndex, my rank, next rank, size
    fprintf(fptr,"%s,%lx,%d,%d,%d,%ld,Ring\n", hostname, comscribeCommId, cudaDeviceIndex, rank, next, (2 * N * (nRanks - 1)) / nRanks);
  }  
}

void logReduce(struct ncclInfo* info) {
  char hostname[HOST_NAME_MAX + 1];
  gethostname(hostname, HOST_NAME_MAX + 1);
  int rank = info->comm->rank; // my rank in the communicator
  int next = info->comm->channels[0].ring.userRanks[1];
  int cudaDeviceIndex = info->comm->cudaDev;
  size_t N = info->nBytes;
  int root = info->root;
  
  char filename[32];
  snprintf(filename, 32, "comscribe_reduce_%d.csv", rank);
  FILE *fptr = fopen(filename,"a");

  if(rank != root) {
    // Hostname, commId, cudaDeviceIndex, my rank, next rank, size
    fprintf(fptr,"%s,%lx,%d,%d,%d,%ld\n", hostname, comscribeCommId, cudaDeviceIndex, rank, next, N);
  }
  fclose(fptr);
}

void logBroadcast(struct ncclInfo* info) {
  char hostname[HOST_NAME_MAX + 1];
  gethostname(hostname, HOST_NAME_MAX + 1);
  int rank = info->comm->rank; // my rank in the communicator
  int next = info->comm->channels[0].ring.userRanks[1];
  int cudaDeviceIndex = info->comm->cudaDev;
  size_t N = info->nBytes;
  int root = info->root;

  char filename[32];
  snprintf(filename, 32, "comscribe_broadcast_%d.csv", rank);
  FILE *fptr = fopen(filename,"a");

  if(next != root) {
    fprintf(fptr,"%s,%lx,%d,%d,%d,%ld,Ring\n", hostname, comscribeCommId, cudaDeviceIndex, rank, next, N);
  }
  fclose(fptr);
}

void logReduceScatter(struct ncclInfo* info) {
  char hostname[HOST_NAME_MAX + 1];
  gethostname(hostname, HOST_NAME_MAX + 1);
  int rank = info->comm->rank; // my rank in the communicator
  int next = info->comm->channels[0].ring.userRanks[1];
  int cudaDeviceIndex = info->comm->cudaDev;
  size_t N = info->nBytes;
  int nRanks = info->comm->nRanks;

  char filename[32];
  snprintf(filename, 32, "comscribe_reducescatter_%d.csv", rank);
  FILE *fptr = fopen(filename,"a");

  fprintf(fptr,"%s,%lx,%d,%d,%d,%ld,Ring\n", hostname, comscribeCommId, cudaDeviceIndex, rank, next, N * (nRanks - 1) / nRanks);
  fclose(fptr);

}

void logAllGather(struct ncclInfo* info) {
  char hostname[HOST_NAME_MAX + 1];
  gethostname(hostname, HOST_NAME_MAX + 1);
  int rank = info->comm->rank; // my rank in the communicator
  int next = info->comm->channels[0].ring.userRanks[1];
  int cudaDeviceIndex = info->comm->cudaDev;
  size_t N = info->nBytes;
  int nRanks = info->comm->nRanks;

  char filename[32];
  snprintf(filename, 32, "comscribe_allgather_%d.csv", rank);
  FILE *fptr = fopen(filename,"a");

  fprintf(fptr,"%s,%lx,%d,%d,%d,%ld,Ring\n", hostname, comscribeCommId, cudaDeviceIndex, rank, next, N * (nRanks - 1) / nRanks);
  fclose(fptr);
}

void logSend(struct ncclInfo* info) {
  char hostname[HOST_NAME_MAX + 1];
  gethostname(hostname, HOST_NAME_MAX + 1);
  int rank = info->comm->rank; // my rank in the communicator
  int peer = info->root; // peer to send
  int cudaDeviceIndex = info->comm->cudaDev;
  size_t N = info->nBytes;

  char filename[32];
  snprintf(filename, 32, "comscribe_send_%d.csv", rank);
  FILE *fptr = fopen(filename,"a");

  fprintf(fptr,"%s,%lx,%d,%d,%d,%ld,Ring\n", hostname, comscribeCommId, cudaDeviceIndex, rank, peer, N);
  fclose(fptr);
}

void logInfo(struct ncclInfo* info) {
  if(strcmp(info->opName, "AllReduce") == 0) {
    logAllReduce(info);
  }else if(strcmp(info->opName, "Reduce") == 0) {
    logReduce(info);
  }else if(strcmp(info->opName, "Broadcast") == 0) {
    logBroadcast(info);
  }else if(strcmp(info->opName, "ReduceScatter") == 0) {
    logReduceScatter(info);
  }else if(strcmp(info->opName, "AllGather") == 0) {
    logAllGather(info);
  }else if(strcmp(info->opName, "Send") == 0) {
    logSend(info);
  }
}

#endif // End include guard
