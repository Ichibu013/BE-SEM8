/*
Name - Ansh Shamra
BE A Computer
Roll No - 85

Problem Statement: Write a CUDA Program for:
1. Addition of two large vectors
2. Matrix Multiplication using CUDA C
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000000  // Vector size
#define TPB 256    // Threads per block

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_a, *h_b, *h_c;    // Host vectors
    float *d_a, *d_b, *d_c;    // Device vectors
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (N + TPB - 1) / TPB;
    vectorAdd<<<blocks, TPB>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < 5; i++) {
        printf("%.2f + %.2f = %.2f\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}