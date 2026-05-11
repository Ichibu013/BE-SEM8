/*
Name - Ansh Shamra
BE A Computer
Roll No - 85

Problem Statement: Write a CUDA Program for:
1. Addition of two large vectors
2. Matrix Multiplication using CUDA C
*/

#include <stdio.h>
#include <stdlib.h>      // Required for malloc() and rand()
#include <cuda_runtime.h>

#define N 1024          // Matrix size NxN
#define BLOCK_SIZE 16   // Reduced to 16 for standard, safer block occupancy (16x16 = 256 threads)

// Macro to catch and print CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void matrixMul(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < n && col < n) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    float *h_a, *h_b, *h_c;    // Host matrices
    float *d_a, *d_b, *d_c;    // Device matrices
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory safely
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // Copy inputs to device safely
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Set up execution configuration
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    // Launch kernel
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // Explicitly check if the kernel failed to launch
    CUDA_CHECK(cudaGetLastError());

    // Wait for the GPU to finish to catch runtime execution errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host safely
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Print small portion of result
    printf("Sample output (top-left 2x2 corner of result):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%.2f ", h_c[i * N + j]);
        }
        printf("\n");
    }

    // Free memory safely
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}