/**
 * 1. Linear array allocation
 * 2. Global memory passing via pointer
 * 3. Host accessing global mem
 */
#include <stdio.h>
#include <stdlib.h>

void check_cuda_errors() {
    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess) {
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));
    }

}

__global__ void incrementor(int* numbers) {
    numbers[threadIdx.x]++;
}

int main(int argc, char** argv) {
    int* start, * device_mem;
    int i, num_elements;

    cudaError_t rc;

    printf("How many elements to increment? ");
    scanf("%d", &num_elements);

    srand(0);

    start = (int*)malloc(num_elements * sizeof(int));
    cudaMalloc((void**)&device_mem, num_elements * sizeof(int));

    printf("Increment input:\n");
    for (i = 0; i < num_elements; i++) {
        start[i] = rand() % 100;
        printf("start[%d] = %d\n", i, start[i]);
    }


    // copy value, and start cuda
    rc = cudaMemcpy(device_mem, start, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    if (rc != cudaSuccess) {
        printf("Could not copy to device. Reason: %s\n", cudaGetErrorString(rc));
    }

    incrementor <<<1, num_elements >>> (device_mem);
    check_cuda_errors();

    // Retrieve data from global memory
    rc = cudaMemcpy(start, device_mem, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    if (rc != cudaSuccess) {
        printf("Could not copy from device. Reason: %s\n", cudaGetErrorString(rc));
    }

    printf("Increment results:\n");
    for (i = 0; i < num_elements; i++) {
        printf("result[%d] = %d\n", i, start[i]);
    }

    free(start);
    cudaFree(device_mem);

    return 0;
}
