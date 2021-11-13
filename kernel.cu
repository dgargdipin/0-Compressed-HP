#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <vector> //for vector  
#include <algorithm> //for generate
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <cub/cub.cuh>
#include <iostream>
#include <stdio.h>
#include <windows.h> //winapi header  

using namespace std;


// method used while debugging for printing an array
__global__ void printArray(int* a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
}
void debugArray(char a[], int* arr, int n) {
    printf("DEBUGGING %s\n", a);
    printArray << <1, 1 >> > (arr, n);
}

// creating a histogram 
__global__ void createHistogram(int* a, int* h, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int pos = a[tid];
    atomicAdd(&h[pos], 1);

}

//compressing an array with 0s, will add to C only if a[tid] > 0
__global__ void compressA01P(int M, int* a, int* a01p, int* c, int* ic)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    if (a[tid])
    {
        int x = a01p[tid];
        int y = a[tid];
        __syncthreads();
        c[x] = y;
        ic[x] = tid;
    }
}

// expanding the c arrays to array B
__global__ void expandToB(int len, int* cp, int* ic, int* b)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) return;

    int x = cp[tid];
    __syncthreads();
    b[x] = ic[tid + 1] - ic[tid];
}

// method used to binarize the array a
__global__ void binarize(int n, int* cnt, int* cnt01)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    cnt01[tid] = (cnt[tid] > 0) ? 1 : 0;
}

// function created to get the prefix-sum of an array using the inclusive scan method
void prefix_sum_on_gpu(int* data, int* output, int size) {
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, data, output, size);
    // Allocate temporary storage for inclusive prefix sum
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, data, output, size);
    //printf("Successfully prefixed sum");
}

int* cubsort(int* d_x, int N)
{
    // Declare, allocate, and initialize device-accessible pointers for sorting data
    int  num_items = N;          // e.g., 7
    int* d_keys_in = d_x;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    int* d_keys_out;        // e.g., [        ...        ]
    cudaMalloc(&d_keys_out, sizeof(int) * N);
    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
    // d_keys_out            <-- [0, 3, 5, 6, 7, 8, 9]
    return d_keys_out;
}
void test_cubsort(int* d_x, int N) {
    float milliseconds = 0;
  
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int* d_y = cubsort(d_x, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Took %f milliseconds using cubsort\n", milliseconds);
    cudaFree(d_y);
}

int main()
{
    //N = number of elements
    //M = maximum element + 1 (histogram size)
    int N, M;
    cin >> N >> M;
    float milliseconds = 0;
    size_t bytesN = sizeof(int) * N;
    size_t bytesM = sizeof(int) * M;

    vector<int> x(N), s(N);
    generate(x.begin(), x.end(), [M]() {return rand() % M; });

    int* d_x, * d_a, * d_b, * d_a01, * d_a01p, * d_c, * d_cp, * d_ic, * d_y;

    /*------------------------START OF THE GPU COMPUTATION--------------------------*/

    // x is input arr
    // d_x is copy of x on gpu
    cudaMalloc(&d_x, bytesN);
    cudaMemcpy(d_x, x.data(), bytesN, cudaMemcpyHostToDevice);
    //start of benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int numThreads = 2;
    int numBlocks = (N + numThreads - 1) / numThreads;
    cudaMalloc(&d_a, bytesM);
    //d_a is histogram of d_x 
    createHistogram << <numThreads, numBlocks >> > (d_x, d_a, N);

    //d_a01 is the binarized form of d_a
    cudaMalloc(&d_a01, bytesM);
    numBlocks = (M + numThreads - 1) / numThreads;
    binarize << <numThreads, numBlocks >> > (M, d_a, d_a01);

    //debugArray("binarized d_a", d_a01, M);
    //da01p is prefix sum of binarized da01
    cudaMalloc(&d_a01p, bytesM);
    prefix_sum_on_gpu(d_a01, d_a01p, M);


    int* len_ptr = new int;
    cudaMemcpy(len_ptr, d_a01p + M - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int len = *len_ptr;
    size_t bytesL = sizeof(int) * (len + 1);

    //d_c is the compressed form of d_a, d_ic is used to store the indices when compressing
    cudaMalloc(&d_c, bytesL);
    cudaMalloc(&d_ic, bytesL);

    numBlocks = (M + numThreads - 1) / numThreads;
    compressA01P << <numThreads, numBlocks >> > (M, d_a, d_a01p, d_c, d_ic);

    //d_cp is the prefix-sum of d_c
    cudaMalloc(&d_cp, bytesL);
    prefix_sum_on_gpu(d_c + 1, d_cp + 1, len);

    cudaMemset(d_cp, 0, sizeof(int));
    cudaMemset(d_ic, 0, sizeof(int));

    //d_b will the semi-final array
    cudaMalloc(&d_b, bytesN + sizeof(int));
    cudaMemset(d_b, 0, sizeof(int) * len);

    numBlocks = (len + 1 + numThreads - 1) / numThreads;
    expandToB << <numThreads, numBlocks >> > (len + 1, d_cp, d_ic, d_b);

    //d_y is the final sorted array
    cudaMalloc(&d_y, bytesN);
    prefix_sum_on_gpu(d_b, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Took %f milliseconds using 0-Compressed\n", milliseconds);
    test_cubsort(d_x, N);
    cudaMemcpy(s.data(), d_y, bytesN, cudaMemcpyDeviceToHost);

    /*------------------------END OF THE GPU COMPUTATION--------------------------*/
    

    // a is the initial array and s is the sorted array.
 /*   for (auto& element : x) cout << element << " ";
    cout << endl;

    for (auto& element : s) cout << element << " ";
    cout << endl;
    return 0;*/
}