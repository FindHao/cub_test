
#include <stdbool.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "utils.cuh"
#include "helper_cuda.h"
// #include "gpu-patch.cuh"

#define GPU_ANALYSIS_DEBUG 1

#if GPU_ANALYSIS_DEBUG
#define PRINT(...)                           \
    if (threadIdx.x == 0 && blockIdx.x == 0) \
    {                                        \
        printf(__VA_ARGS__);                 \
    }
#define PRINT_ALL(...) \
    printf(__VA_ARGS__)
#define PRINT_RECORDS(buffer)                                                                                                     \
    __syncthreads();                                                                                                              \
    if (threadIdx.x == 0)                                                                                                         \
    {                                                                                                                             \
        gpu_patch_analysis_address_t *records = (gpu_patch_analysis_address_t *)buffer->records;                                  \
        for (uint32_t i = 0; i < buffer->head_index; ++i)                                                                         \
        {                                                                                                                         \
            printf("gpu analysis-> merged <%p, %p> (%p)\n", records[i].start, records[i].end, records[i].end - records[i].start); \
        }                                                                                                                         \
    }                                                                                                                             \
    __syncthreads();
#else
#define PRINT(...)
#define PRINT_ALL(...)
#define PRINT_RECORDS(buffer)
#endif

#define ITEMS_PER_THREAD 4
#define GPU_PATCH_ANALYSIS_THREADS 1024

extern "C" __launch_bounds__(GPU_PATCH_ANALYSIS_THREADS, 1)
    __global__ void gp_histogram(
        uint64_t *patch_buffer,
        uint64_t *compact_buffer)
{
    enum
    {
        TILE_SIZE = GPU_PATCH_ANALYSIS_THREADS * ITEMS_PER_THREAD
    };
    auto warp_index = blockDim.x / GPU_PATCH_WARP_SIZE * blockIdx.x + threadIdx.x / GPU_PATCH_WARP_SIZE;
    auto num_warps = blockDim.x / GPU_PATCH_WARP_SIZE;
    auto laneid = get_laneid();

    // gpu_patch_record_address_t *records = (gpu_patch_record_address_t *)patch_buffer->records;

    // PRINT("gpu analysis->full: %u, analysis: %u, head_index: %u, tail_index: %u, size: %u, num_threads: %u",
    //       patch_buffer->full, patch_buffer->analysis, patch_buffer->head_index, patch_buffer->tail_index,
    //       patch_buffer->size, patch_buffer->num_threads)
    // for (auto iter = warp_index; iter < patch_buffer->head_index; iter += num_warps)
    // {
    //     gpu_patch_record_address_t *record = records + iter;
    //     uint64_t address_start = record->address[laneid];
    //     if (((0x1u << laneid) & record->active) == 0)
    //     {
    //         // inactive thread
    //         address_start = 0;
    //     }
    //     addrs[iter + laneid] = address_start;
    // }
    // __syncthreads();

    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef cub::BlockLoad<uint64_t, GPU_PATCH_ANALYSIS_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef cub::BlockStore<uint64_t, GPU_PATCH_ANALYSIS_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
    typedef cub::BlockRadixSort<uint64_t, GPU_PATCH_ANALYSIS_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    // Shared memory
    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
        typename BlockRadixSortT::TempStorage sort;
        // typename BlockScanT::TempStorage scan;
        // typename BlockDiscontinuity::TempStorage disc;
    } temp_storage;
    // Per-thread tile items
    uint64_t items[ITEMS_PER_THREAD];

    // Our current block's offset
    int block_offset = blockIdx.x * TILE_SIZE;
    // Load items into a blocked arrangement
    BlockLoadT(temp_storage.load).Load(patch_buffer + block_offset, items);
    __syncthreads();
    PRINT("items gpu analysis->block_offset: %d, items: %llu, %llu, %llu, %llu\n", block_offset, items[0], items[1], items[2], items[3]);
    // // Sort keys
    BlockRadixSortT(temp_storage.sort).Sort(items);
    __syncthreads();
    PRINT("items gpu analysis->block_offset: %d, items: %llu, %llu, %llu, %llu\n", block_offset, items[0], items[1], items[2], items[3]);
    __syncthreads();
    BlockStoreT(temp_storage.store).Store(compact_buffer + block_offset, items);
}

int main()
{
    int num_item = 4096;
    uint64_t *h_input;
    h_input = (uint64_t *)malloc(num_item * sizeof(uint64_t));
    if (h_input == NULL)
    {
        printf("malloc failed\n");
        return -1;
    }
    printf("malloc size: %d\n", num_item * sizeof(uint64_t));
    // Initialize the host input vectors
    for (int i = 0; i < num_item; ++i)
    {
        h_input[i] = (num_item - i) %100;
    }
    uint64_t *d_input, *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_input, num_item * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc((void **)&d_output, num_item * sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(d_input, h_input, num_item * sizeof(uint64_t), cudaMemcpyHostToDevice));
    // gp_histogram<<<1, GPU_PATCH_ANALYSIS_THREADS>>>(d_input, d_output);
    cub::DoubleBuffer<uint64_t> d_keys(d_input, d_output);
    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_temp_storage, num_item * sizeof(uint64_t)));
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, d_keys, num_item);
    // // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_item);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(h_input, d_keys.Current(), num_item * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < num_item; ++i)
    // {
    //     printf("%d ", h_input[i]);
    // }
    printf("\n========sort end========\n");
    uint64_t *d_unique_out;   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
    int *d_counts_out;   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
    int *d_num_runs_out; // e.g., [ ]
    checkCudaErrors(cudaMalloc((void **)&d_unique_out, num_item * sizeof(uint64_t)));
    checkCudaErrors(cudaMalloc((void **)&d_counts_out, num_item * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_num_runs_out, sizeof(int)));
    // Determine temporary device storage requirements
    temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(NULL, temp_storage_bytes, d_keys.Current(), d_unique_out, d_counts_out, d_num_runs_out, num_item);
    // Allocate temporary storage
    // cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run encoding
    cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_keys.Current(), d_unique_out, d_counts_out, d_num_runs_out, num_item);
    cudaDeviceSynchronize();
    int *h_num_runs_out;
    h_num_runs_out = (int *)malloc(num_item*sizeof(int));
    checkCudaErrors(cudaMemcpy(h_num_runs_out, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost));
    printf("num_runs: %d\n", h_num_runs_out[0]);
    checkCudaErrors(cudaMemcpy(h_input, d_unique_out, h_num_runs_out[0] * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    int *h_counts_out;
    h_counts_out = (int *)malloc(num_item * sizeof(int));
    checkCudaErrors(cudaMemcpy(h_counts_out, d_counts_out, h_num_runs_out[0] * sizeof(int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    for (int i = 0; i < h_num_runs_out[0]; ++i)
    {
        printf("%llu : %d\n", h_input[i], h_counts_out[i]);
    }
    printf("========encode end========\n");
    // for (int i = 0; i < num_item; ++i)
    // {
    //     printf("%llu ", h_input[i]);
    // }
    return 0;
}