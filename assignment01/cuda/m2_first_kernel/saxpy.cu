#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do{                                 \
        cudaError_t err = call;         \
        if(err != cudaSuccess){         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);         \
        }                               \
    }while(0)                           \


__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size>\n", argv[0]);
        return EXIT_FAILURE;
    }
    long long n_ll = atoll(argv[1]);

    if (n_ll < 0) {
        fprintf(stderr, "Size can't be a negative number.\n");
        return EXIT_FAILURE;
    }

    size_t n = static_cast<size_t>(n_ll);

    if (n == 0) {
        printf("SUM=0\n");
        return EXIT_SUCCESS;
    }

    const float a = 2.0f;
    const size_t bytes = n * sizeof(float);

    float *h_x = static_cast<float *>(malloc(bytes));
    float *h_y = static_cast<float *>(malloc(bytes)); 

    if (h_x == nullptr || h_y == nullptr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        free(h_x);
        free(h_y);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < n; ++i) {
        h_x[i] = (static_cast<float>(static_cast<long long>(i % 2048) - 1024)) * 0.5f;
        h_y[i] = (static_cast<float>(static_cast<long long>(i % 1024) - 512));
    }

    float *d_x = nullptr;
    float *d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    saxpy_kernel<<<blocks, threads>>>(static_cast<int>(n), a, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    double s = 0.0;
    for(size_t i = 0; i < n; ++i) {
        s += static_cast<double>(h_y[i]);
    }

    printf("SUM=%.0f n = %zu kernel_time_ms = %.3f\n", s, n, kernel_time_ms);


    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);
    return 0;

}