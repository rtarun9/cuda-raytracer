#include <cuda.h>
#include <iostream>

__global__ void add_vectors(float *vec_a, float *vec_b, size_t N, float *output)
{
    uint32_t thread_id =threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < N)
    {
        output[thread_id] = vec_a[thread_id] + vec_b[thread_id];
    }
}

int main()
{
    cudaDeviceProp device_prop = {};
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "Cuda device name :: " << device_prop.name << '\n';
    constexpr size_t N  = 5;

    float vec_a[] = {1, 2, 3, 4, 5};
    float vec_b[] = {-1.0f, -2.0f, -5.0f, 0.0f, 10.0f};
    float result[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float* dev_vec_a = nullptr;
    float* dev_vec_b = nullptr;
    float* dev_result = nullptr;
    cudaMalloc((void**)&dev_vec_a, sizeof(float) * N);
    cudaMalloc((void**)&dev_vec_b, sizeof(float) * N);
    cudaMalloc((void**)&dev_result, sizeof(float) * N);

    cudaMemcpy(dev_vec_a, vec_a, sizeof(float) * N, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vec_b, vec_b, sizeof(float) * N, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(dev_result, result, sizeof(float) * N, cudaMemcpyKind::cudaMemcpyHostToDevice);

    add_vectors<<<1, N>>>(dev_vec_a, dev_vec_b, N, dev_result);
    cudaDeviceSynchronize();

    cudaMemcpy(result, dev_result, sizeof(float) * N, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < N; i++)
    {
        std::cout << "result[" <<  i << "] = " << result[i] << '\n';
    }

    return 0;
}