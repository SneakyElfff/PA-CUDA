#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>

using namespace std;

vector<int> fillArray(int size) {
    vector<int> array(size);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> range(0, 99);

    for (int i = 0; i < size; i++) {
        array[i] = range(gen);
    }

    return array;
}

void printArray(const vector<int>& array) {
    for (int elem : array) {
        cout << elem << " ";
    }
    cout << "\n";
}

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        cerr << "Error: " << message << " (" << cudaGetErrorString(error) << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

vector<int> integralSumCPU(const vector<int>& array, float &cpu_duration) {
    vector<int> result(array.size());
    auto start = chrono::high_resolution_clock::now();

    result[0] = array[0];
    for (size_t i = 1; i < array.size(); ++i) {
        result[i] = result[i - 1] + array[i];
    }

    auto end = chrono::high_resolution_clock::now();
    cpu_duration = chrono::duration<float, std::milli>(end - start).count();
    return result;
}

// ядро GPU для интегральной суммы с использованием shared memory
__global__ void integralSumKernel(int *d_array, int *d_result, int size) {
    extern __shared__ int shared_data[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size) {
        shared_data[threadIdx.x] = d_array[idx];
    } else {
        shared_data[threadIdx.x] = 0; // Заполнение нулями, если индекс выходит за границы
    }
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int temp = (threadIdx.x >= offset) ? shared_data[threadIdx.x - offset] : 0;
        __syncthreads();
        shared_data[threadIdx.x] += temp;
        __syncthreads();
    }

    if (idx < size) {
        d_result[idx] = shared_data[threadIdx.x];
    }
}

vector<int> integralSumGPU(const vector<int>& array, float &gpu_duration, float &gpu_duration_total) {
    int size = array.size();
    int *d_array, *d_result;

    checkCudaError(cudaMalloc(&d_array, size * sizeof(int)), "cudaMalloc d_array");
    checkCudaError(cudaMalloc(&d_result, size * sizeof(int)), "cudaMalloc d_result");

    cudaEvent_t start_event, stop_event, start_total_event, stop_total_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventCreate(&start_total_event);
    cudaEventCreate(&stop_total_event);

    // замер времени (с учетом копирования данных)
    cudaEventRecord(start_total_event, nullptr);

    checkCudaError(cudaMemcpy(d_array, array.data(), size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy h_array to d_array");

    const int blockSize = 256; // Размер блока
    const int numBlocks = (size + blockSize - 1) / blockSize; // Количество блоков

    // замер времени (без учета копирования данных)
    cudaEventRecord(start_event, nullptr);

    integralSumKernel<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_array, d_result, size);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    cudaEventRecord(stop_event, nullptr);
    cudaEventSynchronize(stop_event);

    float duration;
    cudaEventElapsedTime(&duration, start_event, stop_event);
    gpu_duration = duration;

    vector<int> result(size);
    checkCudaError(cudaMemcpy(result.data(), d_result, size * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d_result to result");

    cudaEventRecord(stop_total_event, nullptr);
    cudaEventSynchronize(stop_total_event);

    cudaEventElapsedTime(&duration, start_total_event, stop_total_event);
    gpu_duration_total = duration;

    checkCudaError(cudaFree(d_array), "cudaFree d_array");
    checkCudaError(cudaFree(d_result), "cudaFree d_result");

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_total_event);
    cudaEventDestroy(stop_total_event);

    return result;
}

bool compareArrays(const vector<int>& array1, const vector<int>& array2) {
    if (array1.size() != array2.size())
        return false;

    for (size_t i = 0; i < array1.size(); i++) {
        if (array1[i] != array2[i]) {
            return false;
        }
    }

    return true;
}

int main() {
    int size = 1024 * 1024;

    vector<int> array = fillArray(size);
    cout << "Original array (first 10 elements)\n";
    printArray(vector<int>(array.begin(), array.begin() + 10));

    float cpu_duration = 0;
    cout << "Integral sum calculation (CPU) started." << endl;
    auto result_cpu = integralSumCPU(array, cpu_duration);
    cout << "Duration (CPU): " << cpu_duration << " ms." << endl;

    float gpu_duration = 0, gpu_duration_total = 0;
    cout << "Integral sum calculation (GPU) started." << endl;
    auto result_gpu = integralSumGPU(array, gpu_duration, gpu_duration_total);
    cout << "Kernel execution time (GPU): " << gpu_duration << " ms." << endl;
    cout << "Total execution time (GPU including data transfer): " << gpu_duration_total << " ms." << endl;

    cout << endl << "GPU realisation is faster than CPU by a factor of " << cpu_duration / gpu_duration << endl;

    if (compareArrays(result_cpu, result_gpu)) {
        cout << "Results are the same." << endl;
        cout << "First matrix element: " << result_cpu[0] << "." << endl;
    } else {
        cout << "Results are NOT the same." << endl;
    }

    cout << "\nTesting results: first 10 elements\n";
    cout << "CPU: ";
    printArray(vector<int>(result_cpu.begin(), result_cpu.begin() + 10));
    cout << "GPU: ";
    printArray(vector<int>(result_gpu.begin(), result_gpu.begin() + 10));

    cout << left << setw(20) << "Size"
         << setw(25) << "Time CPU (ms)"
         << setw(40) << "Time GPU kernel-only (ms)"
         << setw(30) << "Time GPU (ms)" << endl;
    cout << string(98, '-') << endl;

    cout << left << setw(20) << size
         << setw(25) << cpu_duration
         << setw(40) << gpu_duration
         << setw(30) << gpu_duration_total << endl;

    return 0;
}