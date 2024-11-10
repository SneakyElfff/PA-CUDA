#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>

using namespace std;

float duration_gpu_kernel = 0;
float duration_gpu_total = 0;

vector<vector<int>> fillMatrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> range(0, 99);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = range(gen);
        }
    }

    return matrix;
}

void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int elem : row) {
            cout << elem << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        cerr << "Error: " << message << " (" << cudaGetErrorString(error) << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

vector<vector<int>> transformMatrix(const vector<vector<int>> &matrix, int N, int M, int block_size, int window_size) {
    vector<vector<int>> result(N, vector<int>(M, 0));

    for(int i = 0; i < N; i += block_size) {
        for(int j = 0; j < M; j += block_size) {

            for(int iw = 0; iw < window_size; ++iw) {
                for(int jw = 0; jw < window_size; ++jw) {
                    result[i + iw * 2][j + jw * 2] = matrix[i + iw][j + jw];
                    result[i + iw * 2][j + jw * 2 + 1] = matrix[i + iw][j + jw + window_size];
                    result[i + iw * 2 + 1][j + jw * 2] = matrix[i + iw + window_size][j + jw];
                    result[i + iw * 2 + 1][j + jw * 2 + 1] = matrix[i + iw + window_size][j + jw + window_size];
                }
            }
        }
    }

    return result;
}

__global__ void transformMatrixKernel(int *d_matrix, int *d_result, int N, int M, int block_size, int window_size) {
    int block_row = blockIdx.y * block_size;
    int block_col = blockIdx.x * block_size;

    int iw = threadIdx.y;
    int jw = threadIdx.x;

    if (iw < window_size && jw < window_size) {
        int src_idx = (block_row + iw) * M + (block_col + jw);
        int dst_idx = (block_row + iw * 2) * M + (block_col + jw * 2);

        d_result[dst_idx] = d_matrix[src_idx];
        d_result[dst_idx + 1] = d_matrix[src_idx + window_size];
        d_result[dst_idx + M] = d_matrix[src_idx + window_size * M];
        d_result[dst_idx + M + 1] = d_matrix[src_idx + window_size * M + window_size];
    }
}

vector<vector<int>> transformMatrixGPU(const vector<vector<int>> &matrix, int N, int M, int block_size, int window_size) {
    int *h_matrix = new int[N * M];
    int *h_result = new int[N * M];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            h_matrix[i * M + j] = matrix[i][j];

    int *d_matrix;
    int *d_result;

    checkCudaError(cudaMalloc(&d_matrix, N * M * sizeof(int)), "cudaMalloc d_matrix");
    checkCudaError(cudaMalloc(&d_result, N * M * sizeof(int)), "cudaMalloc d_result");

    cudaEvent_t start_event, stop_event, start_total_event, stop_total_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventCreate(&start_total_event);
    cudaEventCreate(&stop_total_event);

    // замер времени (с учетом копирования данных)
    cudaEventRecord(start_total_event, nullptr);

    checkCudaError(cudaMemcpy(d_matrix, h_matrix, N * M * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy h_matrix to d_matrix");
    checkCudaError(cudaMemset(d_result, 0, N * M * sizeof(int)), "cudaMemset d_result");

    dim3 block(32, 32);
    dim3 grid((M + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    // замер времени (без учета копирования данных)
    cudaEventRecord(start_event, nullptr);
    transformMatrixKernel<<<grid, block>>>(d_matrix, d_result, N, M, block_size, window_size);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    cudaEventRecord(stop_event, nullptr);
    cudaEventSynchronize(stop_event);

    cudaEventElapsedTime(&duration_gpu_kernel, start_event, stop_event);

    checkCudaError(cudaMemcpy(h_result, d_result, N * M * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d_result to h_result");

    cudaEventRecord(stop_total_event, nullptr);
    cudaEventSynchronize(stop_total_event);

    cudaEventElapsedTime(&duration_gpu_total, start_total_event, stop_total_event);

    vector<vector<int>> result(N, vector<int>(M));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            result[i][j] = h_result[i * M + j];

    checkCudaError(cudaFree(d_matrix), "cudaFree d_matrix");
    checkCudaError(cudaFree(d_result), "cudaFree d_result");

    delete[] h_matrix;
    delete[] h_result;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_total_event);
    cudaEventDestroy(stop_total_event);

    return result;
}

bool compareMatrices(const vector<vector<int>>& matrix1, const vector<vector<int>>& matrix2) {
    if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size())
        return false;

    for (size_t i = 0; i < matrix1.size(); i++) {
        for (size_t j = 0; j < matrix1[0].size(); j++) {
            if (matrix1[i][j] != matrix2[i][j]) {
                return false;
            }
        }
    }

    return true;
}

int main() {
    int N = 4096;
    int M = 4096;
    int window_size = 2;
    int block_size = N / 2;

    vector<vector<int>> matrix = fillMatrix(N, M);
    cout << "Original Matrix:\n";
//     printMatrix(matrix);

    cout << "Transformation (CPU) started." << endl;
    auto start = chrono::high_resolution_clock::now();
    auto result_cpu = transformMatrix(matrix, N, M, block_size, window_size);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_cpu = end - start;
    cout << "Duration (CPU): " << duration_cpu.count() << " sec." << endl;
//     printMatrix(result_cpu);

    cout << "Transformation (GPU) started." << endl;
    auto result_gpu = transformMatrixGPU(matrix, N, M, block_size, window_size);
    cout << "Kernel execution time (GPU): " << duration_gpu_kernel / 1000.0 << " sec." << endl;
    cout << "Total execution time (GPU including data transfer): " << duration_gpu_total / 1000.0 << " sec." << endl;
//    printMatrix(result_gpu);

    cout << "GPU realisation is faster than CPU by a factor of " << duration_cpu.count() / (duration_gpu_total / 1000.0) << endl;

    if (compareMatrices(result_cpu, result_gpu)) {
        cout << endl << "Results are the same" << endl;
        cout << "First matrix element: " << result_cpu[0][0] << "." << endl;
    }
    else {
        cout << "Results are NOT the same." << endl;
    }

    cout << endl << "Testing results:" << endl;
    cout << left << setw(8) << "M" << setw(8) << "N"
         << setw(19) << "Time CPU (s)" << setw(29) << "Time GPU kernel-only (s)"
         << setw(19) << "Time GPU (s)" << endl;
    cout << string(76, '-') << endl;

    cout << left << setw(8) << M << setw(8) << N
         << setw(19) << duration_cpu.count()
         << setw(29) << duration_gpu_kernel / 1000.0
         << setw(19) << duration_gpu_total / 1000.0 << endl;

    return 0;
}
