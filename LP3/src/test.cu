#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>

using namespace std;

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
    cudaMalloc(&d_matrix, N * M * sizeof(int));
    cudaMalloc(&d_result, N * M * sizeof(int));

    cudaEvent_t start_event, stop_event, start_total_event, stop_total_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventCreate(&start_total_event);
    cudaEventCreate(&stop_total_event);

    // замер времени (с учетом копирования данных)
    cudaEventRecord(start_total_event, 0);

    cudaMemcpy(d_matrix, h_matrix, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, N * M * sizeof(int));

    dim3 block(32, 32);
    dim3 grid((M + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    // замер времени (без учета копирования данных)
    cudaEventRecord(start_event, 0);
    transformMatrixKernel<<<grid, block>>>(d_matrix, d_result, N, M, block_size, window_size);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float duration_gpu_kernel = 0;
    cudaEventElapsedTime(&duration_gpu_kernel, start_event, stop_event);

    cudaMemcpy(h_result, d_result, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_total_event, 0);
    cudaEventSynchronize(stop_total_event);

    float duration_gpu_total = 0;
    cudaEventElapsedTime(&duration_gpu_total, start_total_event, stop_total_event);

    cout << "Kernel execution time (GPU): " << duration_gpu_kernel / 1000.0 << " sec." << endl;
    cout << "Total execution time (GPU including data transfer): " << duration_gpu_total / 1000.0 << " sec." << endl;

    cudaFree(d_matrix);
    cudaFree(d_result);
    delete[] h_matrix;
    delete[] h_result;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_total_event);
    cudaEventDestroy(stop_total_event);

    vector<vector<int>> result(N, vector<int>(M));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            result[i][j] = h_result[i * M + j];

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
    int N = 8;
    int M = 8;
    int window_size = 2;
    int block_size = N / 2;

    vector<vector<int>> matrix = {
        {1, 2, 3, 4, 5, 6, 7, 8},
        {9, 10, 11, 12, 13, 14, 15, 16},
        {17, 18, 19, 20, 21, 22, 23, 24},
        {25, 26, 27, 28, 29, 30, 31, 32},
        {33, 34, 35, 36, 37, 38, 39, 40},
        {41, 42, 43, 44, 45, 46, 47, 48},
        {49, 50, 51, 52, 53, 54, 55, 56},
        {57, 58, 59, 60, 61, 62, 63, 64}
    };

    cout << "Original Matrix:\n";
    printMatrix(matrix);

    cout << "Transformation (CPU) started." << endl;
    auto start = chrono::high_resolution_clock::now();
    auto result_cpu = transformMatrix(matrix, N, M, block_size, window_size);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_cpu = end - start;
    cout << "Duration (CPU): " << duration_cpu.count() << " sec." << endl;
    printMatrix(result_cpu);

    cout << "Transformation (GPU) started." << endl;
    auto result_gpu = transformMatrixGPU(matrix, N, M, block_size, window_size);
    printMatrix(result_gpu);

    if (compareMatrices(result_cpu, result_gpu)) {
        cout << endl << "Results are the same" << endl;
        cout << "First matrix element: " << result_cpu[0][0] << "." << endl;
    }
    else {
        cout << "Results are NOT the same." << endl;

    cout << endl << "Testing results:" << endl;
    cout << left << setw(8) << "M" << setw(8) << "N"
        << setw(19) << "Time CPU (s)" << setw(19) << "Time GPU kernel-only(s)"
        << setw(19) << "Time GPU (s)"
        << endl;
    cout << string(76, '-') << endl;

    cout << left << setw(8) << M << setw(8) << N
        << setw(19) << duration_cpu.count()
//        << setw(19) << duration_gpu_kernel
//        << setw(19) << duration_gpu_total
        << endl;

    return 0;
}
