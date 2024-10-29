#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

using namespace std;

vector<vector<int> > fillMatrix(int rows, int cols) {
    vector<vector<int> > matrix(rows, vector<int>(cols));

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

void printMatrix(const vector<vector<int> > &matrix) {
    for (const auto& row : matrix) {
        for (int elem : row) {
            cout << elem << " ";
        }
        cout << "\n";
    }

    cout << "\n";
}

vector<vector<int> > transformMatrix(const vector<vector<int> > &matrix, int N, int M, int block_size, int window_size) {
    vector<vector<int> > result(N, vector<int>(M, 0));

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

// bool compareMatrices(const vector<vector<int> >& matrix1, const vector<vector<int> >& matrix2) {
//     if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size())
//         return false;
//
//     for (size_t i = 0; i < matrix1.size(); i++) {
//         for (size_t j = 0; j < matrix1[0].size(); j++) {
//             if (matrix1[i][j] != matrix2[i][j]) {
//                 return false;
//             }
//         }
//     }
//
//     return true;
// }

int main()
{
    int N = 8;
    int M = 8;
    int window_size = 2;
    int block_size = N / 2;

    // auto matrix = fillMatrix(N, M);
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3, 4, 5, 6, 7, 8},
        {9, 10, 11, 12, 13, 14, 15, 16},
        {17, 18, 19, 20, 21, 22, 23, 24},
        {25, 26, 27, 28, 29, 30, 31, 32},
        {33, 34, 35, 36, 37, 38, 39, 40},
        {41, 42, 43, 44, 45, 46, 47, 48},
        {49, 50, 51, 52, 53, 54, 55, 56},
        {57, 58, 59, 60, 61, 62, 63, 64}
    };
    printMatrix(matrix);

    cout << "Transformation (CPU) started." << endl;

    auto start = chrono::high_resolution_clock::now();
    auto result = transformMatrix(matrix, N, M, block_size, window_size);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Duration: " << duration.count() << " sec." << endl << endl;
    printMatrix(result);

    cout << "Transformation (GPU) started." << endl;

    start = chrono::high_resolution_clock::now();
    // auto result_gpu = transformMatrix(matrix, N, M, block_size, window_size);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_gpu = end - start;
    cout << "Duration: " << duration_gpu.count() << " sec." << endl << endl;

//    cout << "GPU realization is faster than CPU in " << duration.count() / duration_blocked_vectorised.count() << endl;

    // if (compareMatrices(result, result_gpu)) {
    //     cout << endl << "Results are the same" << endl;
    //     cout << "First matrix element: " << result[0][0] << "." << endl;
    // }
    // else
    //     cout << "Results are NOT the same." << endl;

    cout << endl << "Testing results:" << endl;
    cout << left << setw(8) << "M" << setw(8) << "N"
         << setw(19) << "Time CPU (s)" << setw(19) << "Time GPU (s)"
         << endl;
    cout << string(76, '-') << endl;

    cout << left << setw(8) << M << setw(8) << N
         << setw(19) << duration.count()
         << setw(19) << duration_gpu.count()
         << endl;

    return 0;
}
