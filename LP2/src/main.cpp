#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <immintrin.h>
#include <omp.h>

using namespace std;

const int BLOCK_SIZE = 64;

vector<vector<int>> fillMatrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols)); //двумерный вектор matrix

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 100;
        }
    }

    return matrix;
}

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& matrix1, const vector<vector<int>>&matrix2, int M, int N, int K) {
    vector<vector<int>> result(M, vector<int>(K, 0));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < N; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

vector<vector<int>> blockMultiplyMatrices(const vector<vector<int>>& matrix1, const vector<vector<int>>& matrix2, int M, int N, int K) {
    vector<vector<int>> result(M, vector<int>(K, 0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < K; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                // умножение блоков
                for (int ib = i; ib < min(i + BLOCK_SIZE, M); ++ib) {
                    for (int jb = j; jb < min(j + BLOCK_SIZE, K); ++jb) {
                        __m128i sum = _mm_setzero_si128(); // Инициализация вектора суммы

                        for (int kb = k; kb < min(k + BLOCK_SIZE, N); kb += 4) {
                            // загрузка 4 элементов из matrix1 и matrix2
                            __m128i a = _mm_loadu_si128((__m128i*)&matrix1[ib][kb]);
                            __m128i b = _mm_loadu_si128((__m128i*)&matrix2[kb][jb]);

                            // умножение и накопление
                            sum = _mm_add_epi32(sum, _mm_mullo_epi32(a, b));
                        }

                        // сохранение результата
                        int temp[4];
                        _mm_storeu_si128((__m128i*)temp, sum);
                        result[ib][jb] += temp[0] + temp[1] + temp[2] + temp[3];
                    }
                }
            }
        }
    }

    return result;
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr))); // инициализация генератора случайных чисел

    int M = 2048;
    int N = 4096;
    int K = 2048;

    auto matrix1 = fillMatrix(M, N);
    auto matrix2 = fillMatrix(N, K);

    cout << "Multiplication (traditional) started." << endl;

    auto start = chrono::high_resolution_clock::now();
    auto result = multiplyMatrices(matrix1, matrix2, M, N, K);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    cout << "Duration: " << duration.count() << " sec." << endl << endl;

    cout << "Multiplication (blocked) started." << endl;

    start = chrono::high_resolution_clock::now();
    auto result_blocked = blockMultiplyMatrices(matrix1, matrix2, M, N, K);
    end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration_blocked = end - start;
    cout << "Duration: " << duration_blocked.count() << " sec." << endl;

    cout << endl << "Blocked realization is faster in " << duration.count() / duration_blocked.count() << endl;
    cout << "First matriz element: " << result[0][0] << "." << endl;

    return 0;
}
