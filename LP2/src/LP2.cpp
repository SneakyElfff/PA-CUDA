#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <iomanip>
#include <random>
#include <omp.h>

using namespace std;

const int BLOCK_SIZE = 64;

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

vector<vector<int> > multiplyMatrices(const vector<vector<int> >& matrix1, const vector<vector<int> >&matrix2, int M, int N, int K) {
    vector<vector<int> > result(M, vector<int>(K, 0));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < N; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

vector<vector<int> > blockMultiplyMatrices(const vector<vector<int> >& matrix1, const vector<vector<int> >& matrix2, int M, int N, int K) {
    vector<vector<int> > result(M, vector<int>(K, 0));

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

bool compareMatrices(const vector<vector<int> >& matrix1, const vector<vector<int> >& matrix2) {
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

int main()
{
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

    if (compareMatrices(result, result_blocked)) {
        cout << "Results are the same" << endl;
        cout << "First matrix element: " << result[0][0] << "." << endl;
    }
    else
        cout << "Results are NOT the same." << endl;

    cout << endl << "Testing results:" << endl;
    cout << left << setw(8) << "M" << setw(8) << "N" << setw(8) << "K"
         << setw(20) << "Time (s)" << setw(20) << "L1 (s)"
         << endl;
    cout << string(84, '-') << endl;

    cout << left << setw(8) << M << setw(8) << N << setw(8) << K
         << setw(14) << duration.count()
         << setw(19) << duration_blocked.count()
         << endl;

    return 0;
}
