#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

const int BLOCK_SIZE = 576;

vector<vector<int>> fillMatrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));

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

vector<vector<int>> blockMultiplyMatrices(const vector<vector<int>>& matrix1, const vector<vector<int>>&matrix2, int M, int N, int K) {
    vector<vector<int>> result(M, vector<int>(K, 0));

    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < K; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                // перемножение блоков
                for (int ib = i; ib < min(i + BLOCK_SIZE, M); ++ib) {
                    for (int jb = j; jb < min(j + BLOCK_SIZE, K); ++jb) {
                        for (int kb = k; kb < min(k + BLOCK_SIZE, N); ++kb) {
                            result[ib][jb] += matrix1[ib][kb] * matrix2[kb][jb];
                        }
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

    // каждая матрица > 16 Мб (~18 Мб)
    int M = 2048;
    int N = 2304;
    int K = 2048;

    auto matrix1 = fillMatrix(M, N);
    auto matrix2 = fillMatrix(N, K);

    cout << "Перемножение матриц классическим способом начато." << endl;

    auto start = chrono::high_resolution_clock::now();
    auto result = multiplyMatrices(matrix1, matrix2, M, N, K);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    cout << "Перемножение матриц классическим способом завершено." << endl;
    cout << "Длительность классического перемножения: " << duration.count() << " секунд." << endl << endl;

    cout << "Перемножение матриц блочным способом начато." << endl;

    start = chrono::high_resolution_clock::now();
    auto result_blocked = blockMultiplyMatrices(matrix1, matrix2, M, N, K);
    end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration_blocked = end - start;
    cout << "Перемножение матриц блочным способом завершено." << endl;
    cout << "Длительность блочного перемножения: " << duration_blocked.count() << " секунд." << endl;

    cout << endl << "Блочная реализация быстрее, чем классическая в " << duration.count() / duration_blocked.count() << " раз." << endl;
    cout << "Первый элемент матрицы: " << result[0][0] << "." << endl;

    return 0;
}
