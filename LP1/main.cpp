#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

using namespace std;

constexpr int BLOCK_SIZE = 576;

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

vector<vector<int> > multiplyMatrices(const vector<vector<int> >& matrix1, const vector<vector<int>>&matrix2, int M, int N, int K) {
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

vector<vector<int> > blockMultiplyMatrices(const vector<vector<int> >& matrix1, const vector<vector<int> >&matrix2, int M, int N, int K) {
    vector<vector<int> > result(M, vector<int>(K, 0));

    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < K; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                int i_m = min(i + BLOCK_SIZE, M);
                int j_m = min(j + BLOCK_SIZE, K);
                int k_m = min(k + BLOCK_SIZE, N);

                // перемножение блоков
                for (int ib = i; ib < i_m; ++ib) {
                    for (int jb = j; jb < j_m; ++jb) {
                        for (int kb = k; kb < k_m; ++kb) {
                            result[ib][jb] += matrix1[ib][kb] * matrix2[kb][jb];
                        }
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
    // каждая матрица > 16 Мб
    int M = 4096;
    int N = 16384;
    int K = 1024;

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

    if (compareMatrices(result, result_blocked)) {
        cout << "Результаты реализаций совпадают." << endl;
        cout << "Первый элемент матрицы: " << result[0][0] << "." << endl;
    }
    else
        cout << "Результаты реализаций НЕ совпадают." << endl;

    cout << endl << "Результаты тестирования:" << endl;
    cout << left << setw(10) << "M" << setw(10) << "N" << setw(10) << "N" << setw(10) << "K"
         << setw(20) << "Время-1 (с)" << setw(20) << "Время-2 (с)" << endl;
    cout << string(74, '-') << endl;

    cout << left << setw(10) << M << setw(10) << N << setw(10) << N << setw(10) << K
         << setw(20) << duration.count() << duration_blocked.count() << endl;

    return 0;
}
