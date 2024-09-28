#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

using namespace std;

const int BLOCK_SIZE_L1 = 222;
const int BLOCK_SIZE_L2 = 576;
// const int BLOCK_SIZE_L3 = ;

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

vector<vector<int> > multiplyMatrices(const vector<vector<int> >& matrix1, const vector<vector<int> >& matrix2, int M, int N, int K) {
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

vector<vector<int> > blockMultiplyMatrices(const vector<vector<int> >& matrix1, const vector<vector<int> >& matrix2, int M, int N, int K, int block_size) {
    vector<vector<int> > result(M, vector<int>(K, 0));

    for (int i = 0; i < M; i += block_size) {
        for (int j = 0; j < K; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                int i_m = min(i + block_size, M);
                int j_m = min(j + block_size, K);
                int k_m = min(k + block_size, N);

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
    int M = 2048;
    int N = 8000;
    int K = 1024;

    auto matrix1 = fillMatrix(M, N);
    auto matrix2 = fillMatrix(N, K);

    cout << "Перемножение матриц классическим способом начато." << endl;

    auto start = chrono::high_resolution_clock::now();
    auto result = multiplyMatrices(matrix1, matrix2, M, N, K);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Длительность классического перемножения: " << duration.count() << " секунд." << endl << endl;

    cout << "Перемножение матриц блочным способом начато." << endl;

    start = chrono::high_resolution_clock::now();
    auto result_blocked_L1 = blockMultiplyMatrices(matrix1, matrix2, M, N, K, BLOCK_SIZE_L1);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_blocked_L1 = end - start;
    cout << "Длительность блочного перемножения для L1 кэша: " << duration_blocked_L1.count() << " секунд." << endl;

    start = chrono::high_resolution_clock::now();
    auto result_blocked_L2 = blockMultiplyMatrices(matrix1, matrix2, M, N, K, BLOCK_SIZE_L2);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_blocked_L2 = end - start;
    cout << "Длительность блочного перемножения для L2 кэша: " << duration_blocked_L2.count() << " секунд." << endl;

    // start = chrono::high_resolution_clock::now();
    // auto result_blocked_L3 = blockMultiplyMatrices(matrix1, matrix2, M, N, K, BLOCK_SIZE_L3);
    // end = chrono::high_resolution_clock::now();
    // chrono::duration<double> duration_blocked_L3 = end - start;
    // cout << "Длительность блочного перемножения для L3 кэша: " << duration_blocked_L3.count() << " секунд." << endl;

    cout << endl << "Блочная реализация L1 быстрее, чем классическая в " << duration.count() / duration_blocked_L1.count() << " раз." << endl;
    cout << "Блочная реализация L2 быстрее, чем классическая в " << duration.count() / duration_blocked_L2.count() << " раз." << endl;
    // cout << "Блочная реализация L3 быстрее, чем классическая в " << duration.count() / duration_blocked_L3.count() << " раз." << endl;

    if (compareMatrices(result, result_blocked_L1) && compareMatrices(result, result_blocked_L2)) {
    // if (compareMatrices(result, result_blocked_L1) && compareMatrices(result, result_blocked_L2) && compareMatrices(result, result_blocked_L3)) {
        cout << "Результаты реализаций совпадают." << endl;
        cout << "Первый элемент матрицы: " << result[0][0] << "." << endl;
    }
    else
        cout << "Результаты реализаций НЕ совпадают." << endl;

    cout << endl << "Результаты тестирования:" << endl;
    cout << left << setw(8) << "M" << setw(8) << "N" << setw(8) << "K"
         << setw(20) << "Время-1 (с)" << setw(20) << "L1 (с)"
         << setw(20) << "L2 (с)" << setw(20) << "L3 (с)" << endl;
    cout << string(84, '-') << endl;

    cout << left << setw(8) << M << setw(8) << N << setw(8) << K
         << setw(14) << duration.count()
         << setw(19) << duration_blocked_L1.count()
         << setw(20) << duration_blocked_L2.count()
         // << setw(20) << duration_blocked_L3.count()
         << endl;

    return 0;
}
