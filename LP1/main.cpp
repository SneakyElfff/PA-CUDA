#include <iostream>

using namespace std;

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

int main()
{
    srand(static_cast<unsigned>(time(nullptr))); // инициализация генератора случайных чисел

    // каждая матрица > 16 Мб (~18 Мб)
    int M = 2048;
    int N = 2304;
    int K = 2048;

    auto matrix1 = fillMatrix(M, N);
    auto matrix2 = fillMatrix(N, K);

    cout << "Перемножение матриц начато." << endl;

    auto start = chrono::high_resolution_clock::now();

    auto result = multiplyMatrices(matrix1, matrix2, M, N, K);

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;

    cout << "Перемножение матриц завершено." << endl;
    cout << "Длительность перемножения: " << duration.count() << " секунд." << endl;

    return 0;
}
