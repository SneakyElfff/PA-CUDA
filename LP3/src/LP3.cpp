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

    cout << "Multiplication (GPU) started." << endl;

    start = chrono::high_resolution_clock::now();
//    auto result_blocked = blockMultiplyMatrices(matrix1, matrix2, M, N, K);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_blocked = end - start;
    cout << "Duration: " << duration_blocked.count() << " sec." << endl << endl;

//    cout << "GPU realization is faster than CPU in " << duration.count() / duration_blocked_vectorised.count() << endl;

//    if (compareMatrices(result, result_blocked) & compareMatrices(result, result_blocked_vectorised)) {
//        cout << endl << "Results are the same" << endl;
//        cout << "First matrix element: " << result[0][0] << "." << endl;
//    }
//    else
//        cout << "Results are NOT the same." << endl;

    cout << endl << "Testing results:" << endl;
    cout << left << setw(8) << "M" << setw(8) << "N" << setw(8) << "K"
         << setw(19) << "Time L1-1 (s)" << setw(19) << "Time L1-3 (s)"
         << endl;
    cout << string(76, '-') << endl;

    cout << left << setw(8) << M << setw(8) << N << setw(8) << K
         << setw(19) << duration.count()
         << setw(19) << duration_blocked.count()
         << endl;

    return 0;
}
