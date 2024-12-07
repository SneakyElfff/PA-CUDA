#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

__global__ void laplacianFilter(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int laplacian_value = 0;

    // Обработка центральных пикселей
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        laplacian_value = input[(y - 1) * width + x] +    // верхний сосед
                          input[y * width + (x - 1)] +    // левый сосед
                          input[y * width + (x + 1)] +    // правый сосед
                          input[(y + 1) * width + x] +    // нижний сосед
                          -4 * input[idx];                 // центральный пиксель
    } else {
        // Обработка краевых пикселей
        if (y == 0) { // Верхний край
            if (x == 0) { // Верхний левый угол
                laplacian_value = input[idx] * -4 +
                                  (x < width - 1 ? input[idx + 1] : 0) + // Правый сосед
                                  (y < height - 1 ? input[(y + 1) * width + x] : 0); // Нижний сосед
            } else if (x == width - 1) { // Верхний правый угол
                laplacian_value = input[idx] * -4 +
                                  (x > 0 ? input[idx - 1] : 0) + // Левый сосед
                                  (y < height - 1 ? input[(y + 1) * width + x] : 0); // Нижний сосед
            } else { // Верхний край (не углы)
                laplacian_value = input[idx] * -4 +
                                  (x > 0 ? input[idx - 1] : 0) + // Левый сосед
                                  (x < width - 1 ? input[idx + 1] : 0) + // Правый сосед
                                  (y < height - 1 ? input[(y + 1) * width + x] : 0); // Нижний сосед
            }
        } else if (y == height - 1) { // Нижний край
            if (x == 0) { // Нижний левый угол
                laplacian_value = input[idx] * -4 +
                                  (x < width - 1 ? input[idx + 1] : 0) + // Правый сосед
                                  (y > 0 ? input[(y - 1) * width + x] : 0); // Верхний сосед
            } else if (x == width - 1) { // Нижний правый угол
                laplacian_value = input[idx] * -4 +
                                  (x > 0 ? input[idx - 1] : 0) + // Левый сосед
                                  (y > 0 ? input[(y - 1) * width + x] : 0); // Верхний сосед
            } else { // Нижний край (не углы)
                laplacian_value = input[idx] * -4 +
                                  (x > 0 ? input[idx - 1] : 0) + // Левый сосед
                                  (x < width - 1 ? input[idx + 1] : 0) + // Правый сосед
                                  (y > 0 ? input[(y - 1) * width + x] : 0); // Верхний сосед
            }
        } else if (x == 0) { // Левый край
            if (y > 0 && y < height - 1) { // Не углы
                laplacian_value = input[idx] * -4 +
                                  (y > 0 ? input[(y - 1) * width + x] : 0) + // Верхний сосед
                                  (y < height - 1 ? input[(y + 1) * width + x] : 0) + // Нижний сосед
                                  (x < width - 1 ? input[(y) * width + (x + 1)] : 0); // Правый сосед
            }
        } else if (x == width - 1) { // Правый край
            if (y > 0 && y < height - 1) { // Не углы
                laplacian_value = input[idx] * -4 +
                                  (y > 0 ? input[(y - 1) * width + x] : 0) + // Верхний сосед
                                  (y < height - 1 ? input[(y + 1) * width + x] : 0) + // Нижний сосед
                                  (x > 0 ? input[(y) * width + (x - 1)] : 0); // Левый сосед
            }
        }
    }

    // Ограничение значений результата в диапазоне [0, 255]
    output[idx] = min(max(laplacian_value, 0), 255);
}

void laplacianFilterCPU(const Mat &input, Mat &output) {
    // Обработка центральных пикселей
    for (int y = 1; y < input.rows - 1; ++y) {
        for (int x = 1; x < input.cols - 1; ++x) {
            int laplacian_value = input.at<uchar>(y - 1, x) +    // верхний сосед
                                  input.at<uchar>(y, x - 1) +    // левый сосед
                                  input.at<uchar>(y, x + 1) +    // правый сосед
                                  input.at<uchar>(y + 1, x) +    // нижний сосед
                                  -4 * input.at<uchar>(y, x);    // центральный пиксель
            output.at<uchar>(y, x) = min(max(laplacian_value, 0), 255);
        }
    }

    // Обработка краевых пикселей
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            if (x == 0 && y == 0) { // Верхний левый угол
                output.at<uchar>(y, x) = min(max(
                    input.at<uchar>(y, x) * -4 +
                    (x < input.cols - 1 ? input.at<uchar>(y, x + 1) : 0) +   // Правый сосед
                    (y < input.rows - 1 ? input.at<uchar>(y + 1, x) : 0),    // Нижний сосед
                    0), 255);
            } else if (x == input.cols - 1 && y == 0) { // Верхний правый угол
                output.at<uchar>(y, x) = min(max(
                    input.at<uchar>(y, x) * -4 +
                    (x > 0 ? input.at<uchar>(y, x - 1) : 0) +   // Левый сосед
                    (y < input.rows - 1 ? input.at<uchar>(y + 1, x) : 0),    // Нижний сосед
                    0), 255);
            } else if (x == 0 && y == input.rows - 1) { // Нижний левый угол
                output.at<uchar>(y, x) = min(max(
                    input.at<uchar>(y, x) * -4 +
                    (x < input.cols - 1 ? input.at<uchar>(y, x + 1) : 0) +   // Правый сосед
                    (y > 0 ? input.at<uchar>(y - 1, x) : 0),    // Верхний сосед
                    0), 255);
            } else if (x == input.cols - 1 && y == input.rows - 1) { // Нижний правый угол
                output.at<uchar>(y, x) = min(max(
                    input.at<uchar>(y, x) * -4 +
                    (x > 0 ? input.at<uchar>(y, x - 1) : 0) +   // Левый сосед
                    (y > 0 ? input.at<uchar>(y - 1, x) : 0),    // Верхний сосед
                    0), 255);
            } else if (y == 0) { // Верхний край (не углы)
                output.at<uchar>(y, x) = min(max(
                    input.at<uchar>(y, x) * -4 +
                    (x > 0 ? input.at<uchar>(y, x - 1) : 0) +    // Левый сосед
                    (x < input.cols - 1 ? input.at<uchar>(y, x + 1) : 0) + // Правый сосед
                    (y < input.rows - 1 ? input.at<uchar>(y + 1, x) : 0), // Нижний сосед
                    0), 255);
            } else if (y == input.rows - 1) { // Нижний край (не углы)
                output.at<uchar>(y, x) = min(max(
                    input.at<uchar>(y, x) * -4 +
                    (x > 0 ? input.at<uchar>(y, x - 1) : 0) +    // Левый сосед
                    (x < input.cols - 1 ? input.at<uchar>(y, x + 1) : 0) + // Правый сосед
                    (y > 0 ? input.at<uchar>(y - 1, x) : 0), // Верхний сосед
                    0), 255);
            } else if (x == 0) { // Левый край (не углы)
                output.at<uchar>(y, x) = min(max(
                    input.at<uchar>(y, x) * -4 +
                    (y > 0 ? input.at<uchar>(y - 1, x) : 0) +    // Верхний сосед
                    (y < input.rows - 1 ? input.at<uchar>(y + 1, x) : 0) + // Нижний сосед
                    (x < input.cols - 1 ? input.at<uchar>(y, x + 1) : 0), // Правый сосед
                    0), 255);
            } else if (x == input.cols - 1) { // Правый край (не углы)
                output.at<uchar>(y, x) = min(max(
                    input.at<uchar>(y, x) * -4 +
                    (y > 0 ? input.at<uchar>(y - 1, x) : 0) +    // Верхний сосед
                    (y < input.rows - 1 ? input.at<uchar>(y + 1, x) : 0) + // Нижний сосед
                    (x > 0 ? input.at<uchar>(y, x - 1) : 0), // Левый сосед
                    0), 255);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    Mat inputImage = imread(argv[1], IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    Mat outputImageCPU = inputImage.clone();
    Mat outputImageGPU = inputImage.clone();
    auto startCPU = chrono::high_resolution_clock::now();
    laplacianFilterCPU(inputImage, outputImageCPU);
    auto endCPU = chrono::high_resolution_clock::now();
    auto durationCPU = chrono::duration_cast<chrono::milliseconds>(endCPU - startCPU).count();
    cout << "CPU processing time: " << durationCPU << " ms" << endl;

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    auto startGPU = chrono::high_resolution_clock::now();
    laplacianFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    auto endGPU = chrono::high_resolution_clock::now();
    auto durationGPU = chrono::duration_cast<chrono::milliseconds>(endGPU - startGPU).count();
    cout << "GPU processing time: " << durationGPU << " ms" << endl;

    cudaMemcpy(outputImageGPU.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    imwrite("output_cpu.png", outputImageCPU);
    imwrite("output_gpu.png", outputImageGPU);

    return 0;
}