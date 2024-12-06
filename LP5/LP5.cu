#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

__global__ void laplacianFilter(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;

        // Применяем ядро фильтра Лапласа H1
        int laplacian_value = input[(y - 1) * width + x] +    // верхний сосед
                              input[y * width + (x - 1)] +    // левый сосед
                              input[y * width + (x + 1)] +    // правый сосед
                              input[(y + 1) * width + x] +    // нижний сосед
                              -4 * input[idx];               // центральный пиксель

        // Ограничиваем значение результата в диапазоне [0, 255]
        output[idx] = min(max(laplacian_value, 0), 255);
    }
}

void laplacianFilterCPU(const Mat &input, Mat &output) {
    for (int y = 1; y < input.rows - 1; ++y) {
        for (int x = 1; x < input.cols - 1; ++x) {
            int laplacian_value = input.at<uchar>(y - 1, x) +
                                  input.at<uchar>(y, x - 1) +
                                  input.at<uchar>(y, x + 1) +
                                  input.at<uchar>(y + 1, x) +
                                  -4 * input.at<uchar>(y, x);
            output.at<uchar>(y, x) = min(max(laplacian_value, 0), 255);
        }
    }
}

Mat cropImageIfNeeded(const Mat &image) {
    int width = image.cols;
    int height = image.rows;

    // Проверка на квадратность и кратность 16
    if (width == height || width % 16 == 0 || height % 16 == 0) {
        // Обрезаем изображение, уменьшая ширину и высоту на 1
        int new_width = (width % 16 == 0) ? width - 1 : width;
        int new_height = (height % 16 == 0) ? height - 1 : height;

        // Если изображение квадратное, также уменьшаем его размер
        if (new_width == new_height) {
            new_width -= 1;
            new_height -= 1;
        }

        return image(Rect(0, 0, new_width, new_height)).clone();
    }

    return image; // Возвращаем оригинальное изображение, если оно подходит
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

    // Обрезаем изображение, если оно не подходит
    inputImage = cropImageIfNeeded(inputImage);

    int width = inputImage.cols;
    int height = inputImage.rows;

    Mat outputImageCPU = inputImage.clone();
    Mat outputImageGPU = inputImage.clone();

    // CPU filtering
    auto startCPU = chrono::high_resolution_clock::now();
    laplacianFilterCPU(inputImage, outputImageCPU);
    auto endCPU = chrono::high_resolution_clock::now();
    auto durationCPU = chrono::duration_cast<chrono::milliseconds>(endCPU - startCPU).count();
    cout << "CPU processing time: " << durationCPU << " ms" << endl;

    // GPU filtering
    unsigned char *d_input, *d_output;

    // Allocate GPU memory
    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    // Copy input image to GPU
    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    auto startGPU = chrono::high_resolution_clock::now();
    laplacianFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    auto endGPU = chrono::high_resolution_clock::now();
    auto durationGPU = chrono::duration_cast<chrono::milliseconds>(endGPU - startGPU).count();
    cout << "GPU processing time: " << durationGPU << " ms" << endl;

    // Copy output image back to CPU
    cudaMemcpy(outputImageGPU.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Save results
    imwrite("output_cpu.png", outputImageCPU);
    imwrite("output_gpu.png", outputImageGPU);

    return 0;
}