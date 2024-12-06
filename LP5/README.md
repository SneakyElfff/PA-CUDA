nvcc -o main LP5.cu `pkg-config --cflags --libs opencv4`
./main 1.jpg
