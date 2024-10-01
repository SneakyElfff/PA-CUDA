#include <iostream>
#include <array>
#include <bitset>

void cpuid(int info[4], int level) {
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(level)
    );
}

void printCPUIDFeatures() {
    int cpuInfo[4];

    // Уровень 0: информация о процессоре
    cpuid(cpuInfo, 0);
    std::cout << "Processor ID: " << cpuInfo[1] << std::endl;

    // Уровень 1: основные функции процессора
    cpuid(cpuInfo, 1);
    std::bitset<32> features(cpuInfo[2]);

    std::cout << "Supported features (level 1):" << std::endl;
    std::cout << "SSE: " << features[25] << std::endl;
    std::cout << "SSE2: " << features[26] << std::endl;
    std::cout << "SSE3: " << features[0] << std::endl;
    std::cout << "SSSE3: " << features[9] << std::endl;
    std::cout << "SSE4.1: " << features[19] << std::endl;
    std::cout << "SSE4.2: " << features[20] << std::endl;
    std::cout << "AVX: " << features[27] << std::endl;
    std::cout << "AVX2: " << features[28] << std::endl;
    std::cout << "AVX512F: " << (cpuInfo[2] & (1 << 16)) << std::endl; // AVX-512

    // Уровень 7: расширенные функции
    cpuid(cpuInfo, 7);
    std::cout << "Supported features (level 7):" << std::endl;
    std::cout << "AVX2: " << (cpuInfo[1] & (1 << 5)) << std::endl; // AVX2
    std::cout << "AVX512F: " << (cpuInfo[1] & (1 << 16)) << std::endl; // AVX-512
}

int main() {
    printCPUIDFeatures();
    return 0;
}

// #include <iostream>
// #include <immintrin.h>

// int main() {
//     if (__builtin_cpu_supports("sse")) std::cout << "SSE supported" << std::endl;
//     if (__builtin_cpu_supports("sse2")) std::cout << "SSE2 supported" << std::endl;
//     if (__builtin_cpu_supports("avx")) std::cout << "AVX supported" << std::endl;
//     if (__builtin_cpu_supports("avx2")) std::cout << "AVX2 supported" << std::endl;
//     return 0;
// }