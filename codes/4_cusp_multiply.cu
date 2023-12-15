#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <cub/cub.cuh>
#include <cusp/array2d.h>
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include <cusp/gallery/poisson.h>
#include <cusp/gallery/random.h>
#include <time.h>

#include <chrono>

using namespace std::chrono;

#define N_WARPS_PER_BLOCK (1 << 2)
#define WARP_SIZE (1 << 5)
#define N_THREADS_PER_BLOCK (1 << 7)


int main() {
    for (size_t i = 1; i <= 10; i++)
    {
        std::cout << "--- Matrix size: " << i << " -- - " << std::endl;
        for (size_t j = 5; j < 24; j++)
        {
            std::cout << "  *** Sparse ratio:" << j << "* **" << std::endl;
            int mz = i * 10;
            cusp::csr_matrix<int, float, cusp::device_memory> c_a;
            cusp::gallery::random(c_a, mz, mz, mz * j / 100);
            cusp::csr_matrix<int, float, cusp::device_memory> c_b;
            cusp::gallery::random(c_b, mz, mz, mz * j / 100);
            cusp::csr_matrix<int, float, cusp::device_memory> c_c;

            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();
            cusp::multiply(c_a, c_b, c_c);
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "matrxi size: " << i * 10 << ", cusp::multiply: " << elapsed_seconds.count() << " s" << std::endl;
        }
    }

    return 0;
}