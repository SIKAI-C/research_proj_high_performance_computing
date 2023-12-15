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

#include <chrono>

using namespace std::chrono;

#define N_WARPS_PER_BLOCK (1 << 2)
#define WARP_SIZE (1 << 5)
#define N_THREADS_PER_BLOCK (1 << 7)

void printVector(float* counting, int nrow) {
    for (int i = 0; i < nrow; i++) {
        std::cout << counting[i] << " ";
    }
    std::cout << "\n";
}

int generateASparseMatrixRandomly(int nrow, int ncol, float** result_matrix, int sparse_ratio) {
    float* A = (float*)malloc(sizeof(float) * nrow * ncol);
    int nnz = 0;
    int r;
    float* cur;
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            r = rand();
            cur = A + (i * ncol + j);
            if (r % sparse_ratio == 0) { *cur = 10.0 * (r / (double)RAND_MAX);; }
            else { *cur = 0.0f; }
            if (*cur != 0.0f) { nnz++; }
        }
    }
    *result_matrix = A;
    return nnz;
}

void convertToCSRFormat(float* mat, int nrow, int ncol, int nnz, int** ptr, int** indices, float** data) {
    int* row_ptr = (int*)malloc(sizeof(int) * (nrow + 1));
    int* col_ind = (int*)malloc(sizeof(int) * nnz);
    float* nz_val = (float*)malloc(sizeof(float) * nnz);
    float* cur;
    int count = 0;
    for (int i = 0; i < nrow; i++) {
        row_ptr[i] = count;
        for (int j = 0; j < ncol; j++) {
            cur = mat + (i * ncol + j);
            if (*cur != 0.0f) {
                col_ind[count] = j;
                nz_val[count] = *cur;
                count++;
            }
        }
    }
    row_ptr[nrow] = count;

    *ptr = row_ptr;
    *indices = col_ind;
    *data = nz_val;
    return;
}

__global__
void countingKernel(int n, int* d_counting, int* A_ptr, int* A_ind, int* B_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    if (idx < n) {
        int start = A_ptr[idx];
        int end = A_ptr[idx + 1];
        for (int i = start; i < end; i++) {
            int j = A_ind[i];
            d_counting[idx] = d_counting[idx] + B_ptr[j + 1] - B_ptr[j];
        }
    }
    __syncthreads();
}

__device__
void bubbleSortSwap(int* col_ind, int i, int j) {
    int temp = col_ind[i];
    col_ind[i] = col_ind[j];
    col_ind[j] = temp;
}

__device__
void bubbleSortSwap(float* ety_val, int i, int j) {
    float temp = ety_val[i];
    ety_val[i] = ety_val[j];
    ety_val[j] = temp;
}

__device__
void bubbleSortNetwork(int* col_ind, float* ety_val, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (col_ind[i] > col_ind[j]) {
                bubbleSortSwap(col_ind, i, j);
                bubbleSortSwap(ety_val, i, j);
            }
        }
    }
}

__device__
void contractionOperation(int* col_ind, float* ety_val, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (tid == 0 || col_ind[tid] != col_ind[tid - 1]) {
            float res = ety_val[tid];
            for (int j = tid + 1; j < n && col_ind[j] == col_ind[tid]; j++) {
                res += ety_val[j];
                col_ind[j] = 0;
                ety_val[j] = 0;
            }
            ety_val[tid] = res;
        }
    }
}

__global__
void expansionSortingContractionKernel(int* order, int* counting, int* operations,
                                        int* A_ptr, int* A_ind, float* A_val, int A_row, int A_col, int A_nnz,
                                        int* B_ptr, int* B_ind, float* B_val, int B_row, int B_col, int B_nnz,
                                        int* resulting_row, int* resulting_col, float* resulting_val) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (bid < A_row) {
        int row = order[bid];
        int A_row_start = A_ptr[row];
        int A_row_end = A_ptr[row + 1];
        int num_C_row_nnz = counting[bid];
        __shared__ float C_row_val[768];
        __shared__ int C_row_col[768];
        __shared__ int C_row_ind;
        if (tid == 0) {
            C_row_ind = 0;
        }
        __syncthreads();

        for (int entryIdx = tid + A_row_start; entryIdx < A_row_end; entryIdx += block_size) {
            int k = A_ind[entryIdx];
            int A_ik = A_val[entryIdx];
            int B_rowk_start = B_ptr[k];
            int B_rowk_end = B_ptr[k + 1];
            for (int i = B_rowk_start; i < B_rowk_end; i++) {
                int j = B_ind[i];
                int B_kj = B_val[i];
                int pos = atomicAdd(&C_row_ind, 1);
                C_row_val[pos] = A_ik * B_kj;
                C_row_col[pos] = j;
            }
        }

        __syncthreads();

        if (num_C_row_nnz <= 32) {
            if (tid == 0) {
                bubbleSortNetwork(C_row_col, C_row_val, num_C_row_nnz);
            }
        }
        else {
            typedef cub::BlockRadixSort<int, N_THREADS_PER_BLOCK, 6, float> BlockRadixSort;
            // Allocate shared memory for BlockRadixSort
            __shared__ typename BlockRadixSort::TempStorage temp_storage;
            // Collectively sort the keys
            BlockRadixSort(temp_storage).Sort(*static_cast<int(*)[6]>(static_cast<void*>(C_row_col + 6 * threadIdx.x)), *static_cast<float(*)[6]>(static_cast<void*>(C_row_val + 6 * threadIdx.x)));
        }

        contractionOperation(C_row_col, C_row_val, num_C_row_nnz);
        for (int resIdx = tid; resIdx < num_C_row_nnz; resIdx += block_size) {
            if (C_row_val[resIdx] != 0.f) {
                resulting_row[operations[bid] + resIdx] = row;
                resulting_col[operations[bid] + resIdx] = C_row_col[resIdx];
                resulting_val[operations[bid] + resIdx] = C_row_val[resIdx];
            }
        }
    }
    __syncthreads();
}

int SPMMM(int* A_ptr, int* A_ind, float* A_val, int A_row, int A_col, int A_nnz,
            int* B_ptr, int* B_ind, float* B_val, int B_row, int B_col, int B_nnz,
            int** result_row, int** result_col, float** result_val,
            float* recorded_time) {


    auto tot_start_time = high_resolution_clock::now();

    int* d_A_ptr, * d_A_ind, * d_B_ptr, * d_B_ind;
    float* d_A_val, * d_B_val;
    cudaMalloc(&d_A_ptr, (A_row + 1) * sizeof(int));
    cudaMalloc(&d_A_ind, A_nnz * sizeof(int));
    cudaMalloc(&d_A_val, A_nnz * sizeof(float));
    cudaMalloc(&d_B_ptr, (B_row + 1) * sizeof(int));
    cudaMalloc(&d_B_ind, B_nnz * sizeof(int));
    cudaMalloc(&d_B_val, B_nnz * sizeof(float));
    cudaMemcpy(d_A_ptr, A_ptr, (A_row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_ind, A_ind, A_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_val, A_val, A_nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ptr, B_ptr, (B_row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ind, B_ind, B_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_val, B_val, B_nnz * sizeof(float), cudaMemcpyHostToDevice);


    auto setup_start_time = high_resolution_clock::now();
    int* counting = (int*)malloc(A_row * sizeof(int));
    int* d_counting;
    cudaMalloc(&d_counting, A_row * sizeof(int));
    cudaMemset(d_counting, 0, A_row * sizeof(int));
    int set_up_n_blocks = (A_row + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    countingKernel <<< set_up_n_blocks, N_THREADS_PER_BLOCK >>> (A_row, d_counting, d_A_ptr, d_A_ind, d_B_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(counting, d_counting, A_row * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_ptr<int> d_counting_ptr(d_counting);
    thrust::device_vector<int> d_counting_vec(d_counting_ptr, d_counting_ptr + A_row);
    thrust::device_vector<int> d_order_vec(A_row);
    thrust::sequence(d_order_vec.begin(), d_order_vec.end());
    thrust::stable_sort_by_key(d_counting_vec.begin(), d_counting_vec.end(), d_order_vec.begin(), thrust::greater<int>());
    thrust::device_vector<int> d_operations_vec(A_row + 1);
    thrust::exclusive_scan(d_counting_vec.begin(), d_counting_vec.end(), d_operations_vec.begin());
    int tot_operations = d_counting_vec.back() + d_operations_vec[A_row - 1];
    d_operations_vec[A_row] = tot_operations;
    auto setup_end_time = high_resolution_clock::now();

    int* order = (int*)malloc(A_row * sizeof(int));
    int* operations = (int*)malloc((A_row + 1) * sizeof(int));
    thrust::copy(d_order_vec.begin(), d_order_vec.end(), order);
    thrust::copy(d_operations_vec.begin(), d_operations_vec.end(), operations);
    thrust::copy(d_counting_vec.begin(), d_counting_vec.end(), counting);

    if (counting[0] > 768) {
        return 1;
    }

    int* d_order;
    int* d_operations;
    cudaMalloc(&d_order, A_row * sizeof(int));
    cudaMalloc(&d_operations, (A_row + 1) * sizeof(int));
    cudaMemcpy(d_order, order, A_row * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_operations, operations, (A_row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counting, counting, A_row * sizeof(int), cudaMemcpyHostToDevice);

    int* d_resulting_row, * d_resulting_col;
    float* d_resulting_val;
    cudaMalloc(&d_resulting_row, tot_operations * sizeof(int));
    cudaMalloc(&d_resulting_col, tot_operations * sizeof(int));
    cudaMalloc(&d_resulting_val, tot_operations * sizeof(float));

    int enpansion_and_sorting_n_blocks = A_row;

    auto esc_start_time = high_resolution_clock::now();
    expansionSortingContractionKernel <<< enpansion_and_sorting_n_blocks, N_THREADS_PER_BLOCK >>> (d_order, d_counting, d_operations,
                                                                                                    d_A_ptr, d_A_ind, d_A_val, A_row, A_col, A_nnz,
                                                                                                    d_B_ptr, d_B_ind, d_B_val, B_row, B_col, B_nnz,
                                                                                                    d_resulting_row, d_resulting_col, d_resulting_val);
    cudaDeviceSynchronize();
    
    int* resulting_row = (int*)malloc(tot_operations * sizeof(int));
    int* resulting_col = (int*)malloc(tot_operations * sizeof(int));
    float* resulting_val = (float*)malloc(tot_operations * sizeof(float));
    cudaMemcpy(resulting_row, d_resulting_row, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resulting_col, d_resulting_col, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resulting_val, d_resulting_val, tot_operations * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A_ptr);
    cudaFree(d_A_ind);
    cudaFree(d_A_val);
    cudaFree(d_B_ptr);
    cudaFree(d_B_ind);
    cudaFree(d_B_val);
    cudaFree(d_counting);
    cudaFree(d_order);
    cudaFree(d_operations);
    cudaFree(d_resulting_row);
    cudaFree(d_resulting_col);
    cudaFree(d_resulting_val);

    auto esc_end_time = high_resolution_clock::now();

    free(A_ptr);
    free(A_ind);
    free(A_val);
    free(B_ptr);
    free(B_ind);
    free(B_val);
    free(counting);
    free(order);
    free(operations);

    auto tot_end_time = high_resolution_clock::now();
    
    auto tot_elapsed_time = duration_cast<duration<double>>(tot_end_time - tot_start_time);
    auto setup_elapsed_time = duration_cast<duration<double>>(setup_end_time - setup_start_time);
    auto esc_elapsed_time = duration_cast<duration<double>>(esc_end_time - esc_start_time);
    float tot_elapsed_time_sec = static_cast<float>(tot_elapsed_time.count());
    float setup_elapsed_time_sec = static_cast<float>(setup_elapsed_time.count());
    float esc_elapsed_time_sec = static_cast<float>(esc_elapsed_time.count());

    *result_row = resulting_row;
    *result_col = resulting_col;
    *result_val = resulting_val;

    free(resulting_row);
    free(resulting_col);
    free(resulting_val);

    recorded_time[0] += tot_elapsed_time_sec;
    recorded_time[1] += setup_elapsed_time_sec;
    recorded_time[2] += esc_elapsed_time_sec;

    return 0;

}

int main() {

    int num_matrix_size = 10;
    int* matrix_size_list = (int*)malloc(num_matrix_size * sizeof(int));
    for (int i = 0; i < num_matrix_size; i++) {
        matrix_size_list[i] = 10 * (i + 1);
    }

    int num_sparse_ratio = 10;
    int* sparse_ratio_list = (int*)malloc(num_sparse_ratio * sizeof(int));
    for (int i = 0; i < num_sparse_ratio; i++) {
        sparse_ratio_list[i] = 23 - 2*i;
    }

    int repeat_times = 10;

    float* time1_tot = (float*)malloc(num_matrix_size * num_sparse_ratio * repeat_times * sizeof(float));
    float* time1_setup = (float*)malloc(num_matrix_size * num_sparse_ratio * repeat_times * sizeof(float));
    float* time1_esc = (float*)malloc(num_matrix_size * num_sparse_ratio * repeat_times * sizeof(float));

    for (int i = 0; i < num_matrix_size; i++) {
        std::cout << " --- Matrix size: " << matrix_size_list[i] << " --- \n";
        for (int j = 0; j < num_sparse_ratio; j++) {
            std::cout << "       *** Sparse ratio: " << sparse_ratio_list[j] << " *** \n";
            for (int k = 0; k < repeat_times; k++) {
                int mz = matrix_size_list[i];
                int sr = sparse_ratio_list[j];

                int A_row = mz;
                int A_col = mz;
                int B_row = mz;
                int B_col = mz;
                float* A;
                float* B;
                int A_nnz = generateASparseMatrixRandomly(A_row, A_col, &A, sr);
                int B_nnz = generateASparseMatrixRandomly(B_row, B_col, &B, sr);
                int* A_ptr;
                int* A_ind;
                float* A_val;
                convertToCSRFormat(A, A_row, A_col, A_nnz, &A_ptr, &A_ind, &A_val);
                int* B_ptr;
                int* B_ind;
                float* B_val;
                convertToCSRFormat(B, B_row, B_col, B_nnz, &B_ptr, &B_ind, &B_val);

                float* time = (float*)malloc(3 * sizeof(float));
                for (int i = 0; i < 3; i++) {
                    time[i] = 0;
                }
                int* resulting_row;
                int* resulting_col;
                float* resulting_val;
                int exit_code = SPMMM(A_ptr, A_ind, A_val, A_row, A_col, A_nnz,
                                    B_ptr, B_ind, B_val, B_row, B_col, B_nnz,
                                    &resulting_row, &resulting_col, &resulting_val, time);
                
                if (exit_code == 0) {
                    time1_tot[i * num_sparse_ratio * repeat_times + j * repeat_times + k] = time[0];
                    time1_setup[i * num_sparse_ratio * repeat_times + j * repeat_times + k] = time[1];
                    time1_esc[i * num_sparse_ratio * repeat_times + j * repeat_times + k] = time[2];
                    std::cout << "             Experiment - " << i * num_sparse_ratio * repeat_times + j * repeat_times + k << " | Time: " << time[0] << " | Setup: " << time[1] << " | ESC: " << time[2] << " \n";
                }
                else {
                    time1_tot[i * num_sparse_ratio * repeat_times + j * repeat_times + k] = 0;
                    time1_setup[i * num_sparse_ratio * repeat_times + j * repeat_times + k] = 0;
                    time1_esc[i * num_sparse_ratio * repeat_times + j * repeat_times + k] = 0;
                    std::cout << "             Experiment - " << i * num_sparse_ratio * repeat_times + j * repeat_times + k << " | ERROR -> TRY NEXT \n";
                }
            }
        }
    }

    std::cout << " ####################### \n";
    std::cout << " ### PRINT THE TIME1 ### \n";
    std::cout << " ####################### \n";
    std::cout << " --- Time1 TOT --- \n";
    printVector(time1_tot, num_matrix_size * num_sparse_ratio * repeat_times);
    std::cout << " --- Time1 SETUP --- \n";
    printVector(time1_setup, num_matrix_size * num_sparse_ratio * repeat_times);
    std::cout << " --- Time1 ESC --- \n";
    printVector(time1_esc, num_matrix_size * num_sparse_ratio * repeat_times);

    return 0;
}