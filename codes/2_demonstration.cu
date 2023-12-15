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

#define N_WARPS_PER_BLOCK (1 << 2)
#define WARP_SIZE (1 << 5)
#define N_THREADS_PER_BLOCK (1 << 7)

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

void presentAMatrix(float* mat, int nrow, int ncol, int present_row, int present_col) {
    printf(" --- PRINT THE MATRIX ---- \n");
    for (int i = 0; i < present_row; i++) {
        for (int j = 0; j < present_col; j++) {
            printf("%3.0f", mat[i * ncol + j]);
        }
        printf("...\n");
    }
    printf("...\n");
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

void presentCSR(int* ptr, int* indices, float* data, int nnz, int nrow) {
    printf(" --- PRINT THE CSR FORMAT MATRIX ---- \n");
    printf("ptr - ");
    for (int i = 0; i <= nrow; i++) {
        printf("%+5d", ptr[i]);
    }
    printf("\n");
    printf("indices - ");
    for (int i = 0; i < nnz; i++) {
        printf("%+5d", indices[i]);
    }
    printf("\n");
    printf("data - ");
    for (int i = 0; i < nnz; i++) {
        printf("%+5f", data[i]);
    }
    printf("\n");
}

void print(const thrust::device_vector<int>& v)
{
    for (size_t i = 0; i < v.size(); i++)
        std::cout << " " << v[i];
    std::cout << "\n";
}

void print(const thrust::host_vector<int>& v)
{
    for (size_t i = 0; i < v.size(); i++)
        std::cout << " " << v[i];
    std::cout << "\n";
}

void print(const thrust::device_vector<float>& v)
{
    for (size_t i = 0; i < v.size(); i++)
        std::cout << " " << std::fixed << std::setprecision(1) << v[i];
    std::cout << "\n";
}

void print(const thrust::host_vector<float>& v)
{
    for (size_t i = 0; i < v.size(); i++)
        std::cout << " " << std::fixed << std::setprecision(1) << v[i];
    std::cout << "\n";
}

void print(thrust::device_vector<int>& v1, thrust::device_vector<int>& v2)
{
    for (size_t i = 0; i < v1.size(); i++)
        std::cout << " (" << v1[i] << "," << std::setw(2) << v2[i] << ")";
    std::cout << "\n";
}

void printVector(int* counting, int nrow) {
    for (int i = 0; i < nrow; i++) {
        std::cout << counting[i] << " ";
    }
    std::cout << "\n";
}

void printVector(float* counting, int nrow) {
    for (int i = 0; i < nrow; i++) {
        std::cout << counting[i] << " ";
    }
    std::cout << "\n";
}

__device__
void printVectorDevice(int* vec, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

__device__
void printVectorDevice(float* vec, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", vec[i]);
    }
    printf("\n");
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
    int* phase1_row, int* phase1_col, float* phase1_val,
    int* phase2_row, int* phase2_col, float* phase2_val,
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


        for (int resIdx = tid; resIdx < num_C_row_nnz; resIdx += block_size) {
            phase1_row[operations[bid] + resIdx] = row;
            phase1_col[operations[bid] + resIdx] = C_row_col[resIdx];
            phase1_val[operations[bid] + resIdx] = C_row_val[resIdx];
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

        for (int resIdx = tid; resIdx < num_C_row_nnz; resIdx += block_size) {
            phase2_row[operations[bid] + resIdx] = row;
            phase2_col[operations[bid] + resIdx] = C_row_col[resIdx];
            phase2_val[operations[bid] + resIdx] = C_row_val[resIdx];
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

struct removed_item {
    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) {
        return (thrust::get<2>(t) == 0);
    }
};


int main() {

    int A_row = 7;
    int A_col = 7;
    int B_row = 7;
    int B_col = 7;
    int present_row = 7;
    int present_col = 7;
    float* A;
    float* B;
    int A_nnz = generateASparseMatrixRandomly(A_row, A_col, &A, 4);
    int B_nnz = generateASparseMatrixRandomly(B_row, B_col, &B, 4);
    presentAMatrix(A, A_row, A_col, present_row, present_col);
    presentAMatrix(B, B_row, B_col, present_row, present_col);
    int* A_ptr;
    int* A_ind;
    float* A_val;
    convertToCSRFormat(A, A_row, A_col, A_nnz, &A_ptr, &A_ind, &A_val);
    presentCSR(A_ptr, A_ind, A_val, A_nnz, A_row);
    int* B_ptr;
    int* B_ind;
    float* B_val;
    convertToCSRFormat(B, B_row, B_col, B_nnz, &B_ptr, &B_ind, &B_val);
    presentCSR(B_ptr, B_ind, B_val, B_nnz, B_row);

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

    int* counting = (int*)malloc(A_row * sizeof(int));
    int* d_counting;
    cudaMalloc(&d_counting, A_row * sizeof(int));
    cudaMemset(d_counting, 0, A_row * sizeof(int));
    int set_up_n_blocks = (A_row + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    countingKernel <<< set_up_n_blocks, N_THREADS_PER_BLOCK >>> (A_row, d_counting, d_A_ptr, d_A_ind, d_B_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(counting, d_counting, A_row * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << " --- SET UP --- " << "\n";
    std::cout << "original counting: ";
    printVector(counting, A_row);

    thrust::device_ptr<int> d_counting_ptr(d_counting);
    thrust::device_vector<int> d_counting_vec(d_counting_ptr, d_counting_ptr + A_row);
    thrust::device_vector<int> d_order_vec(A_row);
    thrust::sequence(d_order_vec.begin(), d_order_vec.end());
    thrust::stable_sort_by_key(d_counting_vec.begin(), d_counting_vec.end(), d_order_vec.begin(), thrust::greater<int>());
    thrust::device_vector<int> d_operations_vec(A_row + 1);
    thrust::exclusive_scan(d_counting_vec.begin(), d_counting_vec.end(), d_operations_vec.begin());
    int tot_operations = d_counting_vec.back() + d_operations_vec[A_row - 1];
    d_operations_vec[A_row] = tot_operations;

    std::cout << " --- AFTER REORDER - THRUST VECTOR" << "\n";
    std::cout << "d_counting_vec: ";
    print(d_counting_vec);
    std::cout << "d_order_vec: ";
    print(d_order_vec);
    std::cout << "d_operations_vec: ";
    print(d_operations_vec);

    int* order = (int*)malloc(A_row * sizeof(int));
    int* operations = (int*)malloc((A_row + 1) * sizeof(int));
    thrust::copy(d_order_vec.begin(), d_order_vec.end(), order);
    thrust::copy(d_operations_vec.begin(), d_operations_vec.end(), operations);
    thrust::copy(d_counting_vec.begin(), d_counting_vec.end(), counting);
    std::cout << " --- AFTER REORDER - CONVERT TO ptr" << "\n";
    std::cout << "order: ";
    printVector(order, A_row);
    std::cout << "counting: ";
    printVector(counting, A_row);
    std::cout << "operations: ";
    printVector(operations, A_row + 1);
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

    int* d_phase1_row, * d_phase1_col;
    float* d_phase1_val;
    cudaMalloc(&d_phase1_row, tot_operations * sizeof(int));
    cudaMalloc(&d_phase1_col, tot_operations * sizeof(int));
    cudaMalloc(&d_phase1_val, tot_operations * sizeof(float));

    int* d_phase2_row, * d_phase2_col;
    float* d_phase2_val;
    cudaMalloc(&d_phase2_row, tot_operations * sizeof(int));
    cudaMalloc(&d_phase2_col, tot_operations * sizeof(int));
    cudaMalloc(&d_phase2_val, tot_operations * sizeof(float));


    int enpansion_and_sorting_n_blocks = A_row;
    expansionSortingContractionKernel <<< enpansion_and_sorting_n_blocks, N_THREADS_PER_BLOCK >>> (d_order, d_counting, d_operations,
        d_A_ptr, d_A_ind, d_A_val, A_row, A_col, A_nnz,
        d_B_ptr, d_B_ind, d_B_val, B_row, B_col, B_nnz,
        d_phase1_row, d_phase1_col, d_phase1_val,
        d_phase2_row, d_phase2_col, d_phase2_val,
        d_resulting_row, d_resulting_col, d_resulting_val);
    cudaDeviceSynchronize();
    int* resulting_row = (int*)malloc(tot_operations * sizeof(int));
    int* resulting_col = (int*)malloc(tot_operations * sizeof(int));
    float* resulting_val = (float*)malloc(tot_operations * sizeof(float));
    cudaMemcpy(resulting_row, d_resulting_row, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resulting_col, d_resulting_col, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(resulting_val, d_resulting_val, tot_operations * sizeof(float), cudaMemcpyDeviceToHost);

    int* phase1_row = (int*)malloc(tot_operations * sizeof(int));
    int* phase1_col = (int*)malloc(tot_operations * sizeof(int));
    float* phase1_val = (float*)malloc(tot_operations * sizeof(float));
    cudaMemcpy(phase1_row, d_phase1_row, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phase1_col, d_phase1_col, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phase1_val, d_phase1_val, tot_operations * sizeof(float), cudaMemcpyDeviceToHost);

    int* phase2_row = (int*)malloc(tot_operations * sizeof(int));
    int* phase2_col = (int*)malloc(tot_operations * sizeof(int));
    float* phase2_val = (float*)malloc(tot_operations * sizeof(float));
    cudaMemcpy(phase2_row, d_phase2_row, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phase2_col, d_phase2_col, tot_operations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phase2_val, d_phase2_val, tot_operations * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << " --- AFTER EXPANSION and BEFORE SORTING" << "\n";
    std::cout << "phase1_row: ";
    printVector(phase1_row, tot_operations);
    std::cout << "phase1_col: ";
    printVector(phase1_col, tot_operations);
    std::cout << "phase1_val: ";
    printVector(phase1_val, tot_operations);

    std::cout << " --- AFTER SORTING and BEFORE CONTRACTION" << "\n";
    std::cout << "phase2_row: ";
    printVector(phase2_row, tot_operations);
    std::cout << "phase2_col: ";
    printVector(phase2_col, tot_operations);
    std::cout << "phase2_val: ";
    printVector(phase2_val, tot_operations);

    std::cout << " --- FINALLY --- " << "\n";
    std::cout << "resulting_row: ";
    printVector(resulting_row, tot_operations);
    std::cout << "resulting_col: ";
    printVector(resulting_col, tot_operations);
    std::cout << "resulting_val: ";
    printVector(resulting_val, tot_operations);

    std::cout << "--- COO FORMAT --- " << "\n";
    thrust::host_vector<int> coo_row(resulting_row, resulting_row + tot_operations);
    thrust::host_vector<int> coo_col(resulting_col, resulting_col + tot_operations);
    thrust::host_vector<float> coo_val(resulting_val, resulting_val + tot_operations);

    thrust::device_vector<int> d_coo_row = coo_row;
    thrust::device_vector<int> d_coo_col = coo_col;
    thrust::device_vector<float> d_coo_val = coo_val;

    typedef thrust::device_vector<int>::iterator IntIterator;
    typedef thrust::device_vector<float>::iterator FloatIterator;
    typedef thrust::tuple<IntIterator, IntIterator, FloatIterator> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

    ZipIterator first = thrust::make_zip_iterator(thrust::make_tuple(d_coo_row.begin(), d_coo_col.begin(), d_coo_val.begin()));
    ZipIterator last = thrust::make_zip_iterator(thrust::make_tuple(d_coo_row.end(), d_coo_col.end(), d_coo_val.end()));

    ZipIterator new_last = thrust::remove_if(first, last, removed_item());

    d_coo_row.erase(new_last.get_iterator_tuple().get<0>(), d_coo_row.end());
    d_coo_col.erase(new_last.get_iterator_tuple().get<1>(), d_coo_col.end());
    d_coo_val.erase(new_last.get_iterator_tuple().get<2>(), d_coo_val.end());

    thrust::host_vector<int> h_coo_row = d_coo_row;
    thrust::host_vector<int> h_coo_col = d_coo_col;
    thrust::host_vector<float> h_coo_val = d_coo_val;

    std::cout << "coo_row: ";
    print(h_coo_row);
    std::cout << "coo_col: ";
    print(h_coo_col);
    std::cout << "coo_val: ";
    print(h_coo_val);
    return 0;
}