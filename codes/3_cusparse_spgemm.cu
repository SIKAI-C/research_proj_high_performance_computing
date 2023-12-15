#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <cusparse_v2.h>
#include <cuda.h>

#include <chrono>
using namespace std::chrono;

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

void SPMMM_COMP(int* A_ptr, int* A_ind, float* A_val, int A_row, int A_col, int A_nnz,
    int* B_ptr, int* B_ind, float* B_val, int B_row, int B_col, int B_nnz,
    int** result_row, int** result_col, float** result_val, 
    float* recorded_time) {

    auto tot_start_time = high_resolution_clock::now();

    float alpha = 1.0f, beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    int* dA_csrOffsets, * dA_columns, * dB_csrOffsets, * dB_columns, * dC_csrOffsets, * dC_columns;
    float* dA_values, * dB_values, * dC_values;

    cudaMalloc((void**)&dA_csrOffsets, sizeof(int) * (A_row + 1));
    cudaMalloc((void**)&dA_columns, sizeof(int) * A_nnz);
    cudaMalloc((void**)&dA_values, sizeof(float) * A_nnz);
    cudaMalloc((void**)&dB_csrOffsets, sizeof(int) * (B_row + 1));
    cudaMalloc((void**)&dB_columns, sizeof(int) * B_nnz);
    cudaMalloc((void**)&dB_values, sizeof(float) * B_nnz);
    cudaMalloc((void**)&dC_csrOffsets, sizeof(int) * (A_row + 1));

    cudaMemcpy(dA_csrOffsets, A_ptr, sizeof(int) * (A_row + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, A_ind, sizeof(int) * A_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, A_val, sizeof(float) * A_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB_csrOffsets, B_ptr, sizeof(int) * (B_row + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_columns, B_ind, sizeof(int) * B_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB_values, B_val, sizeof(float) * B_nnz, cudaMemcpyHostToDevice);

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void* dBuffer1 = NULL, * dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    cusparseCreate(&handle);

    cusparseCreateCsr(&matA, A_row, A_col, A_nnz, dA_csrOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matB, B_row, B_col, B_nnz, dB_csrOffsets, dB_columns, dB_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matC, A_row, B_col, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    auto compute_start_time = high_resolution_clock::now();
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);
    cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void**)&dBuffer1, bufferSize1);
    cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1);
    cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL);
    cudaMalloc((void**)&dBuffer2, bufferSize2);
    cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2);
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
    cudaMalloc((void**)&dC_columns, C_nnz1 * sizeof(int));
    cudaMalloc((void**)&dC_values, C_nnz1 * sizeof(float));
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);
    cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);
    auto compute_end_time = high_resolution_clock::now();
    

    int* C_ptr = (int*)malloc(sizeof(int) * (A_row + 1));
    int* C_ind = (int*)malloc(sizeof(int) * C_nnz1);
    float* C_val = (float*)malloc(sizeof(float) * C_nnz1);

    cudaMemcpy(C_ptr, dC_csrOffsets, sizeof(int) * (A_row + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ind, dC_columns, sizeof(int) * C_nnz1, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_val, dC_values, sizeof(float) * C_nnz1, cudaMemcpyDeviceToHost);

    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dB_csrOffsets);
    cudaFree(dB_columns);
    cudaFree(dB_values);
    cudaFree(dC_csrOffsets);
    cudaFree(dC_columns);
    cudaFree(dC_values);

    *result_row = C_ptr;
    *result_col = C_ind;
    *result_val = C_val;

    auto tot_end_time = std::chrono::high_resolution_clock::now();
    auto tot_elapsed_time = duration_cast<duration<double>>(tot_end_time - tot_start_time);
    float tot_elapsed_time_sec = static_cast<float>(tot_elapsed_time.count());

    auto compute_elapsed_time = duration_cast<duration<double>>(compute_end_time - compute_start_time);
    float compute_elapsed_time_sec = static_cast<float>(compute_elapsed_time.count());

    recorded_time[0] += tot_elapsed_time_sec;
    recorded_time[1] += compute_elapsed_time_sec;

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

    float* time2_tot = (float*)malloc(num_matrix_size * num_sparse_ratio * repeat_times * sizeof(float));
    float* time2_compute = (float*)malloc(num_matrix_size * num_sparse_ratio * repeat_times * sizeof(float));

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

                float* time = (float*)malloc(2 * sizeof(float));
                for (int i = 0; i < 2; i++) {
                    time[i] = 0;
                }
                int* resulting_row;
                int* resulting_col;
                float* resulting_val;
                SPMMM_COMP(A_ptr, A_ind, A_val, A_row, A_col, A_nnz,
                        B_ptr, B_ind, B_val, B_row, B_col, B_nnz,
                        &resulting_row, &resulting_col, &resulting_val, time);
                
                time2_tot[i * num_sparse_ratio * repeat_times + j * repeat_times + k] = time[0];
                time2_compute[i * num_sparse_ratio * repeat_times + j * repeat_times + k] = time[1];
                std::cout << "             Experiment - " << i * num_sparse_ratio * repeat_times + j * repeat_times + k << " | Time: " << time[0] << " | Compute: " << time[1] << " \n";

            }
        }
    }
}