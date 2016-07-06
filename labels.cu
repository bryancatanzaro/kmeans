/*
Copyright 2013  Bryan Catanzaro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "labels.h"

namespace kmeans {
namespace detail {

struct cublas_state {
    cublasHandle_t cublas_handle;
    cublas_state() {
        cublasStatus_t stat;
        stat = cublasCreate(&cublas_handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "CUBLAS initialization failed" << std::endl;
            exit(1);
        }
    }
    ~cublas_state() {
        cublasStatus_t stat;
        stat = cublasDestroy(cublas_handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "CUBLAS destruction failed" << std::endl;
            exit(1);
        }
    }
};


cublas_state state;

void gemm(cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k, const float *alpha,
          const float *A, int lda, const float *B, int ldb,
          const float *beta,
          float *C, int ldc) {
    cublasStatus_t status = cublasSgemm(state.cublas_handle, transa, transb,
                                        m, n, k, alpha,
                                        A, lda, B, ldb,
                                        beta,
                                        C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Invalid Sgemm" << std::endl;
        exit(1);
    }
}

void gemm(cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k, const double *alpha,
          const double *A, int lda, const double *B, int ldb,
          const double *beta,
          double *C, int ldc) {
    cublasStatus_t status = cublasDgemm(state.cublas_handle, transa, transb,
                                        m, n, k, alpha,
                                        A, lda, B, ldb,
                                        beta, 
                                        C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Invalid Dgemm" << std::endl;
        exit(1);
    }
}

}
}
