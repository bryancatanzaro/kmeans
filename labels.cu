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
