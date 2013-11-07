#include "labels.h"

namespace kmeans {
namespace detail {

cublasHandle_t cublas_handle;

void labels_init() {
    cublasStatus_t stat;
    stat = cublasCreate(&detail::cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed" << std::endl;
        exit(1);
    }
}

void gemm(cublasOperation_t transa,
          cublasOperation_t transb,
          int m, int n, int k,
          const float *alpha,
          const float *A, int lda,
          const float *B, int ldb,
          const float *beta,
          float *C, int ldc) {
    cublasStatus_t status = cublasSgemm(cublas_handle, transa, transb,
                                        m, n, k, alpha,
                                        A, lda,
                                        B, ldb,
                                        beta,
                                        C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Invalid Sgemm" << std::endl;
        exit(1);
    }
}

void gemm(cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m, int n, int k,
                    const double *alpha,
                    const double *A, int lda,
                    const double *B, int ldb,
                    const double *beta,
                    double *C, int ldc) {
    cublasStatus_t status = cublasDgemm(cublas_handle, transa, transb,
                                        m, n, k, alpha,
                                        A, lda,
                                        B, ldb,
                                        beta, 
                                        C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Invalid Sgemm" << std::endl;
        exit(1);
    }
}

}
}
