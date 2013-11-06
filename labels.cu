#include "labels.h"
#include <cublas_v2.h>

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


template<> void calculate_distances<double>(int n, int d, int k,
                                            thrust::device_vector<double>& data,
                                            thrust::device_vector<double>& centroids,
                                            thrust::device_vector<double>& data_dots,
                                            thrust::device_vector<double>& centroid_dots,
                                            thrust::device_vector<double>& pairwise_distances) {
    detail::make_self_dots(k, d, centroids, centroid_dots);
    detail::make_all_dots(n, k, data_dots, centroid_dots, pairwise_distances);
    //||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
    //pairwise_distances has ||x||^2 + ||y||^2, so beta = 1
    //The dgemm calculates x.y for all x and y, so alpha = -2.0
    double alpha = -2.0;
    double beta = 1.0;
    //If the data were in standard column major order, we'd do a
    //centroids * data ^ T
    //But the data is in row major order, so we have to permute
    //the arguments a little
    cublasStatus_t stat =
        cublasDgemm(detail::cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, k, d, &alpha,
                    thrust::raw_pointer_cast(data.data()),
                    d,//Has to be n or d
                    thrust::raw_pointer_cast(centroids.data()),
                    d,//Has to be k or d
                    &beta,
                    thrust::raw_pointer_cast(pairwise_distances.data()),
                    n); //Has to be n or k
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Invalid Dgemm" << std::endl;
        exit(1);
    }

}

template<> void calculate_distances<float>(int n, int d, int k,
                                           thrust::device_vector<float>& data,
                                           thrust::device_vector<float>& centroids,
                                           thrust::device_vector<float>& data_dots,
                                           thrust::device_vector<float>& centroid_dots,
                                           thrust::device_vector<float>& pairwise_distances) {
    detail::make_self_dots(k, d, centroids, centroid_dots);
    detail::make_all_dots(n, k, data_dots, centroid_dots, pairwise_distances);
    //||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
    //pairwise_distances has ||x||^2 + ||y||^2, so beta = 1
    //The dgemm calculates x.y for all x and y, so alpha = -2.0
    float alpha = -2.0;
    float beta = 1.0;
    //If the data were in standard column major order, we'd do a
    //centroids * data ^ T
    //But the data is in row major order, so we have to permute
    //the arguments a little
    cublasStatus_t stat =
        cublasSgemm(detail::cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, k, d, &alpha,
                    thrust::raw_pointer_cast(data.data()),
                    d,//Has to be n or d
                    thrust::raw_pointer_cast(centroids.data()),
                    d,//Has to be k or d
                    &beta,
                    thrust::raw_pointer_cast(pairwise_distances.data()),
                    n); //Has to be n or k
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Invalid Sgemm" << std::endl;
        exit(1);
    }

}

}
}
