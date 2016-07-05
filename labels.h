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
#pragma once
#include <thrust/device_vector.h>
#include <cfloat>
#include <cublas_v2.h>

namespace kmeans {
namespace detail {

//n: number of points
//d: dimensionality of points
//data: points, laid out in row-major order (n rows, d cols)
//dots: result vector (n rows)
// NOTE:
//Memory accesses in this function are uncoalesced!!
//This is because data is in row major order
//However, in k-means, it's called outside the optimization loop
//on the large data array, and inside the optimization loop it's
//called only on a small array, so it doesn't really matter.
//If this becomes a performance limiter, transpose the data somewhere
template<typename T>
__global__ void self_dots(int n, int d, T* data, T* dots) {
	T accumulator = 0;
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_id < n) {
        for (int i = 0; i < d; i++) {
            T value = data[i + global_id * d];
            accumulator += value * value;
        }
        dots[global_id] = accumulator;
    }    
}


template<typename T>
void make_self_dots(int n, int d,
                    thrust::device_vector<T>& data,
                    thrust::device_vector<T>& dots) {
    self_dots<<<(n-1)/256+1, 256>>>(n, d, thrust::raw_pointer_cast(data.data()),
                                    thrust::raw_pointer_cast(dots.data()));
}

template<typename T>
__global__ void all_dots(int n, int k, T* data_dots, T* centroid_dots, T* dots) {
	__shared__ T local_data_dots[32];
	__shared__ T local_centroid_dots[32];

    int data_index = threadIdx.x + blockIdx.x * blockDim.x;
    if ((data_index < n) && (threadIdx.y == 0)) {
        local_data_dots[threadIdx.x] = data_dots[data_index];
    }
    
    int centroid_index = threadIdx.x + blockIdx.y * blockDim.y;
    if ((centroid_index < k) && (threadIdx.y == 1)) {
        local_centroid_dots[threadIdx.x] = centroid_dots[centroid_index];
    }
       
   	__syncthreads();

	centroid_index = threadIdx.y + blockIdx.y * blockDim.y;
    if ((data_index < n) && (centroid_index < k)) {
        dots[data_index + centroid_index * n] = local_data_dots[threadIdx.x] +
            local_centroid_dots[threadIdx.y];
    }
}


template<typename T>
void make_all_dots(int n, int k, thrust::device_vector<T>& data_dots,
                   thrust::device_vector<T>& centroid_dots,
                   thrust::device_vector<T>& dots) {
    all_dots<<<
        dim3((n-1)/32+1,
             (k-1)/32+1),
        dim3(32, 32)>>>(n, k, thrust::raw_pointer_cast(data_dots.data()),
                        thrust::raw_pointer_cast(centroid_dots.data()),
                        thrust::raw_pointer_cast(dots.data()));
}

void gemm(cublasOperation_t transa,
          cublasOperation_t transb,
          int m, int n, int k,
          const float *alpha,
          const float *A, int lda,
          const float *B, int ldb,
          const float *beta,
          float *C, int ldc);

void gemm(cublasOperation_t transa,
          cublasOperation_t transb,
          int m, int n, int k,
          const double *alpha,
          const double *A, int lda,
          const double *B, int ldb,
          const double *beta,
          double *C, int ldc);

template<typename T>
void calculate_distances(int n, int d, int k,
                         thrust::device_vector<T>& data,
                         thrust::device_vector<T>& centroids,
                         thrust::device_vector<T>& data_dots,
                         thrust::device_vector<T>& centroid_dots,
                         thrust::device_vector<T>& pairwise_distances) {
    detail::make_self_dots(k, d, centroids, centroid_dots);
    detail::make_all_dots(n, k, data_dots, centroid_dots, pairwise_distances);
    //||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
    //pairwise_distances has ||x||^2 + ||y||^2, so beta = 1
    //The dgemm calculates x.y for all x and y, so alpha = -2.0
    T alpha = -2.0;
    T beta = 1.0;
    //If the data were in standard column major order, we'd do a
    //centroids * data ^ T
    //But the data is in row major order, so we have to permute
    //the arguments a little
    gemm(CUBLAS_OP_T, CUBLAS_OP_N,
         n, k, d, &alpha,
         thrust::raw_pointer_cast(data.data()),
         d,//Has to be n or d
         thrust::raw_pointer_cast(centroids.data()),
         d,//Has to be k or d
         &beta,
         thrust::raw_pointer_cast(pairwise_distances.data()),
         n); //Has to be n or k
}

template<typename T>
__global__ void make_new_labels(int n, int k, T* pairwise_distances,
                                int* labels, int* changes,
                                T* distances) {
    T min_distance = DBL_MAX;
    T min_idx = -1;
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id < n) {
        int old_label = labels[global_id];
        for(int c = 0; c < k; c++) {
            T distance = pairwise_distances[c * n + global_id];
            if (distance < min_distance) {
                min_distance = distance;
                min_idx = c;
            }
        }
        labels[global_id] = min_idx;
        distances[global_id] = sqrt(min_distance);
        if (old_label != min_idx) {
            atomicAdd(changes, 1);
        }
    }
}


template<typename T>
int relabel(int n, int k,
            thrust::device_vector<T>& pairwise_distances,
            thrust::device_vector<int>& labels,
            thrust::device_vector<T>& distances) {
    thrust::device_vector<int> changes(1);
    changes[0] = 0;
    make_new_labels<<<(n-1)/256+1,256>>>(
        n, k,
        thrust::raw_pointer_cast(pairwise_distances.data()),
        thrust::raw_pointer_cast(labels.data()),
        thrust::raw_pointer_cast(changes.data()),
        thrust::raw_pointer_cast(distances.data()));
    return changes[0];
}

}
}
