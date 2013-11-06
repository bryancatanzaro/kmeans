#pragma once
#include <thrust/device_vector.h>
#include <cfloat>

namespace kmeans {
namespace detail {

void labels_init();

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

template<typename T>
void calculate_distances(int n, int d, int k,
                         thrust::device_vector<T>& data,
                         thrust::device_vector<T>& centroids,
                         thrust::device_vector<T>& data_dots,
                         thrust::device_vector<T>& centroid_dots,
                         thrust::device_vector<T>& pairwise_distances);

template<> void calculate_distances<float>(int n, int d, int k,
                                           thrust::device_vector<float>& data,
                                           thrust::device_vector<float>& centroids,
                                           thrust::device_vector<float>& data_dots,
                                           thrust::device_vector<float>& centroid_dots,
                                           thrust::device_vector<float>& pairwise_distances);

template<> void calculate_distances<double>(int n, int d, int k,
                                            thrust::device_vector<double>& data,
                                            thrust::device_vector<double>& centroids,
                                            thrust::device_vector<double>& data_dots,
                                            thrust::device_vector<double>& centroid_dots,
                                            thrust::device_vector<double>& pairwise_distances);

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
        distances[global_id] = min_distance;
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
