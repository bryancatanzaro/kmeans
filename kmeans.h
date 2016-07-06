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
#include "centroids.h"
#include "labels.h"
#include <thrust/reduce.h>

namespace kmeans {


//! kmeans clusters data into k groups
/*! 
  
  \param iterations How many iterations to run
  \param n Number of data points
  \param d Number of dimensions
  \param k Number of clusters
  \param data Data points, in row-major order. This vector must have
  size n * d, and since it's in row-major order, data point x occupies
  positions [x * d, (x + 1) * d) in the vector. The vector is passed
  by reference since it is shared with the caller and not copied.
  \param labels Cluster labels. This vector has size n.
  The vector is passed by reference since it is shared with the caller
  and not copied.
  \param centroids Centroid locations, in row-major order. This
  vector must have size k * d, and since it's in row-major order,
  centroid x occupies positions [x * d, (x + 1) * d) in the
  vector. The vector is passed by reference since it is shared
  with the caller and not copied.
  \param distances Distances from points to centroids. This vector has
  size n. It is passed by reference since it is shared with the caller
  and not copied.
  \param init_from_labels If true, the labels need to be initialized
  before calling kmeans. If false, the centroids need to be
  initialized before calling kmeans. Defaults to true, which means
  the labels must be initialized.
  \param threshold This controls early termination of the kmeans
  iterations. If the ratio of the sum of distances from points to
  centroids from this iteration to the previous iteration changes by
  less than the threshold, than the iterations are
  terminated. Defaults to 0.000001
  \return The number of iterations actually performed.
*/

template<typename T>
int kmeans(int iterations,
           int n, int d, int k,
           thrust::device_vector<T>& data,
           thrust::device_vector<int>& labels,
           thrust::device_vector<T>& centroids,
           thrust::device_vector<T>& distances,
           bool init_from_labels=true,
           double threshold=0.000001) {
    thrust::device_vector<T> data_dots(n);
    thrust::device_vector<T> centroid_dots(n);
    thrust::device_vector<T> pairwise_distances(n * k);
    
    detail::make_self_dots(n, d, data, data_dots);

    if (init_from_labels) {
        detail::find_centroids(n, d, k, data, labels, centroids);
    }   
    T prior_distance_sum = 0;
    int i = 0;
    for(; i < iterations; i++) {
        detail::calculate_distances(n, d, k,
                                    data, centroids, data_dots,
                                    centroid_dots, pairwise_distances);

        int changes = detail::relabel(n, k, pairwise_distances, labels, distances);
       
        
        detail::find_centroids(n, d, k, data, labels, centroids);
        T distance_sum = thrust::reduce(distances.begin(), distances.end());
        std::cout << "Iteration " << i << " produced " << changes
                  << " changes, and total distance is " << distance_sum << std::endl;

        if (i > 0) {
            T delta = distance_sum / prior_distance_sum;
            if (delta > 1 - threshold) {
                std::cout << "Threshold triggered, terminating iterations early" << std::endl;
                return i + 1;
            }
        }
        prior_distance_sum = distance_sum;
    }
    return i;
}

}
