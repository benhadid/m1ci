#ifndef _KMEANS_H_
#define _KMEANS_H_

#include <vector>
#include <point.hpp>

//header for the naive implementation of the K-means classification algorithm
void naive_kMeansClustering(std::vector<Point> &points, int iter, int k);

//header for the refactored implementation of the K-means classification algorithm
void refactored_kMeansClustering(std::vector<Point> &points, int iter, int k);

//header for the openmp version of the K-means classification algorithm
void openmp_kMeansClustering(std::vector<Point> &points, int iter, int k);

#endif // #ifndef _KMEANS_H_
