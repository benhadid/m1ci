#include <ctime> // for a random seed
#include <cmath> // for sqrt
#include <vector>
#include <algorithm>
#include <functional>
#include <immintrin.h>

#include <point.hpp>

#include "argmin_ispc.h"

/**
 * @brief computes the squared distance between two points
 *
 * @param p1
 * @param p2
 * @return double
 */

#pragma omp declare simd
static inline double distance(const Point &p1, const Point &p2)
{
    // no need to apply the square root !
    double d1 = p1.x - p2.x;
    double d2 = p1.y - p2.y;

    return d1 * d1 + d2 * d2;
}

#pragma omp declare reduction(vec_double_plus : std::vector<double> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp declare reduction(vec_int_plus : std::vector<int> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

/**
 * @brief Perform k-means clustering
 *
 * @param points - a reference to a vector of points
 * @param stopping_criteria - stopping criteria for k means iterations
 * @param k - number of clusters to identify
 */
void openmp_kMeansClustering(std::vector<Point> &points, int iter, int k)
{
    // Initialize the clusters' centroids to random locations
    // The index of the centroid within the centroids vector
    // represents the cluster label.
    std::vector<Point> centroids;
    srand(1); // need to set the random seed
    std::size_t n = points.size();
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(points.at(rand() % n));
    }

    size_t ptsSz = points.size();

    double epsilon = 1e-8;
    double err = __DBL_MAX__;
    while (err > epsilon && iter > 0)
    {
        // Create vectors to keep track of data needed to compute means
        std::vector<int> nPoints(k, 0);
        std::vector<double> sumX(k, 0.0), sumY(k, 0.0);

// go through all points and compute which centroid is closest
#pragma omp parallel for reduction(vec_double_plus : sumX, sumY) reduction(vec_int_plus : nPoints)
        for (size_t i = 0; i < ptsSz; ++i)
        {
            Point &p = points[i];

            // For each centroid, compute distance from centroid to each point
            // and update the point's cluster field if necessary

            double mindist = __DBL_MAX__;
            int clusterId = -1;
            // #pragma omp simd            
            for (auto j = 0; j < k; ++j)
            {
                double current = distance(centroids[j], p);

                if (current < mindist)
                {
                    mindist = current;
                    clusterId = j;
                }
            }

/*
            std::vector<double> mdist(k);
            #pragma omp simd            
            for (auto j = 0; j < k; ++j)
            {
                mdist[j] = distance(centroids[j], p);
            }
            int clusterId = ispc::argmin(mdist.data(), k);
*/
            




            // update the point's cluster id
            p.cluster = clusterId;

            // update the clusters centroids- first part
            nPoints[clusterId]++;
            sumX[clusterId] += p.x;
            sumY[clusterId] += p.y;
        }

        err = 0.0;
        // update the clusters centroids- final part
#pragma omp simd reduction(+:err)
        for (auto j = 0; j < k; ++j)
        {
            double new_x = sumX[j] / nPoints[j];
            double new_y = sumY[j] / nPoints[j];

            Point newCentroid(new_x, new_y);
            // compute how much the centroids moved from previous positions
            err += distance(centroids[j], newCentroid);

            centroids[j] = newCentroid;
        }
        // use L2 norm for the stopping criteria
        err = sqrt(err);
        iter--;
    }
}