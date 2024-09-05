#include <ctime>    // for a random seed
#include <cmath>    // for sqrt
#include <vector>   // for data manipulation

#include <point.hpp>

/**
 * @brief computes the squared distance between two points
 *
 * @param p1
 * @param p2
 * @return double
 */
static double distance(const Point &p1, const Point &p2)
{
    // no need to apply the square root !
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

/**
 * @brief Perform k-means clustering
 *
 * @param points - a reference to a vector of points
 * @param stopping_criteria - stopping criteria for k means iterations
 * @param k - number of clusters to identify
 */
void naive_kMeansClustering(std::vector<Point>& points, int iter, int k)
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

    // err holds how much the centroids have changed from previous iteration
    double epsilon = 1e-8;
    double  err = __DBL_MAX__;
    while(err > epsilon && iter>0) 
    {
        // For each centroid, compute distance from centroid to each point
        // and update point's cluster if necessary
        for (std::vector<Point>::iterator c = begin(centroids);
             c != end(centroids); ++c)
        {
            // quick hack to get cluster index
            int clusterId = c - begin(centroids);

            // go through all points and check if centroid is closest
            for (std::vector<Point>::iterator it = points.begin();
                 it != points.end(); ++it)
            {

                Point p = *it;
                double dist = distance(*c , p);
                if (dist < p.minDist)
                {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
                *it = p;
            }
        }

        // Create vectors to keep track of data needed to compute means
        std::vector<int> nPoints;
        std::vector<double> sumX, sumY;

        // Initialise with zeroes
        for (int j = 0; j < k; ++j)
        {
            nPoints.push_back(0);
            sumX.push_back(0.0);
            sumY.push_back(0.0);
        }

        // Iterate over all points to update the centroids' coordinates
        for (std::vector<Point>::iterator it = points.begin();
             it != points.end(); ++it)
        {
            int clusterId = it->cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += it->x;
            sumY[clusterId] += it->y;

            it->minDist = __DBL_MAX__; // reset distance
        }

        // Compute the new centroids 
        err = 0.0;
        for (std::vector<Point>::iterator c = std::begin(centroids);
             c != std::end(centroids); ++c)
        {
            // quick hack to get cluster index
            int clusterId = c - begin(centroids);

            double new_x = sumX[clusterId] / nPoints[clusterId];
            double new_y = sumY[clusterId] / nPoints[clusterId];
                                   
            err += distance( *c, Point(new_x, new_y) );
            
            c->x = new_x;
            c->y = new_y;  
        }

        //use L2 norm for the stopping criteria
       err = sqrt( err );
       iter--;
    }
}

/*
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Before clustering
df = pd.read_csv("mall_data.csv", header=None)
df.columns = ["Annual income (k$)", "Spending Score (1-100)"]
sns.scatterplot(x=df["Annual income (k$)"], 
                y=df["Spending Score (1-100)"])
plt.title("Scatterplot of spending (y) vs income (x)")

# After clustering
plt.figure()
df = pd.read_csv("output.csv")
sns.scatterplot(x=df.x, y=df.y, 
                hue=df.c, 
                palette=sns.color_palette("hls", n_colors=5))
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Clustered: spending (y) vs income (x)")

plt.show()
*/