#include <ctime>    // for a random seed
#include <cmath>    // for sqrt
#include <fstream>  // for file-reading
#include <iostream> // for file-reading
#include <sstream>  // for file-reading
#include <vector>   // for data manipulation

struct Point
{
    double x, y;    // point's 2d coordinates
    int cluster;    // point's cluster id
    double minDist; // point's distance to the cluster centroid

    // create a point with coordinates (0,0)
    Point() : x(0.0),
              y(0.0),
              cluster(-1),
              minDist(__DBL_MAX__) {}

    // create a point with coordinates (_x,_y)
    Point(double _x, double _y) : x(_x),
                                y(_y),
                                cluster(-1),
                                minDist(__DBL_MAX__) {}

    // create a new point from a copy of point p
    Point(const Point& p) : x(p.x),
                      y(p.y),
                      cluster(p.cluster),
                      minDist(p.minDist) {}

    // assignment from a point p 
    Point& operator=(const Point& p)
    {
        // Guard self assignment
        if (this != &p)
        {
            x = p.x;
            y = p.y;
            cluster = p.cluster;
            minDist = p.minDist;
        }   
        return *this;
    } 

    // computes the squared distance between this point and point p
    double distance(const Point& p)
    {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

/**
 * @brief Populates a vector with points data from a csv file
 *
 * @param datafilename (std::string): the csv filename to be read
 * @return std::vector<Point>: populated vector with points data
 */
std::vector<Point> readcsv(std::string datafilename)
{
    std::vector<Point> points;        // vector of points to  populate
    std::string line;                 // a placeholder for a text line
    std::ifstream file(datafilename); // associate an input file stream with the csv text file for reading

    // read the csv file line by line
    while (std::getline(file, line))
    {
        std::stringstream lineStream(line);
        std::string bit;
        double x, y;

        std::getline(lineStream, bit, ','); // extract the text associated with the x coordinate
        x = std::stof(bit);                 // and convert to float

        std::getline(lineStream, bit, '\n'); // extract the text associated with the y coordinate
        y = std::stof(bit);                  // an convert to float

        // create a point object with the (x,y) coordinates and store into the vector of points.
        points.push_back(Point(x, y));
    }

    // return the populated vector of points
    return points;
}

/**
 * @brief 
 * 
 * @param outputfilename 
 * @param points 
 */
void writecsv(std::string outputfilename, std::vector<Point>& points) 
{        
    std::ofstream ofile;
    ofile.open(outputfilename);
    ofile << "x,y,c" << std::endl;

    for (std::vector<Point>::iterator it = points.begin();
         it != points.end(); ++it)
    {
        ofile << it->x << "," << it->y << "," << it->cluster << std::endl;
    }
    ofile.close();
}

/**
 * @brief Perform k-means clustering
 *
 * @param points - a reference to a vector of points
 * @param stopping_criteria - stopping criteria for k means iterations
 * @param k - number of clusters to identify
 */
void kMeansClustering(std::vector<Point>& points, double stopping_criteria, int k)
{
    // Initialize the clusters' centroids to random locations
    // The index of the centroid within the centroids vector
    // represents the cluster label.
    std::vector<Point> centroids;
    srand(time(0)); // need to set the random seed
    std::size_t n = points.size();
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(points.at(rand() % n));
    }

    // err holds how much the centroids have changed from previous iteration
    double  err = __DBL_MAX__;
    while(err > stopping_criteria)
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
                double dist = c->distance(p);
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
                                   
            err += c->distance( Point(new_x, new_y) );
            
            c->x = new_x;
            c->y = new_y;  
        }

        //use L2 norm for the stopping criteria
        err = sqrt( err );
    }
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        printf("Usage: %s input_data.csv  output_results.csv \n", argv[0]);
        exit(1);
    }
    
    std::vector<Point> points = readcsv(agv[1]); // "mall_data.csv"
    kMeansClustering(points, 1e-8, 5);           // pass address of points to function
    writecsv(argv[2], points);                   // "output.csv"
}

/*

L_U + k² U = 0

Uxx + Uyy + Uzz + k² U = 0

Uzz + k² U = - Uxx - Uyy 

U = A exp(it)

U' = ( A' + i A t' ) exp(it)

U" = ( A" - A t'² + i ( A t" + 2 A' t' ) ) 

Azz - A tz . tz  + i ( A tzz + 2 Az tz )

(Azz - A tz . tz)² + ( A tzz + 2 Az tz )²  =  (L_A - A G_t . G_t)²  + ( A L_t + 2 G_A . G_t )²


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