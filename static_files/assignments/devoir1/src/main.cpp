#include <cmath>
#include <vector>
#include <iostream>
#include <getopt.h>
#include <stdio.h>
#include <omp.h>

#include <CycleTimer.h>

#include <kmeans.hpp>
#include <utils.hpp>

void usage(const char *progname)
{
    printf("Usage: %s [options]  -i <input_data.csv>  -o <output_results.csv>  \n", progname);
    printf("Program Options:\n");
    // printf("  -i  --input        <input_data.csv>  Use sinput_data.csvS as input data\n");
    // printf("  -o  --output       <output_results.csv>  Write results to soutput_results.csvS\n");

    printf("  -k  --clusters  <VAL>    seek <VAL> clusters, defaults to 5\n");
    printf("  -n  --iterations <ITER>  Run k-means <ITER> times, default to 100\n");
    printf("  -r  --refactor           Run code with implemented refactoring\n");
    printf("  -p  --openmp             Run code with implemented openmp\n");
    printf("  -?  --help         This message\n");
}

int main(int argc, char **argv)
{
    std::string inputfilename;
    std::string outputfilename;

    bool check_refactor = false;
    bool check_openmp = false;

    int k = 5;
    int iter = 100;

    int opt;
    static struct option long_options[] = {
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"refactor", no_argument, NULL, 'r'},
        {"openmp", no_argument, NULL, 'p'},
        {"clusters", optional_argument, NULL, 'k'},
        {"iterations", optional_argument, NULL, 'n'},
        {"help", no_argument, NULL, '?'},
        {NULL, 0, NULL, '\0'}};

    if (argc == 1)
    {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    else
        while ((opt = getopt_long(argc, argv, "i:o:k:n:rp?", long_options, NULL)) != -1)
        {
            switch (opt)
            {
            case 'i':
                inputfilename = std::string(optarg);
                break;

            case 'o':
                outputfilename = std::string(optarg);
                break;

            case 'r':
                check_refactor = true;
                break;

            case 'p':
                check_openmp = true;
                break;

            case 'k':
                k = atoi(optarg);
                break;

            case 'n':
                iter = atoi(optarg);
                break;

            case '?':
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
    // end parsing of commandline options

    //
    // Run the serial implementation. Report the minimum time of three
    // runs for robust timing.
    //
    std::vector<Point> backup = readcsv(inputfilename); // "mall_data.csv"
    std::vector<Point> points;

    double meanSerial = __DBL_MAX__;
    for (int i = 0; i < 5; ++i)
    {
        points = backup;
        double startTime = CycleTimer::currentSeconds();
        naive_kMeansClustering(points, iter, k); // call the naive implementation of kmeans
        double endTime = CycleTimer::currentSeconds();
        meanSerial = std::min(meanSerial, endTime - startTime);
    }
    std::vector<Point> golden_points = points; // these are the golden points other algorithms should reproduce
    
    printf("[k-means serial]:\t\t[%.3f] ms\n", meanSerial * 1000);
    writecsv(outputfilename, points); // "output.csv"

    if (check_refactor)
    {
        double meanRefactored = __DBL_MAX__;
        for (int i = 0; i < 5; ++i)
        {
            points = backup;
            double startTime = CycleTimer::currentSeconds();
            refactored_kMeansClustering(points, iter, k); // call the refactored implementation of kmeans
            double endTime = CycleTimer::currentSeconds();
            meanRefactored = std::min(meanRefactored, endTime - startTime);
        }
        printf("[k-means refactored]:\t\t[%.3f] ms\n", meanRefactored * 1000);

        if (points != golden_points)
        {
            printf("Error : refactored k-means output differs from sequential output\n");
            return 1;
        }

        printf("\t\t\t\t(%.2fx speedup from refactored)\n", meanSerial / meanRefactored);
    }

    if (check_openmp)
    {
        // enter a first parallel region, that is outside of our timed loop -
        // this is needed to create a thread pool       

        omp_set_dynamic(0);   // disable dynamic scheduling
        #pragma omp parallel
        omp_get_num_threads();

        // reset the vector points for the openMP version
        double meanOpenMP = __DBL_MAX__;
        for (int i = 0; i < 5; ++i)
        {
            points = backup;
            double startTime = CycleTimer::currentSeconds();
            openmp_kMeansClustering(points, iter, k); // call the openmp implementation of kmeans
            double endTime = CycleTimer::currentSeconds();
            meanOpenMP = std::min(meanOpenMP, endTime - startTime);
        }
        printf("[k-means openmp]:\t\t[%.3f] ms\n", meanOpenMP * 1000);

        if (points != golden_points)
        {
            printf("Error : openmp k-means output differs from sequential output\n");
            return 1;
        }

        printf("\t\t\t\t(%.2fx speedup from openmp and simd)\n", meanSerial / meanOpenMP);
    }

    return 0;
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