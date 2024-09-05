#include <fstream>  // for file-reading
#include <iostream> // for file-reading
#include <sstream>  // for file-reading

#include <vector>
#include <point.hpp>

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

    std::getline(file, line); // get rid of the header info

    // read the csv file line by line
    while (std::getline(file, line))
    {
        std::stringstream lineStream(line);
        std::string bit;
        double x, y;

        std::getline(lineStream, bit, ','); // extract the id number
        try
        {
            std::getline(lineStream, bit, ','); // extract the annual income (x coordinate)
            x = std::stof(bit);                 // and convert to double

            std::getline(lineStream, bit, '\n'); // extract the spending score (y coordinate)
            y = std::stof(bit);
            // and convert to double
            // create a point object with the (x,y) coordinates and store into the vector of points.
            points.push_back(Point(x, y));
        }
        catch (...)
        {
            // skip errors in csv file
        }
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
void writecsv(std::string outputfilename, std::vector<Point> &points)
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
