#include <string>   // for file-reading
#include <vector>   // for data manipulation

#include <point.hpp>

/**
 * @brief Populates a vector with points data from a csv file
 *
 * @param datafilename (std::string): the csv filename to be read
 * @return std::vector<Point>: populated vector with points data
 */
std::vector<Point> readcsv(std::string datafilename);

/**
 * @brief
 *
 * @param outputfilename
 * @param points
 */
void writecsv(std::string outputfilename, std::vector<Point> &points);
