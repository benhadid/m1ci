#ifndef _KPOINT_H_
#define _KPOINT_H_

struct Point
{
    double x, y; // point's 2d coordinates
    int cluster;   // point's cluster id
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
    Point(const Point &p) : x(p.x),
                            y(p.y),
                            cluster(p.cluster),
                            minDist(p.minDist) {}

    // assignment from a point p
    Point &operator=(const Point &p)
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

    bool operator==(const Point &other) const
    {
        if (typeid(*this) != typeid(other))
            return false;

        if (this == &other)
            return true;

        return ((x == other.x) && (y == other.y) && (cluster == other.cluster));
    }
};

#endif // #ifndef _KPOINT_H_
