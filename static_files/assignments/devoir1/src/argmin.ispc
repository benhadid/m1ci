export uniform int32 argmin(uniform const double values[],
                            uniform const int32 size)
{
    // Vector
    int32  index_local = 0;
    double value_local = values[0];
    foreach (i = 1 ... size)
    {
        double value = values[i];
        if (value < value_local)
        {
            value_local = value;
            index_local = i;
        }
    }

    // Scalar
    uniform int32  index_global = 0;
    uniform double value_global = values[0];
    for (uniform int i = 0; i < programCount; ++i)
    {
        uniform double value = extract(value_local, i);
        if (value < value_global)
        {
            index_global = extract(index_local, i);
            value_global = value;
        }
    }
    return index_global;
}

