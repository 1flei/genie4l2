#pragma once

struct Result
{
    float key_;
    int id_;
};

const static int MAXK = 100;

// -----------------------------------------------------------------------------
template <class ScalarType>
ScalarType calc_l2_sqr(   // calc L2 square distance
    int dim,              // dimension
    const ScalarType *p1, // 1st point
    const ScalarType *p2) // 2nd point
{
    ScalarType diff(0);
    ScalarType ret(0);
    for (int i = 0; i < dim; ++i)
    {
        diff = p1[i] - p2[i];
        ret += diff * diff;
    }
    return ret;
}

// -----------------------------------------------------------------------------
template <class ScalarType>
ScalarType calc_l2_dist(  // calc L2 distance
    int dim,              // dimension
    const ScalarType *p1, // 1st point
    const ScalarType *p2) // 2nd point
{
    return sqrt(calc_l2_sqr(dim, p1, p2));
}

inline double calc_recall(
    std::vector<double> &res,
    std::vector<double> &ground_truth,
    const double eps = 1e-6)
{
    std::sort(res.begin(), res.end());
    std::sort(ground_truth.begin(), ground_truth.end());

    int n = ground_truth.size();
    int res_idx = std::min(res.size(), ground_truth.size());

    auto it = std::upper_bound(res.begin(), res.begin()+res_idx, ground_truth.back()+eps);
    return std::distance(res.begin(), it) * 1. / n;
}