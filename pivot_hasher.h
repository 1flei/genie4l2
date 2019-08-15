#pragma once

//the implementation of projection

#include <vector>
#include <cassert>
#include <random>
#include <algorithm>

//hasher using pivot-based method
//data-dependent method
template<class Scalar, class SigType>
class PivotHasher
{
public:
    //d: dim; sigdim: sigdim, nPivots: #pivots
    PivotHasher(int d, int sigdim, int nPivots, const std::vector<std::vector<Scalar> > & dataset)      //dim of data object, #hasher, radius 
        :dim(d), sigdim(sigdim), nPivots(nPivots)
    {
        assert(d > 0 && sigdim> 0 && nPivots >= sigdim);

        std::uniform_int_distribution<> uniform(0, dataset.size()-1);
        std::random_device rd;
        std::default_random_engine rng(rd());


        pivots.reserve(nPivots);
        for(int i=0;i<nPivots;i++){
            int r = uniform(rng);
            pivots.push_back(dataset[r]);
        }
    }
    ~PivotHasher() {}

    template<class F>
    std::vector<SigType> getSig(const Scalar *data, const F& f) const
    {
        std::vector<SigType> ret(sigdim);
        getSig(data, &ret[0], f);
        return ret;
    }

    //f :: const Scalar * -> const Scalar * -> their distance
    template<class F>
    void getSig(const Scalar *data, SigType* ret, const F& f) const
    {
        std::vector<double> dists(nPivots);
        for(int i=0;i<nPivots;i++){
            dists[i] = f(data, &pivots[i][0]);
        }

        std::vector<int> orders(nPivots);
        for(int i=0;i<nPivots;i++){
            orders[i] = i;
        }
        std::sort(orders.begin(), orders.end(), [&](int a, int b){
            return dists[a] < dists[b];
        });
        std::copy(orders.begin(), orders.begin()+sigdim, ret);
    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & dim;
        ar & nPivots;
        ar & pivots;
    }


protected:
    int dim, sigdim, nPivots;
    std::vector<std::vector<Scalar> > pivots;
};