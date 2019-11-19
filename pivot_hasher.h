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
        std::vector<SigType> ret(nPivots);
        getSig(data, &ret[0], f);
        return ret;
    }

    //f :: int -> const Scalar * -> const Scalar * -> their distance
    template<class F>
    void getSig(const Scalar *data, SigType* ret, const F& f) const
    {
        std::vector<double> dists(nPivots);
        for(int i=0;i<nPivots;i++){
            dists[i] = f(dim, data, &pivots[i][0]);
        }
        std::vector<int> orders(nPivots);
        for(int i=0;i<nPivots;i++){
            orders[i] = i;
        }
        std::partial_sort(orders.begin(), orders.begin()+sigdim, orders.end(), [&](int a, int b){
            return dists[a] < dists[b];
        });
        std::copy(orders.begin(), orders.begin()+sigdim, ret);
        // for(int i=0;i<sigdim;i++){
        //     printf("%d, %f\n", orders[i], dists[orders[i]]);
        // }

        // for(int i=0;;i++){
        //     int start = i*sigdim;
        //     int end = std::min((i+1)*sigdim, nPivots);
        //     std::sort(orders.begin()+start, orders.begin()+end, [&](int a, int b){
        //         return dists[a] < dists[b];
        //     });

        //     if(end==nPivots){
        //         break;
        //     }
        // }

        // std::copy(orders.begin(), orders.end(), ret);
    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int )
    {
        ar & dim;
        ar & sigdim;
        ar & nPivots;
        ar & pivots;
    }


protected:
    int dim, sigdim, nPivots;
    std::vector<std::vector<Scalar> > pivots;
};