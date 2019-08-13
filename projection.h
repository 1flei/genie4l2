#pragma once

//the implementation of projection

#include <vector>
#include <cassert>
#include <random>

//Simple Random Projection
template<class Scalar, class SigType>
class RandProjHasher
{
public:
    RandProjHasher(int d, int K, double r)      //dim of data object, #hasher, radius 
        :dim(d), K(K), r(r)
    {
        assert(d > 0 && K > 0);

        std::normal_distribution<double> normal(0.);
        std::uniform_real_distribution<double> uniform(0., r);
        std::random_device rd;
        std::default_random_engine rng(rd());

        p.resize(K*d);
        b.resize(K);
        for (int i = 0; i < K * d; i++) {
            p[i] = normal(rng);
        }
        for (int i = 0; i < K; i++) {
            b[i] = uniform(rng);
        }
        sigdim = K;
    }
    ~RandProjHasher() {}

    std::vector<SigType> getSig(const Scalar *data) const
    {
        std::vector<SigType> ret(sigdim);
        getSig(data, &ret[0]);
        return ret;
    }

    void getSig(const Scalar *data, SigType* ret) const
    {
        for(int k=0;k<K;k++){
            double projection = 0.;
            for(int i=0;i<dim;i++){
                double x = data[i];
                int pidx = k*dim + i;
                projection += x*p[pidx];
            }
            projection += b[k];

            ret[k] = SigType(floor(projection/r) );
        }
    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & dim;
        ar & K;
        ar & r;
        ar & sigdim;
        ar & p;
        ar & b;
    }

    int dim, K;
    double r;
    int sigdim;
protected:
    std::vector<Scalar> p;
    std::vector<Scalar> b;
};