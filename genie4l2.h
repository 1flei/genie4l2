#pragma once

#include <vector>
#include <memory>
#include "projection.h"


//pure declearation
//to achieve better seperated compilation of cuda and c++
namespace genie
{
    class ExecutionPolicy;
    class InvertedTable;

	namespace query {struct NewQuery;}
	namespace table {struct InvertedTable;}
}


class GenieBucketer
{
public:
    GenieBucketer(int topk, int queryPerBatch, int GPUID, int sigDim);

    void build(const std::vector<std::vector<int> >& sigs);
    //given sigs of queries, return the candidates set for each query
    std::vector<std::vector<int> > batch_query(const std::vector<std::vector<int> >& querySigs);

    std::shared_ptr<genie::ExecutionPolicy> get_genie_policy();

    std::shared_ptr<genie::table::InvertedTable> invTable;
    std::shared_ptr<genie::ExecutionPolicy> geniePolicy;

    int topk;
    int queryPerBatch;
    int GPUID;
    int sigDim;
};

template<class Scalar> 
class Genie4l2
{
public:
    Genie4l2(int dataDim, int nLines, double radius, int topk, int queryPerBatch, int GPUID) 
        :dataDim(dataDim), nLines(nLines), radius(radius), topk(topk), 
        queryPerBatch(queryPerBatch), GPUID(GPUID), hasher(dataDim, nLines, radius), bucketer(3*topk, queryPerBatch, GPUID, nLines)
    {
    }
    ~Genie4l2()
    {
    }

    //take some parameters by default std::vector
    void build(const std::vector<std::vector<Scalar> >& dataObjects)
    {
        //project first
        get_sigs(dataObjects, hashSigs);
        bucketer.build(hashSigs);
    }

    //F :: query-id -> candidate-id -> IO
    template<class F>
    void query(const std::vector<std::vector<Scalar> >& queries, const F& f)
    {
        std::vector<std::vector<int> > querySigs;
        get_sigs(queries, querySigs);
        auto candidatess = bucketer.batch_query(querySigs);
        
        assert(candidatess.size()==queries.size());
        for(int i=0;i<candidatess.size();i++){
            for(int idx:candidatess[i]){
                f(i, idx);
            }
        }
    }


    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & hasher;
    }

private:
    inline void get_sigs(const std::vector<std::vector<Scalar> >& objects, std::vector<std::vector<int> >& sigs) 
    {
        sigs.resize(objects.size());
        for(int i=0;i<objects.size();i++){
            sigs[i].resize(nLines);
            hasher.getSig(&objects[i][0], &sigs[i][0]);
            for(int j=0;j<sigs[i].size();j++){
                sigs[i][j] = sigs[i][j] & 0x7fff;
            }
        }
    }

    int dataDim;
    int nLines;
    double radius;
    int topk;
    int queryPerBatch;
    int GPUID;

    RandProjHasher<Scalar, int> hasher;
    std::vector<std::vector<int> > hashSigs;

    GenieBucketer bucketer;
};