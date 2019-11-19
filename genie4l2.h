#pragma once

#include <vector>
#include <memory>
#include <queue>

#include "projection.h"
#include "pivot_hasher.h"
#include "util.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <cstdio>
// #include <fmt/format.h>

//pure declearation
//to achieve better seperated compilation of cuda and c++
namespace genie
{
    class ExecutionPolicy;

	namespace query {struct NewQuery;}
	namespace table {struct InvertedTable;}
}


class GenieBucketer
{
public:
    GenieBucketer() {};
    GenieBucketer(int topk, int queryPerBatch, int GPUID, int sigDim);

    void build(const std::vector<std::vector<int> >& sigs);
    //given sigs of queries, return the candidates set for each query
    std::vector<std::vector<int> > batch_query(const std::vector<std::vector<int> >& querySigs);

    std::shared_ptr<genie::ExecutionPolicy> get_genie_policy();

    std::shared_ptr<genie::table::InvertedTable> invTable;
    // std::shared_ptr<genie::table::inv_list> invTable;
    std::shared_ptr<genie::ExecutionPolicy> geniePolicy;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version);

    int topk;
    int queryPerBatch;
    int GPUID;
    int sigDim;
};

template<class Scalar>
using Distf = std::function<Scalar(int, const Scalar*, const Scalar*)>;

template<class Scalar>
struct DistFuncScanner
{
    using ResPair = std::pair<Scalar, int>;

    DistFuncScanner(int dim, int topk, const std::vector<std::vector<Scalar> >& queryObjects, 
            const std::vector<std::vector<Scalar> >& dataObjects, 
            Distf<Scalar> distf_=calc_l2_dist<Scalar>):
        dim(dim), topk(topk), queryObjects(queryObjects), dataObjects(dataObjects), 
        distf(std::move(distf_) )
    {
        resQue.resize(queryObjects.size());
        for(int i=0;i<resQue.size();i++){
            resQue.reserve(topk);
        }
    }

    void push(int qid, int candidateId) {
        assert(qid < queryObjects.size() && candidateId < dataObjects.size() && qid < resQue.size());

        double dist = distf(dim, &queryObjects[qid][0], &dataObjects[candidateId][0]);
        if(resQue[qid].size() < topk) {
            resQue[qid].emplace(dist, candidateId);
        } else {
            const auto& p = resQue[qid].top();
            if(p.first > dist) {
                //meaning that current elem should be in the resHeap
                resQue[qid].pop();
                resQue[qid].emplace(dist, candidateId);
            }
        }
    }

    std::vector<std::priority_queue<ResPair> >& fetch_res()
    {
        return resQue;
    }
    std::vector<std::vector<ResPair> > fetch_res_vec()
    {
        std::vector<std::vector<ResPair> > ret(resQue.size());
        for(int qid=0;qid<ret.size();qid++){
            ret[qid].resize(resQue[qid].size());
            int idx = resQue[qid].size()-1;
            while(!resQue[qid].empty()){
                ret[qid][idx] = resQue[qid].top();
                resQue[qid].pop();
                --idx;
            }
        }
        return ret;
    }

    int dim;
    int topk;
    const std::vector<std::vector<Scalar> >& queryObjects;
    const std::vector<std::vector<Scalar> >& dataObjects;
    //max-heap
    std::vector<std::priority_queue<ResPair> > resQue;
    Distf<Scalar> distf;
};

template<class Scalar> 
class Genie4l2
{
public:
    Genie4l2(int dataDim, int nLines, double radius, int topk, int queryPerBatch, int GPUID) 
        :dataDim(dataDim), nLines(nLines), radius(radius), topk(topk), 
        queryPerBatch(queryPerBatch), GPUID(GPUID), hasher(dataDim, nLines, radius), bucketer(3*topk+3*nLines, queryPerBatch, GPUID, nLines)
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
    template<class Scanner>
    void query(const std::vector<std::vector<Scalar> >& queries, const Scanner& f)
    {
        std::vector<std::vector<int> > querySigs;

        get_sigs(queries, querySigs);
        for(int i=0;i * queryPerBatch < queries.size(); i++) {
            int start = i * queryPerBatch;
            int end   = std::min<int>((i+1) * queryPerBatch, querySigs.size());
            std::vector<std::vector<int> > querySigBatch(querySigs.begin() + start, querySigs.begin() + end);
            auto candidatessBatch = bucketer.batch_query(querySigBatch);
            assert(candidatessBatch.size() == querySigBatch.size());

            printf("batch query done!!\n");
            
            for(int i=0;i<candidatessBatch.size();i++){
                for(int idx:candidatessBatch[i]){
                    f(i + start, idx);
                }
            }
        }
    }


    // default version, using DistFuncScanner 
    using ResPair = std::pair<Scalar, int>;
    std::vector<std::vector<ResPair> > query_vec(
        const std::vector<std::vector<Scalar> >& queries, 
        const std::vector<std::vector<Scalar> >& dataObjects)
    {
        DistFuncScanner<Scalar> scanner(dataDim, topk, queries, dataObjects);
        query(queries, [&](int qid, int candidateId){
            scanner.push(qid, candidateId);
        });
        return scanner.fetch_res_vec();
    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int )
    {
        ar & dataDim;
        ar & nLines;
        ar & radius;
        ar & topk;
        ar & queryPerBatch;
        ar & GPUID;
        ar & hasher;
        ar & hashSigs;
        ar & bucketer;
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



template<class Scalar> 
class GeniePivot
{
public:
    GeniePivot(int dataDim, int nPivots, int topk, int queryPerBatch, int GPUID, 
            const std::vector<std::vector<Scalar> >& dataset, 
            Distf<Scalar> distf_=calc_l2_dist<Scalar>)
        :dataDim(dataDim), sigdim(sqrt(nPivots)), nPivots(nPivots), topk(topk), 
        queryPerBatch(queryPerBatch), GPUID(GPUID), hasher(dataDim, sigdim, nPivots, dataset), 
        bucketer(3*topk+3*nPivots, queryPerBatch, GPUID, sqrt(nPivots)), 
        distf(std::move(distf_))
    {
    }
    ~GeniePivot()
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
    template<class Scanner>
    void query(const std::vector<std::vector<Scalar> >& queries, const Scanner& scanner)
    {
        std::vector<std::vector<int> > querySigs;

        get_sigs(queries, querySigs);
        for(int i=0;i * queryPerBatch < queries.size(); i++) {
            int start = i * queryPerBatch;
            int end   = std::min<int>((i+1) * queryPerBatch, querySigs.size());
            std::vector<std::vector<int> > querySigBatch(querySigs.begin() + start, querySigs.begin() + end);
            auto candidatessBatch = bucketer.batch_query(querySigBatch);
            assert(candidatessBatch.size() == querySigBatch.size());

            for(int i=0;i<candidatessBatch.size();i++){
                for(int idx:candidatessBatch[i]){
                    scanner(i + start, idx);
                }
            }
        }
    }

    // default version, using DistFuncScanner 
    using ResPair = std::pair<Scalar, int>;
    std::vector<std::vector<ResPair> > query_vec(
        const std::vector<std::vector<Scalar> >& queries, 
        const std::vector<std::vector<Scalar> >& dataObjects)
    {
        DistFuncScanner<Scalar> scanner(dataDim, topk, queries, dataObjects, distf);
        query(queries, [&](int qid, int candidateId){
            scanner.push(qid, candidateId);
        });
        return scanner.fetch_res_vec();
    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int )
    {
        ar & dataDim;
        ar & sigdim;
        ar & nPivots;
        ar & topk;
        ar & queryPerBatch;
        ar & GPUID;
        ar & hasher;
        ar & hashSigs;
        ar & bucketer;
    }

private:
    inline void get_sigs(const std::vector<std::vector<Scalar> >& objects, std::vector<std::vector<int> >& sigs) 
    {
        sigs.resize(objects.size());
        for(int i=0;i<objects.size();i++){
            sigs[i].resize(sigdim);
            hasher.getSig(&objects[i][0], &sigs[i][0], distf);
            for(int j=0;j<sigs[i].size();j++){
                sigs[i][j] = sigs[i][j] & 0x7fff;
            }
        }
    }

    int dataDim;
    int sigdim;
    int nPivots;
    int topk;
    int queryPerBatch;
    int GPUID;

    PivotHasher<Scalar, int> hasher;
    std::vector<std::vector<int> > hashSigs;

    GenieBucketer bucketer;
    Distf<Scalar> distf;
};
