#pragma once

#include <vector>
#include <memory>
#include "projection.h"
#include "pivot_hasher.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

//pure declearation
//to achieve better seperated compilation of cuda and c++
namespace genie
{
    class ExecutionPolicy;

	namespace query {struct NewQuery;}
	namespace table {struct InvertedTable;}
}

namespace boost {
namespace archive {

class polymorphic_iarchive;
class polymorphic_oarchive;

} // namespace archive
} // namespace boost

class GenieBucketer
{
public:
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

//default Euclidean scanner
struct EuScanner
{
    void operator()(int qid, int candidateIdx)
    {

    }
};

template<class Scalar> 
class Genie4l2
{
public:
    Genie4l2(int dataDim, int nLines, double radius, int topk, int queryPerBatch, int GPUID) 
        :dataDim(dataDim), nLines(nLines), radius(radius), topk(topk), 
        queryPerBatch(queryPerBatch), GPUID(GPUID), hasher(dataDim, nLines, radius), bucketer(3*topk+100, queryPerBatch, GPUID, nLines)
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
        for(int i=0;i * queryPerBatch < queries.size(); i++) {
            int start = i * queryPerBatch;
            int end   = std::min<int>((i+1) * queryPerBatch, querySigs.size());
            std::vector<std::vector<int> > querySigBatch(querySigs.begin() + start, querySigs.begin() + end);
            auto candidatessBatch = bucketer.batch_query(querySigBatch);
            assert(candidatessBatch.size() == querySigBatch.size());
            
            for(int i=0;i<candidatessBatch.size();i++){
                for(int idx:candidatessBatch[i]){
                    f(i + start, idx);
                }
            }
        }
    }


    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
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
    GeniePivot(int dataDim, int nPivots, int topk, int queryPerBatch, int GPUID, const std::vector<std::vector<Scalar> >& dataset) 
        :dataDim(dataDim), sigdim(sqrt(nPivots)), nPivots(nPivots), topk(topk), 
        queryPerBatch(queryPerBatch), GPUID(GPUID), hasher(dataDim, sigdim, nPivots, dataset), bucketer(3*topk+100, queryPerBatch, GPUID, sigdim)
    {
    }
    ~GeniePivot()
    {
    }

    //take some parameters by default std::vector
    template<class F>
    void build(const std::vector<std::vector<Scalar> >& dataObjects, const F& f)
    {
        //project first
        get_sigs(dataObjects, hashSigs, f);
        bucketer.build(hashSigs);
    }

    //F :: query-id -> candidate-id -> IO
    template<class F, class Scanner>
    void query(const std::vector<std::vector<Scalar> >& queries, const F& f, const Scanner& scanner)
    {
        std::vector<std::vector<int> > querySigs;

        get_sigs(queries, querySigs, f);
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

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
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
    template<class F>
    inline void get_sigs(const std::vector<std::vector<Scalar> >& objects, std::vector<std::vector<int> >& sigs, const F& f) 
    {
        sigs.resize(objects.size());
        for(int i=0;i<objects.size();i++){
            sigs[i].resize(nPivots);
            hasher.getSig(&objects[i][0], &sigs[i][0], f);
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
};