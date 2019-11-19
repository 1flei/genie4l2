#pragma once

#include "genie4l2.h"
#include <thread>
#include <numeric>

//bucketer using distgenie
class DistGenieBucketer
{
public:
    DistGenieBucketer() {};
    DistGenieBucketer(int topk, int queryPerBatch, int sigDim);

    //not implemented
    // void build(const std::vector<std::vector<int> >& sigs);
    void build(const std::vector<std::vector<std::vector<int> > >& sigss);
    //given sigs of queries, return the candidates set for each query
    std::vector<std::vector<int> > batch_query(const std::vector<std::vector<int> >& querySigs);

    std::shared_ptr<genie::ExecutionPolicy> get_genie_policy();

    //split sigs into n partitions uniformly. WILL CONSUME sigs
    std::vector<std::vector<std::vector<int> > > split_sigs(std::vector<std::vector<int> >&& sigs, int m) {
        std::vector<std::vector<std::vector<int> > > ret(m);
        int n = sigs.size();
        for(int i=0;i<m;i++){
            int idxbeg = (n*i+m-1)/m;
            int idxend = (n*(i+1)+m-1)/m;
            ret[i].reserve(idxend-idxbeg);
            for(int j=idxbeg;j<idxend;j++){
                ret[i].push_back(std::move(sigs[j]));
            }
        }
        return ret;
    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int )
    {
        ar & topk;
        ar & queryPerBatch;
        ar & sigDim;
        ar & numGPUs;
        ar & extents;

        
        if(Archive::is_loading::value){
            bucketers.clear();
            bucketers.reserve(numGPUs);
            for(int i=0;i<numGPUs;i++){
                bucketers.emplace_back(topk, queryPerBatch, i, sigDim);
                ar & bucketers[i];
            }
        } else{
            for(int i=0;i<numGPUs;i++) {
                ar & bucketers[i];
            }
        }
    }

    int get_num_gpus();

    int topk;
    int queryPerBatch;
    int sigDim;

    int numGPUs;
    std::vector<int> extents;
    std::vector<GenieBucketer> bucketers;

    // std::vector<std::thread> pools;
};

template<class Scalar> 
class DistGenie4l2
{
public:
    DistGenie4l2(int dataDim, int nLines, double radius, int topk, int queryPerBatch) 
        :dataDim(dataDim), nLines(nLines), radius(radius), topk(topk), 
        queryPerBatch(queryPerBatch), hasher(dataDim, nLines, radius), bucketer(topk+30, queryPerBatch, nLines)
    {
    }
    ~DistGenie4l2()
    {
    }

    //take some parameters by default std::vector
    void build(const std::vector<std::vector<Scalar> >& dataObjects)
    {
        //project first
        std::vector<std::vector<int> > hashSigsTmp;
        get_sigs(dataObjects, hashSigsTmp);
        hashSigss = std::move(bucketer.split_sigs(std::move(hashSigsTmp), bucketer.get_num_gpus()) );
        bucketer.build(hashSigss);
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
        ar & hasher;
        ar & hashSigss;
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

    RandProjHasher<Scalar, int> hasher;
    // std::vector<std::vector<int> > hashSigs;
    std::vector<std::vector<std::vector<int> > > hashSigss;

    DistGenieBucketer bucketer;
};