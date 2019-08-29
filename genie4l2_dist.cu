#include "genie4l2_dist.h"

int DistGenieBucketer::get_num_gpus()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    return devCount;
}

DistGenieBucketer::DistGenieBucketer(int topk, int queryPerBatch, int sigDim)
    :topk(topk), queryPerBatch(queryPerBatch), sigDim(sigDim)
{
    numGPUs = get_num_gpus();

    bucketers.reserve(numGPUs);
    for(int i=0;i<numGPUs;i++){
        bucketers.emplace_back(topk, queryPerBatch, i, sigDim);
    }
}


void DistGenieBucketer::build(const std::vector<std::vector<std::vector<int> > >& sigs)
{
    assert(sigs.size()==numGPUs);

    //let each buckets build its own inv_table and so on.
    std::vector<std::thread> pools;
    // pools.clear();
    pools.reserve(numGPUs);

    std::vector<std::vector<std::vector<int> > > sigsForThreads;
    sigsForThreads.resize(numGPUs);

    //calc extents
    extents.resize(sigs.size()+1);
    extents[0] = 0;
    for(int i=0;i<sigs.size();i++){
        extents[i+1] = sigs[i].size();
    }
    //require c++17
    // std::exclusive_scan(extents.begin(), extents.end());
    for(int i=1;i<extents.size();i++) {
        extents[i] += extents[i-1];
    }

    for(int i=0;i<extents.size();i++){
        printf("extents[%d]=%d\n", i, extents[i]);
    }

    //build each bucketer
    for(int threadid=0;threadid<numGPUs;threadid++){
        pools.emplace_back([&sigs, threadid, this](){
            bucketers[threadid].build(sigs[threadid]);
        });
    }
    for(int threadid=0;threadid<numGPUs;threadid++){
        pools[threadid].join();
    }
}

std::vector<std::vector<int> > DistGenieBucketer::batch_query(const std::vector<std::vector<int> >& querySigs)
{
    std::vector<std::thread> pools;
    // pools.clear();
    pools.reserve(numGPUs);

    std::vector<std::vector<std::vector<int> > > candidates(numGPUs);

    //query each bucketer
    for(int threadid=0;threadid<numGPUs;threadid++){
        pools.emplace_back([&candidates, &querySigs, threadid, this](){
            auto ret = bucketers[threadid].batch_query(querySigs);
            candidates[threadid] = std::move(ret);
        });
    }
    for(int threadid=0;threadid<numGPUs;threadid++){
        pools[threadid].join();
    }

    std::vector<std::vector<int> > ret(querySigs.size());
    for(int i=0;i<ret.size();i++){
        for(int threadid=0;threadid<numGPUs;threadid++){
            for(int candidateFromBucketer:candidates[threadid][i]) {
                ret[i].push_back(candidateFromBucketer + extents[threadid]);
            }
        }
    }
    return ret;
}