#include "genie4l2.h"
#include "genie/genie.h"

//let this cu files link all required cuda implementation

std::shared_ptr<genie::ExecutionPolicy> GenieBucketer::get_genie_policy()
{
    genie::Config config = genie::Config()
        .SetK(topk)
        .SetNumOfQueries(queryPerBatch)
        .SetGpuId(GPUID)
        .SetDim(sigDim);
    return genie::MakePolicy(config);
}


GenieBucketer::GenieBucketer(int topk, int queryPerBatch, int GPUID, int sigDim)
    :topk(topk), queryPerBatch(queryPerBatch), GPUID(GPUID), sigDim(sigDim)
{
    geniePolicy = get_genie_policy();
}


void GenieBucketer::build(const std::vector<std::vector<int> >& sigs)
{
    invTable = genie::BuildTable(geniePolicy, sigs);
}


std::vector<std::vector<int> > GenieBucketer::batch_query(const std::vector<std::vector<int> >& querySigs)
{
    auto genieQuery = genie::BuildQuery(geniePolicy, querySigs);
    auto genieResult = genie::Match(geniePolicy, invTable, genieQuery);
    
    //genieResult.first would be the idx and genieResult.second would be the count

    std::vector<std::vector<int> > ret;
    ret.resize(querySigs.size());
    for(int i=0;i<querySigs.size();i++){
        for(int j=0;j<topk;j++){
            int qidx = i*topk + j;
            printf("%d, %d, count=%d\n", i, j, genieResult.second[qidx]);
            ret[i].push_back(genieResult.first[qidx]);
        }
    }
    return ret;
}



template<class Archive>
void GenieBucketer::serialize(Archive & ar, const unsigned int version)
{
    ar & topk;
    ar & queryPerBatch;
    ar & GPUID;
    ar & sigDim;
    if(Archive::is_loading::value){
        geniePolicy = get_genie_policy();
    }
    ar & invTable;
}

// Explicit template instantiation
template
void GenieBucketer::serialize(boost::archive::binary_iarchive & ar, const unsigned int version);
template
void GenieBucketer::serialize(boost::archive::binary_oarchive & ar, const unsigned int version);
template
void GenieBucketer::serialize(boost::archive::text_iarchive & ar, const unsigned int version);
template
void GenieBucketer::serialize(boost::archive::text_oarchive & ar, const unsigned int version);

