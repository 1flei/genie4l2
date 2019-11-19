#include "genie4l2.h"
#include "genie/genie.h"
#include "genie/utility/Logger.h"

//let this cu files link all required cuda implementation

std::shared_ptr<genie::ExecutionPolicy> GenieBucketer::get_genie_policy()
{
    genie::utility::Logger::set_level(genie::utility::Logger::NONE);
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


namespace genie {
    namespace table {
        //the implementation of genie::table::InvertedTable::serialize
        template <class Archive>
        void InvertedTable::load(Archive &ar, const unsigned int )
        {
            ar >> inverted_index_;
            ar >> posting_list_;
            ar >> upperbounds_;
            ar >> lowerbounds_;
            ar >> num_of_rows_;
            ar >> num_of_dimensions_;
        }
        
        template <class Archive>
        void InvertedTable::save(Archive &ar, const unsigned int ) const
        {
            ar << inverted_index_;
            ar << posting_list_;
            ar << upperbounds_;
            ar << lowerbounds_;
            ar << num_of_rows_;
            ar << num_of_dimensions_;
        }
    }
};


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
            // fmt::print("genieResult[{}]=({}, {})", qidx, genieResult.first[qidx][k], genieResult.second[qidx][k]);
            // printf("genieResult[%d]=(%d, %d)\n", qidx, genieResult.first[qidx], genieResult.second[qidx]);
            ret[i].push_back(genieResult.first[qidx]);
        }
    }
    return ret;
}



template<class Archive>
void GenieBucketer::serialize(Archive & ar, const unsigned int )
{
    ar & topk;
    ar & queryPerBatch;
    ar & GPUID;
    ar & sigDim;
    if(Archive::is_loading::value){
        geniePolicy = get_genie_policy();
        cudaSetDevice(GPUID);
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

