#include <boost/program_options.hpp>
#include <iostream>
#include <string> 
#include "projection.h"
#include "genie4l2.h"
#include "util.h"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace std;
using namespace boost::program_options;

// -----------------------------------------------------------------------------
int read_data_binary(						// read data/query set from disk
	int   n,							// number of data/query objects
	int   d,			 				// dimensionality
	const char *fname,					// address of data/query set
	std::vector<std::vector<float> >& data)						// data/query objects (return)
{
	FILE *fp = fopen(fname, "rb");
	if (!fp) {
		printf("Could not open %s\n", fname);
		return 1;
	}

    data.resize(n);
	int i   = 0;
	int tmp = -1;
	while (!feof(fp) && i < n) {
        data[i].resize(d);
		fread(&data[i][0], sizeof(float), d, fp);
		++i;
	}
//	assert(feof(fp) && i == n);
	fclose(fp);

	return 0;
}

// -----------------------------------------------------------------------------
int read_ground_truth(				// read ground truth results from disk
	int qn,								// number of query objects
	const char *fname,					// address of truth set
	std::vector<std::vector<Result> >& R)							// ground truth results (return)
{
	FILE *fp = fopen(fname, "r");
	if (!fp) {
		printf("Could not open %s\n", fname);
		return 1;
	}

	int tmp1 = -1;
	int maxk = -1;
	fscanf(fp, "%d %d\n", &tmp1, &maxk);
//	assert(tmp1 == qn && tmp2 == MAXK);
	assert(maxk == MAXK);

    R.resize(qn);
	for (int i = 0; i < qn; ++i) {
        R[i].resize(maxk);
		for (int j = 0; j < maxk; ++j) {
			fscanf(fp, "%d %f ", &R[i][j].id_, &R[i][j].key_);
		}
		fscanf(fp, "\n");
	}
	fclose(fp);

	return 0;
}


int main(int argc, char **argv)
{
    int n, qn, d, nLines, K, queryPerBatch, GPUID;
	string datasetFilename, queryFilename, weightFilename, groundtruthFilename, outputFilename, indexFilename;
    double r;

	// srand(time(NULL));
	srand(666);

    // Declare the supported options.
    options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")

		("n,n", value(&n)->required(), "the number of data points")
		("d,d", value(&d)->required(), "the dimension of data")
		("qn,q", value(&qn)->required(), "the number of query points")

        ("nLines,L", value(&nLines)->required(), "#projection lines")
        ("r,r", value(&r)->required(), "projection radius")
        ("K,k", value(&K)->required(), "k for top-k")

        ("queryPerBatch,b", value(&queryPerBatch)->required(), "#query per batch")

        ("GPUID", value(&GPUID)->default_value(0), "GPUID used for genie")


        ("dataset_filename,D", value(&datasetFilename)->required(), "path to dataset filename")
		("queryset_filename,Q", value(&queryFilename)->required(), "path to query filename")
		("ground_truth_filename,G", value(&groundtruthFilename)->required(), "path to ground truth filename")
		("output_filename,O", value(&outputFilename)->default_value("output.txt"), "output folder path (with / at the end) or output filename")
        ("index_filename,I", value(&indexFilename)->default_value("index.dat"), "built index")
    ;

    variables_map vm;

    try {
        store(parse_command_line(argc, argv, desc), vm);
        notify(vm);  

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 1;
        }
    } catch (const boost::program_options::required_option & e) {
        std::cout << desc << std::endl;
        if (vm.count("help")) {
            return 1;
        } else {
            throw e;
        }
    }


	// -------------------------------------------------------------------------
	//  read whatever needed
	// -------------------------------------------------------------------------
	std::vector<std::vector<float> > data, queries;
    std::vector<std::vector<Result> > results;

	if(datasetFilename!=""){
        if (read_data_binary(n, d, datasetFilename.c_str(), data) == 1) {
            printf("Reading dataset error!\n");
            return 1;
        }
    }

    if(queryFilename!=""){
        if (read_data_binary(qn, d, queryFilename.c_str(), queries) == 1) {
            printf("Reading query set error!\n");
            return 1;
        }
    }

	if(groundtruthFilename!=""){
		if (read_ground_truth(qn, groundtruthFilename.c_str(), results) == 1) {
			printf("Reading Truth Set Error!\n");
			return 1;
		}
	}
    cout << "finishing reading data, query and ground truth" << endl;


    Genie4l2<float> index(d, nLines, r, K, queryPerBatch, GPUID);
    std::fstream fs(indexFilename, ios_base::out | ios_base::in);

    if(fs.is_open()) {
        boost::archive::binary_iarchive ia(fs);
        ia & index;
    } else{
        index.build(data);
    }
    
    std::vector<std::vector<double> > ress(qn);
    const auto& scanner = [&](int qid, int candidateId){
        double dist = calc_l2_dist(d, &data[candidateId][0], &queries[qid][0]);
        ress[qid].push_back(dist);
        // printf("%d, %d, %f\n", qid, candidateId, dist);
    };
    index.query(queries, scanner);

    double avg_recall = 0.;
    for(int i=0;i<ress.size();i++){
        // printf("ress[i].size()=%d\n", ress[i].size());
        std::vector<double> gti;
        for(int j=0;j<K;j++){
            gti.push_back(results[i][j].key_);
        }
        std::sort(ress[i].begin(), ress[i].end());
        std::sort(gti.begin(), gti.end());

        printf("res=%f, %f, gt=%f, %f\n", ress[i][K/2], ress[i][K-1], gti[K/2], gti[K-1]);

        avg_recall += calc_recall(ress[i], gti);
    }
    avg_recall /= qn;

    printf("avg-recall = %f\n", avg_recall);


    if(!fs.is_open()) {
        fs.open(indexFilename, ios_base::out);
        boost::archive::binary_oarchive oa(fs);
        oa & index;
    }

    return 0;
}