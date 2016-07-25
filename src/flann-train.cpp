#include "data.h"

#include <cstdlib>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <argtable2.h>
#include <boost/dynamic_bitset.hpp>
#include <opencv2/flann/flann.hpp>

int main(int argc, char * argv[])
{
    struct arg_lit  * help       = arg_lit0 ("h", "help", "Print this help and exit");
    struct arg_file * input_file = arg_file0("i", "input" , "<filename>", "Input dataset in libsvm format (default stdin)");
    struct arg_file * index_file = arg_file1("x", "index" , "<filename>", "Output index file");
    struct arg_int  * verbosity  = arg_int0 ("v", "verbosity", "{0..4}", "Log verbosity"
            "\nIndex parameters :");
    struct arg_int  * distance = arg_int0("d", "distance", "{1..9}", "Distance metric"
            "\n\t1=L2 (default), 2=L1, 3=MINKOWSKI,\n\t4=MAX, 5=HIST_INTERSECT, 6=HELLLINGER,"
            "\n\t7=CS, 8=KULLBACK_LEIBLER, 9=HAMMING");
    struct arg_int  * index_type = arg_int0("t", "index-type", "{0..5}", "Constructed index type"
            "\nt=0 - linear brute force search"
            "\nt=1 - kd-tree :");
    // kd-tree params
    struct arg_int * kd_tree_count = arg_int0(NULL, "kd-tree-count", "{1..16+}", "Number of parallel trees (default 4)"
            "\nt=2 - k-means :");
    // k-means params
    struct arg_int * km_branching  = arg_int0(NULL, "km-branching", "n", "Branching factor (default 32)");
    struct arg_int * km_iterations = arg_int0(NULL, "km-iterations", "n", "Maximum iterations (default 11)");
    struct arg_int * km_centers    = arg_int0(NULL, "km-centers", "{0..2}", "Initial cluster centers"
            "\n\t0=CENTERS_RANDOM (default)\n\t1=CENTERS_GONZALES\n\t2=CENTERS_KMEANSPP");
    struct arg_dbl * km_index      = arg_dbl0(NULL, "km-index", "", "Cluster boundary index (default 0.2)"
            "\nt=3 (default) - kd-tree + k-means"
            "\nt=4 - LSH :");
    struct arg_int * lsh_table_count = arg_int0(NULL, "lsh-table-count", "{0..}", "Number of hash tables");
    struct arg_int * lsh_key_size    = arg_int0(NULL, "lsh-key-size", "{0..}", "Hash key bits");
    struct arg_int * lsh_probe_level = arg_int0(NULL, "lsh-probe-level", "{0..}", "Bit shift for neighboring bucket check"
            "\nt=5 - automatically tuned index :");
    // auto params
    struct arg_dbl * auto_precision       = arg_dbl0(NULL, "auto-precision", "[0,1]", "Expected percentage of exact hits");
    struct arg_dbl * auto_build_weight    = arg_dbl0(NULL, "auto-build-weight", "", "");
    struct arg_dbl * auto_memory_weight   = arg_dbl0(NULL, "auto-memory-weight", "", "");
    struct arg_dbl * auto_sample_fraction = arg_dbl0(NULL, "auto-sample-fraction", "[0,1]", "");
    struct arg_end * end = arg_end(20);
    void * argtable[] = {
       help, input_file, index_file, distance, index_type,
       kd_tree_count,
       km_branching, km_iterations, km_centers, km_index,
       lsh_table_count, lsh_key_size, lsh_probe_level,
       auto_precision, auto_build_weight, auto_memory_weight, auto_sample_fraction,
       end };
    if(arg_nullcheck(argtable) != 0)
    {
        fprintf(stderr, "%s: insufficient memory\n", argv[0]);
        return EXIT_FAILURE;
    }
    input_file->filename[0] = nullptr;
    verbosity->ival[0] = 0;
    distance->ival[0] = 1;//L2
    index_type->ival[0] = 3;//kd + km
    // kd-tree
    kd_tree_count->ival[0] = 4;
    // k-means
    km_branching ->ival[0] = 32;
    km_iterations->ival[0] = 11;
    km_centers   ->ival[0] = 0;//CENTERS_RANDOM
    km_index     ->dval[0] = 0.2;
    // auto
    auto_precision      ->dval[0] = 0.9;
    auto_build_weight   ->dval[0] = 0.01;
    auto_memory_weight  ->dval[0] = 0;
    auto_sample_fraction->dval[0] = 0.1;
    int arg_errors = arg_parse(argc, argv, argtable);
    if(help->count > 0)
    {
        printf("Usage: %s", argv[0]);
        arg_print_syntax(stdout, argtable, "\n");
        arg_print_glossary(stdout, argtable,"  %-25s %s\n");
        return EXIT_SUCCESS;
    }
    if(arg_errors > 0)
    {
        arg_print_errors(stderr, end, argv[0]);
        fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
        return EXIT_FAILURE;
    }

    cvflann::log_verbosity(verbosity->ival[0]);

    // -- Index parameters --

    std::unique_ptr<cv::flann::IndexParams> params;
    switch(index_type->ival[0])
    {
        case 0 :
            params = std::make_unique<cv::flann::LinearIndexParams>();
            break;
        case 1 :
            params = std::make_unique<cv::flann::KDTreeIndexParams>(
                kd_tree_count->ival[0]);
            break;
        case 2 :
            params = std::make_unique<cv::flann::KMeansIndexParams>(
                km_branching ->ival[0],
                km_iterations->ival[0],
                static_cast<cvflann::flann_centers_init_t>(km_centers->ival[0]),
                km_index->dval[0]);
            break;
        case 3 :
            params = std::make_unique<cv::flann::CompositeIndexParams>(
                kd_tree_count->ival[0],
                km_branching ->ival[0],
                km_iterations->ival[0],
                static_cast<cvflann::flann_centers_init_t>(km_centers->ival[0]),
                km_index->dval[0]);
            break;
        case 4 :
            if((lsh_table_count->count == 0) || (lsh_key_size == 0) || (lsh_probe_level == 0))
            {
                std::cerr << "For t=4, lsh-table-count, lsh-key-size and lsh-probe-level must be set.\n";
                return EXIT_FAILURE;
            }
            params = std::make_unique<cv::flann::LshIndexParams>(
                lsh_table_count->ival[0],
                lsh_key_size   ->ival[0],
                lsh_probe_level->ival[0]);
            break;
        case 5 :
            params = std::make_unique<cv::flann::AutotunedIndexParams>(
                auto_precision      ->dval[0],
                auto_build_weight   ->dval[0],
                auto_memory_weight  ->dval[0],
                auto_sample_fraction->dval[0]);
            break;
        default :
            std::cerr << "Unknown index type " << index_type->ival[0] << std::endl;
            return EXIT_FAILURE;
    }

    // -- Load data --

    std::cout << "Loading training data ..." << std::flush;

    auto train = load(input_file->filename[0]);

    cv::Mat_<float> mat = cv::Mat_<float>::zeros(train.data.size(), train.dim);

    boost::dynamic_bitset<> train_class_set;
    for(size_t i = 0; i < train.data.size(); ++i)
    {
        size_t const c = train.data[i].first;
        if(c >= train_class_set.size())
            train_class_set.resize(c+1);
        train_class_set.set(c);
        for(auto p : train.data[i].second)
            mat(i, p.first-1) = p.second;
    }

    std::cout << " OK\n"
        "\tdata : " << train.data.size() << 'x' << train.dim << ", " << train_class_set.count() << " classes\n"
        "Training ..." << std::flush;

    cv::flann::Index index(mat, *params, static_cast<cvflann::flann_distance_t>(distance->ival[0]));

    std::cout << " OK\n";

    index.save(index_file->filename[0]);

    return EXIT_SUCCESS;
}
