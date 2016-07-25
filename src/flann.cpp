#include "data.h"

#include <cstdlib>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <argtable2.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/dynamic_bitset.hpp>
#include <opencv2/flann/flann.hpp>

int main(int argc, char * argv[])
{
    struct arg_file * train_file = arg_file1("f", "features", "<filename>", "Training dataset");
    struct arg_file * index_file = arg_file0("x", "index", "<filename>", "Training dataset index");
    struct arg_file * output_index = arg_file0(NULL, "output-index", "<filename>", "Save used index to file");
    struct arg_file * input  = arg_file0("i", "input", "<filename>", "Input dataset in libsvm format");
    struct arg_file * output = arg_file0("o", "output", "<filename>", "");
    struct arg_lit * hist = arg_lit0(NULL, "hist", "");
    struct arg_lit * help   = arg_lit0("h", "help", "Print this help and exit");
    struct arg_int * verbosity = arg_int0 ("v", "verbosity", "{0..4}", "Log verbosity"
            "\nIndex parameters :");
    struct arg_int  * distance = arg_int0("d", "distance", "{1..9}", "Distance metric"
            "\n\t1=L2 (default), 2=L1, 3=MINKOWSKI,\n\t4=MAX, 5=HIST_INTERSECT, 6=HELLLINGER,"
            "\n\t7=CS, 8=KULLBACK_LEIBLER, 9=HAMMING");
    struct arg_int  * index_type = arg_int0("t", "index-type", "{0..5}", "Constructed index type"
            "\n t=0 - linear brute force search"
            "\n t=1 - kd-tree :");
    struct arg_int * kd_tree_count = arg_int0(NULL, "kd-tree-count", "{1..16+}", "Number of parallel trees (default 4)"
            "\n t=2 - k-means :");
    struct arg_int * km_branching  = arg_int0(NULL, "km-branching", "n", "Branching factor (default 32)");
    struct arg_int * km_iterations = arg_int0(NULL, "km-iterations", "n", "Maximum iterations (default 11)");
    struct arg_int * km_centers    = arg_int0(NULL, "km-centers", "{0..2}", "Initial cluster centers"
            "\n\t0=CENTERS_RANDOM (default)\n\t1=CENTERS_GONZALES\n\t2=CENTERS_KMEANSPP");
    struct arg_dbl * km_index      = arg_dbl0(NULL, "km-index", "", "Cluster boundary index (default 0.2)"
            "\n t=3 (default) - kd-tree + k-means"
            "\n t=4 - LSH :");
    struct arg_int * lsh_table_count = arg_int0(NULL, "lsh-table-count", "{0..}", "Number of hash tables");
    struct arg_int * lsh_key_size    = arg_int0(NULL, "lsh-key-size", "{0..}", "Hash key bits");
    struct arg_int * lsh_probe_level = arg_int0(NULL, "lsh-probe-level", "{0..}", "Bit shift for neighboring bucket check"
            "\n t=5 - automatically tuned index :");
    struct arg_dbl * auto_precision       = arg_dbl0(NULL, "auto-precision", "[0,1]", "Expected percentage of exact hits");
    struct arg_dbl * auto_build_weight    = arg_dbl0(NULL, "auto-build-weight", "", "");
    struct arg_dbl * auto_memory_weight   = arg_dbl0(NULL, "auto-memory-weight", "", "");
    struct arg_dbl * auto_sample_fraction = arg_dbl0(NULL, "auto-sample-fraction", "[0,1]", ""
            "\nSearch parameters :");
    struct arg_int * neighbors = arg_int0("n", "neighbors", "n", "Neighbor count (default 1)");
    struct arg_dbl * radius    = arg_dbl0("r", "radius", "r", "Search radius, requests radius search");
    struct arg_int * checks    = arg_int0("c", "checks", "...", "Search checks (default 32)");
    struct arg_end * end = arg_end(20);
    void * argtable[] = {
       train_file, index_file, output_index, input, output, hist, help, verbosity,
       distance, index_type, kd_tree_count,
       km_branching, km_iterations, km_centers, km_index,
       lsh_table_count, lsh_key_size, lsh_probe_level,
       auto_precision, auto_build_weight, auto_memory_weight, auto_sample_fraction,
       neighbors, radius, checks, end };
    if(arg_nullcheck(argtable) != 0)
    {
        fprintf(stderr, "%s: insufficient memory\n", argv[0]);
        return EXIT_FAILURE;
    }
    // -- Defaults --
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
    // search
    neighbors->ival[0] = 1;
    checks->ival[0] = 32;
    // -- Parse --
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

    std::cout << "Loading features '" << train_file->filename[0] << "' ..." << std::flush;

    auto train = load(train_file->filename[0]);

    cv::Mat_<float> mat = cv::Mat_<float>::zeros(train.data.size(), train.dim);

    boost::dynamic_bitset<> train_class_set;
    std::vector<size_t> train_class_hist;
    for(size_t i = 0; i < train.data.size(); ++i)
    {
        size_t const c = train.data[i].first;
        if(c >= train_class_set.size())
        {
            train_class_set.resize(c+1);
            train_class_hist.resize(c+1, 0);
        }
        train_class_set.set(c);
        train_class_hist[c] += 1;

        for(auto p : train.data[i].second)
            mat(i, p.first-1) = p.second;
    }

    std::cout << " OK\n"
        "\tdata : " << train.data.size() << 'x' << train.dim << ", " << train_class_set.count() << " classes\n";
    if(hist->count > 0)
    {
        std::cout << "\thistogram :\n";
        for(size_t i = 0; i < train_class_set.size(); ++i)
            std::cout << '\t' << i << " : " << train_class_hist[i] << " (" << (100.*train_class_hist[i]/train.data.size()) << "%)\n";
    }

    // -- Index --

    cv::flann::Index index;
    if(index_file->count > 0)
    {
        std::cout << "Loading index '" << index_file->filename[0] << "' ..." << std::flush;
        if(!index.load(mat, index_file->filename[0]))
        {
            fprintf(stderr, "Can't load index '%s'.\n", index_file->filename[0]);
            return EXIT_SUCCESS;
        }
    }
    else
    {
        std::cout << "Building index ..." << std::flush;
        // Parameters
        std::unique_ptr<cv::flann::IndexParams> params;
        switch(index_type->ival[0])
        {
            case 0 : // linear brute force search
                params = std::make_unique<cv::flann::LinearIndexParams>();
                break;
            case 1 : // k-d tree
                params = std::make_unique<cv::flann::KDTreeIndexParams>(
                    kd_tree_count->ival[0]);
                break;
            case 2 : // k-means 
                params = std::make_unique<cv::flann::KMeansIndexParams>(
                    km_branching ->ival[0],
                    km_iterations->ival[0],
                    static_cast<cvflann::flann_centers_init_t>(km_centers->ival[0]),
                    km_index->dval[0]);
                break;
            case 3 : // k-d tree + k-means
                params = std::make_unique<cv::flann::CompositeIndexParams>(
                    kd_tree_count->ival[0],
                    km_branching ->ival[0],
                    km_iterations->ival[0],
                    static_cast<cvflann::flann_centers_init_t>(km_centers->ival[0]),
                    km_index->dval[0]);
                break;
            case 4 : // lsh
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
            case 5 : // autotuned index
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
        index.build(mat, *params, static_cast<cvflann::flann_distance_t>(distance->ival[0]));
    }
    std::cout << " OK\n";

    if(output_index->count > 0)
    {
        std::cout << "Saving index '" << output_index->filename[0] << "' ..." << std::flush;
        index.save(output_index->filename[0]);
        std::cout << " OK\n";
    }

    // -- Query --

    if(input->count > 0)
    {
        std::cout << "Loading query '" << input->filename[0] << "' ..." << std::flush;

        auto test = load(input->filename[0]);

        boost::dynamic_bitset<> test_class_set(train_class_set.size());
        std::vector<size_t> test_class_hist(train_class_set.size());
        for(size_t i = 0; i < test.data.size(); ++i)
        {
            size_t const c = test.data[i].first;
            if(c >= test_class_set.size())
            {
                test_class_set.resize(c+1);
                test_class_hist.resize(c+1, 0);
            }
            test_class_set.set(c);
            test_class_hist[c] += 1;
        }

        std::cout << " OK\n"
            "\tdata : " << test.data.size() << 'x' << test.dim << ", " << test_class_set.count() << " classes\n";
        if(!test_class_set.is_subset_of(train_class_set))
            std::cout << "\t!!! " << (test_class_set-train_class_set).count() << " test classes not in training data\n";
        //std::cout << "\thistogram :\n";
        //for(size_t i = 0; i < test_class_set.size(); ++i)
        //    std::cout << '\t' << i << " : " << test_class_hist[i] << " (" << (100.*test_class_hist[i]/test.data.size()) << "%)\n";

        std::cout << "Searching ..." << std::flush;

        cv::flann::SearchParams const params{checks->ival[0]};

        std::ofstream file;
        if(output->count > 0)
        {
            file.open(output->filename[0]);
            if(!file)
            {
                fprintf(stderr, "Can't open output file '%s'\n", output->filename[0]);
                return EXIT_FAILURE;
            }
        }

        auto const n = neighbors->ival[0];
        std::vector<size_t> match_counts(n, 0);
        std::vector<size_t> cumulative_match_counts(n, 0);
        std::vector<size_t> match_hist(n+1, 0);

        namespace acc = boost::accumulators;
        
        std::vector<acc::accumulator_set<double, acc::stats<
            acc::tag::mean, acc::tag::min, acc::tag::max>>> class_matches(test_class_set.size());

        for(size_t i = 0; i < test.data.size(); ++i)
        {
            std::vector<float> query(test.dim, 0);
            std::vector<int  > indices(n);
            std::vector<float> dists(n);

            for(auto p : test.data[i].second)
                query[p.first-1] = p.second;

            if(radius->count > 0)
                index.radiusSearch(query, indices, dists, radius->dval[0], n, params);
            else
                index.knnSearch(query, indices, dists, n, params);

            unsigned matching_neighbors = 0;
            bool found = false;
            file << test.data[i].first;
            for(int j = 0; j < n; ++j)
            {
                file << ' ' << indices[j] << ':' << train.data[indices[j]].first;
                bool ok = abs(test.data[i].first - train.data[indices[j]].first) < 0.1;
                if(ok)
                {
                    ++matching_neighbors;
                    ++match_counts[j];
                }
                found = found || ok;
                if(found)
                    ++cumulative_match_counts[j];
            }
            file << ' ' << matching_neighbors << std::endl;

            ++match_hist[matching_neighbors];

            class_matches[test.data[i].first](matching_neighbors);
        }
        std::cout << " OK\n";

        auto const count = test.data.size();

        for(int i = 0; i < n; ++i)
        {
            std::cout << i << " : " << match_counts[i] << ", total " << cumulative_match_counts[i]
                << " (" << (100.*match_counts[i]/count) << "%, " << (100.*cumulative_match_counts[i]/count) << "%)\n";
        }
        for(int i = 0; i < (n+1); ++i)
        {
            std::cout << i << " : " << match_hist[i] << " (" << (100.*match_hist[i]/count) << "%)\n";
        }

        std::cout << "Class matches :\n";
        for(size_t i = 0; i < class_matches.size(); ++i)
        {
            auto const cnt = acc::count(class_matches[i]);
            if(cnt > 0)
            {
                auto const mean = acc::mean(class_matches[i]);
                std::cout << i << " : " << cnt << " (" << (100.*cnt/count) << "%) - "
                    << mean << " (" << (100.*mean/n)
                    << "%) in [" << acc::min(class_matches[i]) << ',' << acc::max(class_matches[i]) << "]\n";
            }
        }
    }
    return EXIT_SUCCESS;
}
