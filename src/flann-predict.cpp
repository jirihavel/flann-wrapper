#include "data.h"

#include <cstdlib>

#include <fstream>
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
    struct arg_file * train_file = arg_file1("f", "features", "<filename>", "Training dataset");
    struct arg_file * index_file = arg_file1("x", "index", "<filename>", "Training dataset index");
    struct arg_file * input = arg_file1("i", "input", "<filename>", "Input dataset in libsvm format");
    struct arg_file * output = arg_file1("o", "output", "<filename>", "Output index file");
    struct arg_int  * distance = arg_int0("d", "distance", "{1..9}", "Distance metric"
            "\n\t1=L2 (default), 2=L1, 3=MINKOWSKI,\n\t4=MAX, 5=HIST_INTERSECT, 6=HELLLINGER,"
            "\n\t7=CS, 8=KULLBACK_LEIBLER, 9=HAMMING");
    struct arg_int * neighbors = arg_int0("n", "neighbors", "n", "Neighbor count (default 1)");
    struct arg_dbl * radius = arg_dbl0("r", "radius", "r", "Search radius, requests radius search");
    struct arg_int * checks = arg_int0("c", "checks", "...", "Search checks (default 32)");
    struct arg_lit * help = arg_lit0("h", "help", "Print this help and exit");
    struct arg_end * end = arg_end(20);
    void * argtable[] = {
       train_file, index_file, input, output, distance, neighbors, radius, checks,
       help, end };
    if(arg_nullcheck(argtable) != 0)
    {
        fprintf(stderr, "%s: insufficient memory\n", argv[0]);
        return EXIT_FAILURE;
    }
    distance->ival[0] = 1;
    neighbors->ival[0] = 1;
    checks->ival[0] = 32;
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

    std::cout << "Loading training data ..." << std::flush;

    auto train = load(train_file->filename[0]);

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
        "Loading model ..." << std::flush;

    cv::flann::Index index;
    if(!index.load(mat, index_file->filename[0]))
    {
        fprintf(stderr, "Can't load index '%s'.\n", index_file->filename[0]);
        return EXIT_SUCCESS;
    }

    std::cout << " OK\nLoading testing data ..." << std::flush;

    auto test = load(input->filename[0]);

    boost::dynamic_bitset<> test_class_set;
    for(size_t i = 0; i < test.data.size(); ++i)
    {
        size_t const c = test.data[i].first;
        if(c >= test_class_set.size())
            test_class_set.resize(c+1);
        test_class_set.set(c);
    }

    std::cout << " OK\n"
        "\tdata : " << test.data.size() << 'x' << test.dim << ", " << test_class_set.count() << " classes\n";
    if(!test_class_set.is_subset_of(train_class_set))
    {
        std::cout << "\t!!! " << (test_class_set-train_class_set).count() << " test classes not in training data\n";
    }
    std::cout << "Searching ..." << std::flush;

    auto const n = neighbors->ival[0];
    cv::flann::SearchParams const params{checks->ival[0]};

    std::ofstream file(output->filename[0]);
    if(!file)
    {
        fprintf(stderr, "Can't open output file '%s'\n", output->filename[0]);
        return EXIT_FAILURE;
    }

    std::vector<size_t> match_counts(n, 0);
    std::vector<size_t> cumulative_match_counts(n, 0);
    std::vector<size_t> match_hist(n+1, 0);

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

        unsigned m = 0;
        bool found = false;
        file << test.data[i].first;
        for(int j = 0; j < n; ++j)
        {
            file << ' ' << indices[j] << ':' << train.data[indices[j]].first;
            bool ok = abs(test.data[i].first - train.data[indices[j]].first) < 0.1;
            if(ok)
            {
                ++m;
                ++match_counts[j];
            }
            found = found || ok;
            if(found)
                ++cumulative_match_counts[j];
        }
        file << ' ' << m << std::endl;

        ++match_hist[m];
    }

    std::cout << " OK\n";

    auto const count = test.data.size();

    for(int i = 0; i < n; ++i)
    {
        std::cout << i << " : " << match_counts[i] << " of " << count << ", total " << cumulative_match_counts[i]
            << " (" << (100.*match_counts[i]/count) << "%, " << (100.*cumulative_match_counts[i]/count) << "%)\n";
    }
    for(int i = 0; i < (n+1); ++i)
    {
        std::cout << i << " : " << match_hist[i] << " (" << (100.*match_hist[i]/count) << "%)\n";
    }

    return EXIT_SUCCESS;
}
