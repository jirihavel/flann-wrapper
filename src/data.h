#ifndef LIBSVM_DATA_FILE_H_INCLUDED
#define LIBSVM_DATA_FILE_H_INCLUDED

#include <cstdio>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using RowVec = std::vector<std::pair<unsigned, double>>;
using DatVec = std::vector<std::pair<double, RowVec>>;

struct Data
{
    DatVec data;
    unsigned dim;
};

Data load(std::istream & in)
{
    DatVec data;
    unsigned dim = 0;

    size_t line_number = 0;
    std::string line;
    while(std::getline(in, line))
    {
        ++line_number;
        char const * p = line.c_str();

        double label;
        int read;
        if(sscanf(p, "%lf %n", &label, &read) != 1)
        {
            std::cerr << "Line " << line_number << " : Can't read label\n";
            throw std::runtime_error("");
        }
        p += read;

        RowVec row;

        unsigned index;
        double value;
        while(sscanf(p, "%u:%lf %n", &index, &value, &read) == 2)
        {
            p += read;
            row.emplace_back(index, value);
            dim = std::max(dim, index);
        }
        if(*p != '\0')
        {
            std::cerr << "Line " << line_number << " : Invalid data at char " << (p-line.c_str()) << std::endl;
            throw std::runtime_error("");
        }
        data.emplace_back(label, std::move(row));
    }
    return Data{data, dim};
}

Data load(char const * filename)
{
    Data data;
    if(filename)
    {
        std::ifstream file(filename);
        data = load(file);
    }
    else
    {
        data = load(std::cin);
    }
    return data;
}

#endif//LIBSVM_DATA_FILE_H_INCLUDED
