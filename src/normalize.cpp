#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <string>
#include <vector>
#include <utility>

int main()
{
    size_t line_number = 0;
    std::string line;
    while(std::getline(std::cin, line))
    {
        ++line_number;
        char const * p = line.c_str();

        double label;
        int read;
        if(sscanf(p, "%lf %n", &label, &read) != 1)
        {
            std::cerr << "Line " << line_number << " : Can't read label\n";
            return EXIT_FAILURE;
        }
        p += read;

        std::vector<std::pair<unsigned, double>> data;
        double norm = 0;

        unsigned index;
        double value;
        while(sscanf(p, "%u:%lf %n", &index, &value, &read) == 2)
        {
            p += read;
            data.emplace_back(index, value);
            norm += value*value;
        }
        if(*p != '\0')
        {
            std::cerr << "Line " << line_number << " : Invalid data at char " << (p-line.c_str()) << std::endl;
            return EXIT_FAILURE;
        }

        double const scale = 1.0/sqrt(norm);

        std::cout << label;
        for(auto const & x : data)
        {
            std::cout << ' ' << x.first << ':' << (scale*x.second);
        }
        std::cout << std::endl;
    }
    return EXIT_SUCCESS;
}
