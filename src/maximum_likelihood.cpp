/*** COMPILE ON UBUNTU using g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) maximum_likelihood.cpp -o ../maximum_likelihood$(python3-config --extension-suffix) $(gsl-config --cflags) $(gsl-config --libs)
THE g++ Version used is g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
***/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

/*** Start RNG ****/
#include <stdio.h>
#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <vector>
#include <cassert>
/*** End RNG ****/

#include <string>
#include <math.h>
#include <cassert>
#include <stdint.h>

#include <algorithm>

void simulate_dirichlet(std::vector<double> *placeholder_ptr, int size, int K, const double *alpha)
{

    const gsl_rng_type *T;
    gsl_rng *r;

    /* create a generator chosen by the
       environment variable GSL_RNG_TYPE */

    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    for (int i = 0; i < size; i++)
    {
        gsl_ran_dirichlet(r, K, alpha, &(*placeholder_ptr)[K * i]);
    }

    double sum = 0;

    int zero_one_counter = 0;

    for (int i = 0; i < size * K; i++)
    {
        sum = sum + (*placeholder_ptr)[i];

        zero_one_counter = zero_one_counter + ((*placeholder_ptr)[i] > 0.9);
    }
    gsl_rng_free(r);
}

static inline uint32_t log2(const uint32_t x)
{
    uint32_t y;
    asm("\tbsr %1, %0\n"
        : "=r"(y)
        : "r"(x));
    return y;
}

std::vector<std::string> product(std::vector<std::string> vec_1, std::vector<std::string> vec_2)
{

    int size1 = vec_1.size();
    int size2 = vec_2.size();
    int result_size = size1 * size2;

    std::vector<std::string> result(result_size);

    for (int i = 0; i < size1; i++)
    {
        for (int j = 0; j < size2; j++)
        {
            result[j + i * size2] = vec_1[i] + vec_2[j];
        }
    }
    return result;
}

std::vector<std::string> k_fold_product(std::vector<std::string> vec, unsigned int k)
{

    if (k == 1)
    {
        return vec;
    }
    else
    {
        return product(vec, k_fold_product(vec, k - 1));
    }
}

std::vector<std::string> generate_index(unsigned int num_loci)
{
    return k_fold_product({"1", "m", "2"}, num_loci);
}

std::vector<std::string> get_dirichlet_weights(std::vector<double> type_a_frequencies, double c, double weights_placeholder[])
{
    int num_loci = type_a_frequencies.size();

    std::vector<std::string> strains = k_fold_product({"1", "2"}, num_loci);

    std::fill_n(weights_placeholder, strains.size(), c);

    for (int i = 0; i < strains.size(); i++)
    {

        for (int j = 0; j < num_loci; j++)
        {
            if ((strains[i])[j] == '1')
            {
                weights_placeholder[i] = weights_placeholder[i] * type_a_frequencies[j];
            }
            else
            {
                weights_placeholder[i] = weights_placeholder[i] * (1 - type_a_frequencies[j]);
            }
        }
    }

    return strains;
}

std::vector<double> get_independent_locus_frequencies(std::vector<double> hidden_frequencies, std::vector<std::string> idx)
{

    int size = hidden_frequencies.size();

    int num_loci = log2(size);

    // std::cout << "Size: " << size << "\n";
    // std::cout << "Num loci: " << num_loci << "\n";

    std::vector<double> type_a_frequencies(num_loci);

    for (int locus = 0; locus < num_loci; locus++)
    {
        double type_a_freq = 0;

        for (int pattern = 0; pattern < size; pattern++)
        {
            if ((idx[pattern])[locus] == '1')
            {
                type_a_freq = type_a_freq + hidden_frequencies[pattern];
            }
        }
        type_a_frequencies[locus] = type_a_freq;
    }

    return type_a_frequencies;
}

char get_single_locus_infection_pattern(double frequency, double cutoff)
{

    if (frequency <= cutoff)
    {
        return '2';
    }
    else if (frequency < 1-cutoff)
    {
        return 'm';
    }
    else
    {
        return '1';
    }
}

std::string get_full_infection_pattern(std::vector<double> type_a_frequencies, double cutoff)
{

    int size = type_a_frequencies.size();

    std::string infection_pattern = "";

    for (int i = 0; i < size; i++)
    {
        infection_pattern = infection_pattern + get_single_locus_infection_pattern(type_a_frequencies[i], cutoff);
    }

    return infection_pattern;
}

std::string get_full_infection_pattern_from_hidden_frequencies(std::vector<double> hidden_frequencies, std::vector<std::string> idx, double cutoff)
{
    return get_full_infection_pattern(get_independent_locus_frequencies(hidden_frequencies, idx), cutoff);
}

std::vector<double> get_infection_pattern_weights_mc(std::vector<double> type_a_frequencies, double c, double cutoff, int size = 1)
{

    const int num_loci = type_a_frequencies.size();

    const int num_hidden_strains = 1 << num_loci;

    double weight_values_helper[num_hidden_strains];

    std::vector<std::string> strain_names = get_dirichlet_weights(type_a_frequencies, c, weight_values_helper);

    std::vector<double> simulated_rvs(num_hidden_strains * size);
    simulate_dirichlet(&simulated_rvs, size, num_hidden_strains, weight_values_helper);

    std::vector<std::string> infection_patterns(size);

    for (int i = 0; i < size; i++)
    {
        std::vector<double> sub(&simulated_rvs[i * num_hidden_strains], &simulated_rvs[(i + 1) * num_hidden_strains]);
        infection_patterns[i] = get_full_infection_pattern_from_hidden_frequencies(sub, strain_names, cutoff);
    }

    // std::cout << "Size of infection patterns: " << infection_patterns.size() << "\n";

    // std::cout << "The pattern mmmm appears " << std::count(infection_patterns.begin(), infection_patterns.end(), "AAAA") << std::count(infection_patterns.begin(), infection_patterns.end(), "mmmm") << " times"
    // << "\n";

    // std::cout << "The pattern AAAA appears " << std::count(infection_patterns.begin(), infection_patterns.end(), "AAAA") << " times"
    // << "\n";

    const std::vector<std::string> infection_pattern_names = generate_index(num_loci);

    const unsigned int num_patterns = infection_pattern_names.size();

    std::vector<double> infection_pattern_weights(num_patterns);

    for (int i = 0; i < num_patterns; i++)
    {
        infection_pattern_weights[i] = (double)std::count(infection_patterns.begin(), infection_patterns.end(), infection_pattern_names[i]) / (double)size;
        // std::cout << i << ". " << infection_pattern_names[i] << ": " << infection_pattern_weights[i] << "\n";
    }

    return infection_pattern_weights;
}

double likelihood(std::vector<unsigned int> observation, const std::vector<double> type_a_frequencies, double c, double cutoff, int size = 1)
{

    const std::vector<double> weights = get_infection_pattern_weights_mc(type_a_frequencies, c, cutoff, size);

    return gsl_ran_multinomial_pdf(weights.size(), &weights[0], &observation[0]);
}

PYBIND11_MODULE(maximum_likelihood, m)
{
    m.def("likelihood", py::vectorize(likelihood));
    m.def("get_infection_pattern_weigths_mc", &get_infection_pattern_weights_mc);
    m.def("generate_index", &generate_index);
}
