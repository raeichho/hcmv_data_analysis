from multiprocessing import Pool
import maximum_likelihood
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multinomial
import time


def f(cutoff_level):
    
    MONTE_CARLO_SIZE = 6000000   # OLD 2 loci: 600.000 in under 60s; 3 loci: 6.000.000 in 12 min; 4 loci: 6.000.000 in 23 min
                                # NEW 2 loci: 600.000 in 6 mins; 3 loci: 600.000 in 9 min and 6.000.000 in 134 min;

    studied_loci = [1,2,3,4]


    data = pd.read_csv("data/data.csv", na_values=["nd", "cov too low", "no cov"])
    locus_names = ['UL4', 'UL75', 'UL78', 'US27']
    studied_loci = [locus_names[i-1] for i in studied_loci]
    n_studied_loci = len(studied_loci)

    type_a_frequencies = (data[locus_names].sum() / data[locus_names].notna().sum())[studied_loci]


    data = data[studied_loci]
    data = data.dropna()

    assert data.notna().all(axis=None)

    n_samples = data.shape[0]

    # type_a_frequencies = data.sum() / n_samples


    def cutoff(x, cutofflevel=cutoff_level):
        if x>1-cutofflevel:
            return "1"
        elif x<cutofflevel:
            return "2"
        else:
            return "m"

    data = data.applymap(cutoff)

    data = data.agg("".join, axis=1)

    infection_patterns = data.value_counts().to_dict()


    def standardize_infection_patterns(infection_patterns):
        """ """

        from itertools import product
        li = ["".join(i) for i in product(["1", "2", "m"], repeat=n_studied_loci)]

        for key in li:
            if key not in infection_patterns.keys():
                infection_patterns[key] = 0

        def sorting_function(str):
            """this is highly un-intuitive. We want an ''alphabetical'' ordering of strings with
            the modification that an 'm' has to be between the 'A' and the 'B'."""

            return str[0].replace("1","A").replace("2", "C").replace("m", "B")

        return dict(sorted(infection_patterns.items(), key=sorting_function))

    infection_patterns = standardize_infection_patterns(infection_patterns)

    def count_num_mixed_loci(str):
        return str.count("m")

    def get_num_mixed_loci_dict(infection_patterns):
        num_loci = len(studied_loci)
        return {n: sum([v for k, v in infection_patterns.items() if count_num_mixed_loci(k)==n]) for n in range(num_loci+1)}

    print("Type A frequencies:\n")
    print(type_a_frequencies)

    print("\nNumbers of mixed loci:\n")
    print(get_num_mixed_loci_dict(infection_patterns))


    possible_c = np.linspace(0.01,0.21,100)

    infection_patterns_list = list(infection_patterns.values())
    type_a_frequencies_list = list(type_a_frequencies)



    likelihood_values = maximum_likelihood.likelihood(infection_patterns_list,
                                            type_a_frequencies_list,
                                            possible_c, cutoff_level, MONTE_CARLO_SIZE)

    ml_estimator = possible_c[likelihood_values.argmax()]


    return (cutoff_level, possible_c, likelihood_values)


if __name__ == '__main__':

    start = time.time()

    possible_cutoffs = np.linspace(0.005, 0.1, 20)
    
    with Pool(5) as p:
        result = p.map(f, possible_cutoffs)

    possible_c = np.linspace(0.01,0.21,100)

    df = pd.DataFrame(columns=possible_cutoffs, index=possible_c)
    df.index.name="possible c"
    
    for res in result:
        df[res[0]] = res[2]
    
    df.to_csv("big_analysis.csv")

    end = time.time()
    elapsed_time = end - start

    print(elapsed_time)