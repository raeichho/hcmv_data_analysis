import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import maximum_likelihood
from scipy.stats import multinomial

np.random.seed(seed=1)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "monospace"
})

LATEX_TEXTWIDTH = 5.87 # inches



def plot_theta_thetamodel(ax):

    MONTE_CARLO_SIZE = 60000
    possible_c = np.linspace(0,10,300)

    def get_plot_points(bar_x):

        plot_points = np.zeros((len(possible_c), 4))

        for i, c in enumerate(possible_c):
            weights = maximum_likelihood.get_infection_pattern_weigths_mc([bar_x], c, 0.1, MONTE_CARLO_SIZE)
            plot_points[i, 0] =c 
            plot_points[i, 1] = weights[0]
            plot_points[i, 2] = weights[1]
            plot_points[i, 3] = weights[2]

        return (plot_points[:,1], plot_points[:,3])

    plot_points03 = get_plot_points(0.3)
    plot_points04 = get_plot_points(0.4)
    plot_points05 = get_plot_points(0.5)

    ax.fill_between([0,1],[0,0], [1,0], color="lightgray", hatch='/')
    ax.scatter(plot_points03[0], plot_points03[1], linewidths=0.5, s=14, marker="x", color="black", label ="$0.3$")
    ax.scatter(plot_points04[0], plot_points04[1], linewidths=0.5, s=40, marker="4", color="black", label ="$0.4$")
    ax.scatter(plot_points05[0], plot_points05[1], linewidths=0.5, s=40,marker="2", color="black", label="$0.5$")
    ax.set_title("$\\Theta_{model}$ and $\\Theta$")
    ax.set_xlabel("$p_A$")
    ax.set_ylabel("$p_B$")
    ax.legend()



def plot_df(ax1, df):
    im = ax1.imshow(df.loc[:, df.columns != 'possible c'].transpose(), aspect='auto',extent=[0.01,0.21,0.00, 0.10], origin='lower') # extent is l, r, b, t
    ax1.set_xlabel("c")
    ax1.set_ylabel("cutoff")


        
    est_cutoff_index = 1 + df.loc[:, df.columns != 'possible c'].max().argmax()
    est_cutoff_string = df.columns[est_cutoff_index] 
    est_cutoff = float(est_cutoff_string)

    est_c = (df[df.columns[0]])[df[est_cutoff_string].argmax()]
    ax1.scatter(est_c, est_cutoff, marker="x", color="red")


   # ax2.axis('off')
   # ax2.text(0.5, 3, "ml estimator for $c$: " + "{:.3f}".format(est_c) + ", ml estimator for cutoff: ""{:.2f}".format(est_cutoff), ha='center', va='top')
   # ax2.text(0.5, 2, "values for $c$: (0.01, 0.21) discretized in $100$ steps", ha='center', va='top')
   #ax2.text(0.5, 1, "values for cutoff: (0.005, 0.1) discretized in $20$ steps", ha='center', va='top')



if __name__ == "__main__":

    fig = plt.figure(figsize=(3/4*LATEX_TEXTWIDTH, 3/4*LATEX_TEXTWIDTH))
    ax = fig.add_axes([0.13, 0.13, 0.74, 0.74])  # [left, bottom, width, height]

    plot_theta_thetamodel(ax)
    plt.show()

    fig.savefig("figures/theta_theta-model.pgf")


    fig = plt.figure(figsize=(LATEX_TEXTWIDTH, 2.5))
    ax1 = fig.add_axes([0.1, 0.15, 0.45, 0.7])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.65, 0.15, 0.34, 0.7])  # [left, bottom, width, height]


    #df = pd.read_csv("joint_inference/joint_inference.csv")
    df = pd.read_csv("big_analysis.csv")
    plot_df(ax1, df)
    ax1.hlines([.072, 0.023, 0.024, 1-0.916632274040446],0.01,0.21, colors=['white'], linewidth=1)
    ax1.set_title("(a) Likelihood")


    ax2.set_title("(b) No. mixed loci")



    MONTE_CARLO_SIZE = 600000 # 2 loci: 600.000 in under 60s; 3 loci: 6.000.000 in 12 min; 4 loci: 6.000.000 in 23 min
    CUTOFF = 0.07
    C = 0.147


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



    def cutoff(x, cutofflevel=CUTOFF):
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



    infection_patterns_list = list(infection_patterns.values())
    type_a_frequencies_list = list(type_a_frequencies)



    C_infection_pattern_weights = maximum_likelihood.get_infection_pattern_weigths_mc(type_a_frequencies_list, C, CUTOFF, MONTE_CARLO_SIZE)


    def multinomial_p_value(observation, weights, n_draws=1000000):
        
        N = np.sum(observation)
        
        sample = multinomial.rvs(N, weights, size=n_draws)
        
        probs = multinomial.pmf(sample, N, weights)
        
        prob_observation =  multinomial.pmf(observation, N, weights)
        
        p = (probs<=prob_observation).sum() / n_draws
        
        return p

    observation = list(infection_patterns.values())
    weights = C_infection_pattern_weights
    p_value = multinomial_p_value(observation, weights)
    print("p-value: ", p_value)

    sorted_infection_patterns_dict = dict(sorted(infection_patterns.items(), key=lambda item: item[1]))
    sorted_infection_pattern_weights_dict = {k : v / n_samples for k, v in sorted_infection_patterns_dict.items()}
    C_infection_pattern_weights_dict = dict(zip(maximum_likelihood.generate_index(n_studied_loci), C_infection_pattern_weights))
    sorted_C_infection_pattern_weights_dict = {k: C_infection_pattern_weights_dict[k] for k in sorted_infection_patterns_dict.keys()}

    barwidth =0.4
    x_ticks = np.arange(n_studied_loci+1)
    ax2.bar(x_ticks+0.2, np.array(list(get_num_mixed_loci_dict(infection_patterns).values()))/n_samples, width=barwidth,  color="black", edgecolor="black", label="observation")
    ax2.bar(x_ticks-0.2, get_num_mixed_loci_dict(dict(zip(infection_patterns.keys(), C_infection_pattern_weights))).values(), width=barwidth, hatch="///",edgecolor="black", color="white", label="inference")
    ax2.set_xlabel("no. mixed loci")
    ax2.set_ylabel("frequency")
    ax2.legend()
    ax2.set_xticks(x_ticks)

    plt.show()

    fig.savefig("figures/likelihood.pgf")


  



