from jmetal.lab.experiment import (
    generate_kolmogorov_smirnov_latex_tables,
)

if __name__ == "__main__":
    """
        Generate Latex tables with the results of the Kolmogorov-Smirnov test. The last algorithm is considered as 
        the reference algorithm, and the cells include a symbol with the p-value < 0.05.
    """
    generate_kolmogorov_smirnov_latex_tables(filename="QualityIndicatorSummary.csv")
