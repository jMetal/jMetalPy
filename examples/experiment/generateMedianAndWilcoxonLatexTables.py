from jmetal.lab.experiment import (
     generate_median_and_wilcoxon_latex_tables,
)

if __name__ == "__main__":
    """
        Generate Latex tables including medians and IQRs. Additionally, the last algorithm is considered as the reference
        algorithm, and the cells include a symbol indicating whether the differences with the reference algorithm
        are significant or not according to the Wilcoxon rank sum test. 
    """
    generate_median_and_wilcoxon_latex_tables(filename="QualityIndicatorSummary.csv")
