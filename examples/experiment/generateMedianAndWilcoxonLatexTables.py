from jmetal.lab.experiment import (
     generate_median_and_wilcoxon_latex_tables,
)

if __name__ == "__main__":
    # Generate Median & IQR tables
    generate_median_and_wilcoxon_latex_tables(filename="QualityIndicatorSummary.csv")
