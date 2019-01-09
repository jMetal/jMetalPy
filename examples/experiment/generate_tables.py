from jmetal.util.graphic import CDplot
from jmetal.util.laboratory import compute_median_iqr_tables, compute_mean_indicator

if __name__ == '__main__':
    base_directory = 'data'

    compute_median_iqr_tables(filename='data/QualityIndicatorSummary2.csv')
    avg = compute_mean_indicator(filename='data/QualityIndicatorSummary2.csv', indicator_name='EP')
    CDplot(avg.T, alpha=0.05)
