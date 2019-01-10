from jmetal.util.visualization import CDplot
from jmetal.util.laboratory import compute_median_iqr_tables, compute_mean_indicator
from jmetal.util.statistical_test.functions import friedman_test, friedman_ph_test, quade_test

if __name__ == '__main__':
    base_directory = 'data'

    # compute_median_iqr_tables(filename='data/QualityIndicatorSummary2.csv')
    avg = compute_mean_indicator(filename='data/QualityIndicatorSummary2.csv', indicator_name='EP')

    CDplot(avg.T, alpha=0.05)

    print(friedman_test(avg))
    print(quade_test(avg))

    z, p_val, adj_pval = friedman_ph_test(avg, control=0, apv_procedure='Bonferroni')
    print(z, '\n', p_val, '\n', adj_pval)
