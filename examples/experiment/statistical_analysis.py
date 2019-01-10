from jmetal.util.visualization import CDplot
from jmetal.util.laboratory import compute_median_iqr_tables, compute_mean_indicator
from jmetal.util.statistical_test.functions import *

if __name__ == '__main__':
    base_directory = 'data'

    compute_median_iqr_tables(filename='data/QualityIndicatorSummary2.csv')
    avg = compute_mean_indicator(filename='data/QualityIndicatorSummary2.csv', indicator_name='IGD+')

    print(avg)

    CDplot(avg.T, alpha=0.05)

    # Non-parametric test
    print('-------- Sign Test --------')
    print(sign_test(avg[['MOCell', 'SMPSO']]))
    print('-------- Friedman Test --------')
    print(friedman_test(avg))
    print('-------- Friedman Aligned Rank Test --------')
    print(friedman_aligned_rank_test(avg))
    print('-------- Quade Test --------')
    print(quade_test(avg))

    # Post-hoc tests
    print('-------- Friedman Post-Hoc Test --------')
    z, p_val, adj_pval = friedman_ph_test(avg, control=0, apv_procedure='Bonferroni')
    print('z values \n', z)
    print('p-values \n', p_val)
    print('adjusted p-values \n', adj_pval)
    print('-------- Friedman Aligned Rank Post-Hoc Test --------')
    z, p_val, adj_pval = friedman_aligned_ph_test(avg, apv_procedure='Shaffer')
    print('z values \n', z)
    print('p-values \n', p_val)
    print('adjusted p-values \n', adj_pval)
    print('-------- QuadeTest Post-Hoc Test --------')
    z, p_val, adj_pval = quade_ph_test(avg, apv_procedure='Holm')
    print('z values \n', z)
    print('p-values \n', p_val)
    print('adjusted p-values \n', adj_pval)
