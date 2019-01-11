from jmetal.util.statistical_test.bayesian import bayesian_sign_test, bayesian_signed_rank_test
from jmetal.util.visualization import CDplot, plot_posterior
from jmetal.util.laboratory import compute_median_iqr_tables, compute_mean_indicator
from jmetal.util.statistical_test.functions import *

if __name__ == '__main__':
    # Compute Median and IQR tables from the experiment
    compute_median_iqr_tables(filename='QualityIndicatorSummary.csv')

    # Statistical analysis
    avg = compute_mean_indicator(filename='QualityIndicatorSummary.csv', indicator_name='Fitness')
    print(avg)

    # Non-parametric test
    print('-------- Sign Test --------')
    print(sign_test(avg[['ssGA', 'gGA']]))
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

    # Plot critical distance
    CDplot(avg.T, alpha=0.05)

    print('-------- Bayesian Sign Test --------')
    bst, DProcess = bayesian_sign_test(avg[['ssGA', 'gGA']],
                             prior_strength=0.5, return_sample=True)
    print('Pr(MOCell < SMPSO) = %.3f' % bst[0])
    print('Pr(MOCell ~= SMPSO) = %.3f' % bst[1])
    print('Pr(MOCell > SMPSO) = %.3f' % bst[2])

    plot_posterior(DProcess)

    print('-------- Bayesian Signed Rank Test --------')
    bst, DProcess = bayesian_signed_rank_test(
        avg[['MOCell', 'SMPSO']], prior_strength=0.5, return_sample=True)
    print('Pr(MOCell < SMPSO) = %.3f' % bst[0])
    print('Pr(MOCell ~= SMPSO) = %.3f' % bst[1])
    print('Pr(MOCell > SMPSO) = %.3f' % bst[2])

    plot_posterior(DProcess)
