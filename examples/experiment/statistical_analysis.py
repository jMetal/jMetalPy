from jmetal.lab.experiment import generate_boxplot, generate_latex_tables, compute_mean_indicator, compute_wilcoxon
from jmetal.lab.statistical_test.bayesian import *
from jmetal.lab.statistical_test.functions import *
from jmetal.lab.visualization import CDplot, plot_posterior

if __name__ == '__main__':
    # Generate Median & IQR tables
    generate_latex_tables(filename='QualityIndicatorSummary.csv')

    # Generate boxplots
    generate_boxplot(filename='QualityIndicatorSummary.csv')

    # Wilcoxon
    compute_wilcoxon(filename='QualityIndicatorSummary.csv')

    # Statistical lab

    avg = compute_mean_indicator(filename='QualityIndicatorSummary.csv', indicator_name='HV')
    print(avg)

    # Non-parametric test
    print('-------- Sign Test --------')
    print(sign_test(avg[['NSGAII', 'SMPSO']]))
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

    CDplot(avg.T, alpha=0.15, higher_is_better=True)

    print('-------- Bayesian Sign Test --------')
    bst, DProcess = bayesian_sign_test(avg[['NSGAII', 'SMPSO']], rope_limits=[-0.002, 0.002],
                                       prior_strength=0.5, return_sample=True)
    plot_posterior(DProcess, higher_is_better=True, alg_names=['NSGAII', 'SMPSO'])

    print('Pr(NSGAII < SMPSO) = %.3f' % bst[0])
    print('Pr(NSGAII ~= SMPSO) = %.3f' % bst[1])
    print('Pr(NSGAII > SMPSO) = %.3f' % bst[2])

    print('-------- Bayesian Signed Rank Test --------')
    bst, DProcess = bayesian_signed_rank_test(avg[['NSGAII', 'SMPSO']], rope_limits=[-0.002, 0.002],
                                              prior_strength=0.5, return_sample=True)
    plot_posterior(DProcess, higher_is_better=True, alg_names=['NSGAII', 'SMPSO'])

    print('Pr(NSGAII < SMPSO) = %.3f' % bst[0])
    print('Pr(NSGAII ~= SMPSO) = %.3f' % bst[1])
    print('Pr(NSGAII > SMPSO) = %.3f' % bst[2])
