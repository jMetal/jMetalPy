import pandas as pd

from jmetal.component.critical_distance import CDplot
from jmetal.util.laboratory import create_tables_from_experiment

if __name__ == '__main__':
    base_directory = 'data'

    create_tables_from_experiment(base_dir=base_directory, filename='QualityIndicatorSummary.csv')

    # Plot CD
    results_hv = []
    labels = ['NSGAIIa', 'NSGAIIb', 'NSGAIIc', 'NSGAIId']

    for algorithm in labels:
        # Read .csv file for selected algorithm
        values = pd.read_csv(base_directory + '/QualityIndicator' + algorithm + '.csv', sep='\t')
        values = values.set_index(['Problem', 'IndicatorName', 'ExecutionId'])

        # Get only Epsilon values
        values = values.xs('EP', level='IndicatorName', axis=0)

        # Compute mean
        values = values.groupby(level=0).mean()['IndicatorValue'].tolist()

        results_hv.append(values)

    CDplot(results_hv, alg_names=labels)
