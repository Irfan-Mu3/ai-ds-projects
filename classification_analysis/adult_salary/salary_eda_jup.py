import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# for jupyter:
# import sys
# sys.path.append('..')

from classification_funcs.classification_plots import create_dummy_matrix, create_classification_plots, create_andrews_plot, \
    create_rel_polarity_matrix, create_quantitative_mekko_charts, create_qualitative_mekko_charts


def data_preparation(datafile):
    df = pd.read_csv(datafile)

    # step: remove rows that have unknowns (cells with '?')
    unknown_mask = (df.T != '?').all()
    df = df[unknown_mask]

    return df


if __name__ == '__main__':
    # step: Start
    df = data_preparation('adult.csv')

    df_qual = df.select_dtypes(exclude=np.number)
    df_quant = df.select_dtypes(include=np.number)
    df_dummies = create_dummy_matrix(df_qual)

    # substep: change 'Salary' to dummy variable (less than 50K).
    y_variable = pd.get_dummies(df['Salary'], prefix='Salary', prefix_sep=': ')['Salary: <=50K']

    # step: study all variables: see if Andrew curves yields anything revealing
    # ordered based on correlation (since Andrew plots depend on order of variables)
    create_andrews_plot(df_quant[['Education-num', 'Age', 'Hours-per-week', 'Capital-gain', 'Capital-loss', 'Fnlwgt']],
                        y_variable)
    plt.show()

    # step: study first the quantitative variables
    # substep: study R^2 matrix
    df_quant_r2 = (pd.concat([df_quant, y_variable], axis=1).corr() ** 2).sort_values(by=['Salary: <=50K'],
                                                                                      ascending='True')

    pd.set_option('display.max_rows', len(df_quant_r2))
    print("Quant R^2 rel. to Salary:", df_quant_r2['Salary: <=50K'])
    # warn: Comments: PCA is worthless due to lack of correlations if partitioning via PCA.

    # substep: see if polarity exists between quant. variables
    polarity_matrix = create_rel_polarity_matrix(df_quant, y_variable, classes=[1, 0])
    print(polarity_matrix)

    # substep: we can confirm this via scatterplots
    create_classification_plots(df_quant.copy(), y_variable, jitter=True, num_samples=4000, jitter_size=0.05,
                                hide_legend=True)
    plt.show()

    # step: study the qualitative variables
    # substep: determine correlations of dummies (will include Salary)

    df_dummies_r2 = (df_dummies.corr() ** 2).sort_values(by=['Salary: <=50K'], ascending='True')
    pd.set_option('display.max_rows', len(df_dummies_r2))
    print("Dummy R^2 rel. to Salary:", df_dummies_r2['Salary: <=50K'][-10:])

    # substep: study polarity of qual. variables
    polarity_matrix = create_rel_polarity_matrix(df_dummies, y_variable, classes=[1, 0])
    print(polarity_matrix)

    # plot correlation graph, assuming > 0.3 signifies an existing correlation
    polarity_graph_vals = polarity_matrix[polarity_matrix > 0.3].fillna(0)
    polarity_graph_vals = polarity_graph_vals.round(3)

    # remove diagonal entries, and remove variables with no strong correlation
    cols = polarity_graph_vals.columns
    for c in cols: polarity_graph_vals[c][c] = 0
    var_mask = (polarity_graph_vals.T != 0).any()
    polarity_graph_vals = polarity_graph_vals[var_mask]
    polarity_graph_vals = polarity_graph_vals.T[var_mask].T

    ax = sns.heatmap(polarity_graph_vals, xticklabels=1, yticklabels=1)
    plt.show()

    corr_graph = nx.from_pandas_adjacency(polarity_graph_vals)

    pos = nx.spring_layout(corr_graph)
    nx.draw_networkx(corr_graph, pos)
    weights = nx.get_edge_attributes(corr_graph, 'weight')
    nx.draw_networkx_edge_labels(corr_graph, pos, edge_labels=weights)
    plt.show()

    # substep:
    create_qualitative_mekko_charts(x_qual=df_qual.drop(columns=['Salary']), y_var=y_variable, share_y='row')
    plt.show()

    create_qualitative_mekko_charts(x_qual=df_qual[['Native-country', 'Marital-status']], y_var=y_variable)
    plt.show()

    # substep: create stripplots
    create_quantitative_mekko_charts(df_quant.copy(), df_qual.drop(columns=['Salary']).copy(), y_variable,
                                     num_samples=3000)
    plt.show()

    ####################################################################################################################

    # # substep: see if Mekko charts yields anything useful
    # create_mekko_charts(df_qual[['Marital-status', 'Relationship', 'Sex', 'Occupation', 'Salary']], 'Salary')
    # plt.show()
    # Comments: better to plot seperately the above
