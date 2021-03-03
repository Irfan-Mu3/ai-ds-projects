import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.decomposition import PCA
from regression_plots import create_regression_plots, create_partial_plots, \
    create_partial_residual_plots


def price_data_preparation(datafile):
    df = pd.read_csv(datafile)
    df = df.drop(columns=['Id'])

    # step: remove NAs from numeric variables as zero.
    df = df.fillna(value=0)

    return df


def transform_house_pd(df: pd.DataFrame):
    # step: transforms and normalization

    # step: power transforms
    # warn: constants should be added to each variable below if zero values are encountered, or to make them positive.

    # continuous transforms
    df['SalePrice'] = np.log(df['SalePrice'])
    df['GrLivArea'] = (np.log(df['GrLivArea']))
    df['GarageArea'] = np.power(df['GarageArea'], 0.8)
    df['TotalBsmtSF'] = np.power(df['TotalBsmtSF'], 0.41)
    df['1stFlrSF'] = np.power(df['1stFlrSF'], -0.1)
    df['MasVnrArea'] = np.power(df['MasVnrArea'], 0.25)

    # ordinal transforms
    df['OverallQual'] = np.power(df['OverallQual'], 1.35)
    df['GarageCars'] = np.power(df['GarageCars'], 0.8)
    df['TotRmsAbvGrd'] = np.power(df['TotRmsAbvGrd'], 0.375)

    # step: Replace categorical variables with dummies
    # df = create_dummy_matrix(df)

    # step: normalizations
    # df = pd.DataFrame(data=skp.minmax_scale(df, feature_range=(0, 1), axis=0, copy=True), columns=df.columns)

    return df


if __name__ == '__main__':
    df = price_data_preparation('train.csv')

    # note, if we sort by salePrice now, this will affect the heatmap
    corr_matrix = df.corr()
    R_squared = np.multiply(corr_matrix, corr_matrix)

    # plot heatmap
    ax = sns.heatmap(R_squared, xticklabels=1, yticklabels=1)
    plt.show()

    # plot correlation graph, assuming > 0.3 signifies an existing correlation
    corr_graph_vals = R_squared[R_squared > 0.3].fillna(0)
    corr_graph_vals = corr_graph_vals.round(3)


    # remove diagonal entries, and remove variables with no strong correlation
    cols = corr_graph_vals.columns
    for c in cols: corr_graph_vals[c][c] = 0
    var_mask = (corr_graph_vals.T != 0).any()
    corr_graph_vals = corr_graph_vals[var_mask]
    corr_graph_vals = corr_graph_vals.T[var_mask].T
    corr_graph = nx.from_pandas_adjacency(corr_graph_vals)

    pos = nx.spring_layout(corr_graph)
    nx.draw_networkx(corr_graph,pos)
    weights = nx.get_edge_attributes(corr_graph, 'weight')
    nx.draw_networkx_edge_labels(corr_graph,pos,edge_labels=weights)
    plt.show()

    # print correlated variables to SalePrice
    pd.set_option('display.max_rows', len(corr_matrix))
    print("R^2:", R_squared.sort_values(by=['SalePrice'], ascending='True'))

    # step: Choosen highest R^2 variables with respect to Y (SalePrice)
    # Note, these are chosen without first transforming the variables
    primary_vars = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']
    secondary_vars = ['SalePrice', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', ]
    comb_vars = np.append(primary_vars, secondary_vars[1:])

    # step: regression scatterplots
    create_regression_plots(df[primary_vars], cols=None, use_cdf=True, hide_legend=True)
    create_regression_plots(df[secondary_vars], cols=None, use_cdf=True, hide_legend=True)
    plt.show()

    # step: transform vars to reduce skewness
    old_df = df.copy()
    df = transform_house_pd(df)

    # substep: replot regression scatterplots
    create_regression_plots(df[primary_vars], cols=None, use_cdf=True, hide_legend=True)
    create_regression_plots(df[secondary_vars], cols=None, use_cdf=True, hide_legend=True)
    plt.show()

    # step: component-residual plots
    create_partial_residual_plots(df[primary_vars], 'SalePrice')
    create_partial_residual_plots(df[secondary_vars], 'SalePrice')
    plt.show()

    # step: partial regression plots
    threshold = 0.25
    outlier_idxs = create_partial_plots(df[primary_vars], 'SalePrice', drop_rows=None, threshold=threshold)
    create_partial_plots(df[primary_vars], 'SalePrice', drop_rows=outlier_idxs, threshold=threshold)
    print("outliers:", list(outlier_idxs))
    plt.show()

    ######################################################################################

    # step: PCA study

    pca = PCA()
    pca.fit(df[comb_vars].drop(columns=['SalePrice']))
    print(pca.explained_variance_ratio_)

    # Since the first two components are 'large enough', let us see if there are any clusters
    # in our dataset via PCA
    # Let us also use the outlier idxs earlier found, to see if these outliers correspond to anything significant
    # within the clusters

    # substep: 2D projection
    orig_trans_Xs = pca.fit_transform(df[comb_vars].drop(columns=['SalePrice']))
    trans_Xs = orig_trans_Xs[:, :2]
    plt.scatter(trans_Xs[:, 0], trans_Xs[:, 1], label='first 2 components of PCA')
    plt.scatter(trans_Xs[outlier_idxs, 0], trans_Xs[outlier_idxs, 1], label='outliers')
    plt.legend()
    plt.xlabel('Comp. 1')
    plt.ylabel('Comp. 2')
    plt.show()

    # Interestingly, we find 3 clusters, [Comp1 < -100,:], [ -100 < Comp1 < 75, Comp2 > 12], [Rest].

    # substep: 3D projection, with outliers
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(trans_Xs[:, 0], trans_Xs[:, 1], df['SalePrice'], label='first 2 components of PCA')
    ax.scatter(trans_Xs[outlier_idxs, 0], trans_Xs[outlier_idxs, 1], df['SalePrice'][outlier_idxs], label='outliers')
    for j in outlier_idxs:
        ax.text(trans_Xs[j, 0], trans_Xs[j, 1], df['SalePrice'][j], j)

    ax.set_xlabel('Comp. 1')
    ax.set_ylabel('Comp. 2')
    ax.set_zlabel('SalePrice')

    plt.legend()
    plt.show()

    ##########################################################################################

    # step: Decision Tree classification
    # 1) Let us see if these clusters above, correspond to any meaningful relationship within the original dataset
    # We first first the indices of the rows for each cluster
    # 1) [Comp1 < -100,:], 2) [ -100 < Comp1 < 75, Comp2 > 12], 3) [Rest].
    clf = tree.DecisionTreeClassifier()
    clust1_idx = np.flatnonzero(trans_Xs[:, 0] < -100)
    clust2_idx = np.flatnonzero(((trans_Xs[:, 0] < 75) & (-100 < trans_Xs[:,0])) & (trans_Xs[:, 1] > 12))

    num_samples = trans_Xs.shape[0]
    clust3_idx = np.setdiff1d(np.arange(num_samples),np.append(clust1_idx,clust2_idx))

    plt.scatter(trans_Xs[clust1_idx, 0], trans_Xs[clust1_idx, 1], label='Clust 1')
    plt.scatter(trans_Xs[clust2_idx, 0], trans_Xs[clust2_idx, 1], label='Clust 2')
    plt.scatter(trans_Xs[clust3_idx, 0], trans_Xs[clust3_idx, 1], label='Clust 3')
    plt.legend()
    plt.show()

    # Create a new ordinal variable for the clusters [0,1,2].
    Y_class = np.empty(num_samples,dtype=int)
    Y_class[clust1_idx] = 0
    Y_class[clust2_idx] = 1
    Y_class[clust3_idx] = 2

    Xs_comb = old_df[comb_vars].drop(columns='SalePrice')
    clf = clf.fit(Xs_comb.to_numpy(),Y_class)
    tree.plot_tree(clf,feature_names=Xs_comb.columns,class_names=['Left','Top','Rest'],filled=True,proportion=False)
    plt.show()

    # We find that the clusters can be identified well via two variables: GarageCars or TotalBsmtSF

    # 2) Let us also see if the outliers have a specific pattern relative to other points
    # We already have the indices of the outlier, let us create a new dichotomous variable classifying
    # between outliers and non-outliers [0,1]
    Y_out = np.zeros(num_samples,dtype=int)
    Y_out[outlier_idxs] = 1

    # Xs_comb = old_df[comb_vars] # keep SalePrice in
    clf = clf.fit(Xs_comb.to_numpy(),Y_out)
    tree.plot_tree(clf,feature_names=Xs_comb.columns,class_names=['Reg.','Outlier'],filled=True,proportion=False)
    plt.show()
    # Looking at the graph, there appears to be no simple explanation

    ############################################################################################################