import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import tree

# warn: this import below is needed for successfully importing RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from classification_plots import create_dummy_matrix, create_classification_plots, \
    create_quantitative_mekko_charts, create_qualitative_mekko_charts

from salary_eda_jup import data_preparation


def k_fold_cross_val(Xs, y_var, k=10):
    clf = tree.DecisionTreeClassifier()
    clf_forest = RandomForestClassifier(n_estimators=10)
    clf_boost = HistGradientBoostingClassifier()

    num_folds = k
    N = Xs.shape[0]
    test_size = int(N / num_folds)

    test_idxs = np.random.permutation(N)[:num_folds * test_size].reshape(num_folds, test_size)

    total_score = np.asarray([0., 0., 0.])
    total_F1_score = np.asarray([0., 0., 0.])

    for i in range(num_folds):
        print("Iteration " + str(i) + ":")
        test_i = Xs.index.isin(test_idxs[i])
        df_train, df_test = Xs[~test_i], Xs[test_i]
        y_train, y_test = y_var[~test_i], y_var[test_i]

        clf = clf.fit(df_train.to_numpy(), y_train.to_numpy().ravel())
        score_b = clf.score(df_test.to_numpy(), y_test.to_numpy().ravel())

        clf_forest = clf_forest.fit(df_train.to_numpy(), y_train.to_numpy().ravel())
        score_f = clf_forest.score(df_test.to_numpy(), y_test.to_numpy().ravel())

        clf_boost = clf_boost.fit(df_train.to_numpy(), y_train.to_numpy().ravel())
        score_h = clf_boost.score(df_test.to_numpy(), y_test.to_numpy().ravel())

        y_hat = clf.predict(df_test.to_numpy())
        f1_b = f1_score(y_test.to_numpy().ravel(), y_hat, average='binary')
        print("F1 score (tree):", f1_b)

        y_hat = clf_forest.predict(df_test.to_numpy())
        f1_f = f1_score(y_test.to_numpy().ravel(), y_hat, average='binary')
        print("F1 score (forest):", f1_f)

        y_hat = clf_boost.predict(df_test.to_numpy())
        f1_boost = f1_score(y_test.to_numpy().ravel(), y_hat, average='binary')
        print("F1 score (boost):", f1_boost)

        print("Prediction scores for (tree,forest,boost):", score_b, score_f, score_h)
        total_score += np.asarray([score_b, score_f, score_h])
        total_F1_score += np.asarray([f1_b, f1_f, f1_boost])

    print("Avg. accuracy scores for (tree,forest,boost):", total_score / num_folds)
    print("Avg. F1 scores for (tree,forest,boost):", total_F1_score / num_folds)

    return clf, clf_forest, clf_boost


def plot_biplot_w_class(bi_df: pd.DataFrame, idx_class_true, class_labels, colors=None):
    if colors is None:
        colors = [None, None]

    LN = np.sum(idx_class_true)
    MN = len(y_variable) - LN

    cols = bi_df.columns
    jitter_size = 0.01

    bi_df = bi_df.to_numpy()

    fig, ax = plt.subplots(1)

    ax.scatter(bi_df[idx_class_true, 0] + np.random.normal(0, jitter_size, LN),
               bi_df[idx_class_true, 1], label=class_labels[0],
               marker='x', alpha=0.7, color=colors[0])
    ax.scatter(bi_df[~idx_class_true, 0] + np.random.normal(0, jitter_size, MN),
               bi_df[~idx_class_true, 1],
               label=class_labels[1], alpha=0.9, color=colors[1])
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    plt.legend()


if __name__ == '__main__':
    # step: Start
    df = data_preparation('adult.csv')

    df_qual = df.select_dtypes(exclude=np.number)
    df_quant = df.select_dtypes(include=np.number)
    df_dummies = create_dummy_matrix(df_qual)

    # substep: change 'Salary' to dummy variable (less than 50K).
    y_variable = pd.get_dummies(df['Salary'], prefix='Salary', prefix_sep=': ')['Salary: <=50K']

    ###############################################################################################################

    # step: Decision Trees, forests, gradient boosting: Testing and training via cross-validate

    # substep: quantative variables only
    k_fold_cross_val(df_quant, y_variable, 10)
    # Comments: The accuracy ranges from 75 to 85 percent, depending on the technique used.

    # substep:  qualitative variables only
    df_dummy_qual = create_dummy_matrix(df_qual.drop(columns=['Salary']), )
    k_fold_cross_val(df_dummy_qual, y_variable, 10)

    # substep: all variables
    k_fold_cross_val(df_dummies.drop(columns=['Salary: <=50K']), y_variable, 10)

    # substep: EDA variables
    df2 = df[['Capital-gain', 'Capital-loss', 'Education', 'Relationship', 'Occupation']]
    df2_dummies = create_dummy_matrix(df2, )
    clf, clf_forest, clf_boost = k_fold_cross_val(df2_dummies, y_variable, 10)

    fig, ax = plt.subplots(1)

    _ = tree.plot_tree(clf, feature_names=df2_dummies.columns, class_names=['> 50k', '<=50k'],
                       filled=True, proportion=False, ax=ax)
    plt.show()

    ##############################################################################################################

    # step: start of manual approach

    o_y_variable = y_variable.copy()  # Needed, as we will update the y_variable.

    total_individuals = len(y_variable)
    num_less = np.sum(y_variable)
    num_more = len(y_variable) - num_less
    print("N:", total_individuals)
    print("num of <=50k:", num_less)
    print("num of >50k:", num_more)
    print("ratio:", num_more / num_less)
    y_pred = np.ones_like(y_variable)
    print("F1 score of naive classifer:", f1_score(y_pred, y_variable, ))
    print("Accuracy of naive classifer:", num_less / total_individuals)

    # step: study main variable: Capital-gain (vs Age)
    less_than_index = y_variable == 1
    # plot_biplot_w_class(df_quant[['Capital-gain', 'Age']], less_than_index,
    #                     ['less than', 'more than'])
    #
    # plt.show()

    # substep: Remove points of 'more than' that are easy to discriminate,
    #  where Capital-gain is more than 0.6 (and the points are strictly > 50K)
    clust1_idx = df_quant['Capital-gain'] > 7000
    clust2_idx = ~clust1_idx

    plot_biplot_w_class(df_quant[['Capital-gain', 'Age']], clust2_idx,
                        ['less than', 'more than'], ['royalblue', 'red'])
    plt.show()

    outlyings = clust1_idx.sum()
    print("Number of points classified as >=50k: ", outlyings)
    misclass = y_variable[clust1_idx].sum()
    print("Miss-classifications:", misclass)
    print("True classifcation percent:", (outlyings - misclass) / outlyings)

    y_pred[clust1_idx] = 0
    print("Potential F1 score of (naive classifer + Capital-gain discriminant):", f1_score(y_pred, y_variable, ))
    print("Potential accuracy of (naive classifer + Capital-gain discriminant):",
          (num_less + outlyings - misclass) / total_individuals)

    ##################################################################################################################
    # step: deal with remaining points
    tree_clust2_idx = clust2_idx
    tree_clust1_idx = clust1_idx
    o_df = df.copy()
    prev_total_outlyings = outlyings
    prev_total_misclass = misclass

    # substep:  remove all points previously classified as > 50k:
    df = df[clust2_idx]
    df_qual = df.select_dtypes(exclude=np.number)
    df_quant = df.select_dtypes(include=np.number)
    df_dummies = create_dummy_matrix(df_qual, )
    y_variable = pd.get_dummies(df['Salary'], prefix='Salary', prefix_sep=': ')['Salary: <=50K']
    df_qual_wo_sal = df_qual.drop(columns=['Salary'])
    less_than_index = y_variable == 1

    # substep: re-study bi-relationships
    create_quantitative_mekko_charts(df_quant.copy(), df_qual_wo_sal.copy(), y_variable, num_samples=1000,
                                     jitter=False, )
    create_classification_plots(df_quant.copy(), y_variable, jitter=True, num_samples=5000,
                                jitter_size=0.01)
    plt.show()

    ######################################################################################################

    # step: Capital-loss and Education-num
    plot_biplot_w_class(df_quant[['Education-num', 'Capital-loss']], less_than_index,
                        ['less than', 'more than'])
    plt.show()

    # substep: cluster as (> 1800) for Capital-Loss, (>= 12) (or 13) for Education-num
    clust1_idx = (df_quant['Education-num'] >= 12) & (df_quant['Capital-loss'] >= 1800)
    clust2_idx = ~clust1_idx

    plot_biplot_w_class(df_quant[['Education-num', 'Capital-loss']], clust2_idx,
                        ['less than', 'more than'], ['royalblue', 'red'])
    plt.show()

    outlyings = clust1_idx.sum()
    print("Number of points classified as >=50k: ", outlyings)
    misclass = y_variable[clust1_idx].sum()
    print("Miss-classifications:", misclass)
    print("True classifcation percent:", (outlyings - misclass) / outlyings)

    tree_clust1_idx |= (o_df['Education-num'] >= 12) & (o_df['Capital-loss'] >= 1800)
    y_pred[tree_clust1_idx] = 0
    print("Potential F1 score of (naive classifer + Capital-gain + Capital-loss):", f1_score(y_pred, o_y_variable))
    print("Potential accuracy of (naive classifer + Capital-gain + Capital-loss):",
          (num_less + outlyings - misclass + prev_total_outlyings - prev_total_misclass) / total_individuals)

    ###############################################################

    # step: Capital-loss and others

    plot_biplot_w_class(df_quant[['Education-num', 'Hours-per-week']], less_than_index,
                        ['less than', 'more than'])
    plot_biplot_w_class(pd.concat((df_quant['Capital-loss'], df_qual['Occupation']), axis=1), less_than_index,
                        ['less than', 'more than'])
    plot_biplot_w_class(pd.concat((df_quant['Capital-loss'], df_qual['Native-country']), axis=1), less_than_index,
                        ['less than', 'more than'])
    plt.show()

    # Comments: None seem useful

    ###############################################################################################
    # step: again, deal with remaining points
    prev_total_outlyings += outlyings
    prev_total_misclass += misclass

    # substep: remove all points previously classified as > 50k:
    df = df[clust2_idx]
    df_qual = df.select_dtypes(exclude=np.number)
    df_quant = df.select_dtypes(include=np.number)
    df_dummies = create_dummy_matrix(df_qual, )
    y_variable = pd.get_dummies(df['Salary'], prefix='Salary', prefix_sep=': ')['Salary: <=50K']
    df_qual_wo_sal = df_qual.drop(columns=['Salary'])
    less_than_index = y_variable == 1

    # substep: study last variables
    create_qualitative_mekko_charts(x_qual=df_qual[['Relationship', 'Education', 'Occupation']], y_var=y_variable,
                                    share_y=False)
    plt.show()

    ###################################################################################################################

    # step: finding discriminants via: Education, Relationship and Occupation

    # Relationship (Wife,Husband) and Occupation (Doctorate, Prof-school, Masters)
    clust1_idx = (df_qual['Relationship'].isin(['Wife', 'Husband'])) & (
        df_qual['Education'].isin(['Doctorate', 'Prof-school', ]))

    clust2_idx = ~clust1_idx

    outlyings = clust1_idx.sum()
    print("Number of points classified as >=50k: ", outlyings)
    misclass = y_variable[clust1_idx].sum()
    print("Miss-classifications:", misclass)
    print("True classifcation percent:", (outlyings - misclass) / outlyings)

    tree_clust1_idx_pot = (o_df['Relationship'].isin(['Wife', 'Husband'])) & (
        o_df['Education'].isin(['Doctorate', 'Prof-school', ]))

    y_pred_pot = y_pred.copy()
    y_pred_pot[tree_clust1_idx | tree_clust1_idx_pot] = 0
    print("Potential F1 score of (naive classifer + Capital-gain + Capital-loss + ...):",
          f1_score(y_pred_pot, o_y_variable, ))
    print("Potential accuracy of (naive classifer + Capital-gain + Capital-loss + ...):",
          (num_less + outlyings - misclass + prev_total_outlyings - prev_total_misclass) / total_individuals)

    ###################################

    # Relationship (Wife) and Occupation (Doctorate, Prof-school, Masters)
    clust1_idx = (df_qual['Relationship'].isin(['Wife', ])) & (
        df_qual['Education'].isin(['Doctorate', 'Prof-school', ]))

    clust2_idx = ~clust1_idx

    outlyings = clust1_idx.sum()
    print("Number of points classified as >=50k: ", outlyings)
    misclass = y_variable[clust1_idx].sum()
    print("Miss-classifications:", misclass)
    print("True classifcation percent:", (outlyings - misclass) / outlyings)
    print("Overall accuracy:", str((22654 + 1330 - 18 + 485 - 64 + 30 - 5) / 30162 * 100) + '%')

    tree_clust1_idx_pot = (o_df['Relationship'].isin(['Wife', ])) & (
        o_df['Education'].isin(['Doctorate', 'Prof-school', ]))

    y_pred_pot = y_pred.copy()
    y_pred_pot[tree_clust1_idx | tree_clust1_idx_pot] = 0
    print("Potential F1 score of (naive classifer + Capital-gain + Capital-loss + ...):",
          f1_score(y_pred_pot, o_y_variable, ))
    print("Potential accuracy of (naive classifer + Capital-gain + Capital-loss + ...):",
          (num_less + outlyings - misclass + prev_total_outlyings - prev_total_misclass) / total_individuals)

    ###################################

    # Relationship (Husband, Wife) and Occupation (Exec-managerial, Prof-speciality)
    clust1_idx = (df_qual['Relationship'].isin(['Wife', 'Husband'])) & (
        df_qual['Occupation'].isin(['Prof-specialty', 'Exec-managerial', ]))

    clust2_idx = ~clust1_idx

    outlyings = clust1_idx.sum()
    print("Number of points classified as >=50k: ", outlyings)
    misclass = y_variable[clust1_idx].sum()
    print("Miss-classifications:", misclass)
    print("True classifcation percent:", (outlyings - misclass) / outlyings)
    print("Overall accuracy:", str((22654 + 1330 - 18 + 485 - 64 + 3834 - 1335) / 30162 * 100) + '%')

    tree_clust1_idx_pot = (o_df['Relationship'].isin(['Wife', 'Husband'])) & (
        o_df['Occupation'].isin(['Prof-specialty', 'Exec-managerial', ]))

    y_pred_pot = y_pred.copy()
    y_pred_pot[tree_clust1_idx | tree_clust1_idx_pot] = 0
    print("Potential F1 score of (naive classifer + Capital-gain + Capital-loss + ...):",
          f1_score(y_pred_pot, o_y_variable, ))
    print("Potential accuracy of (naive classifer + Capital-gain + Capital-loss + ...):",
          (num_less + outlyings - misclass + prev_total_outlyings - prev_total_misclass) / total_individuals)

    ####################################

    # Relationship (Wife) and Occupation (Prof-speciality)
    clust1_idx = (df_qual['Relationship'].isin(['Wife', ])) & (
        df_qual['Occupation'].isin(['Prof-specialty', ]))

    clust2_idx = ~clust1_idx

    outlyings = clust1_idx.sum()
    print("Number of points classified as >=50k: ", outlyings)
    misclass = y_variable[clust1_idx].sum()
    print("Miss-classifications:", misclass)
    print("True classifcation percent:", (outlyings - misclass) / outlyings)
    print("Overall accuracy:", str((22654 + 1330 - 18 + 485 - 64 + 261 - 78) / 30162 * 100) + '%')

    tree_clust1_idx_pot = (o_df['Relationship'].isin(['Wife', ])) & (
        o_df['Occupation'].isin(['Prof-specialty', ]))

    y_pred_pot = y_pred.copy()
    y_pred_pot[tree_clust1_idx | tree_clust1_idx_pot] = 0
    print("Potential F1 score of (naive classifer + Capital-gain + Capital-loss + ...):",
          f1_score(y_pred_pot, o_y_variable))
    print("Potential accuracy of (naive classifer + Capital-gain + Capital-loss + ...):",
          (num_less + outlyings - misclass + prev_total_outlyings - prev_total_misclass) / total_individuals)

    #########################################################

    # 3D: Relationship (Wife,Husband) & Occupation (Prof-Speciality, Exec-Managerial)
    # & Education (Doctorate, Prof-School,Masters)

    clust1_idx = (df_qual['Relationship'].isin(['Wife', 'Husband'])) & (
            df_qual['Occupation'].isin(['Prof-specialty', 'Exec-Managerial'])
            & df_qual['Education'].isin(['Doctorate', 'Prof-school', 'Masters']))

    clust2_idx = ~clust1_idx

    outlyings = clust1_idx.sum()
    print("Number of points classified as >=50k: ", outlyings)
    misclass = y_variable[clust1_idx].sum()
    print("Miss-classifications:", misclass)
    print("True classifcation percent:", (outlyings - misclass) / outlyings)

    tree_clust1_idx_pot = (df_qual['Relationship'].isin(['Wife', 'Husband'])) & (
            df_qual['Occupation'].isin(['Prof-specialty', 'Exec-Managerial'])
            & df_qual['Education'].isin(['Doctorate', 'Prof-school', 'Masters', ]))

    y_pred_pot = y_pred.copy()
    y_pred_pot[tree_clust1_idx | tree_clust1_idx_pot] = 0
    print("Potential F1 score of (naive classifer + Capital-gain + Capital-loss + ...):",
          f1_score(y_pred_pot, o_y_variable))
    print("Potential accuracy of (naive classifer + Capital-gain + Capital-loss + ...):",
          (num_less + outlyings - misclass + prev_total_outlyings - prev_total_misclass) / total_individuals)

    # clust1_idx = (df_qual['Relationship'].isin(['Wife', 'Husband'])) & (
    #     df_qual['Occupation'].isin(['Prof-specialty', 'Exec-managerial', ]))
    #
    # clust2_idx = ~clust1_idx
    #
    # outlyings = clust1_idx.sum()
    # print("Number of points classified as >=50k: ", outlyings)
    # misclass = y_variable[clust1_idx].sum()
    # print("Miss-classifications:", misclass)
    # print("True classifcation percent:", (outlyings - misclass) / outlyings)
    #
    #
    # tree_clust1_idx &= (o_df['Relationship'].isin(['Wife', 'Husband'])) & (
    #     o_df['Occupation'].isin(['Prof-specialty', 'Exec-managerial', ]))
    #
    # y_pred[tree_clust1_idx] = 0
    # print("Potential F1 score of (naive classifer + Capital-gain + Capital-loss + Rel/Occ):", f1_score(y_pred, o_y_variable, ))
    # print("Potential accuracy of (naive classifer + Capital-gain + Capital-loss + Rel/Occ):",
    #       (num_less + outlyings - misclass + prev_total_outlyings - prev_total_misclass) / total_individuals)
    #
    #
    #

    # create_quantitative_mekko_charts(df_quant[['Hours-per-week', 'Fnlwgt', 'Education-num', 'Age']].copy(),
    # df_qual_wo_sal[['Education', 'Workclass', 'Relationship']].copy(), y_variable,
    # num_samples=5000)

    # create_quantitative_mekko_charts(df_quant.copy(), df_qual_wo_sal.copy(), y_variable, num_samples=1000,
    #                                  jitter=False, )

    # create_quantitative_mekko_charts(df_quant[['Education-num', 'Age']].copy(),
    #                                  df_qual_wo_sal[['Education','Workclass']].copy(), y_variable,jitter=False,
    #                                  num_samples=5000)
    # plt.show()

    # TODO: Remove Doctorate, Prof-School
    # TODO: count losses function (for selected points do a count on the y var (since this is < 50))

    # create_quantitative_mekko_charts(df_quant[['Hours-per-week', 'Capital-loss']].copy(),
    #                                  df_qual_wo_sal[['Workclass,Occupation', 'Relationship']].copy(), y_variable,
    #                                  num_samples=5000)
    # plt.show()

    # TODO: Capital-gain, Relationship (Not-in-family)
    # create_quantitative_mekko_charts(df_quant[['Capital-gain', ]].copy(),
    #                                  df_qual_wo_sal[['Relationship']].copy(), y_variable,
    #                                  num_samples=5000)
    # plt.show()

    # TODO: For each, determine the total number of points,
    # TODO: ratio of gain when classifying as >50k (the probability of miss-classification)
    # TODO: Then, order the OR statements in terms of probability of miss-classification.

    # TODO fit decision tree of finite depth, to see if we get the same
