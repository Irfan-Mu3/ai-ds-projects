import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
import pandas as pd
from matplotlib.patches import Patch
from statsmodels.graphics import mosaicplot
from costats import bipolarity


# warn: the plots earlier only worked for 2+ variables
# todo: use SVM for biscatterplots (for linear discrimination aid)
# todo: see if there is an analogy for partial_classification vs (Decision-Tree approach)?
# todo: see if there is an analogy for partial_residuals vs (probability of missclassification)?

def create_dummy_matrix(data_matrix: pd.DataFrame,drop_last_var=True):
    # remove column becoming a dummy, and append new dummies
    nominal_df: pd.DataFrame = data_matrix.select_dtypes(exclude=np.number)
    cols = nominal_df.columns

    for c in cols:
        dummies = pd.get_dummies(nominal_df[c], prefix=c, prefix_sep=': ')
        if drop_last_var: dummies = dummies.iloc[:, :-1]  # remember: drop last column (to prevent multicollinearity) [Not always advised]
        nominal_df = pd.concat([nominal_df, dummies], axis=1)
        nominal_df = nominal_df.drop(columns=[c])

    return pd.concat([nominal_df, data_matrix.drop(columns=cols)], axis=1)


def andrew_func(Xs: pd.DataFrame, points):
    N = Xs.shape[1]

    # initial part of the andrew curve
    f_vals = np.outer(Xs.iloc[:, 0] / np.sqrt(2), np.ones(len(points)))

    # remaining portion
    for i in range(1, N - 1):
        if i % 2 == 1:
            f_vals += np.outer(Xs.iloc[:, i], np.sin(np.ceil(i / 2.0) * points))
        else:
            f_vals += np.outer(Xs.iloc[:, i], np.cos((i / 2.0) * points))

    return f_vals


def create_andrews_plot(Xs: pd.DataFrame, y_var: pd.Series, num_points=20, num_samples=1000):
    # Xs: x variables.
    # y_var: y variable (categorical but numerical, with each number representing a class).
    # num_points: the number of points between [-pi,pi] to plot the Andrew curve for.
    # num_samples: number of samples used for each class. This is needed for big-data.

    # remember: provides a seperate subplot for each class

    unq_classes = np.unique(y_var)
    class_masks = np.empty((len(unq_classes), y_var.shape[0]), dtype=bool)

    for i in range(len(unq_classes)):
        class_masks[i] = (y_var == unq_classes[i])

    fig, axs = plt.subplots(len(unq_classes))

    linx = np.linspace(-np.pi, np.pi, num_points)

    for i in range(len(unq_classes)):
        samples = np.random.choice(np.arange(len(class_masks[i])), num_samples)
        andrew_curves = andrew_func(Xs.iloc[samples, :], linx)

        plot_color = np.random.rand(3, )
        for k in range(len(andrew_curves)):
            axs[i].plot(linx, andrew_curves[k], color=plot_color)

    for i in range(len(unq_classes)): plt.setp(axs[i], ylabel='class ' + str(unq_classes[i]))


def create_rel_polarity_matrix(Xs: pd.DataFrame, y_var: pd.Series, classes):
    num_vars = Xs.shape[1]
    rel_polarity_matrix = np.zeros((num_vars, num_vars))

    Xs_n = Xs.to_numpy()
    y_var_n = y_var.to_numpy().flatten()

    # remember: symmetric
    for i in range(num_vars):
        for j in range(i, num_vars):
            rel_polarity_matrix[i, j] = bipolarity(Xs_n[:, i], Xs_n[:, j], y_var_n, classes, rel=True)
            rel_polarity_matrix[j, i] = rel_polarity_matrix[i, j]

    return pd.DataFrame(rel_polarity_matrix, Xs.columns, Xs.columns)


def create_mekko_charts(df: pd.DataFrame, class_variable_y_name: str):
    # remember: expects that the y variable (class variable) is part of the dataframe df.

    num_vars = df.shape[1] - 1
    cols = df.columns
    fig, axs = plt.subplots(num_vars)

    for i in range(num_vars):
        bivariables = [cols[i], class_variable_y_name]
        mosaicplot.mosaic(df[bivariables], index=bivariables, ax=axs[i], labelizer=lambda _: "", gap=0)

    for i in range(num_vars):
        plt.setp(axs[-1], xlabel=cols[i])


def create_probability_hypermatrix(x1: pd.Series, x2: pd.Series, y_var: pd.Series):
    # returns a (num_category_x1 x num_category_x2 x num_classes) hypermatrix
    # the matrix is indexed by the categories
    # todo: remove any categories too small?

    multidf = pd.get_dummies(x2).groupby([x1, y_var]).sum()
    return multidf / multidf.sum(level=[0])


def prop_helper(class_colors: np.ndarray, xph: pd.DataFrame, colnames: np.ndarray, rownames: np.ndarray,
                classes: np.ndarray):
    prop = {}

    class_idxs = np.arange(0, len(classes))
    class_dict = dict(zip(classes, class_idxs))

    # remember: xph is asymmetric, hence:
    for cn in colnames:
        for rn in rownames:
            max_class_val = (xph[cn][rn]).idxmax()  # chooses first largest class
            if np.isnan(max_class_val): continue  # for when using two same variables
            prop[(str(cn), str(rn))] = {'color': class_colors[class_dict[max_class_val]], 'alpha': xph[cn][rn].max()}

    return prop


def create_qualitative_mekko_charts(x_qual: pd.DataFrame, y_var: pd.Series,
                                    hide_legend=False, share_y=False):
    N, num_vars_qual = x_qual.shape
    cols_qual = x_qual.columns

    unq_classes = np.unique(y_var)

    fig, axs = plt.subplots(num_vars_qual, num_vars_qual, sharex='col', sharey=share_y)
    if axs.ndim == 1: axs = axs[..., np.newaxis]  # test this

    # plot_colors = np.random.rand(len(unq_classes), 3)
    plot_colors = plt.cm.viridis(np.linspace(0, 1, len(unq_classes)))

    # remember: symmetric plotting
    for i in range(num_vars_qual):
        for j in range(i, num_vars_qual):

            xph = create_probability_hypermatrix(x_qual.iloc[:, i], x_qual.iloc[:, j], y_var)
            cols = xph.columns
            rows, _ = xph.index.levels
            prop = prop_helper(plot_colors, xph, cols, rows, unq_classes)

            var_names = [cols_qual[j], cols_qual[i]]
            xs = x_qual.iloc[:, [i, j]]

            if i != j:
                mosaicplot.mosaic(xs, var_names, ax=axs[i, j]
                                  , properties=prop,
                                  labelizer=lambda _: "")

                prop = {(k[1], k[0]): v for k, v in prop.items()}
                mosaicplot.mosaic(x_qual.iloc[:, [j, i]], [cols_qual[i], cols_qual[j]], ax=axs[j, i]
                                  , properties=prop,
                                  labelizer=lambda _: "", )
            else:
                var_names[0] += str('_')
                xs.columns = var_names
                mosaicplot.mosaic(xs, var_names, ax=axs[i, j]
                                  , properties=prop,
                                  labelizer=lambda _: "")

            # if i == j == 0:
            #     labels = axs[i,j].get_yticklabels()
            #     axs[i, j].set_yticks(np.linspace(0.1, 1, len(cols),endpoint=False))
            #     axs[i, j].set_yticklabels(labels,rotation=45)
            # elif i == j:
            #     axs[i, j].set_yticks(np.linspace(0.1, 1, len(cols),endpoint=False))
            #     axs[i, j].set_yticklabels([])
            #
            # if j != 0:
            #     axs[i, j].set_yticklabels([])
            #     if i != 0:
            #         axs[j,i].set_yticklabels([]),

        # labels = axs[i,0].get_yticklabels()
        # axs[i, 0].set_yticks(np.linspace(0.1, 1, len(labels), endpoint=False))
        # axs[i, 0].set_yticklabels(labels, rotation=45)

        plt.setp(axs[i, 0], ylabel=cols_qual[i])
        plt.setp(axs[-1, i], xlabel=cols_qual[i])
        plt.setp(axs[-1, i], xlabel=cols_qual[i])

    for i in range(num_vars_qual):
        # axs[i,0].set_yticklabels(axs[i,0].get_yticklabels(),rotation=30)
        axs[-1, i].set_xticklabels(axs[-1, i].get_xticklabels(), rotation=45)

    legend_elements = [Patch(facecolor=plot_colors[i],
                             label=unq_classes[i]) for i in range(len(unq_classes))]

    fig.legend(handles=legend_elements, loc='upper center', ncol=len(unq_classes))


def create_quantitative_mekko_charts(x_quant: pd.DataFrame, x_qual: pd.DataFrame, y_var: pd.Series,
                                     hide_legend=False, jitter=True, num_samples=1000, jitter_size=0.01):
    # remember: df_qual is NOT a matrix of (artificial) dummy variables.
    num_vars_qual = x_qual.shape[1]
    num_vars_quant = x_quant.shape[1]

    cols_qual = x_qual.columns
    cols_quant = x_quant.columns

    unq_classes = np.unique(y_var)
    class_masks = np.empty((len(unq_classes), y_var.shape[0]), dtype=bool)

    for i in range(len(unq_classes)):
        class_masks[i] = y_var == unq_classes[i]

    # step: normalize to [0,1], (and convert to numpy matrix if not converted by normalization)
    x_quant = skp.minmax_scale(x_quant, feature_range=(0, 1), axis=0, copy=True)
    x_qual = x_qual.to_numpy()

    # step: jitter
    if jitter: x_quant += np.random.normal(0, jitter_size, (x_quant.shape))

    fig, axs = plt.subplots(num_vars_qual, num_vars_quant)
    if num_vars_qual == 1 or num_vars_quant == 1:   axs = axs[..., np.newaxis]  # test this

    sample_idxs = np.zeros((len(unq_classes), num_samples), dtype=int)

    # warn: Samples chosen for each row must be fixed to maintain y-tick label ordering, hence:
    for l in range(len(unq_classes)):
        sample_idxs[l] = np.random.choice(np.flatnonzero(class_masks[l]), num_samples)

    # remember: strictly asymmetric
    for i in range(num_vars_qual):
        y_vals = x_qual[:, i]
        for j in range(num_vars_quant):
            x_points = x_quant[:, j]
            for l in range(len(unq_classes)):
                xs, ys = x_points[sample_idxs[l]], y_vals[sample_idxs[l]]
                axs[i, j].scatter(xs, ys, label='class ' + str(unq_classes[l]), s=1.7, alpha=0.8)

            if j != 0: axs[i, j].set_yticks([])  # only keep category labels for LHS of graph
            if not hide_legend: axs[i, j].legend()

    for i in range(num_vars_quant): plt.setp(axs[-1, i], xlabel=cols_quant[i])
    for j in range(num_vars_qual): plt.setp(axs[j, 0], ylabel=cols_qual[j])


def create_classification_plots(x_quant: pd.DataFrame, y_var: pd.Series, cols=None, hide_legend=False,
                                jitter=True, num_samples=1000, jitter_size=0.01):
    # Xs: the set of explanatory variables
    # y_var: the classfication variable
    # cols: Columns of Xs
    # use_cdf: Use cdf to represent distribution of ea. var, instead of pdf
    # jitter: Jitter the values in each biplot
    # num_samples: the maximum number of samples for each class to be used. Needed for big-data
    # jitter_size: If jitter == True, use this value to determine jitter (normal noise) amount.

    N, num_vars = x_quant.shape

    unq_classes = np.unique(y_var)
    class_masks = np.empty((len(unq_classes), N), dtype=bool)

    for i in range(len(unq_classes)):  class_masks[i] = y_var == unq_classes[i]

    labels = cols if cols is not None else x_quant.columns

    fig, axs = plt.subplots(num_vars, num_vars)
    if num_vars == 1:   axs = axs[..., np.newaxis]  # test this

    # step: normalize to [0,1], (and convert to numpy matrix if not converted by normalization)
    x_quant = skp.minmax_scale(x_quant, feature_range=(0, 1), axis=0, copy=True)

    # step: jitter
    if jitter: x_quant += np.random.normal(0, jitter_size, (x_quant.shape))

    # remember: symmetric plotting
    for i in range(num_vars):
        y_points = x_quant[:, i]
        for j in range(i, num_vars):
            x_points = x_quant[:, j]
            for l in range(len(unq_classes)):
                idxs = np.flatnonzero(class_masks[l])
                samples = np.random.choice(idxs, num_samples)
                xs, ys = x_points[samples], y_points[samples]
                class_label = 'class ' + str(unq_classes[l])
                axs[i, j].scatter(xs, ys, label=class_label, s=1.7, alpha=0.8)
                axs[i, j].scatter(np.mean(x_points[idxs]), np.mean(y_points[idxs]), s=30,
                                  marker='x', label=class_label, zorder=3 + l)
                if i != j:
                    axs[j, i].scatter(ys, xs, label=class_label, s=1.7, alpha=0.8)
                    axs[j, i].scatter(np.mean(y_points[idxs]), np.mean(x_points[idxs]), s=30,
                                      marker='x', label=class_label, zorder=3 + l)

            if not hide_legend:
                axs[i, j].legend()

    for i in range(num_vars):
        plt.setp(axs[-1, i], xlabel=labels[i])
        plt.setp(axs[i, 0], ylabel=labels[i])


if __name__ == '__main__':
    # np.random.seed(1024536)

    # step: example for Polarity

    y_boy = pd.Series(np.random.choice([0, 1, 100], 100, replace=True), name='y_boy')
    woo = pd.Series(np.random.choice([1, 2, 3, 4, 5], 100, replace=True), name='woo')
    bla = pd.Series(np.random.choice(['a', 'b', 'c'], 100, replace=True), name='bla')
    tressss = pd.Series(np.random.choice(['abc', 'def', 'geh'], 100, replace=True), name='tressss')

    xph = create_probability_hypermatrix(woo, bla, y_boy)

    print(xph)

    cols = xph.columns
    rows = xph.index.levels[0]
    print("cols:", cols)
    print("rows:", rows)
    print("xph shape:", xph.shape)

    plot_colors = np.random.rand(len(y_boy.unique()), 3)
    prop = prop_helper(plot_colors, xph, cols, rows, np.asarray([0, 1, 100]))

    print(prop)

    create_qualitative_mekko_charts(pd.concat((woo, bla, tressss), axis=1), y_boy)
    plt.show()
