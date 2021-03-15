import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, multivariate_normal, gaussian_kde
from scipy.signal import savgol_filter
import statsmodels.api as sm
import sklearn.preprocessing as skp
import pandas as pd
from sklearn import linear_model

lowess = sm.nonparametric.lowess


def create_dummy_matrix(data_matrix: pd.DataFrame):
    # remove column becoming a dummy, and append new dummies
    nominal_df: pd.DataFrame = data_matrix.select_dtypes(exclude=np.number)
    cols = nominal_df.columns

    for c in cols:
        # print("c", c)
        dummies = pd.get_dummies(nominal_df[c], prefix=c, prefix_sep=': ')
        dummies = dummies.iloc[:, :-1]  # remember: drop last column
        nominal_df = pd.concat([nominal_df, dummies], axis=1)
        nominal_df = nominal_df.drop(columns=[c])

    return pd.concat([nominal_df, data_matrix.drop(columns=cols)], axis=1)


def create_partial_residual_plots(data_matrix: pd.DataFrame, response_variable_y_name):
    # step: normalized to [0,1]
    data_matrix = pd.DataFrame(data=skp.minmax_scale(data_matrix, feature_range=(0, 1), axis=0, copy=True),
                               columns=data_matrix.columns)

    multiple_regression = linear_model.LinearRegression()
    # print(data_matrix.drop(columns=[response_variable_y_name]).shape, data_matrix[response_variable_y_name].shape)
    multiple_regression.fit(data_matrix.drop(columns=[response_variable_y_name]), data_matrix[response_variable_y_name])
    slope_coeffs = multiple_regression.coef_
    # print("slope coeffs:",slope_coeffs)

    err_diff = multiple_regression.predict(data_matrix.drop(columns=[response_variable_y_name])) - data_matrix[
        response_variable_y_name]

    dsize = data_matrix.shape
    num_vars = dsize[1]
    num_Xs = num_vars - 1

    fig, axs = plt.subplots(2, num_Xs)
    Xs_cols = data_matrix.columns.drop(response_variable_y_name)

    for i in range(len(Xs_cols)):
        col = Xs_cols[i]

        x_var = data_matrix[col]
        y_var = data_matrix[response_variable_y_name]

        axs[0, i].scatter(x_var, y_var)

        lingress_res = linregress(x_var, y_var)
        axs[0, i].plot(x_var, (x_var * lingress_res.slope) + lingress_res.intercept,
                       label='slope b = ' + str(np.around(lingress_res.slope, 3)),
                       color='black',
                       linestyle='--', linewidth=2)

        lowess_res = lowess(y_var, x_var)
        axs[0, i].plot(lowess_res[:, 0], lowess_res[:, 1], label='lowess', color='grey')

        # lingress_res = linregress(x_var, partial_error)
        # axs[i].plot(x_var, (x_var * lingress_res.slope) + lingress_res.intercept,
        #             label='slope b = ' + str(np.around(lingress_res.slope, 3)),
        #             color='black',
        #             linestyle='--', linewidth=2)

        partial_error = (slope_coeffs[i] * x_var) + err_diff
        axs[1, i].scatter(x_var, partial_error)

        axs[1, i].plot(x_var, (x_var * slope_coeffs[i]),
                       label='orig. slope b = ' + str(np.around(slope_coeffs[i], 3)),
                       color='red',
                       linestyle=':', linewidth=2)

        lowess_res = lowess(partial_error, x_var)
        axs[1, i].plot(lowess_res[:, 0], lowess_res[:, 1], label='lowess', color='grey')

    for i in range(num_Xs):
        plt.setp(axs[1, i], xlabel=str(Xs_cols[i]))

    axs[1, 0].set_ylabel('Comp + Residuals')
    axs[0, 0].set_ylabel('SalePrice')


def create_partial_plots(data_matrix: pd.DataFrame, response_variable_y_name, drop_rows=None, mixed_contours=True,
                         threshold=None):
    # step: normalized to [0,1]

    data_matrix = pd.DataFrame(data=skp.minmax_scale(data_matrix, feature_range=(0, 1), axis=0, copy=True),
                               columns=data_matrix.columns)

    dsize = data_matrix.shape
    num_vars = dsize[1]

    if drop_rows is not None:
        data_matrix = data_matrix.drop(drop_rows)
        assert data_matrix.shape[0] != 0, "no rows remaining after dropped rows"

    num_Xs = num_vars - 1
    fig, axs = plt.subplots(1, num_Xs)
    Xs_cols = data_matrix.columns.drop(response_variable_y_name)

    outliers = []

    for i in range(len(Xs_cols)):
        col = Xs_cols[i]

        linreg_y = linear_model.LinearRegression()
        linreg_x = linear_model.LinearRegression()

        linreg_y.fit(data_matrix.drop(columns=[response_variable_y_name, col]), data_matrix[response_variable_y_name])
        linreg_x.fit(data_matrix.drop(columns=[response_variable_y_name, col]), data_matrix[col])

        y_hat = linreg_y.predict(data_matrix.drop(columns=[response_variable_y_name, col]))
        x_hat = linreg_x.predict(data_matrix.drop(columns=[response_variable_y_name, col]))

        y_res = data_matrix[response_variable_y_name] - y_hat
        x_res = data_matrix[col] - x_hat

        x_res = x_res.to_numpy()
        y_res = y_res.to_numpy()

        mu = [np.mean(x_res), np.mean(y_res)]
        cov = np.cov(x_res, y_res)

        lingress_res = linregress(x_res, y_res)

        axs[i].plot(x_res, (x_res * lingress_res.slope) + lingress_res.intercept,
                    label='slope b = ' + str(np.around(lingress_res.slope, 3)),
                    color='black',
                    linestyle='--', linewidth=2)

        axs[i].scatter(x_res, y_res)

        if mixed_contours:
            X, Y = np.meshgrid(np.linspace(min(x_res), max(x_res), 100), np.linspace(min(y_res), max(y_res), 100))
            dat = np.vstack((x_res, y_res))
            rv = gaussian_kde(dat)
            positions = np.vstack([X.ravel(), Y.ravel()])
            if threshold:
                outlier_mask = rv(np.vstack([x_res, y_res])) < threshold
            else:
                outlier_mask = rv(np.vstack([x_res, y_res])) < 0.25
            axs[i].contour(X, Y, np.reshape(rv(positions).T, X.shape), levels=10)

        else:
            rv = multivariate_normal(mu, cov, allow_singular=False)
            X, Y = np.meshgrid(np.linspace(min(x_res), max(x_res), 100), np.linspace(min(y_res), max(y_res), 100))
            locs = np.dstack((X, Y))
            axs[i].contour(X, Y, rv.pdf(locs), levels=10)
            if threshold:
                outlier_mask = rv.pdf(list(zip(x_res, y_res))) < threshold
            else:
                outlier_mask = rv.pdf(list(zip(x_res, y_res))) < 0.1

        axs[i].scatter(x_res[outlier_mask], y_res[outlier_mask], color='hotpink', )

        outlier_idxs = np.flatnonzero(outlier_mask)

        for j in outlier_idxs:
            axs[i].annotate(j, (x_res[j], y_res[j]))

        axs[i].legend()
        outliers = np.unique(np.append(outliers, outlier_idxs))

    for i in range(len(axs)):
        plt.setp(axs[i], xlabel=str(Xs_cols[i]) + '|rem. xs' + '')

    axs[0].set_ylabel(str(response_variable_y_name) + '|' + "rem. xs ")
    return outliers.astype(int)


def create_ecdf(points, m):
    # points: vector of x values
    # m: discretization interval

    # remember: assumes data normalized in [0,1] range

    F_dist = np.empty(m)
    N = len(points)
    lins, dx = np.linspace(0, 1, m, retstep=True)

    for k in range(m):
        F_dist[k] = (1 / N) * np.sum(points < lins[k])

    return F_dist


def create_ecdf_2d(points, m):
    # points: matrix of point locations (x,y)
    # m: discretization interval

    # remember: assumes data normalized in [0,1] range
    # remember: assumes points and grid are 2D matrices: 2 x m

    F_dist = np.empty((m, m))
    N = points.shape[1]
    lins, dx = np.linspace(0, 1, m, retstep=True)

    for k in range(m):
        for l in range(m):
            F_dist[k, l] = np.sum((points[0, :] < lins[k]) & (points[1, :] < lins[l]))

    return F_dist / (N)


def create_regression_plots(data_matrix: pd.DataFrame, cols=None, use_cdf=True, hide_legend=False):
    dsize = data_matrix.shape
    N = dsize[0]
    num_vars = dsize[1]

    if cols is not None:
        labels = cols
    else:
        labels = data_matrix.columns

    fig, axs = plt.subplots(num_vars, num_vars)

    # step: normalize to [0,1]
    data_matrix = skp.minmax_scale(data_matrix, feature_range=(0, 1), axis=0, copy=True)

    # step: since, all points are in [0,1], then for cdf approximation, we use the discretization:
    m_points = 200
    lins_x, dx = np.linspace(0, 1, m_points, retstep=True)

    base_N = 100000
    gauss_d = np.random.normal(0, 1, base_N)
    norm_normed = (gauss_d - min(gauss_d)) / (max(gauss_d) - min(gauss_d))
    norm_F_dist = np.empty(m_points)

    for k in range(m_points):
        norm_F_dist[k] = (1 / base_N) * np.sum([norm_normed < lins_x[k]])

    if not use_cdf:
        norm_f_dist = savgol_filter(np.gradient(norm_F_dist, dx), 31, 3)

    for i in range(num_vars):
        for j in range(num_vars):
            y_points = data_matrix[:, i]
            if i == j:
                x_F_dist = np.empty(m_points)
                for k in range(m_points):
                    x_F_dist[k] = (1 / N) * np.sum([y_points < lins_x[k]])

                if use_cdf:
                    axs[i, i].plot(lins_x, x_F_dist, label='ecdf')
                    axs[i, i].plot(lins_x, norm_F_dist, label='normal cdf')
                else:
                    x_f_dist = np.gradient(x_F_dist, dx)
                    axs[i, i].plot(lins_x, x_f_dist, label='epdf')
                    axs[i, i].plot(lins_x, norm_f_dist, label='true pdf')
            else:
                x_points = data_matrix[:, j]
                axs[i, j].scatter(x_points, y_points)
                lingress_res = linregress(x_points, y_points)

                # warn: lowess may fail (returns Runtime Warning with jup.) when using discrete data
                lowess_res = lowess(y_points, x_points, frac=2.0 / 3.0)
                axs[i, j].plot(lins_x, lins_x * lingress_res.slope + lingress_res.intercept, label='linear regression',
                               color='black',
                               linestyle='--', linewidth=2)
                axs[i, j].plot(lowess_res[:, 0], lowess_res[:, 1], label='lowess', color='grey')

            if not hide_legend:
                axs[i, j].legend()

    for i in range(num_vars):
        plt.setp(axs[-1, i], xlabel=labels[i])
        plt.setp(axs[i, 0], ylabel=labels[i])


if __name__ == '__main__':
    pass
    # test: common parameters

    # N = 1000
    # Xs = 4
    # fake_labels = np.arange(Xs)

    ########################################################
    # test: linear regression with contours and x,y from multivariate norm

    # covs = np.asarray([[4, 3.5],
    #                    [3.5, 6]])
    # norm_rv = multivariate_normal([0, 0], covs)
    # data_matrix = norm_rv.rvs(N)
    #
    # w, v = np.linalg.eig(covs)
    # print(w)
    # print(v)
    # print("v*cov", covs @ v)
    #
    # data_matrix = (v @ data_matrix.T).T
    #
    # pointsx = data_matrix[:, 0]
    # pointsy = data_matrix[:, 1]
    #
    # res = linregress(pointsx, pointsy)
    #
    # lins = np.linspace(min(pointsx), max(pointsx), 30000)
    #
    # print("slope:", res.slope)
    # print("intercept:", res.intercept)
    # plt.plot(lins, (lins * res.slope) + 0, color='black')
    #
    # plt.scatter(pointsx, pointsy)
    # m = 100
    # X, Y = np.meshgrid(np.linspace(min(pointsx), max(pointsx), m), np.linspace(min(pointsy), max(pointsy), m))
    # locs = np.dstack((X, Y))
    # mu = [0, 0]
    #
    # rv = multivariate_normal(mu, np.cov(data_matrix.T), allow_singular=True)
    #
    # plt.contour(X, Y, rv.pdf(locs))
    # plt.show()

    #########################################################
    # test: rotated linear regression via PCA, plot contours also, and potential outliers

    # points_x = np.random.normal(0, 1, N)
    # points_y = np.random.normal(0, 1, N)
    #
    # m = 200
    #
    # points_x = (points_x - min(points_x)) / (max(points_x) - min(points_x))
    # points_y = (points_y - min(points_y)) / (max(points_y) - min(points_y))
    #
    # normed_points = np.vstack([points_x, points_y])
    #
    # norm_rv = multivariate_normal([0, 0, 0], [[0.2, 3, 0.1], [0.2, 0.7, 0.3], [0.1, 0.3, 0.5]])
    # data_matrix = norm_rv.rvs(N)
    # data_matrix = skp.minmax_scale(data_matrix, feature_range=(0, 1), axis=0, copy=True)
    #
    # f_labels = [0, 1, 2]
    # create_partial_plots(pd.DataFrame(data=data_matrix, columns=f_labels), f_labels[0])
    # plt.show()
    #
    # X, Y = np.meshgrid(np.linspace(min(points_x), max(points_x), m), np.linspace(min(points_y), max(points_y), m))
    # locs = np.dstack((X, Y))
    # mu = [np.mean(points_x), np.mean(points_y)]
    # cov = np.cov(points_x, points_y)
    # print(mu, cov)
    # rv = multivariate_normal(mu, cov, allow_singular=True)
    #
    # w, v = np.linalg.eig(cov)
    # print("w:", w)
    # print("v:", v)
    #
    # plt.contour(X, Y, rv.pdf(locs))
    # plt.scatter(points_x, points_y)
    # mu = np.mean(normed_points, axis=1)
    # plt.scatter(mu[0], mu[1], color='red')
    # plt.show()
    #
    # print(points_x)
    # print(points_y)

    ######################################################################################
    # test: create bivariate cdf

    # Fdist_2D = create_ecdf_2d(normed_points, m)
    # print(Fdist_2D)
    # lins = np.linspace(0, 1, m)
    # Fdist_2D_spline = RectBivariateSpline(lins, lins, Fdist_2D)
    #
    # X, Y = np.meshgrid(lins, lins)
    # print(X.shape, Y.shape, Fdist_2D.shape)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Fdist_2D, color='yellow')
    # ax.contour(X, Y, Fdist_2D)
    #
    # plt.show()
    ######################################################
    # test: check create_regression_plots

    # lowess = sm.nonparametric.lowess
    # coeffs = np.random.rand(Xs)
    #
    # data_matrix = np.ones((N, Xs))
    # for i in range(1, Xs):
    #     data_matrix[:, i] = np.random.normal(0, 1, N)  # weibull(100, N)
    #
    # # remember: the zeroth label is the y label
    # data_matrix[:, 0] = data_matrix @ coeffs.T
    #
    # data_matrix_pd = pd.DataFrame(data=data_matrix, columns=fake_labels)
    # print(data_matrix[:10, :])
    # print(data_matrix_pd.head(10))
    #
    # create_regression_plots( pd.DataFrame(data=data_matrix,columns=fake_labels),None,False)
    # plt.show()
    #
    # # cov matrix might be singular:
    # create_partial_plots(data_matrix_pd, fake_labels[0])
    # plt.show()

    ########################################################

    # test: play with transforms

    # q = 1 / 3
    # x_points = np.append(np.random.normal(-25, 1, int(N * q)),
    #                      np.append(np.random.normal(-10, 1, int(N * q)), np.random.normal(10, 1, int(q * N))))
    #
    # L_r = 0
    # U_r = np.pi
    # x_points = skp.minmax_scale(x_points, feature_range=(L_r, U_r), axis=0, copy=True)
    #
    # # warn: integral and derivative are directional/ordered. Therefore not element-wise computations.
    # # warn: Convolution is also directional.
    #
    # x_points = np.exp(x_points)
    # x_points = skp.minmax_scale(x_points, feature_range=(0, 1), axis=0, copy=True)
    # print(x_points)
    # print(np.exp(x_points))
    #
    # m_points = 1000
    # lins_x, dx = np.linspace(0, 1, m_points, retstep=True)
    # x_F_dist = np.empty(m_points)
    # for k in range(m_points):  x_F_dist[k] = (1 / N) * np.sum([x_points < lins_x[k]])
    #
    # plt.plot(lins_x, x_F_dist, label='ecdf')
    #
    # base_N = 30000
    # gauss_d = np.random.normal(0, 1, base_N)
    # norm_normed = (gauss_d - min(gauss_d)) / (max(gauss_d) - min(gauss_d))
    # norm_F_dist = np.empty(m_points)
    #
    # for k in range(m_points): norm_F_dist[k] = (1 / base_N) * np.sum([norm_normed < lins_x[k]])
    #
    # plt.plot(lins_x, norm_F_dist, label=' norm (0,1) cdf')
    #
    # gauss_d = np.random.normal(0, 5, base_N)
    # norm_normed = (gauss_d - min(gauss_d)) / (max(gauss_d) - min(gauss_d))
    # norm_F_dist = np.empty(m_points)
    #
    # for k in range(m_points): norm_F_dist[k] = (1 / base_N) * np.sum([norm_normed < lins_x[k]])
    #
    # plt.plot(lins_x, norm_F_dist, label='norm (0,5) cdf')
    #
    # gauss_d = np.random.normal(0, 10, base_N)
    # norm_normed = (gauss_d - min(gauss_d)) / (max(gauss_d) - min(gauss_d))
    # norm_F_dist = np.empty(m_points)
    #
    # for k in range(m_points):
    #     norm_F_dist[k] = (1 / base_N) * np.sum([norm_normed < lins_x[k]])
    #
    # plt.plot(lins_x, norm_F_dist, label='norm (0,10) cdf')
    # plt.legend()
    # plt.show()
    #
    # # substep: pdfs
    # x_f_dist = np.gradient(x_F_dist, dx)
    # norm_f_dist = savgol_filter(np.gradient(norm_F_dist, dx), 31, 3)
    # plt.plot(lins_x, x_f_dist, label='epdf')
    # plt.plot(lins_x, norm_f_dist, label='norm (0,1) pdf')
    # plt.legend()
    # plt.show()
