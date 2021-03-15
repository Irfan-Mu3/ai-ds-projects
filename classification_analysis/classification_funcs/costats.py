import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
from scipy.stats import multivariate_normal

'''
This file is used to define polarity: A measure to determine how well points (in a Euclidean space)
belonging to one of n classes are seperated relative to each other. 

The main function of use in this file is bipolarity (with rel=True) which returns a value between [0,1], where 0 means not seperated 
,i.e. when the points of each classes overlap, and 1 means well seperated, there is a strong discriminant possible between
the points.

Further notes: 
1) It is  possible for univariate polarity to better (larger) than bipolarity.

2) Bipolarity (bivariate-polarity) appears to have a form univariate_polarity_x + univariate_poly_y = bipolarity_xy.
Thus bipolarity_xy / (univariate_polarity_x + univariate_poly_y) = 1. 

3) As a consequence, the bipolarity of two identical variable yields twice the univariate_polarity, i.e.
bipolarity_xx = 2* univariate_pol_x 

4) If some classes have too few points, the results can be misleading.
'''


def compute_correlation(x, y):
    covar = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x))
    return covar * (1 / (np.std(x) * np.std(y)))


def polar_means(x, y, classes):
    # computes centroids for each class (self-means),  as well centroids of other classes (other_means)
    # assumes: The mean of two clusters is the mean of their centroids

    self_means = np.empty(len(classes))
    other_means = np.empty_like(self_means)
    for l in range(len(classes)):
        self_means[l] = np.mean(x[y == classes[l]])
    for l in range(len(classes)):
        other_means[l] = np.mean(np.delete(self_means, l))

    return self_means, other_means


def univariate_polarity(x, y, classes, penalize=False):
    # computes average squared distance from the 'centroid of the other classes'
    # x: 1D x values
    # y: 1D y values (of same shape as x)
    # classes: unique values y can take
    # remove_selfs: penalizes polarity by subtracting the squared distances from the 'centroid of the same class'

    self_means, other_means = polar_means(x, y, classes)

    polarity = 0
    rev_pol = 0
    for l in range(len(classes)):
        x_cl = x[y == classes[l]]
        polarity += np.sum(((x_cl - other_means[l]) ** 2)) / len(x_cl)
        rev_pol += np.sum(((x_cl - self_means[l]) ** 2)) / len(x_cl)

    if penalize:
        return polarity - rev_pol
    else:
        return polarity


# remember: the definition of polarity is fine: as the distance of the centroid increases, polarity increases
# remember: copolarity - does not appear to do what we want: measures whether the average distance away from
# the other centroid,

def bipolarity(x1, x2, y, classes, penalize=False, rel=False):
    # Bi-dimensional polarity: computes the average squared distance (the norm) from the 2d
    # 'centroid of the other classes'
    # x1: 1D values for dim-1
    # x2: 1D values for dim-2 (with same shape as above)
    # y: 1D values (with same shapen as above)
    # classes: unique values y can take
    # remove_selfs: penalizes polarity by subtracting the squared distances from the 'centroid of the same class'

    self_means_x1, other_means_x1 = polar_means(x1, y, classes)
    self_means_x2, other_means_x2 = polar_means(x2, y, classes)

    other_euclid = 0
    self_euclid = 0

    for l in range(len(classes)):
        mask = (y == classes[l])
        x1_cl = x1[mask]
        x2_cl = x2[mask]
        # Since we square the norm, the square and the square roots cancel, leaving us with the form below
        other_euclid += np.sum(((x1_cl - other_means_x1[l]) ** 2 + (x2_cl - other_means_x2[l]) ** 2)) / (
            len(x1_cl))
        self_euclid += np.sum(((x1_cl - self_means_x1[l]) ** 2 + (x2_cl - self_means_x2[l]) ** 2)) / (
            len(x1_cl))

    # remember: 0.5 * co-euclid = pol_x + pol_y, even in the removed case.
    if penalize:
        return (other_euclid - self_euclid)
    elif rel:
        return (other_euclid - self_euclid) / other_euclid
    else:
        return other_euclid


# def relative_bipolarity(x1, x2, y, classes):
#     # Returns the relative bipolarity. This appears to give a value between [0,1], with values nearer to zero
#     # being less 'polarized' (clusters of classes being less seperated, i.e. overlapping more),
#     # and values nearer '1' being more 'polarized' (clusters clearly seperated)
#
#     # x1: 1D values for dim-1
#     # x2: 1D values for dim-2 (with same shape as above)
#     # y: 1D values (with same shapen as above)
#     # classes: unique values y can take
#
#     num = bipolarity(x1, x2, y, classes, penalize=True)
#     denom = bipolarity(x1, x2, y, classes, penalize=False)
#
#     return num / denom


def copolarity(x1, x2, y, classes):
    # Similar to covariance: copolarity is high only when the clusters are diagonally
    # seperated, and small otherwise.
    # For example, if pol_x is high but pol_y is low. Then copolarity is low.
    # warn: this function has not been of use to us unfortunately.

    # x1: 1D values for dim-1
    # x2: 1D values for dim-2 (with same shape as above)
    # y: 1D values (with same shapen as above)
    # classes: unique values y can take
    # warn: this function was not found useful to have penalty as of late.

    self_means_x1, other_means_x1 = polar_means(x1, y, classes)
    self_means_x2, other_means_x2 = polar_means(x2, y, classes)

    copolarity = 0
    # rev_copol = 0

    for l in range(len(classes)):
        mask = (y == classes[l])
        x1_cl = x1[mask]
        x2_cl = x2[mask]
        copolarity += np.sum((x1_cl - other_means_x1[l]) * (x2_cl - other_means_x2[l])) / (len(x1_cl))
        # rev_copol += np.sum((x1_cl - self_means_x1[l]) * (x2_cl - self_means_x2[l])) / (len(x1_cl))

    # if penalize:
    #     return copolarity - rev_copol
    # else:
    return copolarity


def polarization_coefficient(x1, x2, y, classes):
    # attempts to normalizes copolarity between [-1,1]
    pol_coeff = copolarity(x1, x2, y, classes) / np.sqrt(
        univariate_polarity(x1, y, classes) * univariate_polarity(x2, y, classes))
    return pol_coeff


if __name__ == '__main__':
    np.random.seed(12315234)

    ####################################################################
    # step: experiment with polarity
    # np.random.seed(None)
    # print(np.random.get_state)

    Na = int(np.random.rand() * 30000)
    Nb = int(np.random.rand() * 10000)

    print("Na,Nb", Na, Nb)

    classes = [0, 1]

    mu = [0, 0]
    cov = [[150, 80], [80, 120]]
    x, y = np.random.multivariate_normal(mu, cov, Na).T

    rv = multivariate_normal(mu, cov, allow_singular=False)
    X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
    locs = np.dstack((X, Y))
    plt.contour(X, Y, rv.pdf(locs), levels=10)
    plt.scatter(x, y, label='x1', s=1.5)

    mu = [100, 0]
    cov = [[150, 80], [100, 120]]

    x2, y2 = np.random.multivariate_normal(mu, cov, Nb).T
    rv = multivariate_normal(mu, cov, allow_singular=False)
    X, Y = np.meshgrid(np.linspace(min(x2), max(x2), 100), np.linspace(min(y2), max(y2), 100))
    locs = np.dstack((X, Y))
    plt.contour(X, Y, rv.pdf(locs), levels=10)
    plt.scatter(x2, y2, label='x2', s=1.5)

    xs_1 = np.append(x, x2)
    xs_2 = np.append(y, y2)
    ys = np.asarray(([0] * Na) + ([1] * Nb))

    pol_x = univariate_polarity(xs_1, ys, classes, True)
    pol_y = univariate_polarity(xs_2, ys, classes, True)
    print("polarity_x (penalized):", pol_x)
    print("polarity_y (penalized):", pol_y)
    print("copolarity:", copolarity(xs_1, xs_2, ys, classes))
    print("bipolar (penalized):", bipolarity(xs_1, xs_2, ys, classes, True))
    print("relative bipolar:", relative_bipolarity(xs_1, xs_2, ys, classes))
    print("polarization coefficient (penalized):", polarization_coefficient(xs_1, xs_2, ys, classes))

    plt.legend()
    plt.show()
