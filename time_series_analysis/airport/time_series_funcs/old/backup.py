import numpy as np
import matplotlib.pyplot as plt
from sympy import poly, lambdify, O, series
from sympy.abc import B, a, b, c, d
from scipy.optimize import minimize, basinhopping, brute, least_squares, Bounds
from numba import jit, njit


# step: MA: Define e_t = Theita(B**t)Theta(B**s)theta(B)a_t = at - ... - ...
# step: AR and Integration: Define Omaiga(B**t)Omega(B**s)omega(B)(Differenced[z_t - mu]) = e_t
# step: Therefore Differenced[zt] = delta + et + ... + ..., where delta is a function of the mean
# step: Delta: Set mean of w_t to mu. If no integration occurs, E[w_t] = E[y_t] = mu

def sampleARIMA(p_vec, q_vec, sd, N):
    a_t = np.random.normal(0, sd, N)

    print("a_t:", a_t[:10])

    assert len(p_vec) < N and len(q_vec) < N

    # step: define e_t = Theta(B)a_t
    e_t = a_t.copy()

    q = len(q_vec)
    for i in range(q):
        e_t[i + 1:] -= q_vec[i] * a_t[:-(i + 1)]

    # step: define Omega(B)z_t = e_t
    z_t = e_t.copy()

    rev_p_vec = np.flip(p_vec)

    p = len(p_vec)

    for k in range(0, N):
        prev_zs = z_t[max(0, k - p):k]
        if prev_zs.size == 0:
            continue

        z_t[k] += prev_zs @ rev_p_vec[-prev_zs.size:]

    return z_t


def sampleSeasonalARIMA(p_poly: poly, d_poly: poly, q_poly: poly, sd, N, mean_mu=0.0):
    a_t = np.random.normal(0, sd, N)

    plt.plot(a_t, label='orig a_t')

    print("true exp,std of a_t:", np.mean(a_t), np.std(a_t))

    # step: MA filter
    e_t = MA_helper(a_t, q_poly, False)

    # step: Delta: Set mean of w_t to mu. If no integration occurs, E[w_t] = E[y_t] = mu
    delta = mean_mu if p_poly is None else sum(np.asarray(p_poly.coeffs()).astype(float)) * mean_mu
    e_t += delta

    # step: AR filter

    w_t = integrate_series(e_t.copy(), p_poly, True)
    print("mean and std of w_t:", np.mean(w_t), np.std(w_t))

    # step Integration filter

    z_t = integrate_series(w_t.copy(), d_poly, True)

    # step: AR-Integration filter

    # z_t = integrate_series(e_t, d_poly*p_poly, True)

    return z_t


def calculateSeasonalARIMA_error(y_t, p_poly, d_poly, q_poly):
    # z_t: the differenced series, with mean subtracted, i.e.
    # z_t = w_t - mean(w_t), where w_t = difference_series(y_t)
    # and y_t is the original series.
    # p_poly: the AR polynomial. Must have constant 1.
    # q_poly: the MA polynomial. Must have constant 1.

    # step: de-Integration filter
    w_t = MA_helper(y_t, d_poly, False)
    print("w_t[:10]", w_t[:10])

    # step: de-AR filter
    zed_t = MA_helper(w_t, p_poly, False)
    print("z_t[:10]", zed_t[:10])

    # print("estimated mean & std of zed_t", np.mean(zed_t), np.std(zed_t))

    # step: de-AR, de-Integration
    # zed_t = MA_helper(y_t,d_poly*p_poly, False)

    # step: de-Delta
    delta = np.mean(w_t) if p_poly is None else sum(np.asarray(p_poly.coeffs()).astype(float)) * np.mean(w_t)
    zed_t -= delta
    print("delta:", delta)
    # plt.plot(zed_t, label='e_t')

    # step: de-MA filter
    a_t = integrate_series(zed_t, q_poly, True)
    # print("a_t[:10]",a_t[:100])

    return a_t


def forecastSeasonalARIMA(y_t, a_t, h, p_poly, d_poly, q_poly):
    a_t = np.append(a_t, np.zeros(h))

    # step: MA filter
    e_t = MA_helper(a_t, q_poly, False)

    # step: Delta
    mu = np.mean(difference_series(y_t, d_poly))
    e_t += mu if p_poly is None else sum(np.asarray(p_poly.coeffs()).astype(float)) * mu
    # print("mu", mu)

    # step: AR filter
    # w_t = integrate_series(e_t, p_poly, True)

    # step Integration filter
    # z_t = integrate_series(w_t.copy(), d_poly, True)

    # print("e_t", list(e_t))

    # step: AR,Integration filter
    # print("p_poly", p_poly)
    # print("d_poly", d_poly)
    # print("pd_poly", p_poly * d_poly)
    # pd_poly = (p_poly * d_poly)
    # warn: the below does not work,strangely
    z_t = integrate_series(e_t.copy(), p_poly * d_poly)

    # z_t = integrate_series((integrate_series(e_t.copy(), p_poly)), d_poly)

    return z_t.copy()


# def backcastSeasonalARIMA(y_t_rev: np.ndarray, a_t_rev:np.ndarray, h: int, p_poly: poly, d_poly: poly, q_poly: poly):
#     # assumes a_t and y_t are reversed
#
#     a_t_rev = np.append(a_t_rev, np.zeros(h))
#
#     # step: MA filter
#     e_t = MA_helper(a_t_rev, q_poly, False)
#
#     # step: Delta
#     mu = np.mean(difference_series(y_t_rev, d_poly))
#     e_t += mu if p_poly is None else sum(np.asarray(p_poly.coeffs()).astype(float)) * mu
#     print("mu backcast", mu)
#
#     # step: AR filter
#     P_coeffs = np.asarray(p_poly.coeffs()).astype(float)
#     max_coeff = P_coeffs[0]
#
#     print("p poly before:",p_poly)
#     max_deg = O(B**(p_poly.degree()))
#     p_poly = poly((p_poly.expr + max_deg).removeO()/max_coeff)
#     print("max coeff:",max_coeff)
#     print("p poly after:",p_poly)
#     # print("p poly / max coeff:", p_poly/max_coeff)
#
#     w_t = - integrate_series(e_t / max_coeff, p_poly, True)
#
#     # step Integration filter
#     z_t = integrate_series(w_t.copy(), d_poly, True)
#
#     return z_t


def difference_series(y_t: np.ndarray, d_poly: poly):
    return MA_helper(y_t, d_poly, False)


# todo: this function causes problems
def integrate_series(y_t: np.ndarray, p_poly: poly, subtract=True):
    N = len(y_t)

    if p_poly is not None:
        assert p_poly.degree() < N

    if p_poly is not None:

        P_coeffs = np.asarray(p_poly.coeffs()[:-1]).astype(float)
        p_lags = (np.round((p_poly.diff().coeffs() / np.asarray(P_coeffs)).astype(float))).astype(int)

        print("P_coeffs", P_coeffs)
        print("p_lags", p_lags)

        for k in range(0, N):
            loc_Ys = k - p_lags
            prev_Ys = loc_Ys[loc_Ys >= 0]
            if prev_Ys.size > 0:
                # since we are bringing y_ts on the RHS, we have to subtract
                if subtract:
                    y_t[k] -= y_t[prev_Ys] @ P_coeffs[-prev_Ys.size:]
                else:
                    y_t[k] += y_t[prev_Ys] @ P_coeffs[-prev_Ys.size:]

    return y_t


#
# def backcast_integrate_series(y_t: np.ndarray, p_poly: poly, degree, subtract=True):
#     N = len(y_t)
#
#     if p_poly is not None:
#         assert p_poly.degree() < N
#
#     if p_poly is not None:
#
#         P_coeffs = np.asarray(p_poly.coeffs()[:-1]).astype(float)
#         p_lags = (p_poly.diff().coeffs() / np.asarray(P_coeffs)).astype(int)
#
#         for k in range(0, N):
#             loc_Ys = k - p_lags
#             prev_Ys = loc_Ys[loc_Ys >= 0]
#             if prev_Ys.size > 0:
#                 # since we are bringing y_ts on the RHS, we have to subtract
#                 if subtract:
#                     y_t[k] -= y_t[prev_Ys] @ P_coeffs[-prev_Ys.size:]
#                 else:
#                     y_t[k] += y_t[prev_Ys] @ P_coeffs[-prev_Ys.size:]
#
#     return y_t


def MA_helper(a_t: np.ndarray, q_poly: poly, subtract=False):
    N = len(a_t)

    if q_poly is not None:
        assert q_poly.degree() < N

    e_t = a_t.copy()

    if q_poly is not None:
        Q_coeffs = np.asarray(q_poly.coeffs()[:-1]).astype(float)
        q_lags = (np.round((q_poly.diff().coeffs() / np.asarray(Q_coeffs)).astype(float))).astype(int)

        Q = len(Q_coeffs)
        for i in range(Q):
            if subtract:
                e_t[q_lags[i]:] -= Q_coeffs[i] * a_t[:-q_lags[i]]
            else:
                e_t[q_lags[i]:] += Q_coeffs[i] * a_t[:-q_lags[i]]
    return e_t


def lambdify_poly_coeff_creator(poly_coeffs, symbols):
    # the symbols need not be ordered, however poly_coeffs should be with the largest lag first,
    # and zeroth lag not included
    lambds = [lambdify(symbols, coeff, modules='numpy') for coeff in poly_coeffs]
    return np.asarray(lambds)


def calculateSeasonalARIMA_error_minimization_form(ARIMA_coeffs: np.asarray, w_t: np.ndarray, p_coeff_lambds,
                                                   q_coeff_lambds, p_lags,
                                                   q_lags, p_coeff_size, q_coeff_size):
    # ARIMA_coeffs: AR and MA coeffs, with AR coeffs listed first in order they appear.
    # z_t: the differenced series, with mean subtracted, i.e.
    # z_t = w_t - mean(w_t), where w_t = difference_series(y_t)
    # and y_t is the original series.
    # p_coeff_lambds: lambdas to obtain the (SAR) polynomial coefficients via the AR coeffs
    # q_coeff_lambds: lambdas to obtain the (SMA) polynomial coefficients via the MA coeffs
    # p_lags: the lags for AR parameters
    # q_lags: the lags for MA parameters
    # p_coeff_size: the number of AR params
    # q_coeff_size: the number of MA params

    # step: obtain ordered coefficients, with furthest lag being in index 0
    N = w_t.size

    # step: de-AR,de-Integration filter

    # step: de-Delta

    #

    # step: Define  zed_t = Omaiga(B**t)Omega(B**s)omega(B)z_t = z_t - ... - ...

    zed_t = w_t.copy()

    if p_coeff_size != 0:
        p_ARIMA_coeffs = ARIMA_coeffs[:p_coeff_size]
        P_coeffs = np.asarray([p_coeff_lambds[i](*p_ARIMA_coeffs) for i in range(len(p_coeff_lambds))])

        P = len(P_coeffs)
        for i in range(P):
            # we add here, since we do not move the since we do not move z_t to the RHS, but keep it left
            zed_t[p_lags[i]:] += P_coeffs[i] * w_t[:-p_lags[i]]

    # step: AR: Define Omaiga(B**t)Omega(B**s)omega(B)a_t = zed_t
    # step: Therefore at = zed_t + ... + ...
    a_t = zed_t.copy()

    # step: Delta: subtract mean from w_t, (this sets the mean of a_t to zero)
    # subtract here
    a_t -= np.mean(w_t) if p_coeff_size == 0 else (1 + sum(P_coeffs)) * np.mean(w_t)

    if q_coeff_size != 0:
        q_ARIMA_coeffs = ARIMA_coeffs[-q_coeff_size:]
        Q_coeffs = np.asarray([q_coeff_lambds[j](*q_ARIMA_coeffs) for j in range(len(q_coeff_lambds))])

        for k in range(0, N):
            loc_As = k - q_lags
            prev_As = loc_As[loc_As >= 0]
            if prev_As.size > 0:
                # we sub here, since we move a_t terms to the LHS
                a_t[k] -= a_t[prev_As] @ Q_coeffs[-prev_As.size:]

    return a_t


# def calculateSeasonalARIMA_error_minimization_form(ARIMA_coeffs: np.asarray, w_t: np.ndarray, p_coeff_lambds,
#                                                    q_coeff_lambds, p_lags,
#                                                    q_lags, p_coeff_size, q_coeff_size):
#     # ARIMA_coeffs: AR and MA coeffs, with AR coeffs listed first in order they appear.
#     # z_t: the differenced series, with mean subtracted, i.e.
#     # z_t = w_t - mean(w_t), where w_t = difference_series(y_t)
#     # and y_t is the original series.
#     # p_coeff_lambds: lambdas to obtain the (SAR) polynomial coefficients via the AR coeffs
#     # q_coeff_lambds: lambdas to obtain the (SMA) polynomial coefficients via the MA coeffs
#     # p_lags: the lags for AR parameters
#     # q_lags: the lags for MA parameters
#     # p_coeff_size: the number of AR params
#     # q_coeff_size: the number of MA params
#
#     # step: obtain ordered coefficients, with furthest lag being in index 0
#     N = w_t.size
#
#     # step: Define  zed_t = Omaiga(B**t)Omega(B**s)omega(B)z_t = z_t - ... - ...
#
#     zed_t = w_t.copy()
#
#     if p_coeff_size != 0:
#         p_ARIMA_coeffs = ARIMA_coeffs[:p_coeff_size]
#         P_coeffs = np.asarray([p_coeff_lambds[i](*p_ARIMA_coeffs) for i in range(len(p_coeff_lambds))])
#
#         P = len(P_coeffs)
#         for i in range(P):
#             # we add here, since we do not move the since we do not move z_t to the RHS, but keep it left
#             zed_t[p_lags[i]:] += P_coeffs[i] * w_t[:-p_lags[i]]
#
#     # step: AR: Define Omaiga(B**t)Omega(B**s)omega(B)a_t = zed_t
#     # step: Therefore at = zed_t + ... + ...
#     a_t = zed_t.copy()
#
#     # step: Delta: subtract mean from w_t, (this sets the mean of a_t to zero)
#     # subtract here
#     a_t -= np.mean(w_t) if p_coeff_size == 0 else (1 + sum(P_coeffs)) * np.mean(w_t)
#
#     if q_coeff_size != 0:
#         q_ARIMA_coeffs = ARIMA_coeffs[-q_coeff_size:]
#         Q_coeffs = np.asarray([q_coeff_lambds[j](*q_ARIMA_coeffs) for j in range(len(q_coeff_lambds))])
#
#         for k in range(0, N):
#             loc_As = k - q_lags
#             prev_As = loc_As[loc_As >= 0]
#             if prev_As.size > 0:
#                 # we sub here, since we move a_t terms to the LHS
#                 a_t[k] -= a_t[prev_As] @ Q_coeffs[-prev_As.size:]
#
#     return a_t


def full_form_squared(ARIMA_coeffs: np.asarray, w_t: np.ndarray, p_coeff_lambds,
                      q_coeff_lambds, p_lags,
                      q_lags, p_coeff_size, q_coeff_size):
    # wraps error as summed squared error, needed for particular minimization functions.
    res = calculateSeasonalARIMA_error_minimization_form(ARIMA_coeffs, w_t, p_coeff_lambds,
                                                         q_coeff_lambds, p_lags,
                                                         q_lags, p_coeff_size, q_coeff_size)
    return np.sum(res ** 2)


# todo: test, with known structures
def compute_interval(a_t, p_poly, d_poly, q_poly, num_coeffs, max_lag=100):
    # num_coeffs includes the estimated mean too

    v = len(a_t) - num_coeffs
    var = sum(a_t ** 2) * (1 / v)

    if p_poly is None and q_poly is None and d_poly is None:
        const_shock = 1.96 * np.sqrt((var).astype(np.float))
        return np.asarray([const_shock] * max_lag)

    # compute_memory_shock_terms

    AR_series = poly((series((1 / p_poly), x=B, x0=0, n=max_lag + 1)).as_expr().removeO()) if p_poly is not None else 1

    integrate_series = \
        poly((series((1 / d_poly), x=B, x0=0, n=max_lag + 1)).as_expr().removeO()) if d_poly is not None else 1

    if q_poly is not None:
        shock_poly: poly = AR_series * integrate_series * q_poly
    else:
        shock_poly = AR_series * integrate_series

    # shock_poly: poly = (series(1 / (d_poly*p_poly), x=B, x0=0, n=max_lag + 1)).as_expr().removeO().as_poly()

    shock_coeffs = shock_poly.coeffs()

    # print("AR_series:",AR_series)
    # print("integrate_series:",integrate_series)
    # print("AR*integrate_series:",AR_series*integrate_series)
    # print("shock_poly:", shock_poly)
    # print("shock_poly  coeffs:",shock_coeffs)

    s_t = np.cumsum(np.flip(shock_coeffs) ** 2)

    # print("var*s_t", list(var * s_t))

    std_errors = 1.96 * np.sqrt((var * s_t).astype(np.float))
    # print("std_errors:", list(std_errors))

    return std_errors


if __name__ == '__main__':
    np.random.seed(1236)

    ###################################################################################################

    # Define process parameters: 1) AR, D & MA polynomials, 2) mean of process which is a (deviations from mean)
    # 3) std of shock/gaussian error terms

    mu = 2
    std = 3

    p_poly = poly(1 - (a * B) - (d * B ** 2), B) * poly(1 - (b * B ** 8) - (c * B ** 16), B)  # + poly(d * B ** 7),B)
    q_poly = poly(1 - a * B, B) * poly(1 - (b * B ** 7), B) * poly(1 - c * B ** 31, B)
    d_poly = poly((1 - B), B)

    #
    p_symbols = [a, d, b, c]
    q_symbols = [a, b, c]

    # Coefficients in the order they appear above
    p_ARIMA_coeffs = [0.5, 0.4, 0.3, 0.4]
    q_ARIMA_coeffs = [-0.3, -0.6, 0.2]

    # assert p
    # assert len(p_ARIMA_coeffs) ==

    sample_p_poly = poly(p_poly.subs(dict(zip(p_symbols, p_ARIMA_coeffs[:len(p_ARIMA_coeffs)]))))
    sample_q_poly = poly(q_poly.subs(dict(zip(q_symbols, q_ARIMA_coeffs[-len(q_ARIMA_coeffs):]))))

    print(sample_p_poly)
    print(sample_q_poly)

    # param bounds
    p_param_abs_bounds = [2]
    q_param_abs_bounds = [2]

    # assumes multiplicative model
    total_params = np.append(p_param_abs_bounds, q_param_abs_bounds)
    total_param_bounds = (-total_params, total_params)

    ##########################################################################################

    # Sample a random process with the coefficients above

    N = 2000

    y_t_sampled = sampleSeasonalARIMA(sample_p_poly, d_poly, sample_q_poly, std, N, mean_mu=mu)
    a_t_error = calculateSeasonalARIMA_error(y_t_sampled, sample_p_poly, d_poly, sample_q_poly)

    plt.plot(y_t_sampled, label="sampled y_t")
    plt.plot(a_t_error, label='estimated a_t error', marker='.', markevery=10)

    # remember: delta converges slowly with increased N
    print("Mean & std of y_t:", np.mean(y_t_sampled), np.std(y_t_sampled))
    # remember: don't square error to determine the sd of a_t
    print("estimated mean & std of a_t", np.mean(a_t_error), np.std(a_t_error))

    plt.legend()
    plt.show()

    #########################################################################################

    # Learn the coefficients from the sampled process

    p_poly_coeffs = np.asarray(p_poly.coeffs()[:-1])
    q_poly_coeffs = np.asarray(q_poly.coeffs()[:-1])

    p_lags = (p_poly.diff().coeffs() / np.asarray(p_poly_coeffs)).astype(int)
    q_lags = (q_poly.diff().coeffs() / np.asarray(q_poly_coeffs)).astype(int)

    p_coeff_lambds = lambdify_poly_coeff_creator(p_poly_coeffs, p_symbols)
    q_coeff_lambds = lambdify_poly_coeff_creator(q_poly_coeffs, q_symbols)

    initial_guess = np.random.random(len(p_ARIMA_coeffs) + len(q_ARIMA_coeffs))
    initial_guess /= np.sum(initial_guess)

    est_w_t_flipped = difference_series(y_t_sampled, d_poly)

    # res = minimize(full_form_squared, x0=myx0, args=(calc_w_t, p_coeff_lambds, q_coeff_lambds, p_lags, q_lags, 2,2),
    #                method='Nelder-Mead', options={'maxiter': 5000, 'disp': True})
    # print("res_minimize:", res.x)

    # warn: lm method does not allow bounds, therefore good initial guess is needed
    #  (i.e. satisfying conditions like invertibility)

    res = least_squares(calculateSeasonalARIMA_error_minimization_form, bounds=(-2, 2), x0=initial_guess,
                        args=(est_w_t_flipped, p_coeff_lambds, q_coeff_lambds, p_lags, q_lags, len(p_ARIMA_coeffs),
                              len(q_ARIMA_coeffs)),
                        )
    print("Learning success? :", res.success)
    print("learnt params", res.x)
    print("true params:", np.append(p_ARIMA_coeffs, q_ARIMA_coeffs))
    print("initial param guess:", initial_guess)

    learnt_coeffs = res.x

    est_p_poly: poly = poly(p_poly.subs(dict(zip(p_symbols, learnt_coeffs[:len(p_ARIMA_coeffs)]))), B)
    est_q_poly: poly = poly(q_poly.subs(dict(zip(q_symbols, learnt_coeffs[-len(q_ARIMA_coeffs):]))), B)

    print("est_p_poly:", est_p_poly)
    print("est_q_poly:", est_q_poly)
    print("d_poly:", d_poly)

    ##########################################################################################################

    # Forecast with 95% confidence interval
    # p_pol = poly(
    #     0.131068068331084 * B ** 18 + 0.232053307911907 * B ** 17 - 0.399731589577912 * B ** 16 + 0.0964199635920732 * B ** 10 + 0.170709554090376 * B ** 9 - 0.294061748253928 * B ** 8 - 0.327890193690927 * B ** 2 - 0.580522815714762 * B + 1.0,
    #     B)
    # q_pol = poly(-0.0255340479087502*B**39 - 0.122704627686806*B**38 - 0.0418684801379178*B**32 - 0.201200228240157*B**31 + 0.126908642858358*B**8 + 0.609863262880315*B**7 + 0.208093601603387*B + 1.0, B)

    # print("p_pol:",p_pol)
    # print("q_pol:",q_pol)
    #
    # print(p_pol.expr)
    # print(est_p_poly.as_expr)

    h_max = int(N / 2)  # forecast length
    # a_t_error_2p0 = calculateSeasonalARIMA_error(y_t_sampled, p_pol, d_poly, q_pol)
    a_t_error_2p0 = calculateSeasonalARIMA_error(y_t_sampled, est_p_poly, d_poly, est_q_poly)
    plt.plot(np.arange(0, N), a_t_error_2p0, label='a_t_error_2p0')

    yt_forecasts = forecastSeasonalARIMA(y_t_sampled, a_t_error_2p0, h_max, est_p_poly, d_poly, est_q_poly)
    plt.plot(np.arange(0, N + h_max), yt_forecasts, label='forecast z_t, h = ' + str(h_max))

    # Backcast w/o relearning
    y_t_sampled_flipped = np.flip(y_t_sampled)

    a_t_error_2p0_flipped = calculateSeasonalARIMA_error(y_t_sampled_flipped, est_p_poly, d_poly, est_q_poly)
    yt_backcasts = forecastSeasonalARIMA(y_t_sampled_flipped, a_t_error_2p0_flipped, h_max, est_p_poly, d_poly,
                                         est_q_poly)

    plt.plot(np.arange(0, N), a_t_error_2p0_flipped, label='a_t_error_2p0 flipped? w0 relearning')
    plt.plot(np.arange(-h_max, N), np.flip(yt_backcasts), label='backcast w0 relearning z_t,  h = ' + str(h_max),
             marker='x',
             markevery=20)

    #################################

    # Backcast with relearning

    initial_guess = np.random.random(len(p_ARIMA_coeffs) + len(q_ARIMA_coeffs))
    initial_guess /= np.sum(initial_guess)

    est_w_t_flipped = difference_series(y_t_sampled_flipped, d_poly)

    # remember: With multiplicative models, flipping param polynomials for backcasting yields the same lags as forecasting.
    # remember: e.g. 1) res_b = polyb.diff().coeffs() / np.asarray(polyb.coeffs())[:-1]
    #  Then 2) np.flip(res_b) - res_b[-1] + 1 === res_b
    # Although, something can be left to be said about the parameters

    p_poly = poly(1 - (a * B) - (d * B ** 2), B) * poly(1 - (b * B ** 8) - (c * B ** 16), B)  # + poly(d * B ** 7),B)
    q_poly = poly(1 - a * B, B) * poly(1 - (b * B ** 7), B) * poly(1 - c * B ** 31, B)
    #
    # p_lags = (p_poly.diff().coeffs() / np.asarray(p_poly_coeffs)).astype(int)
    # q_lags = (q_poly.diff().coeffs() / np.asarray(q_poly_coeffs)).astype(int)
    #
    # p_coeff_lambds = lambdify_poly_coeff_creator(p_poly_coeffs, p_symbols)
    # q_coeff_lambds = lambdify_poly_coeff_creator(q_poly_coeffs, q_symbols)
    #

    res = least_squares(calculateSeasonalARIMA_error_minimization_form, bounds=(-2, 2), x0=initial_guess,
                        args=(est_w_t_flipped, p_coeff_lambds, q_coeff_lambds, p_lags, q_lags, len(p_ARIMA_coeffs),
                              len(q_ARIMA_coeffs)),
                        )
    print("Learning success? :", res.success)
    print("learnt params", res.x)
    print("true params?:", np.append(np.asarray(p_ARIMA_coeffs)/p_ARIMA_coeffs[0], np.asarray(q_ARIMA_coeffs)/p_ARIMA_coeffs[0]))
    print("initial param guess:", initial_guess)

    learnt_coeffs = res.x

    est_p_poly_back: poly = poly(p_poly.subs(dict(zip(p_symbols, learnt_coeffs[:len(p_ARIMA_coeffs)]))), B)
    est_q_poly_back: poly = poly(q_poly.subs(dict(zip(q_symbols, learnt_coeffs[-len(q_ARIMA_coeffs):]))), B)

    a_t_error_2p0_flipped = calculateSeasonalARIMA_error(y_t_sampled_flipped, est_p_poly_back, d_poly, est_q_poly_back)
    yt_backcasts = forecastSeasonalARIMA(y_t_sampled_flipped, a_t_error_2p0_flipped, h_max, est_p_poly_back, d_poly,
                                         est_q_poly_back)

    plt.plot(np.arange(0, N), a_t_error_2p0_flipped, label='a_t_error_2p0 flipped?')

    plt.plot(np.arange(-h_max, N), np.flip(yt_backcasts), label='backcast z_t,  h = ' + str(h_max), marker='x',
             markevery=20)
    plt.plot(np.arange(0, N), y_t_sampled, label='original zt sampled')

    # Error Intervals for fore/back casting

    num_coeffs = len(p_ARIMA_coeffs) + len(q_ARIMA_coeffs)
    num_coeffs += 1 if mu != 0 else 0

    max_lag = 200
    error_bound = compute_interval(a_t_error_2p0, est_p_poly, d_poly, est_q_poly, num_coeffs, max_lag=max_lag)

    plt.plot(np.arange(N, N + max_lag), yt_forecasts[N:N + max_lag] + error_bound[:max_lag], label='forecast upper 95%')
    plt.plot(np.arange(N, N + max_lag), yt_forecasts[N:N + max_lag] - error_bound[:max_lag], label='forecast lower 95%')

    plt.plot(np.arange(-max_lag, 0), np.flip(yt_backcasts[N:N + max_lag]) + np.flip(error_bound[:max_lag]),
             label='backcast upper 95%')
    plt.plot(np.arange(-max_lag, 0), np.flip(yt_backcasts[N:N + max_lag]) - np.flip(error_bound[:max_lag]),
             label='backcast lower 95%')

    plt.legend()
    plt.show()

    ###########################################################################################################

    ##########################################################################################################

    #
    # p_poly = poly(1 - 0.5 * B, B)  # * poly(1 - 0.5 * B ** 4, B)
    # q_poly = poly(1 - (-0.3) * B, B) * poly(1 - (-0.6) * B ** 4, B)
    # d_poly = poly((1 - B), B)
    #
    #
    # p_poly = poly(1 - a * B, B)  # * poly(1 - b * B ** 4, B)  # + poly(0.1 * x ** 7))
    # q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 4, B)  # * poly(1 - 0.15 * x ** 7)
    #
    # p_ARIMA_coeffs = [0.5]
    # q_ARIMA_coeffs = [-0.3, -0.6]
    #
    # p_poly_coeffs = np.asarray(p_poly.coeffs()[:-1])
    # q_poly_coeffs = np.asarray(q_poly.coeffs()[:-1])
    #
    # p_lags = (p_poly.diff().coeffs() / np.asarray(p_poly_coeffs)).astype(int)
    # q_lags = (q_poly.diff().coeffs() / np.asarray(q_poly_coeffs)).astype(int)
    #
    # p_coeff_lambds = lambdify_poly_coeff_creator(p_poly_coeffs, p_symbols)
    # q_coeff_lambds = lambdify_poly_coeff_creator(q_poly_coeffs, q_symbols)
    #
    # print("estimated Expected value of shock a_t", np.mean(a_t_error))
    # print("estimated sd of shock a_t", np.std(a_t_error))
    #
    # ###################################################################
    #
    #
    #
    # print(p_poly)
    # print(q_poly)
    #
    # p_poly = poly(p_poly.subs(dict(zip(p_symbols, learnt_coeffs[:len(p_ARIMA_coeffs)]))))
    # q_poly = poly(q_poly.subs(dict(zip(q_symbols, learnt_coeffs[-len(q_ARIMA_coeffs):]))))
    #
    # print(p_poly)
    # print(q_poly)
    #
    # a_t_error_2p0 = calculateSeasonalARIMA_error(y_t_sampled, p_poly, d_poly, q_poly)
    # h_max = int(N / 2)
    # yt_forecasts = forecastSeasonalARIMA(y_t_sampled, a_t_error_2p0, h_max, p_poly, d_poly, q_poly)
    # plt.plot(yt_forecasts, label='forecasted z_t, h = ' + str(h_max))
    # plt.plot(y_t_sampled, label='original zt sampled')
    #
    # _num_coeffs = len(p_ARIMA_coeffs) + len(q_ARIMA_coeffs)
    # _num_coeffs += 1 if mu != 0 else 0
    #
    # max_lag = h_max
    # error_bound = compute_interval(a_t_error_2p0, p_poly, d_poly, q_poly, _num_coeffs, max_lag=max_lag)
    #
    # print(len(yt_forecasts[N:N + max_lag]), len(error_bound[:max_lag]))
    #
    # plt.plot(np.arange(N, N + max_lag), yt_forecasts[N:N + max_lag] + error_bound[:max_lag], label='forecast upper 95%')
    # plt.plot(np.arange(N, N + max_lag), yt_forecasts[N:N + max_lag] - error_bound[:max_lag], label='forecast lower 95%')
    #
    # y_t_sampled_flipped = np.flip(y_t_sampled)
    # a_t_error_2p0_flipped = calculateSeasonalARIMA_error(y_t_sampled_flipped, p_poly, d_poly, q_poly)
    # yt_backcasts = forecastSeasonalARIMA(y_t_sampled_flipped, a_t_error_2p0_flipped, h_max, p_poly, d_poly, q_poly)
    #
    # plt.plot(np.arange(0, N + h_max), yt_forecasts, label='forecast z_t, h = ' + str(h_max))
    # plt.plot(np.arange(-h_max, N), np.flip(yt_backcasts), label='backcast z_t,  h = ' + str(h_max))
    # plt.plot(np.arange(0, N), y_t_sampled, label='original zt sampled')
    #
    # plt.plot(np.arange(-max_lag, 0), np.flip(yt_backcasts)[:max_lag] + np.flip(error_bound[:max_lag]),
    #          label='backcast upper 95%')
    # plt.plot(np.arange(-max_lag, 0), np.flip(yt_backcasts)[:max_lag] - np.flip(error_bound[:max_lag]),
    #          label='backcast lower 95%')
    #
    # plt.legend()
    # plt.show()

    # print(p_poly,d_poly)

    # p_poly = poly(0.2288841035668*B**5 - 0.520739168326687*B**4 - 0.439536945727134*B + 1.0, B)  # + poly(0.1 * B** 7,B)
    # d_poly = poly((1 - B), B)  # * poly(1 - B ** 35, B)  * poly(1 - B ** 365, B)
    # print(p_poly,d_poly)

    # randry = np.random.rand(1000)
    # plt.plot(integrate_series(integrate_series(a_t_error_2p0.copy(),p_poly),d_poly),label='2')
    #
    # plt.plot(integrate_series(a_t_error_2p0.copy(),p_poly*d_poly),label='1')
    #
    # plt.legend()
    # plt.show()
    #########################################################################################################

    # todo: backcasting is solving the problem twice: 1) normally, and then backcast
    # todo: 2) add the backcastpoints to the time series and solve again

    # y_t_sampled_flipped = np.flip(y_t_sampled)
    # a_t_error_2p0_flipped = calculateSeasonalARIMA_error(y_t_sampled_flipped, p_poly, d_poly, q_poly)
    # yt_backcasts = forecastSeasonalARIMA(y_t_sampled_flipped, a_t_error_2p0_flipped, h_max, p_poly, d_poly, q_poly)
    #
    # plt.plot(np.arange(0, N + h_max), yt_forecasts, label='forecast z_t, h = ' + str(h_max))
    # plt.plot(np.arange(-h_max, N), np.flip(yt_backcasts), label='backcast z_t,  h = ' + str(h_max))
    # plt.plot(np.arange(0, N), y_t_sampled, label='original zt sampled')
    #
    # myx0 = np.random.random(len(p_ARIMA_coeffs) + len(q_ARIMA_coeffs))
    # myx0 /= np.sum(myx0)
    # calc_w_t_flipped = difference_series(y_t_sampled_flipped, d_poly)
    #
    # res = least_squares(calculateSeasonalARIMA_error_minimization_form, bounds=(-2, 2), x0=myx0,
    #                     args=(calc_w_t_flipped, p_coeff_lambds, q_coeff_lambds, p_lags, q_lags, len(p_ARIMA_coeffs),
    #                           len(q_ARIMA_coeffs)),
    #                     )

    # print("res", res)
    # print("true params:", np.append(p_ARIMA_coeffs, q_ARIMA_coeffs))
    # print("initial guess:", myx0)

    # learnt_coeffs = res.x

    # p_poly = poly(1 - a * B, B) * poly(1 - b * B ** 4, B)
    # q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 4, B)

    # print(p_poly)
    # print(q_poly)

    # p_poly = poly(p_poly.subs(dict(zip(pq_symbols, learnt_coeffs[:len(p_ARIMA_coeffs)]))))
    # q_poly = poly(q_poly.subs(dict(zip(pq_symbols, learnt_coeffs[-len(q_ARIMA_coeffs):]))))
    # print(p_poly)
    # print(q_poly)

    # ###############################################################################################################

    # a_t_error_2p0_flipped = calculateSeasonalARIMA_error(y_t_sampled_flipped, p_poly, d_poly, q_poly)
    # yt_backcasts_2 = forecastSeasonalARIMA(y_t_sampled_flipped, a_t_error_2p0_flipped, h_max, p_poly, d_poly, q_poly)
    # plt.plot(np.arange(-h_max, N), np.flip(yt_backcasts_2), label='backcasting alone z_t,  h = ' + str(h_max),
    #          linestyle=':')

    # # a_t_error_3p0 = calculateSeasonalARIMA_error(np.flip(yt_backcasts_2), p_poly, d_poly, q_poly)
    # # yt_forecasts_2 = forecastSeasonalARIMA(np.flip(yt_backcasts_2), a_t_error_3p0, h_max, p_poly, d_poly, q_poly)
    # # plt.plot(np.arange(-h_max, N + h_max), np.flip(yt_forecasts_2), label='backcasting then forecasting z_t, h = ' + str(h_max),
    # #          linestyle=':')

    # plt.legend()
    # plt.show()

    ##########################################################################################################

    # remember: if the solution is off, this is because we are sampling the process, not truly using the expected value
    # res = minimize(calculateSeasonalARIMA_minimization_form, x0=myx0, args=(z_t_sampled, p_lags, q_lags),
    #                method='Powell',options={'maxiter':5000,'disp':True})
    #
    # print(res)

    # res2 = basinhopping(calculateSeasonalARIMA_minimization_form,x0=myx0,niter=2000,minimizer_kwargs={'args':(z_t_sampled, p_lags, q_lags)})
    # print(res2)

    # res3 = brute(calculateSeasonalARIMA_minimization_form,ranges=([-2]*len(p_lags),[2]*len(p_lags)),args=(z_t_sampled, p_lags, q_lags))
    # print(res3)

    # N = 2000
    # p_params = 3
    # q_params = 5
    # T = p_params + q_params
    #
    # A = np.random.random((T, N))
    # b = np.random.rand(p_params + q_params)
    #
    # x = lstsq(A, b, rcond=None)
