import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import statsmodels.tsa.stattools as stt
from sympy import poly
from sympy.abc import a, b, B, c, d
import time_series_funcs.sarima_plus_plus as tsf
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as holtwinters


# todo tommorrow: solve problem, forecast sol, plot autocorr, test if MA(1)
# todo: after finding weekly correlate, look at averages of days of the week over the years
# todo: study particular crimes


def compute_rks(xs):
    # computes the autocorrelation
    N = len(xs)
    n = N - 1
    cnorm = (1 / N)

    c_ks = np.empty(int(N / 2))

    for k in range(int(N / 2)):
        c_ks[k] = cnorm * (xs[:n - (2 * k)] @ xs[k:n - k].T)

    return c_ks / c_ks[0]


def compute_rks2(xs, mean):
    # alternative algorithm to computes autocorrelation
    N = len(xs)
    xs = np.append(xs, np.ones(N) * (-mean))
    c_ks = np.empty(int(N))
    for k in range(N):
        c_ks[k] = (1 / (N - k))
        temp = 0
        for t in range(0, N - k):
            temp += xs[t] * xs[t + k]
        c_ks[k] *= temp

    return c_ks / c_ks[0]


def compute_rks3(ts):
    # alternative algorithm to computes autocorrelation
    dev_from_mean = ts - np.mean(ts)
    autocorr_f = np.correlate(dev_from_mean, dev_from_mean, mode='full')
    rks3 = autocorr_f[int(autocorr_f.size / 2):] / autocorr_f[int(autocorr_f.size / 2)]
    return rks3


def compute_rolling_mean(ts, order=30):
    return pd.Series(ts).rolling(order).mean()


def compute_corr_95(autocorr):
    # computes 95% conf. interval for autocorrelations. Should not be used for the residuals (naively anyway).
    temp = 2 * (autocorr ** 2)
    temp[0] = 1
    temp = np.cumsum(temp)
    return np.sqrt(temp / len(temp))


def compute_safe_conf_95(autocorr):
    # computes 95% conf. interval of autocorrelations. Can be used for ts, or its residuals.
    # plots the ts, the autocorr, and the partial autocorr together.

    # ts: time-series
    # pacf_max: compute pacf for the interval [0,1,2,...,pacf_max]
    # corr_max: present autocorr. for the autocorrelations in the interval [0,1,...,corr_max]
    # ts_is_residuals: Is the time-series the residuals of an original series? (This affects confidence intervals)
    # y_label: label for the y_axis of the time series
    # ts_label: label for the time series itself
    return np.asarray([2 / np.sqrt(len(autocorr))] * len(autocorr))


def create_series_plot(ts: np.ndarray, pacf_max, corr_max=None, ts_is_residuals=False, y_label=None, ts_label=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    if ts_label:
        ax1.plot(ts, label=ts_label)
    else:
        ax1.plot(ts, label='ts')

    ax1.plot(compute_rolling_mean(ts), label='rolling mean every 30t')

    rks3 = compute_rks3(ts)
    if ts_is_residuals:
        se_95 = [2 / np.sqrt(len(rks3))] * len(rks3)
    else:
        se_95 = np.sqrt(compute_corr_95(rks3))

    ax2.plot(rks3, label='corellogram')
    ax2.vlines(np.arange(0, len(rks3)), 0, rks3, color="tab:red")
    ax2.plot(se_95, label='est. est. upper 95% conf. interval')
    ax2.plot(-se_95, label='est. est. lower 95% conf. interval')

    ax2.set_ylabel('Estimated corr. coeff.')
    ax2.set_xlabel('t')
    if corr_max is not None:
        ax2.set_xlim([0, corr_max])

    # remember: pacf calculation can be unstable. Lag_max is a function of convergence: the faster it converges,
    #  the smaller the lag_max possible.
    xs_PACF = stt.pacf(rks3, nlags=pacf_max)
    xs_pacf_SE = np.sqrt(1 / len(rks3))

    ax3.plot(xs_PACF, label='pacf')
    ax3.plot([xs_pacf_SE] * pacf_max, label='est. upper 95% conf. interval')
    ax3.plot([-xs_pacf_SE] * pacf_max, label='est. lower 95% conf. interval')
    ax3.vlines(np.arange(0, len(xs_PACF)), 0, xs_PACF, color="tab:red")

    if y_label:
        ax1.set_ylabel(y_label)
    ax1.set_xlabel('t')
    ax1.legend()

    ax2.set_ylabel('Estimated corr. coeff.')
    ax2.set_xlabel('t')
    ax2.set_ylim([-1, 1])
    ax2.legend()

    ax3.set_ylabel('Pacf.')
    ax3.set_xlabel('t')
    ax3.legend()

    return fig, (ax1, ax2, ax3)


def learn_model(ts, p_pol, q_pol, p_symbs, q_symbs, bounds=(-1, 1)):
    initial_guess = np.random.random(len(p_symbs) + len(q_symbs))
    initial_guess /= np.sum(initial_guess)

    res = least_squares(tsf.calculateSeasonalARIMA_error_minimization_form_slow, bounds=bounds, x0=initial_guess,
                        args=(ts, p_pol, q_pol, p_symbs, q_symbs, len(p_symbs), len(q_symbs)))

    return res


def plot_res(ts, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=100, ):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ts = ts.astype(float)

    if p_poly is not None:
        est_p_poly: poly = poly(p_poly.subs(dict(zip(p_symbols, p_ARIMA_coeffs))), B)
    else:
        est_p_poly = None
    if q_poly is not None:
        est_q_poly: poly = poly(q_poly.subs(dict(zip(q_symbols, q_ARIMA_coeffs))), B)
    else:
        est_q_poly = None

    print("est_p_poly:", est_p_poly)
    print("est_q_poly:", est_q_poly)

    # remember: use non-differenced series
    model_err = tsf.calculateSeasonalARIMA_error(ts, est_p_poly, d_poly, est_q_poly)

    # step: differencing error, to test if if the noise term is describable by a MA(1) process

    model_err_sec = model_err[:-1] - model_err[1:]

    ax1.plot(model_err_sec, label='consec. differenced residuals')

    rks3 = compute_rks3(model_err_sec)
    res_95 = compute_safe_conf_95(rks3)
    ax2.plot(rks3, label='corellogram')
    ax2.vlines(np.arange(0, len(rks3)), 0, rks3, color="tab:red")
    ax2.plot(res_95, label='est. upper 95% conf. interval, ')
    ax2.plot(-res_95, label='est. lower 95% conf. interval,')

    # remember: pacf calculation can be unstable. Lag_max is a function of convergence: the faster it converges,
    #  the smaller the pacf_max possible.
    ts_pacf = stt.pacf(rks3, nlags=pacf_max)
    ts_pacf_95 = np.sqrt(1 / len(rks3))

    ax3.plot(ts_pacf, label='pacf')
    ax3.plot([ts_pacf_95] * pacf_max, label='est. upper 95% conf. interval')
    ax3.plot([-ts_pacf_95] * pacf_max, label='est. lower 95% conf. interval')
    ax3.vlines(np.arange(0, len(ts_pacf)), 0, ts_pacf, color="tab:red")

    ax1.set_ylabel('Error')
    ax1.set_xlabel('t')
    ax1.legend()

    ax2.set_ylabel('Estimated corr. coeff.')
    ax2.set_xlabel('t')
    ax2.legend()

    ax3.set_ylabel('pacf.')
    ax3.set_xlabel('t')
    ax3.legend()

    return fig, (ax1, ax2, ax3)


if __name__ == '__main__':
    df = pd.read_csv('Airport_Monthly_Operational_Report.csv')
    df = df.sort_values(by=['Month'])
    cargo_total_series = df['Cargo Totals (Cargo + Mail + Belly Freight)'].to_numpy()
    total_op_series = df['Total Operations'].to_numpy()
    total_ppl_series = df['Total Passengers'].to_numpy()

    ###################################################################################################################

    # step: study different series

    # plt.plot(cargo_total_series,label='cargo total')
    # plt.plot(total_op_series,label='operations total')
    plt.plot(total_ppl_series, label='passengers total')
    plt.xlabel("Month")
    plt.legend()
    plt.show()

    # step: stationarize series

    create_series_plot(total_ppl_series, pacf_max=60, corr_max=int(len(total_ppl_series) / 4), ts_is_residuals=False,
                       y_label='No. incidents')
    plt.show()

    tps_diff = total_ppl_series[:-1] - total_ppl_series[1:]
    create_series_plot(tps_diff, pacf_max=60, corr_max=int(len(tps_diff) / 4), ts_is_residuals=False,
                       y_label='No. incidents, diff 1')
    plt.show()

    tps_diff_bi_year = tps_diff[:-6] - tps_diff[6:]
    create_series_plot(tps_diff_bi_year, pacf_max=50, corr_max=int(len(tps_diff_bi_year) / 4), ts_is_residuals=False,
                       y_label='No. incidents, diff 1')
    plt.show()

    tps_diff_month = tps_diff[:-12] - tps_diff[12:]
    create_series_plot(tps_diff_month, pacf_max=50, corr_max=int(len(tps_diff_month) / 4), ts_is_residuals=False,
                       y_label='No. incidents, diff 1')
    plt.show()



    # Comments: It would appear that a differencing of 1,12 is all that is needed.

    ############################################################################################################

    # step: split data
    # Last seven points correspond to 2020, with last 5 points appear to be affected
    total_ppl_series = total_ppl_series[:-7]
    # For testing, we further remove 2 years
    total_ppl_series = total_ppl_series[:-24]

    ts_diff = total_ppl_series[:-1] - total_ppl_series[1:]
    ts_diff_bi_year = ts_diff[:-6] - ts_diff[6:]
    ts_diff_month = ts_diff[:-12] - ts_diff[12:]

    # step: forecast with mul. SARIMA

    p_poly = None
    q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 12, B)
    d_poly = poly(1 - B, B) * poly(1 - B ** 12, B)

    p_symbols = []
    q_symbols = [a, b]

    res = learn_model(ts_diff_month, p_poly, q_poly, p_symbols, q_symbols)

    ARIMA_coeffs = res.x
    p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]

    plot_res(total_ppl_series, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=30)
    plt.show()

    # Comments: The differenced result is almost white noise. However the first lag of the autocorr. is 0.25, not 0.5

    # step: forecast with add. SARIMA

    q_poly = poly(1 - a * B - b * B ** 12, B)

    res = learn_model(ts_diff_month, p_poly, q_poly, p_symbols, q_symbols)

    ARIMA_coeffs = res.x
    p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]

    plot_res(total_ppl_series, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=30)
    plt.show()


    # step: forecast with Holt-winters

    # comment: the trend grows, so it is multiplicative
    hw_model = holtwinters(total_ppl_series, seasonal_periods=12, trend='add')
    hw_model = hw_model.fit()

    h_forecasts = hw_model.forecast(steps=12)
    plt.plot(total_ppl_series)
    plt.plot(np.arange(len(total_ppl_series),len(total_ppl_series)+ len(h_forecasts)), h_forecasts)
    plt.show()
