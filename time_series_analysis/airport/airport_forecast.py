import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as stt
from matplotlib import pyplot as plt
from sympy import poly
from sympy.abc import a, b, B

import time_series_funcs.sarima_plus_plus as spp


# import time_series_funcs.holt_winters_plus_plus as hwpp

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


def plot_residuals(ts, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=100, ):
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
    model_err = spp.calculateSeasonalARIMA_error(ts, est_p_poly, d_poly, est_q_poly)

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
    # cargo_total_series = df['Cargo Totals (Cargo + Mail + Belly Freight)'].to_numpy()
    # total_op_series = df['Total Operations'].to_numpy()
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
    # total_ppl_series = np.log(total_ppl_series) # remember: for stability of variance

    create_series_plot(total_ppl_series, pacf_max=60, corr_max=int(len(total_ppl_series) / 4), ts_is_residuals=False,
                       ts_label='Passengers total')
    plt.legend()
    plt.show()

    # remember: need to remove intervention to determine autocorr properly
    #  Last seven points correspond to 2020, with last 5 points appear to be affected
    orig_total_ppl_series = total_ppl_series
    total_ppl_series = total_ppl_series[:-7]

    tps_diff = total_ppl_series[:-1] - total_ppl_series[1:]
    create_series_plot(tps_diff, pacf_max=50, corr_max=int(len(tps_diff) / 4), ts_is_residuals=False,
                       ts_label='Passengers total, diff 1')
    plt.legend()
    plt.show()

    tps_diff_bi_year = tps_diff[:-6] - tps_diff[6:]
    create_series_plot(tps_diff_bi_year, pacf_max=35, corr_max=int(len(tps_diff_bi_year) / 4), ts_is_residuals=False,
                       ts_label='Passengers total, diff 6')
    plt.legend()
    plt.show()

    tps_diff_month = tps_diff[:-12] - tps_diff[12:]
    create_series_plot(tps_diff_month, pacf_max=35, corr_max=int(len(tps_diff_month) / 4), ts_is_residuals=False,
                       ts_label='Passengers total, diff 12')
    plt.legend()
    plt.show()

    # Comments: It would appear that a differencing of 1,12 is all that is needed.

    ############################################################################################################

    # step: split data

    # For testing, we further remove 2 years
    total_ppl_series = total_ppl_series[:-24]

    ts_diff = total_ppl_series[:-1] - total_ppl_series[1:]
    ts_diff_year = ts_diff[:-12] - ts_diff[12:]

    # step: forecast with mul. SARIMA

    p_poly = None
    q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 12, B)
    d_poly = poly(1 - B, B) * poly(1 - B ** 12, B)

    p_symbols = []
    q_symbols = [a, b]

    res = spp.learn_model(ts_diff_year, p_poly, q_poly, p_symbols, q_symbols)

    ARIMA_coeffs = res.x
    p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]

    plot_residuals(total_ppl_series, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=30)
    plt.show()

    #####
    h = 24
    est_q_poly: poly = poly(q_poly.subs(dict(zip(q_symbols, q_ARIMA_coeffs))), B)

    y_h_pieces = spp.stepwise_forecastSeasonalARIMA(total_ppl_series.astype(float), h, None, d_poly, est_q_poly,
                                                    orig_total_ppl_series[-31:])

    y_h_multistep, y_h_step_stds, y_h_steps_means = spp.batch_stepwise_forecastSeasonalARIMA(
        total_ppl_series.astype(float), h, None,
        d_poly, est_q_poly, orig_total_ppl_series[-31:],
        num_samples=400)

    fig, ax = plt.subplots(1)

    for i in range(y_h_multistep.shape[0]):
        for j in range(y_h_multistep.shape[1]):
            ax.plot(np.arange((i * h), (i + 1) * h), y_h_multistep[i, j], linewidth=0.25, linestyle='--')
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + y_h_step_stds[i], color='green',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - y_h_step_stds[i], color='green',
                linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + 2 * y_h_step_stds[i], color='orange',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - 2 * y_h_step_stds[i], color='orange',
                linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + 3 * y_h_step_stds[i], color='red',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - 3 * y_h_step_stds[i], color='red',
                linewidth=2)

    for i in range(y_h_multistep.shape[0]):
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps_means[i], color='hotpink', linewidth=4)

    ax.plot(y_h_pieces, label='forecasting with lag:' + str(h), color='black', linewidth=3, linestyle='--')
    ax.plot(orig_total_ppl_series[-31:], label='original ts', color='royalblue', linewidth=3)
    plt.legend()
    plt.show()

    # Comments: The differenced result is almost white noise. However the first lag of the autocorr. is 0.25, not 0.5

    # step: forecast with add. SARIMA

    q_poly = poly(1 - a * B - b * B ** 12, B)

    res = spp.learn_model(ts_diff_year, p_poly, q_poly, p_symbols, q_symbols)

    ARIMA_coeffs = res.x
    p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]

    plot_residuals(total_ppl_series, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=30)
    plt.show()

    ###
    h = 24
    est_q_poly: poly = poly(q_poly.subs(dict(zip(q_symbols, q_ARIMA_coeffs))), B)

    y_h_pieces = spp.stepwise_forecastSeasonalARIMA(total_ppl_series.astype(float), h, None, d_poly, est_q_poly,
                                                    orig_total_ppl_series[-31:])

    y_h_multistep, y_h_step_stds, y_h_steps_means = spp.batch_stepwise_forecastSeasonalARIMA(
        total_ppl_series.astype(float), h, None,
        d_poly, est_q_poly, orig_total_ppl_series[-31:],
        num_samples=400)

    fig, ax = plt.subplots(1)

    for i in range(y_h_multistep.shape[0]):
        for j in range(y_h_multistep.shape[1]):
            ax.plot(np.arange((i * h), (i + 1) * h), y_h_multistep[i, j], linewidth=0.25, linestyle='--')
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + y_h_step_stds[i], color='green',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - y_h_step_stds[i], color='green',
                linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + 2 * y_h_step_stds[i], color='orange',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - 2 * y_h_step_stds[i], color='orange',
                linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + 3 * y_h_step_stds[i], color='red',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - 3 * y_h_step_stds[i], color='red',
                linewidth=2)

    for i in range(y_h_multistep.shape[0]):
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps_means[i], color='hotpink', linewidth=4)

    ax.plot(y_h_pieces, label='forecasting with lag:' + str(h), color='black', linewidth=3, linestyle='--')
    ax.plot(orig_total_ppl_series[-31:], label='original ts', color='royalblue', linewidth=3)
    plt.legend()
    plt.show()
    ####################################################################################################################

    # step: see if other series can be blind forecasted with (mul) airline model :

    q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 12, B)

    # substep: cargo totals

    orig_cargo_total_series = df['Cargo Totals (Cargo + Mail + Belly Freight)'].to_numpy()
    cargo_total_series = orig_cargo_total_series[:-(7 + 24)]

    res = spp.learn_model(ts_diff_year, p_poly, q_poly, p_symbols, q_symbols)

    ARIMA_coeffs = res.x
    p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]

    plot_residuals(cargo_total_series, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=30)
    plt.show()

    ###
    h = 24
    est_q_poly: poly = poly(q_poly.subs(dict(zip(q_symbols, q_ARIMA_coeffs))), B)

    y_h_pieces = spp.stepwise_forecastSeasonalARIMA(cargo_total_series.astype(float), h, None, d_poly, est_q_poly,
                                                    orig_cargo_total_series[-31:])

    y_h_multistep, y_h_step_stds, y_h_steps_means = spp.batch_stepwise_forecastSeasonalARIMA(
        cargo_total_series.astype(float), h, None,
        d_poly, est_q_poly, orig_cargo_total_series[-31:],
        num_samples=400)

    fig, ax = plt.subplots(1)

    for i in range(y_h_multistep.shape[0]):
        for j in range(y_h_multistep.shape[1]):
            ax.plot(np.arange((i * h), (i + 1) * h), y_h_multistep[i, j], linewidth=0.25, linestyle='--')
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + y_h_step_stds[i], color='green',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - y_h_step_stds[i], color='green',
                linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + 2 * y_h_step_stds[i], color='orange',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - 2 * y_h_step_stds[i], color='orange',
                linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + 3 * y_h_step_stds[i], color='red',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - 3 * y_h_step_stds[i], color='red',
                linewidth=2)

    for i in range(y_h_multistep.shape[0]):
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps_means[i], color='hotpink', linewidth=4)

    ax.plot(y_h_pieces, label='forecasting with lag:' + str(h), color='black', linewidth=3, linestyle='--')
    ax.plot(orig_cargo_total_series[-31:], label='original ts', color='royalblue', linewidth=3)
    plt.legend()
    plt.show()

    # Comments: The fit isn't perfect. Looking at the residuals, it appears there is a correlation to study.

    ###################################################################################################################

    # substep: Total operations

    orig_total_op_series = df['Total Operations'].to_numpy()
    total_op_series = orig_total_op_series[:-(7 + 24)]

    res = spp.learn_model(ts_diff_year, p_poly, q_poly, p_symbols, q_symbols)

    ARIMA_coeffs = res.x
    p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]

    plot_residuals(total_op_series, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=30)
    plt.show()

    ###
    h = 24
    est_q_poly: poly = poly(q_poly.subs(dict(zip(q_symbols, q_ARIMA_coeffs))), B)

    y_h_pieces = spp.stepwise_forecastSeasonalARIMA(total_op_series.astype(float), h, None, d_poly, est_q_poly,
                                                    orig_total_op_series[-31:])

    y_h_multistep, y_h_step_stds, y_h_steps_means = spp.batch_stepwise_forecastSeasonalARIMA(
        total_op_series.astype(float), h, None,
        d_poly, est_q_poly, orig_total_op_series[-31:],
        num_samples=400)

    fig, ax = plt.subplots(1)

    for i in range(y_h_multistep.shape[0]):
        for j in range(y_h_multistep.shape[1]):
            ax.plot(np.arange((i * h), (i + 1) * h), y_h_multistep[i, j], linewidth=0.25, linestyle='--')
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + y_h_step_stds[i], color='green',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - y_h_step_stds[i], color='green',
                linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + 2 * y_h_step_stds[i], color='orange',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - 2 * y_h_step_stds[i], color='orange',
                linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] + 3 * y_h_step_stds[i], color='red',
                linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_pieces[(i * h):(i + 1) * h] - 3 * y_h_step_stds[i], color='red',
                linewidth=2)

    for i in range(y_h_multistep.shape[0]):
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps_means[i], color='hotpink', linewidth=4)

    ax.plot(y_h_pieces, label='forecasting with lag:' + str(h), color='black', linewidth=3, linestyle='--')
    ax.plot(orig_total_op_series[-31:], label='original ts', color='royalblue', linewidth=3)
    plt.legend()
    plt.show()

    # Comments: This appears to be well-fit.

    ###################################################################################################################

    # step: forecast with Holt-winters mult

    # # comment: the trend grows, so it is multiplicative
    # res_mul = hwpp.learn_mul_hw(total_ppl_series)
    # print("holt-winter mul est. params:", res_mul.x)
    #
    # _, est_a_t = hwpp.mul_hw_error(total_ppl_series, *(res_mul.x), )
    # est_ts_mhw, _ = hwpp.sample_mul_hw(len(total_ppl_series), *(res_mul.x), e_t=est_a_t, )
    # mul_forecast, _ = hwpp.forecast_mul_hw(est_a_t, *res_mul.x,period=12,h=24+7,random=False)
    #
    # plt.plot(total_ppl_series, label='orig', linestyle=':', linewidth=5)
    # plt.plot(est_ts_mhw, label='est es.', linestyle=':', )
    # plt.plot(mul_forecast,label='forecast')
    # plt.plot(orig_total_ppl_series,label='orig series')
    # plt.plot(est_a_t,label='est error',alpha=0.5)
    #
    # # plt.plot(est_a_t, label='estimated err')
    # plt.legend()
    # plt.show()
    #
    # # step: forecast with Holt-winters add
    #
    # res_add = hwpp.learn_add_hw(total_ppl_series)
    # print("holt-winter mul est. params:", res_add.x)
    #
    # _, est_a_t = hwpp.add_hw_error(total_ppl_series, *(res_add.x), )
    # est_ts_ahw, _ = hwpp.sample_add_hw(len(total_ppl_series), *(res_add.x), e_t=est_a_t, )
    # add_forecast, _ = hwpp.forecast_add_hw(est_a_t, *res_add.x,period=12,h=24+7,random=False)
    #
    # plt.plot(total_ppl_series, label='orig', linestyle=':', linewidth=5)
    # plt.plot(est_ts_ahw, label='est es.', linestyle=':', )
    # plt.plot(add_forecast,label='forecast')
    # plt.plot(orig_total_ppl_series,label='orig series')
    # plt.plot(est_a_t,label='est error',alpha=0.5)
    #
    # # plt.plot(est_a_t, label='estimated err')
    # plt.legend()
    # plt.show()

    # h_forecasts = hw_model.forecast(steps=12)
    # plt.plot(total_ppl_series)
    # plt.plot(np.arange(len(total_ppl_series),len(total_ppl_series)+ len(h_forecasts)), h_forecasts)
    # plt.show()
