import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import statsmodels.tsa.stattools as stt
from sympy import poly
from sympy.abc import a, b, B, c, d
import time_series_funcs.sarima_plus_plus as tsf

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


if __name__ == '__main__':
    df = pd.read_csv('Police_Department_Incident_Reports__2018_to_Present.csv')
    df = df.sort_values(by=['Incident Date'])

    # Get only calls related to crime
    temp_df = df[~df['Incident Category'].isin(['Non-Criminal'])]

    # Sum up number of calls per day
    crim_sum_df = temp_df.value_counts(['Incident Date']).reset_index(name='num_incidents').sort_values(
        by=['Incident Date'])

    # Obtain time series of number of calls over a 2 year period
    series_data = crim_sum_df['num_incidents'].to_numpy()

    ######################################################################################

    plt.plot(series_data)
    plt.xlabel("Days since 1st January 2018")
    plt.ylabel("No. incidents")
    plt.show()

    ######################################################################
    # step: Plot each year on top of each other, and see if there is a relationship

    split = 365
    series_data_chunks = [series_data[i:i + split] for i in range(0, len(series_data), split)]
    roll_order = 30

    for i in range(len(series_data_chunks)):
        N = len(series_data_chunks[i])
        plt.plot(np.arange(0, N), series_data_chunks[i], label=str(2018 + i), linestyle=':')
        plt.plot(compute_rolling_mean(series_data_chunks[i], roll_order),
                 label='rolling ' + str(roll_order) + ', ' + str(2018 + i))

    plt.ylabel('No. incidents')
    plt.xlabel('Days')
    plt.legend()
    plt.show()

    # Comments: Yearly seasonal, lockdown intervention.
    # Comments: First plot original study, then proceed to removing both, and studying relationships

    ######################################################################################
    # step: Plot ORIGINAL time-series, autocorrelation, and partial autocorr.

    # remember: original series, and rolling mean of 30 days

    create_series_plot(series_data, pacf_max=100, corr_max=int(len(series_data) / 4), ts_is_residuals=False,
                       y_label='No. incidents')
    plt.show()
    # Comments: Low convergence, implies non-stationarity caused by the intervention (lockdown for Covid)

    #######################################################################
    # step: Cutting out non-stationary data (lockdown intervention), and yearly differencing

    week = 7
    month = 30
    year = 365  # warn: setting this to 364 yields different autocorrelation peaks, hiding weekly autocorrs.
    cut_date = 70
    total_series_length = 365 * 2 + cut_date  # warn: the cut date extends the ts enough for the 3rd-order models

    # substep: remove intervention and yearly differencing
    ts_cut = series_data[:total_series_length]
    ts_cut_year = ts_cut[:-year] - ts_cut[year:]
    ts_cut_week = ts_cut[:-week] - ts_cut[week:]
    ts_cut_month = ts_cut[:-month] - ts_cut[month:]

    #############
    plt.plot(ts_cut, label='cut ts')
    plt.legend()
    plt.show()
    #############

    _, (a1, a2, a3) = create_series_plot(ts_cut, pacf_max=100, corr_max=int(len(series_data) / 4),
                                         ts_is_residuals=False,
                                         y_label='No. incidents',
                                         ts_label='ts with intervention period removed')

    # substep: let us plot also the correlogram of the ts that also has yearly differencing applied
    rks3 = compute_rks3(ts_cut_year)
    ts_95 = compute_corr_95(rks3)

    a2.plot(rks3, label='corellogram with yr diff. ', )
    a2.vlines(np.arange(0, len(rks3)), 0, rks3, color="tab:blue")
    a2.plot(ts_95, label='est. upper 95% conf. interval (with yr diff.)')
    a2.plot(-ts_95, label='est. lower 95% conf. interval (with yr diff.)')
    a2.legend()

    plt.show()
    # Comments: Appears to have peaks every 7th day, which implies seasonality

    ##############################################################################################
    # step: We study two differencing schemes: 1)  weekly and consec., 2) weekly, monthly, and consec.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 2)

    # substep: Model 1 - Differencing: Weekly, and Consec
    ts_cut_year_week = ts_cut_year[:-week] - ts_cut_year[week:]
    ts_cut_year_week_sec = ts_cut_year_week[:-1] - ts_cut_year_week[1:]

    ax1[0].plot(ts_cut_year_week_sec, label='ts with diff:1,7,365')
    ax1[0].plot(compute_rolling_mean(ts_cut_year_week_sec, 30), label='rolling 30 of ts', linewidth=4)

    rks3 = compute_rks3(ts_cut_year_week_sec)
    ts_95 = compute_corr_95(rks3)

    ax2[0].plot(rks3, label='corellogram via rk3')
    ax2[0].vlines(np.arange(0, len(rks3)), 0, rks3, color="tab:red")
    ax2[0].plot(ts_95, label='upper SE, weekly')
    ax2[0].plot(-ts_95, label='lower SE, weekly')

    lag_max = 100
    ts_pacf = stt.pacf(rks3, nlags=lag_max)
    ts_pacf_95 = np.sqrt(1 / len(rks3))

    ax3[0].plot(ts_pacf, label='pacf')
    ax3[0].plot([ts_pacf_95] * lag_max, label='upper SE')
    ax3[0].plot([-ts_pacf_95] * lag_max, label='lower SE')
    ax3[0].vlines(np.arange(0, len(ts_pacf)), 0, ts_pacf, color="tab:red")

    ax1[0].set_ylabel('No. incidents')
    ax1[0].set_xlabel('t')
    ax1[0].legend()

    ax2[0].set_ylabel('Estimated corr. coeff.')
    ax2[0].set_xlabel('t')
    ax2[0].set_xlim([0, 400])
    ax2[0].legend()

    ax3[0].set_ylabel('pacf.')
    ax3[0].set_xlabel('t')
    ax3[0].legend()

    # # substep: Model 2

    ts_cut_year_week_month = ts_cut_year_week[:-month] - ts_cut_year_week[month:]
    ts_cut_year_week_month_sec = ts_cut_year_week_month[:-1] - ts_cut_year_week_month[1:]

    ax1[1].plot(ts_cut_year_week_month_sec, label='ts with diff:1,7,30,365')
    ax1[1].plot(compute_rolling_mean(ts_cut_year_week_month_sec, 30),
                label='rolling 30 of ts', linewidth=4)

    rks3 = compute_rks3(ts_cut_year_week_month_sec)
    ts_95 = compute_corr_95(rks3)

    ax2[1].plot(rks3, label='corellogram via rk3')
    ax2[1].vlines(np.arange(0, len(rks3)), 0, rks3, color="tab:red")
    ax2[1].plot(ts_95, label='upper SE')
    ax2[1].plot(-ts_95, label='lower SE')

    lag_max = 100
    ts_pacf = stt.pacf(rks3, nlags=lag_max)
    ts_pacf_95 = np.sqrt(1 / len(rks3))

    ax3[1].plot(ts_pacf, label='pacf')
    ax3[1].vlines(np.arange(0, len(ts_pacf)), 0, ts_pacf, color="tab:red")
    ax3[1].plot([ts_pacf_95] * lag_max, label='upper SE')
    ax3[1].plot([-ts_pacf_95] * lag_max, label='lower SE')

    ax1[1].set_ylabel('No. incidents')
    ax1[1].set_xlabel('t')
    ax1[1].legend()

    ax2[1].set_ylabel('Estimated corr. coeff.')
    ax2[1].set_xlabel('t')
    ax2[1].set_xlim([0, 400])
    ax2[1].legend()

    ax3[1].set_ylabel('pacf.')
    ax3[1].set_xlabel('t')
    ax3[1].legend()
    plt.show()

    # Comments: Model one appears to be atleast a SMA(1)xMA(1) model, whilst Model 2 at least a SMA_7(1)xSMA_30(1)XMA(1)
    # For both, it does not appear that a 365 model is neccessary
    # Note that differencing by year and not having a param for the year lag is fine, since
    # having a non-stationary series is not a problem. However, params-wise, we need to difference the series.
    # Thus for model 1: we have a random walk, at the level of years, but a SMA process, for week, and MA for consec.

