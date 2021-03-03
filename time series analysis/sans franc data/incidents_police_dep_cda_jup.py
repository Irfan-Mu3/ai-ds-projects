import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import statsmodels.tsa.stattools as stt
from sympy import poly
from sympy.abc import a, b, B, c, d
import time_series_funcs.sarima_plus_plus as tsf
from incidents_police_dep_eda_jup import compute_rks3, compute_safe_conf_95


# todo: after finding weekly correlate, look at averages of days of the week over the years
# todo: study particular crimes

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
    df = pd.read_csv('Police_Department_Incident_Reports__2018_to_Present.csv')
    df = df.sort_values(by=['Incident Date'])

    # Get only calls related to crime
    temp_df = df[~df['Incident Category'].isin(['Non-Criminal'])]

    # Sum up number of calls per day
    crim_sum_df = temp_df.value_counts(['Incident Date']).reset_index(name='num_incidents').sort_values(
        by=['Incident Date'])

    # Obtain time series of number of calls over a 2 year period
    series_data = crim_sum_df['num_incidents'].to_numpy()
    ############################################################################
    week = 7
    month = 30
    year = 365
    cut_date = 70
    total_series_length = 365 * 2 + cut_date  # warn: the cut date extends the ts enough for the 3rd-order models

    # substep: remove intervention and yearly differencing
    ts_cut = series_data[:total_series_length]
    ts_cut_sec = ts_cut[:-1] - ts_cut[1:]
    ts_cut_year = ts_cut[:-year] - ts_cut[year:]
    ts_cut_week = ts_cut[:-week] - ts_cut[week:]
    ts_cut_month = ts_cut[:-month] - ts_cut[month:]

    # substep: Model 1 - Differencing: Weekly, and Consec
    ts_cut_year_week = ts_cut_year[:-week] - ts_cut_year[week:]
    ts_cut_year_week_sec = ts_cut_year_week[:-1] - ts_cut_year_week[1:]

    # # substep: Model 2
    ts_cut_year_week_month = ts_cut_year_week[:-month] - ts_cut_year_week[month:]
    ts_cut_year_week_month_sec = ts_cut_year_week_month[:-1] - ts_cut_year_week_month[1:]

    ###########################################################################
    # step: Learning models

    # substep: Model 1  (Differencing: sec,weeky & yearly) : SMA_7(1)xMA(1)

    # p_poly = None
    # q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 7, B)
    # d_poly = poly(1 - B, B) * poly(1 - B ** 7, B) * poly(1 - B ** 365, B)
    #
    # p_symbols = []
    # q_symbols = [a, b]
    #
    # res = learn_model(ts_cut_year_week_sec, p_poly, q_poly, p_symbols, q_symbols)
    #
    # ARIMA_coeffs = res.x
    # p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    # q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]
    #
    # plot_res(ts_cut, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=380)
    # plt.show()
    #
    # # Comments: Studying the autocorrelations of the differenced (1) residuals, the process is almost MA(1), with the first
    # # lag a value of 0.4 (in contrast to a true MA(1) autocor. which has a value of 0.5
    # # However, we now see a significant spike at lag 365, suggesting that the error is not truly white noise.
    # # Perhaps an improvement to this model is to use another seasonal parameter - a moving average: SMA_365(0,1,1). Let
    # # us investigate this now.
    #
    # #############################################################################################################
    # # substep: Model 1 + yearly param = Model 1b:  SMA_365(1)xSMA_7(1)xMA(1)
    #
    # # test: extra sec.differencing
    # # d_poly *= poly(1-B,B)
    # # ts_cut_year_week_sec = ts_cut_year_week_sec[:-1] - ts_cut_year_week_sec[1:]
    #
    # # test: extra month differencing
    # # d_poly *= poly(1-B**week,B)
    # # ts_cut_year_week_sec = ts_cut_year_week_sec[:-week] - ts_cut_year_week_sec[week:]
    #
    # p_poly = None
    # q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 7, B) * poly(1 - c * B ** 365, B)
    # d_poly = poly(1 - B, B) * poly(1 - B ** 7, B) * poly(1 - B ** 365, B)
    #
    # p_symbols = []
    # q_symbols = [a, b, c]
    #
    # res = learn_model(ts_cut_year_week_sec, p_poly, q_poly, p_symbols, q_symbols)
    #
    # ARIMA_coeffs = res.x
    # p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    # q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]
    #
    # plot_res(ts_cut, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=380)
    # plt.show()
    #
    # # Comments: The autocorrelation's 1st lag value is around 0.39, and now the 365th lag is muted. However, it is
    # # not perfect. Let us look at what happens, if we instead choose to remove the yearly differencing
    # # and the parameter therefore.
    #
    # #######################################################################################################
    # # substep: Model 1 - yearly differencing, SMA_7(1)xMA(1)
    #
    # ts_cut_week_sec = ts_cut_week[:-1] - ts_cut_week[1:]
    #
    # p_poly = None
    # q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 7, B)
    # d_poly = poly(1 - B, B) * poly(1 - B ** 7, B)
    #
    # p_symbols = []
    # q_symbols = [a, b]
    #
    # res_main = learn_model(ts_cut_week_sec, p_poly, q_poly, p_symbols, q_symbols)
    #
    # ARIMA_coeffs = res_main.x
    # p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    # q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]
    #
    # plot_res(ts_cut, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=380)
    # plt.show()
    #
    # # Comments: We find the process almost an MA(1) having a 1st lag value of an 0.413  for the autocorr. There is no
    # # spike for the 365th value. The autocorr. suggests this is a nicer, and perhaps more parsimonous result.
    # # However, it may be argued that this is artefact due to our model using only two years.
    # # If we had used many years, we might find the earlier model to be better.
    #
    # # Comments: The model above appears to have significant autocorr values on lag 4 (having value 0.118), and 7 (-0.0111). Let us see
    # # if an extra weekly differencing can remove this.
    #
    # #############################################################################################################
    #
    # # substep: Model 1c learning (without yr diff), and extra week diff, SMA_7(1)xMA(1)
    #
    # ts_cut_week_sec = ts_cut_week[:-1] - ts_cut_week[1:]
    # ts_cut_week2_sec = ts_cut_week_sec[:-week] - ts_cut_week_sec[week:]
    #
    # p_poly = None
    # q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 7, B)
    # d_poly = poly(1 - B, B) * poly(1 - B ** 7, B) * poly(1 - B, B)
    #
    # p_symbols = []
    # q_symbols = [a, b]
    #
    # res = learn_model(ts_cut_week2_sec, p_poly, q_poly, p_symbols, q_symbols)
    #
    # ARIMA_coeffs = res.x
    # p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    # q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]
    #
    # plot_res(ts_cut, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=380)
    # plt.show()
    # #
    # # # Comments: Almost an MA(1), but it has around 0.6 as the 1st lag of the autocorr., and the spikes
    # # # at 4,7 have been muted to values 0.085, -0.075 respectively. However, looking at the first lag,
    # # # it may be that this is over-differencing.
    # #
    # # # Comments: We now proceed to learning models with monthly differencing included.
    # # #############################################################################
    # #
    # # # substep: Model 2 (Differencing: sec,weeky,monthly & yearly), SMA_30(1)xSMA_7(1)xMA(1)
    # #
    # p_poly = None
    # q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 7, B) * poly(1 - c * B ** 30, B)
    # d_poly = poly(1 - B, B) * poly(1 - B ** 7, B) * poly(1 - B ** 365, B) * poly(1 - B ** 30, B)
    #
    # p_symbols = []
    # q_symbols = [a, b, c]
    #
    # res = learn_model(ts_cut_year_week_month_sec, p_poly, q_poly, p_symbols, q_symbols)
    #
    # ARIMA_coeffs = res.x
    # p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    # q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]
    #
    # plot_res(ts_cut, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=380)
    # plt.show()
    # #
    # # # Comments: The process is not MA(1), since the autocorr. has spikes at lags 7,30,365.
    # # # Let us remove the yearly differencing and see what happens.
    # #
    # # ########################################################################################
    # #
    # # # substep: Model 2 - yr diff, SMA_30(1)xSMA_7(1)xMA(1)
    # #
    # ts_cut_month_week = ts_cut_month[:-week] - ts_cut_month[week:]
    # ts_cut_month_week_sec = ts_cut_month_week[:-1] - ts_cut_month_week[1:]
    #
    # p_poly = None
    # q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 7, B) * poly(1 - c * B ** 30, B)
    # d_poly = poly(1 - B, B) * poly(1 - B ** 7, B) * poly(1 - B ** 30, B)
    #
    # p_symbols = []
    # q_symbols = [a, b, c]
    #
    # res = learn_model(ts_cut_month_week_sec, p_poly, q_poly, p_symbols, q_symbols)
    #
    # ARIMA_coeffs = res.x
    # p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    # q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]
    #
    # plot_res(ts_cut, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=100)
    # plt.show()
    # #
    # # # Comments: We find similar to before, the spike at the 365 lag removed. There are still spikes
    # # # for the 7th and 30th lag however, but not too large.
    # #
    # # # Let us instead of removing the yearly difference, we introduce a yearly param.
    # #
    # # #############################################################################
    # #
    # # # substep: Model 2 + yearly param,   SMA_365(1)xSMA_30(1)xSMA_7(1)xMA(1)
    # #
    # p_poly = None
    # q_poly = poly(1 - (a * B) - (b * B ** 7) - (c * B ** 30) - (d * B ** 365), B)
    # d_poly = poly(1 - B, B) * poly(1 - B ** 7, B) * poly(1 - B ** 365, B) * poly(1 - B ** 30, B)
    #
    # p_symbols = []
    # q_symbols = [a, b, c, d]
    #
    # res = learn_model(ts_cut_year_week_month_sec, p_poly, q_poly, p_symbols, q_symbols)
    #
    # ARIMA_coeffs = res.x
    # p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    # q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]
    #
    # plot_res(ts_cut, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=100)
    # plt.show()
    # #
    # # # Comments: The autocorr. is far from an MA(1) process. Whilst the 365th lag has been muted, there are spikes
    # # # at lag 1,7-1,7,7+1,365+30.
    # #
    # # # And for our last model (just for studying), we will remove the monthly param, and yearly param.
    # #
    # # #############################################################################################################
    # #
    # # # substep: Model 2 - monthly param, SMA_7(1)xMA(1)
    # #
    # p_poly = None
    # q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 7, B)
    # d_poly = poly(1 - B, B) * poly(1 - B ** 7, B) * poly(1 - B ** 365, B) * poly(1 - B ** 30, B)
    #
    # p_symbols = []
    # q_symbols = [a, b]
    #
    # res = learn_model(ts_cut_year_week_month_sec, p_poly, q_poly, p_symbols, q_symbols)
    #
    # ARIMA_coeffs = res.x
    # p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    # q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]
    #
    # plot_res(ts_cut, p_poly, q_poly, d_poly, p_ARIMA_coeffs, q_ARIMA_coeffs, pacf_max=100)
    # plt.show()

    # Comments: The process is far from an MA(1), due to having spikes at lag 30, and 365 and also at (30-1,30+1), and (365-30,365+30).
    # From earlier, seeing that removing the yearly differencing yields better results than introducing a yearly parameter, it
    # is perhaps the case that we have overdifferenced and that the monthly SMA parameter is excessive.

    ################################################################################################################

    # step: Forecast

    # Let us use the model: Model 1 - yearly differencing, SMA_7(1)xMA(1)
    # And forecast with it

    ts_cut = series_data[:(2 * 365)]
    ts_cross_val = series_data[2 * 365:(2 * 365) + 70]
    ts_cut_week = ts_cut[:-week] - ts_cut[week:]
    ts_cut_week_sec = ts_cut_week[:-1] - ts_cut_week[1:]

    p_poly = None
    q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 7, B)
    d_poly = poly(1 - B, B) * poly(1 - B ** 7, B)

    p_symbols = []
    q_symbols = [a, b]

    res_main = learn_model(ts_cut_week_sec, p_poly, q_poly, p_symbols, q_symbols)

    ARIMA_coeffs = res_main.x
    p_ARIMA_coeffs = ARIMA_coeffs[:len(p_symbols)]
    q_ARIMA_coeffs = ARIMA_coeffs[-len(p_symbols):]

    N = len(ts_cut)
    h_max = 125  # forecast length

    est_q_poly: poly = poly(q_poly.subs(dict(zip(q_symbols, q_ARIMA_coeffs))), B)
    print("est_q_poly:", est_q_poly)

    a_t = tsf.calculateSeasonalARIMA_error(ts_cut.astype(float), None, d_poly, est_q_poly)
    yt_forecasts = tsf.forecastSeasonalARIMA(ts_cut.astype(float), a_t, h_max, None, d_poly, est_q_poly)

    y_multiforecasts, y_stds = tsf.sample_multiforecastSeasonalARIMA(ts_cut.astype(float), a_t, h_max, None, d_poly,
                                                                     est_q_poly, num_samples=500)
    for i in range(y_multiforecasts.shape[0]):
        plt.plot(y_multiforecasts[i], linewidth=0.5, linestyle='--')

    yt_forecast_initial_val = ts_cut[-1]
    plt.plot(yt_forecast_initial_val + y_stds, label='upper std, h = ' + str(h_max), linewidth=2, color='green')
    plt.plot(yt_forecast_initial_val - y_stds, label='lower std, h = ' + str(h_max), linewidth=2, color='green')

    plt.plot(yt_forecast_initial_val + 2 * y_stds, label='upper 2 * std, h = ' + str(h_max), linewidth=2,
             color='orange')
    plt.plot(yt_forecast_initial_val - 2 * y_stds, label='lower 2 * std, h = ' + str(h_max), linewidth=2,
             color='orange')

    plt.plot(yt_forecast_initial_val + 3 * y_stds, label='upper 3 * std, h = ' + str(h_max), linewidth=2, color='red')
    plt.plot(yt_forecast_initial_val - 3 * y_stds, label='lower 3 * std, h = ' + str(h_max), linewidth=2, color='red')

    plt.plot(yt_forecasts, label='forecast average, h = ' + str(h_max), linewidth=3, linestyle=':', color='black')

    plt.legend()
    plt.show()

    # Comments: Proves inadequate for long term forecasting

    ###############################################################################################################

    h = 7
    y_h_steps = tsf.forecastSeasonalARIMA_stepwise_slow(ts_cut.astype(float), h, None, d_poly, est_q_poly, ts_cross_val)

    y_h_multistep, y_h_step_stds, y_h_steps_means = tsf.sample_multiforecastSeasonalARIMA_stepwise_slow(
        ts_cut.astype(float), h, None,
        d_poly, est_q_poly, ts_cross_val,
        num_samples=200)

    ts_full = np.append(ts_cut, ts_cross_val)
    final_idx_ts_cut = len(ts_cut) - 1

    fig, ax = plt.subplots(1)

    for i in range(y_h_multistep.shape[0]):
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] + y_h_step_stds[i], color='green',linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] - y_h_step_stds[i], color='green',linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] + 2 * y_h_step_stds[i], color='orange',linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] - 2 * y_h_step_stds[i], color='orange',linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] + 3 * y_h_step_stds[i], color='red',linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] - 3 * y_h_step_stds[i], color='red',linewidth=2)
        for j in range(y_h_multistep.shape[1]):
            ax.plot(np.arange((i * h), (i + 1) * h), y_h_multistep[i, j], linewidth=0.25, linestyle='--')

    for i in range(y_h_multistep.shape[0]):
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps_means[i], color='hotpink', linewidth=4)

    ax.plot(y_h_steps, label='forecasting with lag:' + str(h), color='black', linewidth=3, linestyle='--')
    ax.plot(ts_cross_val, label='original ts', color='royalblue', linewidth=3)
    plt.legend()
    plt.show()

    # Comments: Appears useful for short-term forecasting

    #############################################################################################################

    h = 2
    y_h_steps = tsf.forecastSeasonalARIMA_stepwise_slow(ts_cut.astype(float), h, None, d_poly, est_q_poly, ts_cross_val)

    y_h_multistep, y_h_step_stds, y_h_steps_means = tsf.sample_multiforecastSeasonalARIMA_stepwise_slow(
        ts_cut.astype(float), h, None,
        d_poly, est_q_poly, ts_cross_val,
        num_samples=50)

    ts_full = np.append(ts_cut, ts_cross_val)
    final_idx_ts_cut = len(ts_cut) - 1

    fig, ax = plt.subplots(1)

    for i in range(y_h_multistep.shape[0]):
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] + y_h_step_stds[i], color='green',linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] - y_h_step_stds[i], color='green',linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] + 2 * y_h_step_stds[i], color='orange',linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] - 2 * y_h_step_stds[i], color='orange',linewidth=2)

        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] + 3 * y_h_step_stds[i], color='red',linewidth=2)
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps[(i * h)] - 3 * y_h_step_stds[i], color='red',linewidth=2)
        for j in range(y_h_multistep.shape[1]):
            plt.plot(np.arange((i * h), (i + 1) * h), y_h_multistep[i, j], linewidth=0.25, linestyle='--')

    for i in range(y_h_multistep.shape[0]):
        ax.plot(np.arange((i * h), (i + 1) * h), y_h_steps_means[i], color='hotpink', linewidth=3)

    ax.plot(y_h_steps, label='forecasting with lag:' + str(h), color='black', linewidth=3, linestyle='--')
    ax.plot(ts_cross_val, label='original ts', color='royalblue', linewidth=3)
    ax.legend()
    plt.show()
