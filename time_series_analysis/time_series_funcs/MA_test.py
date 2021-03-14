import numpy as np
import matplotlib.pyplot as plt
from sympy import poly
from sympy.abc import B, a, b, c, d
from scipy.optimize import minimize, basinhopping, brute, least_squares

from sarima_plus_plus import sampleSeasonalARIMA, calculateSeasonalARIMA_error, \
    calculateSeasonalARIMA_error_minimization_form_slow, \
    full_form_squared, difference_series

if __name__ == '__main__':
    p_poly = None
    q_poly = poly(1 - a * B, B) * poly(1 - b * B ** 4, B) * poly(1 - c * B ** 30, B)
    d_poly = poly(1 - B, B)

    q_symbols = [a, b, c]
    q_ARIMA_coeffs = [0.6, 0.2, 0.1]
    est_p_poly = None
    est_q_poly: poly = poly(q_poly.subs(dict(zip(q_symbols, q_ARIMA_coeffs))), B)

    y_t = sampleSeasonalARIMA(est_p_poly, d_poly, est_q_poly, 1, 1000)
    plt.plot(y_t, label="sampled ARIMA", marker='x')

    a_t_error = calculateSeasonalARIMA_error(y_t, est_p_poly, d_poly, est_q_poly)
    plt.plot(a_t_error, label='sampled ARIMA error', marker='.', markevery=10)

    print("est_q_poly", est_q_poly)
    print("Expected value of shock a_t", np.mean(a_t_error))
    print("sd of shock a_t", np.sqrt(np.var(a_t_error)))
    # remember: don't square error to determine the sd of a_t

    plt.legend()
    plt.show()
    ###################################################################

    myx0 = np.random.random(len(q_ARIMA_coeffs))
    myx0 /= np.sum(myx0)

    w_t = difference_series(y_t, d_poly)

    res = minimize(full_form_squared, x0=myx0,
                   args=(w_t, p_poly, q_poly, [], q_symbols, 0, len(q_ARIMA_coeffs)),
                   method='Nelder-Mead', options={'maxiter': 5000, 'disp': True})
    print("estimated params (via minimize):", res.df)
    print("true params:", q_ARIMA_coeffs)
    print("initial guess:", myx0)

    # warn: lm method does not allow bounds, therefore good initial guess is needed (?) (i.e. satisfying conditions like inveritibility)
    res = least_squares(calculateSeasonalARIMA_error_minimization_form_slow, bounds=(-2, 2), x0=myx0,
                        args=(w_t, p_poly, q_poly, [], q_symbols, 0, len(q_ARIMA_coeffs)),
                        )
    print("estimated params (via least_squares):", res.x)
    print("true params:", q_ARIMA_coeffs)
    print("initial guess:", myx0)
