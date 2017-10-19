from functools import partial
from os import path
from scipy.optimize import leastsq

import math
import numpy as np
import pandas as pd
import sys

def main():
    """
    python3 feature_extraction.py ~/Datasets/ogle/ogle4/OCVS/smc/rrlyr/RRab.csv ~/Datasets/ogle/ogle4/OCVS/smc/rrlyr/curves/ RRab_smc_colors.csv
    """
    data_file = sys.argv[1]
    curves_dir = sys.argv[2]
    output_file = sys.argv[3]

    data = load_data(data_file)

    data = setup_new_columns(data)

    star_id_col = "id"
    period_col = "period"
    phi31_col = "phi31"

    extract = lambda d: extract_with_curve(curves_dir, star_id_col, period_col, phi31_col, d)
    new_data = data.apply(extract, axis=1)

    new_data.to_csv(output_file, index=False)

    print(new_data.info())

def load_data(data_file):
    data = pd.read_csv(data_file)

    return data

def setup_new_columns(data):
    columns = ["log_p", "metalicity_smol", "metalicity_sand", "phi31_v_skow"
            , "phi31_v_deb", "metalicity_neme", "metalicity_jk_v"
            , "metalicity_jkzw_v", "phi31_i_sine", "V-I_min", "V-I_max"
            , "V_amplitude"
            ]
    data = pd.concat([data, pd.DataFrame(columns=columns)])

    return data

def extract_with_curve(curves_dir, star_id_col, period_col, phi31_col, data):
    """
    Extracts the features from the given star's data with its light curve.

    Parameters
    ----------
    curves_dir : str
        The directory that the curve files are stored in.
    star_id_col : str
        The name of the column containing the star id.
    period_col : str
        The name of the column containing the light curve period.
    phi31_col : str
        The name of the column containing the cosine phi31.
    data : pandas.core.frame.DataFrame
        The exisiting data on the given star.

    Returns
    -------
    new_data : pandas.core.frame.DataFrame
        The existing and extracted information on the given star.
    """
    star_id = data[star_id_col]
    curve_paths = [
        get_curve_path(curves_dir, "I", star_id),
        get_curve_path(curves_dir, "V", star_id)
    ]

    if path.exists(curve_paths[0]) and path.exists(curve_paths[1]):
        curves = [get_curves(p) for p in curve_paths]

        return extract_features(data, period_col, phi31_col, curves)
    else:
        return data

def get_curve_path(curves_dir, band, star_id):
    """
    Returns the file path of the curve in the given curve file directory for
    the given star id.

    Parameters
    ----------
    curves_dir : str
        The directory that the curve files are stored in.
    star_id : str
        The id of the given star.

    Returns
    -------
    curve_path : str
        The file path of the curve file for the given star.
    """
    curve_file = "%s.csv" % star_id
    curve_path = path.join(curves_dir, band, curve_file)

    return curve_path

def get_curves(curve_path):
    """
    Gets the light curve from the file at the specified curve_path.

    Uses a custom csv processing method in order to load in data from files
    faster than `pandas.read_csv`.

    Assumes that the data file follows a csv structure where the columns are
    the time, magnitude, and error in that order.

    Parameters
    ----------
    curve_path : str
        The file path of the curve file for the given star.

    Returns
    -------
    light_curve : numpy.ndarray
        The times, magnitudes, and errors of the light curve.
    """
    with open(curve_path, "r") as f:
        lines = f.read().split("\n")
        parts = [line.split(",")[0:4] for line in lines]

        return np.array(parts[1:-1], dtype="float64")

def extract_features(data, period_col, phi31_col, curves):
    period = data[period_col]
    phi31_i_cosine = data[phi31_col]

    num_obs_I = curves[0].shape[0]
    times_I = curves[0][:,0].reshape(num_obs_I, 1)
    magnitudes_I = curves[0][:,1].reshape(num_obs_I, 1)
    errors_I = curves[0][:,2].reshape(num_obs_I, 1)

    num_obs_V = curves[1].shape[0]
    times_V = curves[1][:,0].reshape(num_obs_V, 1)
    magnitudes_V = curves[1][:,1].reshape(num_obs_V, 1)
    errors_V = curves[1][:,2].reshape(num_obs_V, 1)

    phase_times_I = phase_fold(times_I, period)
    phase_times_V = phase_fold(times_V, period)

    fourier_order = 4

    if num_obs_V >= fourier_order * 2 + 1:
        fourier_coef_I = fourier_decomposition(phase_times_I, magnitudes_I, fourier_order)
        fourier_amplitude_I = fourier_R(fourier_coef_I, 1)

        fourier_coef_V = fourier_decomposition(phase_times_V, magnitudes_V, fourier_order)
        fourier_amplitude_V = fourier_R(fourier_coef_V, 1)

        amplitude_V = np.max(magnitudes_V) - np.min(magnitudes_V)

        phase_times = np.linspace(0.0, 1.0, 1000)
        phase_times = phase_times.reshape(phase_times.size, 1)

        fourier_magnitudes_I = fourier_series(phase_times, fourier_coef_I, fourier_order)
        fourier_magnitudes_V = fourier_series(phase_times, fourier_coef_V, fourier_order)

        phase_max_V_index = np.argmax(fourier_magnitudes_V)
        phase_min_V_index = np.argmin(fourier_magnitudes_V)

        V_I_max = (fourier_magnitudes_V[phase_max_V_index] - fourier_magnitudes_I[phase_max_V_index])[0]
        V_I_min = (fourier_magnitudes_V[phase_min_V_index] - fourier_magnitudes_I[phase_min_V_index])[0]

        phi31_i_sine = phi31_sine_from_cosine(phi31_i_cosine)

        log_p = math.log10(period)

        metalicity_smol = metalicity_smolec(period, phi31_i_sine)
        metalicity_sand = metalicity_sandage(amplitude_V, log_p)

        phi31_v_deb = phi31_sine_from_cosine(
            phi31_v_from_i_deb(phi31_i_cosine))
        phi31_v_skow = phi31_v_from_i_skow(phi31_i_sine)

        metalicity_neme = metalicity_nemec(period, phi31_v_skow)
        metalicity_jk_v = metalicity_jk_v_band(period, phi31_v_skow)
        metalicity_jkzw_v = metalicity_jkzw_v_band(metalicity_jk_v)

        columns = ["log_p", "metalicity_smol", "metalicity_sand", "phi31_v_skow"
            , "phi31_v_deb", "metalicity_neme", "metalicity_jk_v"
            , "metalicity_jkzw_v", "phi31_i_sine", "V-I_max", "V-I_min"
            , "V_amplitude"
            ]

        data[columns] = [log_p, metalicity_smol, metalicity_sand
            , phi31_v_skow, phi31_v_deb, metalicity_neme, metalicity_jk_v
            , metalicity_jkzw_v, phi31_i_sine, V_I_max, V_I_min
            , amplitude_V
            ]

    return data

def phase_fold(times, period):
    """
    Folds the given light curve over its period to express the curve in terms
    of phase rather than time.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    period : numpy.float64
        The light curve period.

    Returns
    -------
    phase_times : numpy.ndarray
        The light curve times in terms of phase.
    """
    phase_times = (times % period) / period

    return phase_times

def fourier_decomposition(times, magnitudes, order):
    """
    Fits the given light curve to a cosine fourier series of the given order
    and returns the fit amplitude and phi weights. The coefficents are
    calculated using a least squares fit.

    The fourier series that is fit is the following:

    n = order
    f(time) = A_0 + sum([A_k * cos(2pi * k * time + phi_k) for k in range(1, n + 1)])

    The fourier coeeficients are returned in a list of the following form:

    [A_0, A_1, phi_1, A_2, phi_2, ...]

    Each of the A coefficients will be positive.

    The number of (time, magnitude) values provided must be greater than or
    equal to the order * 2 + 1. This is a requirement of the least squares
    function used for calculating the coefficients.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    order : int
        The order of the fourier series to fit.

    Returns
    -------
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.
    """
    times = times[:,0]
    magnitudes = magnitudes[:,0]

    num_examples = times.shape[0]
    num_coef = order * 2 + 1

    if num_coef > num_examples:
        raise Exception("Too few examples for the specified order. Number of examples must be at least order * 2 + 1. Required: %d, Actual: %d" % (num_coef, num_examples))

    initial_coef = np.ones(num_coef)

    cost_function = partial(fourier_series_cost, times, magnitudes, order)

    fitted_coef, success = leastsq(cost_function, initial_coef)

    final_coef = correct_coef(fitted_coef, order)

    return final_coef

def correct_coef(coef, order):
    """
    Corrects the amplitudes in the given fourier coefficients so that all of
    them are positive.

    This is done by taking the absolute value of all the negative amplitude
    coefficients and incrementing the corresponding phi weights by pi.

    Parameters
    ----------
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.
    order : int
        The order of the fourier series to fit.

    Returns
    -------
    cor_fourier_coef : numpy.ndarray
        The corrected fit fourier coefficients.
    """
    coef = coef[:]
    for k in range(order):
        i = 2 * k + 1
        if coef[i] < 0.0:
            coef[i] = abs(coef[i])
            coef[i + 1] += math.pi

    return coef

def fourier_series_cost(times, magnitudes, order, coef):
    """
    Returns the error of the fourier series of the given order and coefficients
    in modeling the given light curve.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    order : int
        The order of the fourier series to fit.
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.

    Returns
    -------
    error : numpy.float64
        The error of the fourier series in modeling the curve.
    """
    return magnitudes - fourier_series(times, coef, order)

def fourier_series(times, coef, order):
    """
    Returns the magnitude values given by applying the fourier series described
    by the given order and coefficients to the given time values.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    order : int
        The order of the fourier series to fit.
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.

    Returns
    -------
    magnitudes : numpy.ndarray
        The calculated light curve magnitudes.
    """
    cos_vals = [coef[2 * k + 1] * np.cos(2 * np.pi * (k + 1) * times + coef[2 * k + 2])
            for k in range(order)]
    cos_sum = np.sum(cos_vals, axis=0)

    return coef[0] + cos_sum

def fourier_R(coef, n):
    return coef[2 * (n - 1) + 1]

def phi31_sine_from_cosine(phi31_cosine):
    shifted = phi31_cosine - math.pi
    shifted_ranged = shifted % (2 * math.pi)

    if shifted_ranged < math.pi:
        return shifted_ranged + 2 * math.pi
    else:
        return shifted_ranged

def metalicity_jk_v_band(period, phi31_v):
    """
    Returns the JK96 caclulated metalicity of the given RRab RR Lyrae.

    (Jurcsik and Kovacs, 1996) (3)
    (Szczygiel et al., 2009) (6)
    (Skowron et al., 2016) (1)

    Parameters
    ----------
    period : float64
        The period of the star.
    phi31_v : float64
        The V band phi31 of the star.

    Returns
    -------
    metalicity_jk_v : float64
        The JK96 metalicity of the star.
    """
    return -5.038 - 5.394 * period + 1.345 * phi31_v

def metalicity_jkzw_v_band(metalicity_jk_v):
    """
    Returns the JK96 caclulated metalicity in the Zinn and West system of the
    given RRab RR Lyrae.

    (Sandage, 2004) (1)
    (Szczygiel et al., 2009) (7)

    Parameters
    ----------
    metalicity_jk_v : float64
        The JK96 metalicity of the star.

    Returns
    -------
    metalicity_jkzw_v : float64
        The JK96 metalicity of the star in the Zinn and West system.
    """
    return 1.05 * metalicity_jk_v - 0.20

def metalicity_sandage(amplitude_v, log_p):
    """
    Returns the Sandage formula metalicity of the given RRab RR Lyrae.

    Note that this formula "produces an unphysically bimodal distribution of
    photometric metalicities, which is a reflection of the Oosterhoff
    dichotomy". (Skowron et al., 2016)

    (Sandage, 2004) (7)
    (Szczygiel et al., 2009) (8)
    (Skowron et al., 2016) (2)

    Parameters
    ----------
    amplitude_v : float64
        The V band amplitude of the star.
    log_p : float64
        The log10 scaled period of the star.

    Returns
    -------
    metalicity_sand : float64
        The Sandage formula metalicity of the star.
    """
    return -1.453 * amplitude_v - 7.990 * log_p - 2.145

def metalicity_smolec(period, phi31_i):
    """
    Returns the Smolec formula metalicity of the given RRab RR Lyrae.

    (Smolec, 2005) (2)
    (Skowron et al., 2016) (3)

    Parameters
    ----------
    period : float64
        The period of the star.
    phi31_i : float64
        The I band phi31 of the star.

    Returns
    -------
    metalicity_smol : float64
        The Smolec formula metalicity of the star.
    """
    return -3.142 - 4.902 * period + 0.824 * phi31_i

def metalicity_nemec(period, phi31_v):
    """
    Returns the Nemec formula metalicity of the given RRab RR Lyrae.

    The version of the formula used is from Skowron et al. (2016), where I band
    phi31 is used, as opposed to the original usage of Kp-passband phi31.

    (Nemec et al., 2013) (2)
    (Skowron et al., 2016) (5)

    Parameters
    ----------
    period : float64
        The period of the star.
    phi31_v : float64
        The V band phi31 of the star.

    Returns
    -------
    metalicity_neme : float64
        The Nemec formula metalicity of the star.
    """
    return -7.63 - 39.03 * period + 5.71 * phi31_v + 6.27 * phi31_v * period \
            - 0.72 * phi31_v ** 2

def phi31_v_from_i_deb(phi31_i):
    """
    Calculates the V band phi31 for the given I band phi3 using the
    relationship found in Deb and Singh (2010).

    (Deb and Singh, 2010) (Table 3)
    (Skowron et al., 2016) (4)

    Parameters
    ----------
    phi31_i : float64
        The I band phi31 of the star.

    Returns
    -------
    phi31_v : float64
        The calculated V band phi31 of the star.
    """
    return 0.568 * phi31_i + 0.436

def phi31_v_from_i_skow(phi31_i):
    """
    Calculates the V band phi31 for the given I band phi3 using the
    relationship found in Skowron et al. (2016).

    (Skowron et al., 2016) (6)

    Parameters
    ----------
    phi31_i : float64
        The I band phi31 of the star.

    Returns
    -------
    phi31_v : float64
        The calculated V band phi31 of the star.
    """
    return 0.122 * phi31_i ** 2 - 0.750 * phi31_i + 5.331

if __name__ == "__main__":
    main()
