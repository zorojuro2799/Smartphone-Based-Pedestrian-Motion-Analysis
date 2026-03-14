"""
filters.py — Signal Filtering Methods
=======================================
Implements three digital signal processing filters for noise
reduction in IMU-based pedestrian motion analysis:

    1. Moving Average Filter (FIR)
    2. Butterworth Low-Pass Filter (IIR, zero-phase)
    3. Kalman Filter (recursive state-space estimator)

All filters use uniform parameters defined in config.py to
ensure fair comparison across datasets.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from config import FS, MOVING_AVG_WINDOW, BUTTERWORTH_CUTOFF, BUTTERWORTH_ORDER, \
                   KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE


def moving_average_filter(signal, window_size=MOVING_AVG_WINDOW):
    """
    Moving Average Filter — simple FIR smoothing.

    Replaces each sample with the average of its N nearest neighbors.
    Effective for high-frequency noise but introduces lag and can
    attenuate sharp peaks.

    Transfer function: H(z) = (1/N) * sum(z^(-k)), k=0..N-1

    Parameters:
        signal (np.ndarray): Input signal
        window_size (int): Averaging window (must be odd)

    Returns:
        np.ndarray: Smoothed signal (same length)
    """
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')


def butterworth_filter(signal, cutoff=BUTTERWORTH_CUTOFF, fs=FS,
                       order=BUTTERWORTH_ORDER):
    """
    Butterworth Low-Pass Filter — maximally flat passband.

    4th-order IIR filter with zero-phase implementation via
    scipy.signal.filtfilt (forward-backward filtering).

    Cutoff at 3.0 Hz preserves gait fundamental (1.4–2.2 Hz) and
    first harmonic while rejecting noise above ~4 Hz.

    Parameters:
        signal (np.ndarray): Input signal
        cutoff (float): Cutoff frequency (Hz)
        fs (float): Sampling frequency (Hz)
        order (int): Filter order

    Returns:
        np.ndarray: Filtered signal (zero-phase)
    """
    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def kalman_filter(signal, process_noise=KALMAN_PROCESS_NOISE,
                  measurement_noise=KALMAN_MEASUREMENT_NOISE):
    """
    1-D Kalman Filter — optimal recursive state estimator.

    Models the signal using a constant-velocity state-space model:
        State: x = [position, velocity]^T
        Transition: x_k = F * x_{k-1} + w   (w ~ N(0, Q))
        Observation: z_k = H * x_k + v       (v ~ N(0, R))

    The Kalman gain K adapts over time to balance prediction trust
    (low Q) against measurement trust (low R).

    Parameters:
        signal (np.ndarray): Input measurements
        process_noise (float): Q scaling factor
        measurement_noise (float): R value

    Returns:
        np.ndarray: Filtered state estimates
    """
    n = len(signal)
    dt = 1.0 / FS

    # State: [position, velocity]
    x = np.array([signal[0], 0.0])
    P = np.eye(2)

    # System matrices
    F = np.array([[1.0, dt], [0.0, 1.0]])           # State transition
    Q = process_noise * np.array([                    # Process noise
        [dt**3 / 3, dt**2 / 2],
        [dt**2 / 2, dt]
    ])
    H = np.array([[1.0, 0.0]])                       # Observation
    R = np.array([[measurement_noise]])               # Measurement noise
    I = np.eye(2)

    filtered = np.zeros(n)

    for i in range(n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        z = signal[i]
        y = z - (H @ x_pred)[0]                      # Innovation
        S = (H @ P_pred @ H.T + R)[0, 0]             # Innovation covariance
        K = (P_pred @ H.T) / S                        # Kalman gain

        x = x_pred + K.flatten() * y
        P = (I - K @ H) @ P_pred

        filtered[i] = x[0]

    return filtered


def apply_all_filters(signal):
    """
    Apply all three filters to an input signal.

    Parameters:
        signal (np.ndarray): Raw acceleration magnitude

    Returns:
        dict: {'ma': ..., 'bw': ..., 'kf': ...} filtered signals
    """
    return {
        'ma': moving_average_filter(signal),
        'bw': butterworth_filter(signal),
        'kf': kalman_filter(signal),
    }
