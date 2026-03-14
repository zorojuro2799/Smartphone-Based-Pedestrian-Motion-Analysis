"""
step_detection.py — Step Detection Algorithms
===============================================
Implements step counting algorithms for pedestrian motion analysis:

    1. Accelerometer-based: peak detection on filtered acceleration magnitude
    2. Gyroscope-based: peak detection on angular velocity magnitude
    3. Sensor fusion: combined accelerometer + gyroscope approach

All methods use adaptive thresholds computed from signal statistics
to handle varying amplitudes across carrying positions and speeds.
"""

import numpy as np
from scipy.signal import find_peaks
from config import FS, STEP_MIN_DISTANCE_S, STEP_HEIGHT_FACTOR, STEP_PROMINENCE_FACTOR


def detect_steps_accel(signal, fs=FS):
    """
    Detect steps from filtered acceleration magnitude using peak detection.

    The algorithm identifies positive peaks in the signal that correspond
    to footstrike events during walking. Three criteria filter valid steps:

    1. Height: peak must exceed mean + 0.35 * std (adaptive to signal level)
    2. Distance: minimum 0.40s between peaks (max ~2.5 steps/s)
    3. Prominence: peak must rise at least 0.20 * std above surroundings

    Parameters:
        signal (np.ndarray): Filtered acceleration magnitude
        fs (float): Sampling frequency (Hz)

    Returns:
        np.ndarray: Indices of detected step peaks
        dict: Peak properties from scipy.signal.find_peaks
    """
    min_distance = int(STEP_MIN_DISTANCE_S * fs)

    sig_mean = np.mean(signal)
    sig_std = np.std(signal)
    min_height = sig_mean + STEP_HEIGHT_FACTOR * sig_std
    min_prominence = STEP_PROMINENCE_FACTOR * sig_std

    peaks, properties = find_peaks(
        signal,
        height=min_height,
        distance=min_distance,
        prominence=min_prominence,
    )

    return peaks, properties


def detect_steps_gyro(signal, fs=FS):
    """
    Detect steps from gyroscope magnitude using peak detection.

    During walking, the phone experiences rotational motion with each
    step. The gyroscope magnitude shows peaks corresponding to the
    leg swing phase of the gait cycle. This provides an independent
    step detection modality.

    Parameters are slightly adjusted for gyroscope characteristics:
    - Lower height factor (0.30) due to smaller signal amplitude
    - Same minimum distance constraint

    Parameters:
        signal (np.ndarray): Filtered gyroscope magnitude (rad/s)
        fs (float): Sampling frequency (Hz)

    Returns:
        np.ndarray: Indices of detected step peaks
        dict: Peak properties
    """
    min_distance = int(STEP_MIN_DISTANCE_S * fs)

    sig_mean = np.mean(signal)
    sig_std = np.std(signal)
    # Gyroscope signals have different characteristics — use adjusted threshold
    # Gyro peaks are less pronounced than accel, especially for pocket position
    min_height = sig_mean + 0.35 * sig_std
    min_prominence = 0.18 * sig_std

    peaks, properties = find_peaks(
        signal,
        height=min_height,
        distance=min_distance,
        prominence=min_prominence,
    )

    return peaks, properties


def detect_steps_fusion(accel_signal, gyro_signal, fs=FS):
    """
    Sensor fusion step detection using both accelerometer and gyroscope.

    Strategy: detect peaks independently from both sensors, then
    validate: a step is counted only if both sensors detect a peak
    within a tolerance window (±0.15s). This reduces false positives
    from noise spikes that affect only one sensor.

    Parameters:
        accel_signal (np.ndarray): Filtered acceleration magnitude
        gyro_signal (np.ndarray): Filtered gyroscope magnitude
        fs (float): Sampling frequency

    Returns:
        np.ndarray: Validated peak indices (from accelerometer)
        dict: Detection statistics
    """
    accel_peaks, _ = detect_steps_accel(accel_signal, fs)
    gyro_peaks, _ = detect_steps_gyro(gyro_signal, fs)

    tolerance_samples = int(0.10 * fs)  # ±100ms matching window (tight)

    # Validate: keep accel peak only if a gyro peak exists nearby
    validated_peaks = []
    for ap in accel_peaks:
        distances = np.abs(gyro_peaks - ap)
        if len(distances) > 0 and np.min(distances) <= tolerance_samples:
            validated_peaks.append(ap)

    stats = {
        'accel_only': len(accel_peaks),
        'gyro_only': len(gyro_peaks),
        'fused': len(validated_peaks),
        'rejected': len(accel_peaks) - len(validated_peaks),
    }

    return np.array(validated_peaks), stats
