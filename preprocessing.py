"""
preprocessing.py — Data Loading and Preparation
=================================================
Handles CSV import, data cleaning, synchronization checks,
and computation of derived signals (acceleration magnitude,
gyroscope magnitude).
"""

import os
import numpy as np
import pandas as pd
from config import DATA_DIR, FS, GRAVITY


def load_dataset(name):
    """
    Load a CSV sensor recording and perform basic preprocessing.

    The Physics Toolbox Suite app exports CSV files with columns:
        time_s, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

    Preprocessing steps:
        1. Remove duplicate timestamps (sensor glitches)
        2. Check for missing samples (timestamp gaps)
        3. Verify data integrity

    Parameters:
        name (str): Dataset identifier (e.g., 'normal_hand')

    Returns:
        pd.DataFrame: Cleaned sensor data
        dict: Data quality metrics
    """
    path = os.path.join(DATA_DIR, f'{name}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Quality metrics
    quality = {'name': name, 'raw_samples': len(df)}

    # Remove duplicate timestamps
    n_before = len(df)
    df = df.drop_duplicates(subset='time_s').reset_index(drop=True)
    quality['duplicates_removed'] = n_before - len(df)

    # Check for missing samples (gaps > 1.5x expected interval)
    dt_expected = 1.0 / FS
    dt_actual = np.diff(df['time_s'].values)
    n_gaps = int(np.sum(dt_actual > 1.5 * dt_expected))
    quality['gaps_detected'] = n_gaps

    # Basic statistics
    quality['duration_s'] = round(df['time_s'].iloc[-1] - df['time_s'].iloc[0], 2)
    quality['clean_samples'] = len(df)
    quality['actual_fs'] = round(len(df) / quality['duration_s'], 1)

    return df, quality


def compute_accel_magnitude(df):
    """
    Compute orientation-independent acceleration magnitude.

    Combines three-axis accelerometer data using Euclidean norm,
    then removes the static gravitational component. The resulting
    signal reflects only the dynamic acceleration from walking.

    Formula: magnitude = sqrt(ax² + ay² + az²) - g

    Parameters:
        df (pd.DataFrame): Sensor data with acc_x, acc_y, acc_z

    Returns:
        np.ndarray: Dynamic acceleration magnitude (m/s²)
    """
    mag = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    return mag.values - GRAVITY


def compute_gyro_magnitude(df):
    """
    Compute total angular velocity magnitude from gyroscope.

    The gyroscope magnitude captures the overall rotational motion
    of the phone during walking. Peaks in gyroscope magnitude
    correspond to the swing phases of the gait cycle.

    Formula: gyro_mag = sqrt(gx² + gy² + gz²)

    Parameters:
        df (pd.DataFrame): Sensor data with gyro_x, gyro_y, gyro_z

    Returns:
        np.ndarray: Angular velocity magnitude (rad/s)
    """
    return np.sqrt(
        df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2
    ).values


def get_speed_from_name(name):
    """Extract walking speed category from dataset name."""
    if name.startswith('slow'):
        return 'slow'
    elif name.startswith('fast'):
        return 'fast'
    return 'normal'


def get_position_from_name(name):
    """Extract carrying position from dataset name."""
    return name.split('_', 1)[1]
