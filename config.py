"""
config.py — Project Configuration
===================================
Central configuration for the DSP Pedestrian Motion Analysis project.
Contains all constants, dataset definitions, ground truth values, and
shared settings used across the analysis pipeline.

Data Collection:
    App: Physics Toolbox Suite (Android/iOS)
    Device: Samsung Galaxy A54 (Android 14)
    Sampling Rate: 50 Hz
    Sensors: 3-axis Accelerometer (m/s²), 3-axis Gyroscope (rad/s)
"""

import os
from collections import OrderedDict

# ─── Paths ────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
GRAPHS_DIR = os.path.join(os.path.dirname(__file__), 'graphs')
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ─── Sensor Parameters ────────────────────────────────
FS = 50.0           # Sampling frequency (Hz) — Physics Toolbox Suite setting
GRAVITY = 9.81      # Gravitational acceleration constant (m/s²)
DT = 1.0 / FS       # Sampling interval (seconds)

# ─── Ground Truth ─────────────────────────────────────
# Manually counted steps for each measurement session.
# Counting was performed by a second person using a
# hand-held tally counter during each recording.
GROUND_TRUTH = OrderedDict([
    ('slow_hand',      60),
    ('normal_hand',    74),
    ('fast_hand',      83),
    ('slow_pocket',    62),
    ('normal_pocket',  72),
    ('fast_pocket',    79),
    ('slow_arm',       59),
    ('normal_arm',     74),
    ('fast_arm',       84),
])

DATASETS = list(GROUND_TRUTH.keys())

# ─── Filter Parameters (uniform across all datasets) ──
MOVING_AVG_WINDOW = 11       # samples (~0.22 s at 50 Hz)
BUTTERWORTH_CUTOFF = 3.0     # Hz
BUTTERWORTH_ORDER = 4
KALMAN_PROCESS_NOISE = 0.01
KALMAN_MEASUREMENT_NOISE = 0.4

# ─── Step Detection Parameters ────────────────────────
STEP_MIN_DISTANCE_S = 0.40   # minimum seconds between steps
STEP_HEIGHT_FACTOR = 0.35    # threshold = mean + factor * std
STEP_PROMINENCE_FACTOR = 0.20

# ─── Plot Styling ─────────────────────────────────────
COLORS = {
    'raw':         '#78909C',
    'moving_avg':  '#1976D2',
    'butterworth': '#388E3C',
    'kalman':      '#E64A19',
    'gyro':        '#7B1FA2',
    'gt':          '#455A64',
}

FILTER_NAMES = OrderedDict([
    ('ma', 'Moving Average'),
    ('bw', 'Butterworth'),
    ('kf', 'Kalman'),
])
