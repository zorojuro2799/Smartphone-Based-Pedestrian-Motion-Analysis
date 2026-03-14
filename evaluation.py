"""
evaluation.py — Step Detection Evaluation
==========================================
Computes accuracy metrics by comparing detected step counts
against manually counted ground truth values. Generates
summary statistics across datasets, filters, and positions.
"""

import numpy as np
import pandas as pd
from config import GROUND_TRUTH, FILTER_NAMES


def evaluate(detected, ground_truth):
    """
    Compute step detection accuracy metrics.

    Parameters:
        detected (int): Number of detected steps
        ground_truth (int): Manually counted steps

    Returns:
        dict: error, abs_error, pct_error, detected, ground_truth
    """
    error = detected - ground_truth
    abs_error = abs(error)
    pct_error = abs_error / ground_truth * 100 if ground_truth > 0 else 0
    return {
        'detected': detected,
        'ground_truth': ground_truth,
        'error': error,
        'abs_error': abs_error,
        'pct_error': round(pct_error, 1),
    }


def build_results_dataframe(all_results):
    """
    Compile all results into a structured DataFrame for analysis.

    Parameters:
        all_results (list[dict]): Per-dataset analysis results

    Returns:
        pd.DataFrame: Columns include dataset, speed, position, filter,
                      ground_truth, detected, error, pct_error
    """
    rows = []
    for r in all_results:
        name = r['name']
        speed = name.split('_')[0]
        position = name.split('_', 1)[1]
        gt = GROUND_TRUTH[name]

        for fkey, fname in [('eval_raw', 'Raw'), ('eval_ma', 'Moving Average'),
                            ('eval_bw', 'Butterworth'), ('eval_kf', 'Kalman')]:
            rows.append({
                'Dataset': name,
                'Speed': speed.title(),
                'Position': position.title(),
                'Filter': fname,
                'Ground_Truth': gt,
                'Detected': r[fkey]['detected'],
                'Error': r[fkey]['error'],
                'Abs_Error': r[fkey]['abs_error'],
                'Error_Pct': r[fkey]['pct_error'],
            })

        # Gyroscope-based results
        if 'eval_gyro' in r:
            rows.append({
                'Dataset': name, 'Speed': speed.title(), 'Position': position.title(),
                'Filter': 'Gyroscope', 'Ground_Truth': gt,
                'Detected': r['eval_gyro']['detected'],
                'Error': r['eval_gyro']['error'],
                'Abs_Error': r['eval_gyro']['abs_error'],
                'Error_Pct': r['eval_gyro']['pct_error'],
            })

        # Fusion results
        if 'eval_fusion' in r:
            rows.append({
                'Dataset': name, 'Speed': speed.title(), 'Position': position.title(),
                'Filter': 'Fusion (Accel+Gyro)', 'Ground_Truth': gt,
                'Detected': r['eval_fusion']['detected'],
                'Error': r['eval_fusion']['error'],
                'Abs_Error': r['eval_fusion']['abs_error'],
                'Error_Pct': r['eval_fusion']['pct_error'],
            })

    return pd.DataFrame(rows)


def print_results_table(all_results):
    """Print formatted results to console."""
    print("\n" + "=" * 105)
    print(f"{'Dataset':<18} {'GT':>4} | "
          f"{'MA':>4} {'Err%':>6} | {'BW':>4} {'Err%':>6} | "
          f"{'KF':>4} {'Err%':>6} | {'Raw':>4} {'Err%':>6} | "
          f"{'Gyro':>4} {'Err%':>6}")
    print("=" * 105)

    for r in all_results:
        gt = GROUND_TRUTH[r['name']]
        gyro_det = r.get('eval_gyro', {}).get('detected', '-')
        gyro_err = r.get('eval_gyro', {}).get('pct_error', '-')
        gyro_str = f"{gyro_det:>4} {gyro_err:>5.1f}%" if isinstance(gyro_err, float) else f"{'':>4} {'N/A':>6}"

        print(f"{r['name']:<18} {gt:>4} | "
              f"{r['eval_ma']['detected']:>4} {r['eval_ma']['pct_error']:>5.1f}% | "
              f"{r['eval_bw']['detected']:>4} {r['eval_bw']['pct_error']:>5.1f}% | "
              f"{r['eval_kf']['detected']:>4} {r['eval_kf']['pct_error']:>5.1f}% | "
              f"{r['eval_raw']['detected']:>4} {r['eval_raw']['pct_error']:>5.1f}% | "
              f"{gyro_str}")

    print("=" * 105)

    # Averages
    print("\n  Average Error by Filter:")
    for key, fname in [('eval_ma', 'Moving Average'), ('eval_bw', 'Butterworth'),
                       ('eval_kf', 'Kalman'), ('eval_raw', 'Raw (unfiltered)')]:
        avg = np.mean([r[key]['pct_error'] for r in all_results])
        print(f"    {fname:<25}: {avg:.2f}%")

    if 'eval_gyro' in all_results[0]:
        avg_gyro = np.mean([r['eval_gyro']['pct_error'] for r in all_results])
        print(f"    {'Gyroscope-based':<25}: {avg_gyro:.2f}%")
    if 'eval_fusion' in all_results[0]:
        avg_fusion = np.mean([r['eval_fusion']['pct_error'] for r in all_results])
        print(f"    {'Fusion (Accel+Gyro)':<25}: {avg_fusion:.2f}%")

    # By position
    print("\n  Average Error by Position (Butterworth):")
    for pos in ['hand', 'pocket', 'arm']:
        vals = [r['eval_bw']['pct_error'] for r in all_results if pos in r['name']]
        print(f"    {pos.title():<25}: {np.mean(vals):.2f}%")

    # By speed
    print("\n  Average Error by Speed (Butterworth):")
    for spd in ['slow', 'normal', 'fast']:
        vals = [r['eval_bw']['pct_error'] for r in all_results if r['name'].startswith(spd)]
        print(f"    {spd.title():<25}: {np.mean(vals):.2f}%")
