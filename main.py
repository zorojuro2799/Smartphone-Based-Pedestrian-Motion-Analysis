"""
main.py — DSP Project
=========================================
Smartphone-Based Pedestrian Motion Analysis

Orchestrates the complete analysis workflow:
    1. Load and preprocess all datasets
    2. Apply three signal filtering methods
    3. Detect steps using accelerometer, gyroscope, and fusion
    4. Evaluate accuracy against ground truth
    5. Generate all report-ready visualizations
    6. Export results to CSV

Usage:
    python main.py

Project Structure:
    config.py          — Constants, parameters, ground truth
    preprocessing.py   — Data loading, cleaning, magnitude computation
    filters.py         — Moving Average, Butterworth, Kalman filters
    step_detection.py  — Peak-based step detection algorithms
    evaluation.py      — Accuracy metrics and result tables
    visualization.py   — All plotting functions
    main.py            — This file (pipeline orchestrator)

Author : [Srivardhan Varma / Nani deepak ]
Date   : March 2026
Course : Digital Signal Processing
"""

import os
import numpy as np

from config import DATASETS, GROUND_TRUTH, GRAPHS_DIR
from preprocessing import load_dataset, compute_accel_magnitude, compute_gyro_magnitude
from filters import apply_all_filters, butterworth_filter
from step_detection import detect_steps_accel, detect_steps_gyro, detect_steps_fusion
from evaluation import evaluate, build_results_dataframe, print_results_table
from visualization import (
    plot_raw_sensor_data, plot_filter_comparison, plot_step_detection_detail,
    plot_gyro_vs_accel, plot_fusion_comparison,
    plot_accuracy_summary, plot_error_comparison,
    plot_position_comparison, plot_speed_comparison,
    plot_frequency_analysis, plot_error_heatmap,
)


def analyse_dataset(name):
    """
    Run the complete analysis pipeline on a single dataset.

    Pipeline:
        1. Load CSV, compute accel & gyro magnitudes
        2. Apply all three filters to accelerometer signal
        3. Apply Butterworth to gyroscope signal
        4. Detect steps: raw, MA, BW, KF, gyro, fusion
        5. Evaluate all against ground truth

    Returns:
        dict: Signals, peaks, and evaluation metrics
    """
    df, quality = load_dataset(name)
    t = df['time_s'].values

    accel_mag = compute_accel_magnitude(df)
    gyro_mag = compute_gyro_magnitude(df)

    filtered = apply_all_filters(accel_mag)
    gyro_bw = butterworth_filter(gyro_mag, cutoff=4.0)

    peaks_raw, _ = detect_steps_accel(accel_mag)
    peaks_ma, _  = detect_steps_accel(filtered['ma'])
    peaks_bw, _  = detect_steps_accel(filtered['bw'])
    peaks_kf, _  = detect_steps_accel(filtered['kf'])
    peaks_gyro, _ = detect_steps_gyro(gyro_bw)
    peaks_fusion, fusion_stats = detect_steps_fusion(filtered['bw'], gyro_bw)

    gt = GROUND_TRUTH[name]

    return {
        'name': name, 'df': df, 't': t, 'quality': quality,
        'mag': accel_mag, 'gyro_mag': gyro_mag,
        'ma_sig': filtered['ma'], 'bw_sig': filtered['bw'], 'kf_sig': filtered['kf'],
        'gyro_bw': gyro_bw,
        'peaks_raw': peaks_raw, 'peaks_ma': peaks_ma, 'peaks_bw': peaks_bw,
        'peaks_kf': peaks_kf, 'peaks_gyro': peaks_gyro, 'peaks_fusion': peaks_fusion,
        'fusion_stats': fusion_stats,
        'eval_raw':    evaluate(len(peaks_raw), gt),
        'eval_ma':     evaluate(len(peaks_ma), gt),
        'eval_bw':     evaluate(len(peaks_bw), gt),
        'eval_kf':     evaluate(len(peaks_kf), gt),
        'eval_gyro':   evaluate(len(peaks_gyro), gt),
        'eval_fusion': evaluate(len(peaks_fusion), gt),
    }


if __name__ == '__main__':
    print("=" * 65)
    print("  DSP Project — Smartphone-Based Pedestrian Motion Analysis")
    print("  Pipeline: Load -> Filter -> Detect -> Evaluate -> Visualize")
    print("=" * 65)

    all_results = []

    print("\n[1/5] Per-dataset analysis...")
    for name in DATASETS:
        print(f"  Processing: {name}")
        res = analyse_dataset(name)
        all_results.append(res)
        plot_raw_sensor_data(name, res['df'])
        plot_filter_comparison(res)
        plot_step_detection_detail(res)

    print("\n[2/5] Gyroscope & sensor fusion analysis...")
    for name in ['normal_hand', 'normal_pocket', 'fast_hand']:
        res = next(r for r in all_results if r['name'] == name)
        plot_gyro_vs_accel(res)

    print("\n[3/5] Frequency analysis...")
    for name in ['normal_hand', 'normal_pocket', 'fast_hand']:
        res = next(r for r in all_results if r['name'] == name)
        plot_frequency_analysis(res)

    print("\n[4/5] Summary plots...")
    plot_accuracy_summary(all_results)
    plot_error_comparison(all_results)
    plot_position_comparison(all_results)
    plot_speed_comparison(all_results)
    plot_error_heatmap(all_results)
    plot_fusion_comparison(all_results)

    print("\n[5/5] Results...")
    print_results_table(all_results)

    df_results = build_results_dataframe(all_results)
    csv_path = os.path.join(GRAPHS_DIR, 'results_table.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n    [saved] {csv_path}")

    n_plots = len([f for f in os.listdir(GRAPHS_DIR) if f.endswith('.png')])
    print(f"\n{'='*65}")
    print(f"  COMPLETE: {n_plots} plots + CSV saved to: {GRAPHS_DIR}/")
    print(f"{'='*65}")
