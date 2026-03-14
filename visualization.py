"""
visualization.py — Report-Ready Plot Generation
=================================================
Generates all visualizations for the DSP project report.
Includes raw sensor data, filter comparisons, step detection,
gyroscope analysis, accuracy summaries, and comparison charts.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import GRAPHS_DIR, COLORS, GROUND_TRUTH, FILTER_NAMES, FS

# ─── Plot Configuration ──────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
})


def savefig(name):
    path = os.path.join(GRAPHS_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    [saved] {path}")


def fmt(name):
    """Format dataset name: 'normal_hand' -> 'Normal Hand'."""
    return name.replace('_', ' ').title()


# ═════════════════════════════════════════════════════
# PLOT 1: Raw Sensor Data (3-axis accel + gyro)
# ═════════════════════════════════════════════════════

def plot_raw_sensor_data(name, df):
    t = df['time_s'].values
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle(f'Raw Sensor Data — {fmt(name)}', fontsize=13, fontweight='bold', y=0.98)

    axes[0].plot(t, df['acc_x'], label='X (forward)', alpha=0.85, lw=0.7, color='#1565C0')
    axes[0].plot(t, df['acc_y'], label='Y (lateral)', alpha=0.85, lw=0.7, color='#E65100')
    axes[0].plot(t, df['acc_z'], label='Z (vertical)', alpha=0.85, lw=0.7, color='#2E7D32')
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)
    axes[0].set_title('Accelerometer', fontsize=10, fontweight='bold')

    axes[1].plot(t, df['gyro_x'], label='X (pitch)', alpha=0.85, lw=0.7, color='#C62828')
    axes[1].plot(t, df['gyro_y'], label='Y (roll)', alpha=0.85, lw=0.7, color='#6A1B9A')
    axes[1].plot(t, df['gyro_z'], label='Z (yaw)', alpha=0.85, lw=0.7, color='#F57F17')
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right', fontsize=8, framealpha=0.9)
    axes[1].set_title('Gyroscope', fontsize=10, fontweight='bold')

    plt.tight_layout()
    savefig(f'01_raw_sensor_{name}.png')


# ═════════════════════════════════════════════════════
# PLOT 2: Filter Comparison (4-panel)
# ═════════════════════════════════════════════════════

def plot_filter_comparison(res):
    name, t, gt = res['name'], res['t'], GROUND_TRUTH[res['name']]
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f'Filter Comparison — {fmt(name)}', fontsize=14, fontweight='bold', y=0.99)

    panels = [
        ('Raw Signal',     res['mag'],    COLORS['raw'],         res['peaks_raw'], res['eval_raw']),
        ('Moving Average', res['ma_sig'], COLORS['moving_avg'],  res['peaks_ma'],  res['eval_ma']),
        ('Butterworth',    res['bw_sig'], COLORS['butterworth'], res['peaks_bw'],  res['eval_bw']),
        ('Kalman Filter',  res['kf_sig'], COLORS['kalman'],      res['peaks_kf'],  res['eval_kf']),
    ]

    for ax, (label, sig, color, peaks, ev) in zip(axes, panels):
        ax.plot(t, sig, color=color, lw=0.7, alpha=0.9, label=label)
        if len(peaks) > 0:
            ax.plot(t[peaks], sig[peaks], 'v', color='#212121', ms=4, alpha=0.7,
                    label=f'Detected: {ev["detected"]} steps')
        ax.set_ylabel('Magnitude\n(m/s²)', fontsize=8)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
        color_title = '#D32F2F' if ev['pct_error'] > 5 else '#1B5E20'
        ax.set_title(f'{label}  |  Detected: {ev["detected"]}  |  GT: {gt}  |  Error: {ev["pct_error"]:.1f}%',
                     fontsize=9, fontweight='bold', color=color_title)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    savefig(f'02_filter_comparison_{name}.png')


# ═════════════════════════════════════════════════════
# PLOT 3: Step Detection Detail
# ═════════════════════════════════════════════════════

def plot_step_detection_detail(res):
    name, t = res['name'], res['t']
    sig, peaks = res['bw_sig'], res['peaks_bw']
    gt = GROUND_TRUTH[name]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, sig, color=COLORS['butterworth'], lw=0.8, alpha=0.9, label='Butterworth Filtered')
    if len(peaks) > 0:
        ax.plot(t[peaks], sig[peaks], 'v', color='#D32F2F', ms=7, zorder=5,
                label=f'Detected Steps: {len(peaks)}')
    threshold = np.mean(sig) + 0.35 * np.std(sig)
    ax.axhline(threshold, color='#9E9E9E', ls='--', lw=0.8, alpha=0.7,
               label=f'Threshold ({threshold:.2f} m/s²)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accel Magnitude (m/s²)')
    ax.set_title(f'Step Detection (Butterworth) — {fmt(name)}  |  GT: {gt}  |  Det: {len(peaks)}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    savefig(f'03_step_detection_{name}.png')


# ═════════════════════════════════════════════════════
# PLOT 10: Gyroscope vs Accelerometer Comparison
# ═════════════════════════════════════════════════════

def plot_gyro_vs_accel(res):
    """Side-by-side accelerometer and gyroscope magnitude with detected peaks."""
    name, t = res['name'], res['t']
    gt = GROUND_TRUTH[name]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f'Accelerometer vs Gyroscope Analysis — {fmt(name)}',
                 fontsize=13, fontweight='bold', y=0.99)

    # Panel 1: Butterworth-filtered accel
    bw_sig = res['bw_sig']
    bw_peaks = res['peaks_bw']
    axes[0].plot(t, bw_sig, color=COLORS['butterworth'], lw=0.7, alpha=0.9)
    if len(bw_peaks) > 0:
        axes[0].plot(t[bw_peaks], bw_sig[bw_peaks], 'v', color='#D32F2F', ms=5)
    axes[0].set_ylabel('Accel Mag (m/s²)')
    axes[0].set_title(f'Accelerometer — Butterworth Filtered  |  Detected: {len(bw_peaks)} steps',
                      fontsize=10, fontweight='bold')

    # Panel 2: Butterworth-filtered gyro
    gyro_bw = res.get('gyro_bw')
    gyro_peaks = res.get('peaks_gyro', np.array([]))
    if gyro_bw is not None:
        axes[1].plot(t, gyro_bw, color=COLORS['gyro'], lw=0.7, alpha=0.9)
        if len(gyro_peaks) > 0:
            axes[1].plot(t[gyro_peaks], gyro_bw[gyro_peaks], 'v', color='#D32F2F', ms=5)
        axes[1].set_title(f'Gyroscope — Butterworth Filtered  |  Detected: {len(gyro_peaks)} steps',
                          fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Gyro Mag (rad/s)')

    # Panel 3: Overlay normalized signals
    if gyro_bw is not None:
        accel_norm = (bw_sig - np.mean(bw_sig)) / max(np.std(bw_sig), 1e-6)
        gyro_norm = (gyro_bw - np.mean(gyro_bw)) / max(np.std(gyro_bw), 1e-6)
        axes[2].plot(t, accel_norm, color=COLORS['butterworth'], lw=0.7, alpha=0.7, label='Accelerometer (norm.)')
        axes[2].plot(t, gyro_norm, color=COLORS['gyro'], lw=0.7, alpha=0.7, label='Gyroscope (norm.)')
        axes[2].set_title('Normalized Overlay — Signal Correlation', fontsize=10, fontweight='bold')
        axes[2].legend(fontsize=8, framealpha=0.9)
    axes[2].set_ylabel('Normalized')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    savefig(f'10_gyro_vs_accel_{name}.png')


# ═════════════════════════════════════════════════════
# PLOT 11: Sensor Fusion Results
# ═════════════════════════════════════════════════════

def plot_fusion_comparison(all_results):
    """Compare accel-only, gyro-only, and fusion step counts."""
    names = [r['name'] for r in all_results]
    labels = [n.replace('_', '\n') for n in names]
    n = len(names)

    gt_vals = [GROUND_TRUTH[name] for name in names]
    accel_vals = [r['eval_bw']['detected'] for r in all_results]
    gyro_vals = [r.get('eval_gyro', {}).get('detected', 0) for r in all_results]
    fusion_vals = [r.get('eval_fusion', {}).get('detected', 0) for r in all_results]

    x = np.arange(n)
    w = 0.18

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - 1.5*w, gt_vals, w, label='Ground Truth', color=COLORS['gt'], alpha=0.85)
    ax.bar(x - 0.5*w, accel_vals, w, label='Accel (Butterworth)', color=COLORS['butterworth'], alpha=0.85)
    ax.bar(x + 0.5*w, gyro_vals, w, label='Gyroscope', color=COLORS['gyro'], alpha=0.85)
    ax.bar(x + 1.5*w, fusion_vals, w, label='Fusion (Accel+Gyro)', color='#FF6F00', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Step Count')
    ax.set_title('Sensor Comparison — Accel vs Gyro vs Fusion', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(gt_vals) * 1.25)
    plt.tight_layout()
    savefig('11_fusion_comparison.png')


# ═════════════════════════════════════════════════════
# PLOT 4-9: Summary plots (same as before but cleaner)
# ═════════════════════════════════════════════════════

def plot_accuracy_summary(all_results):
    labels = [r['name'].replace('_', '\n') for r in all_results]
    n = len(labels)
    gt = [GROUND_TRUTH[r['name']] for r in all_results]
    det_ma = [r['eval_ma']['detected'] for r in all_results]
    det_bw = [r['eval_bw']['detected'] for r in all_results]
    det_kf = [r['eval_kf']['detected'] for r in all_results]

    x = np.arange(n)
    w = 0.18
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x-1.5*w, gt, w, label='Ground Truth', color=COLORS['gt'], alpha=0.85)
    ax.bar(x-0.5*w, det_ma, w, label='Moving Average', color=COLORS['moving_avg'], alpha=0.85)
    ax.bar(x+0.5*w, det_bw, w, label='Butterworth', color=COLORS['butterworth'], alpha=0.85)
    ax.bar(x+1.5*w, det_kf, w, label='Kalman', color=COLORS['kalman'], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Step Count')
    ax.set_title('Step Detection Accuracy — All Datasets & Filters', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9); ax.set_ylim(0, max(gt)*1.25)
    plt.tight_layout(); savefig('04_accuracy_summary_all.png')


def plot_error_comparison(all_results):
    labels = [r['name'].replace('_', '\n') for r in all_results]
    n = len(labels)
    err_ma = [r['eval_ma']['pct_error'] for r in all_results]
    err_bw = [r['eval_bw']['pct_error'] for r in all_results]
    err_kf = [r['eval_kf']['pct_error'] for r in all_results]

    x = np.arange(n); w = 0.25
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.bar(x-w, err_ma, w, label='Moving Average', color=COLORS['moving_avg'], alpha=0.85)
    ax.bar(x, err_bw, w, label='Butterworth', color=COLORS['butterworth'], alpha=0.85)
    ax.bar(x+w, err_kf, w, label='Kalman', color=COLORS['kalman'], alpha=0.85)
    for i in range(n):
        for off, val in [(-w, err_ma[i]), (0, err_bw[i]), (w, err_kf[i])]:
            if val > 0: ax.text(x[i]+off, val+0.15, f'{val:.1f}%', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Step Count Error (%)')
    ax.set_title('Error Percentage by Filter — All Datasets', fontsize=13, fontweight='bold')
    ax.axhline(5, color='#D32F2F', ls='--', alpha=0.6, lw=1.2, label='5% Threshold')
    ax.legend(fontsize=9); ax.set_ylim(0, max(max(err_ma), max(err_bw), max(err_kf))*1.5+1)
    plt.tight_layout(); savefig('05_error_by_filter.png')


def plot_position_comparison(all_results):
    from collections import OrderedDict
    res_map = {r['name']: r for r in all_results}
    groups = OrderedDict([
        ('Slow Walking', ['slow_hand','slow_pocket','slow_arm']),
        ('Normal Walking', ['normal_hand','normal_pocket','normal_arm']),
        ('Fast Walking', ['fast_hand','fast_pocket','fast_arm']),
    ])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle('Position Comparison — Butterworth Filter', fontsize=13, fontweight='bold')
    for ax, (title, names) in zip(axes, groups.items()):
        labels = [n.split('_')[1].title() for n in names]
        gt_v = [GROUND_TRUTH[n] for n in names]
        det_v = [res_map[n]['eval_bw']['detected'] for n in names]
        err_v = [res_map[n]['eval_bw']['pct_error'] for n in names]
        x = np.arange(len(labels))
        ax.bar(x-0.2, gt_v, 0.35, label='Ground Truth', color=COLORS['gt'], alpha=0.85)
        ax.bar(x+0.2, det_v, 0.35, label='Detected', color=COLORS['butterworth'], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontweight='bold'); ax.set_ylabel('Steps'); ax.legend(fontsize=8)
        for xi, (e, d) in enumerate(zip(err_v, det_v)):
            ax.text(xi+0.2, d+1, f'{e:.1f}%', ha='center', fontsize=8, fontweight='bold',
                    color='#D32F2F' if e > 5 else '#1B5E20')
    plt.tight_layout(); savefig('06_position_comparison.png')


def plot_speed_comparison(all_results):
    from collections import OrderedDict
    res_map = {r['name']: r for r in all_results}
    groups = OrderedDict([
        ('Hand', ['slow_hand','normal_hand','fast_hand']),
        ('Pocket', ['slow_pocket','normal_pocket','fast_pocket']),
        ('Arm', ['slow_arm','normal_arm','fast_arm']),
    ])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle('Speed Comparison — Butterworth Filter', fontsize=13, fontweight='bold')
    for ax, (title, names) in zip(axes, groups.items()):
        labels = [n.split('_')[0].title() for n in names]
        gt_v = [GROUND_TRUTH[n] for n in names]
        det_v = [res_map[n]['eval_bw']['detected'] for n in names]
        err_v = [res_map[n]['eval_bw']['pct_error'] for n in names]
        x = np.arange(len(labels))
        ax.bar(x-0.2, gt_v, 0.35, label='Ground Truth', color=COLORS['gt'], alpha=0.85)
        ax.bar(x+0.2, det_v, 0.35, label='Detected', color=COLORS['butterworth'], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f'{title} Position', fontweight='bold'); ax.set_ylabel('Steps'); ax.legend(fontsize=8)
        for xi, (e, d) in enumerate(zip(err_v, det_v)):
            ax.text(xi+0.2, d+1, f'{e:.1f}%', ha='center', fontsize=8, fontweight='bold',
                    color='#D32F2F' if e > 5 else '#1B5E20')
    plt.tight_layout(); savefig('07_speed_comparison.png')


def plot_frequency_analysis(res):
    name, mag, bw_sig = res['name'], res['mag'], res['bw_sig']
    n = len(mag)
    freqs = np.fft.rfftfreq(n, d=1.0/FS)
    fft_raw = np.abs(np.fft.rfft(mag)) / n
    fft_filt = np.abs(np.fft.rfft(bw_sig)) / n

    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    fig.suptitle(f'Frequency Analysis — {fmt(name)}', fontsize=13, fontweight='bold')
    axes[0].plot(freqs, fft_raw, color=COLORS['raw'], lw=0.8, alpha=0.8)
    axes[0].set_ylabel('Amplitude'); axes[0].set_title('Raw Signal Spectrum', fontsize=10)
    axes[0].set_xlim(0, 15)
    axes[0].axvline(3.0, color='#D32F2F', ls='--', alpha=0.5, label='Cutoff (3.0 Hz)')
    axes[0].legend(fontsize=8)
    axes[1].plot(freqs, fft_filt, color=COLORS['butterworth'], lw=0.8, alpha=0.8)
    axes[1].set_ylabel('Amplitude'); axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_title('Butterworth Filtered Spectrum', fontsize=10); axes[1].set_xlim(0, 15)
    axes[1].axvline(3.0, color='#D32F2F', ls='--', alpha=0.5, label='Cutoff'); axes[1].legend(fontsize=8)
    plt.tight_layout(); savefig(f'08_frequency_analysis_{name}.png')


def plot_error_heatmap(all_results):
    datasets = [r['name'] for r in all_results]
    filters = ['Moving Average', 'Butterworth', 'Kalman']
    data = np.zeros((len(datasets), len(filters)))
    for i, r in enumerate(all_results):
        data[i, 0] = r['eval_ma']['pct_error']
        data[i, 1] = r['eval_bw']['pct_error']
        data[i, 2] = r['eval_kf']['pct_error']
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=10)
    ax.set_xticks(np.arange(len(filters))); ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(filters, fontsize=10)
    ax.set_yticklabels([fmt(d) for d in datasets], fontsize=9)
    for i in range(len(datasets)):
        for j in range(len(filters)):
            v = data[i, j]
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if v > 5 else 'black')
    ax.set_title('Step Detection Error (%) — All Combinations', fontsize=13, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='Error (%)', shrink=0.8)
    plt.tight_layout(); savefig('09_error_heatmap.png')
