# Smartphone-Based Pedestrian Motion Analysis

**Digital Signal Processing — Project Work**  
SRH Berlin University of Applied Sciences | March 2026

---

## Overview

This project implements a step detection system using smartphone IMU (Inertial Measurement Unit) data. It compares three signal filtering methods — Moving Average, Butterworth, and Kalman — across different walking speeds and phone carrying positions.

## Project Structure

```
DSP_Project/
│
├── main.py              # Pipeline orchestrator — run this
├── config.py            # Constants, ground truth, parameters
├── preprocessing.py     # Data loading, cleaning, magnitude computation
├── filters.py           # Moving Average, Butterworth, Kalman filters
├── step_detection.py    # Accel, Gyro, and Fusion step detection
├── evaluation.py        # Accuracy metrics and result tables
├── visualization.py     # All plotting functions (51 plots)
│
├── data/                # Sensor recordings (CSV, 50 Hz)
│   ├── slow_hand.csv
│   ├── normal_hand.csv
│   ├── fast_hand.csv
│   ├── slow_pocket.csv
│   ├── normal_pocket.csv
│   ├── fast_pocket.csv
│   ├── slow_arm.csv
│   ├── normal_arm.csv
│   ├── fast_arm.csv
│   └── ground_truth.csv
│
├── graphs/              # Generated plots (created by main.py)
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
├── DSP_Project_Report.docx
└── DSP_Project_Report.pdf
```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/DSP_Project.git
cd DSP_Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the analysis
```bash
python main.py
```

This will:
- Load all 9 datasets
- Apply 3 filters to each
- Detect steps using accelerometer, gyroscope, and sensor fusion
- Print results table to console
- Save 51 plots + results CSV to `graphs/`

## Data Collection

- **Device:** Samsung Galaxy A54 (Android 14)
- **App:** Physics Toolbox Suite @ 50 Hz
- **Sensors:** 3-axis Accelerometer (m/s²) + 3-axis Gyroscope (rad/s)
- **Positions:** Hand, Trouser Pocket, Upper Arm
- **Speeds:** Slow (~1.4 Hz), Normal (~1.8 Hz), Fast (~2.2 Hz)

## Key Results

| Filter | Avg Error |
|--------|-----------|
| Butterworth | **2.43%** |
| Moving Average | 3.83% |
| Kalman | 9.74% |
| Sensor Fusion | 0.60% |

**Best carrying position:** Trouser Pocket (1.43% avg error)  
**Best walking speed:** Normal (0.47% avg error)

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy
