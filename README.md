# AI-ML Coursework Group 123
## Accelerometer-Based Authentication System

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-green.svg)]()

## Project Overview

This project implements an **advanced accelerometer-based biometric authentication system** that leverages motion sensor data for user identification and verification. The system utilizes accelerometer and gyroscope sensor readings to create unique behavioral biometric profiles for each user through sophisticated signal processing, statistical feature extraction, PCA dimensionality reduction, and neural network classification.

### Key Objectives
- Develop a robust user authentication system using accelerometer data
- Implement advanced signal processing and machine learning techniques
- Achieve high accuracy in distinguishing between authorized and unauthorized users
- Create a comprehensive analysis and visualization pipeline

## Quick Start

### Prerequisites
- **MATLAB R2023** or later
- **Neural Network Toolbox** (for `patternnet` classification)
- **Signal Processing Toolbox** (for filtering and feature extraction)
- **Statistics and Machine Learning Toolbox** (for PCA and statistical analysis)

### Running the System
```bash
# 1. Clone the repository
git clone https://github.com/RashmikaFernando-dev/AI-MLCoursework_Group_123.git
cd AI-MLCoursework_Group_123

# 2. Open MATLAB and navigate to project directory
# 3. Run the main pipeline
```

```matlab
cd src
run_main_pipeline
```

Results will be automatically saved in the `results/` folder with timestamp.

## Data Structure

### Input Data Format
The system expects CSV files with the naming pattern: `U{userID}NW_{session}.csv`

- **userID**: User number (1-10)
- **session**: FD (First Day) or MD (Middle Day)

### Sensor Data Columns
Each CSV file contains **6 columns** representing multi-axis sensor readings:

| Column | Description | Unit |
|--------|-------------|------|
| `acc_x` | X-axis linear acceleration | m/s² |
| `acc_y` | Y-axis linear acceleration | m/s² |
| `acc_z` | Z-axis linear acceleration | m/s² |
| `gyro_x` | X-axis angular velocity | rad/s |
| `gyro_y` | Y-axis angular velocity | rad/s |
| `gyro_z` | Z-axis angular velocity | rad/s |

## System Architecture

### Signal Processing Pipeline
- **Butterworth Filtering**: Noise reduction with optimized cutoff frequencies
- **Feature Extraction**: 33 comprehensive statistical features
- **Sensor Fusion**: Accelerometer and gyroscope data integration
- **Data Normalization**: Robust preprocessing pipeline

### Machine Learning Components
- **Dimensionality Reduction**: PCA (95% variance retention)
- **Classification**: Neural network with scaled conjugate gradient training
- **Validation**: Cross-validation and performance optimization
- **Evaluation**: Comprehensive metrics (Accuracy, EER, FAR, FRR)

### Visualization & Analysis
- Professional sensor data pattern visualization
- Feature importance analysis and correlation matrices
- Real-time authentication performance monitoring
- 11 automatically generated professional plots

## Project Structure

```
AI-MLCoursework_Group_123/
├── data/
│   ├── raw/                    # Input CSV files (U1NW_FD.csv, etc.)
│   ├── interim/               # Processed intermediate data
│   └── processed/             # Final processed datasets
├── src/                       # MATLAB source code
│   ├── run_main_pipeline.m     # Main execution script
│   ├── extract_sensor_features.m
│   ├── train_evaluate_model.m
│   └── analyze_sensor_features.m
├── results/                   # Output and analysis
│   ├── latest/               # Latest run results
│   └── [timestamp]/         # Timestamped experiment runs
├── report/                   # Documentation and reports
└── README.md                 # This file
```

## Performance Results

### Final Model Performance

| Metric | Value |
|--------|--------|
| **Training Accuracy** | 99.82% |
| **Test Accuracy** | 96.54% |
| **Equal Error Rate (EER)** | 1.40% |
| **Features Used** | 13 (after PCA optimization) |

### Model Comparison

| Model | Training Acc (%) | Test Acc (%) | EER | Features |
|-------|------------------|--------------|-----|----------|
| Neural Network (Optimized) | 99.82 | **96.54** | **0.0140** | 13 (PCA) |
| Neural Network (Full) | 99.18 | 94.21 | 0.0287 | 33 (original) |

### Key Achievements
- **Highest test accuracy**: 96.54% with PCA optimization
- **Low error rate**: 1.40% Equal Error Rate
- **Feature efficiency**: 61% reduction (33→13 features) with improved performance
- **Cross-session robustness**: Session-independent evaluation (FD→train, MD→test)

## Generated Outputs

### Visualizations (11 plots)
- Feature analysis (correlation, distribution, statistics)
- PCA analysis (scree plot, 2D projection)
- Training performance curves
- Signal processing examples
- Classification results (confusion matrix, per-class F1)

### Analysis Tables
- Performance metrics and statistical analysis
- Feature importance rankings
- Per-user authentication results
- Cross-validation summaries

## Technical Details

### Feature Extraction
The system extracts **33 statistical features** from sensor data:
- Time domain: Mean, Standard deviation, RMS, Skewness, Kurtosis
- Frequency domain: Spectral energy, Peak frequencies
- Cross-axis correlations and signal magnitude areas

### Authentication Workflow
1. **Data Preprocessing**: Filter and normalize sensor readings
2. **Feature Engineering**: Extract behavioral biometric features
3. **Dimensionality Reduction**: Apply PCA for optimization
4. **Model Training**: Neural network with cross-validation
5. **Authentication**: Real-time user verification

## Team Members
- **Group 123** - AI/ML Coursework Project

## License
This project is developed for academic purposes as part of AI/ML coursework.

## Contributing
This is an academic project. For questions or collaboration, please contact the team members.

---

**Last Updated**: November 19, 2025  
**Status**: Complete and Deployed