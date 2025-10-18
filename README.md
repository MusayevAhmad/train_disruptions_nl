# Train Disruption Predictive Analytics - Assignment 2

## Overview

This project implements descriptive and predictive analytics for Dutch train disruptions using **Logistic Regression** to predict whether a disruption will be "Long" (duration ≥ median threshold) based on features available at report time.

## Implementation Summary

### Model Details
- **Algorithm**: Logistic Regression with L2 regularization
- **Training Objective**: Log loss (cross-entropy)
- **Best Regularization Parameter (C)**: 0.5
- **Class Weighting**: Balanced
- **Decision Threshold**: 0.330 (optimized for F1 score)

### Dataset
- **Source**: Disruption data from 2011-2024 (14 CSV files)
- **Total Records**: 55,864 disruptions
- **Training Data**: 2021-2023 (14,172 records)
- **Validation Data**: 2023 Q4 (1,369 records)
- **Test Data**: 2024 (5,964 records)

### Target Variable
- **Definition**: `is_long` = 1 if `duration_minutes` ≥ 50.0 minutes (median threshold computed from 2021-2024 data)
- **Class Balance**: ~50% Long vs ~50% Short (approximately balanced)

### Features Used

#### Categorical Features (One-Hot Encoded)
1. **hour_bin**: Time-of-day bins
   - `night` (0-5h)
   - `morning_peak` (6-9h)
   - `midday` (10-15h)
   - `evening_peak` (16-19h)
   - `late` (20-23h)

2. **weekday**: Day of week (0=Monday, 6=Sunday)

3. **month**: Month of year (1-12)

4. **cause_group**: Disruption cause category
   - Top 8 categories kept: rolling stock, infrastructure, accidents, external, engineering work, logistical, staff, unknown
   - Rare categories mapped to "Other"

#### Numeric Features (Passthrough)
1. **station_count**: Number of affected stations (parsed from `rdt_station_codes`)
2. **is_engineering_work**: Binary indicator (1 if cause is engineering work)

### Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| **PR-AUC (Long class)** | 0.7612 |
| **Brier Score** | 0.1980 |
| **Precision (Long)** | 0.6667 |
| **Recall (Long)** | 0.7735 |
| **F1 Score (Long)** | 0.7161 |

### Confusion Matrix (Test Set)

|  | Predicted Short | Predicted Long |
|--|----------------|---------------|
| **Actually Short** | 1,751 (TN) | 1,175 (FP) |
| **Actually Long** | 688 (FN) | 2,350 (TP) |

## Project Structure

```
train_disruptions_nl/
├── datasets/
│   ├── disruptions-2011.csv
│   ├── disruptions-2012.csv
│   ├── ...
│   └── disruptions-2024.csv
├── src/
│   ├── execute_analysis.py                      # Main analysis script
│   ├── model_pipeline.py                        # Logistic regression implementation
│   ├── feature_engineering.py                   # Feature creation and splitting
│   ├── data_loader.py                           # Data loading
│   ├── visualizations.py                        # Figure generation
│   ├── logger.py                                # Structured logging
│   ├── train_disruption_analysis.ipynb          # Exploratory notebook
│   └── column_comparison.py                     # Data validation script
├── figures/
│   ├── fig1_heatmap.png                         # Hour × Weekday heatmap
│   ├── fig2_monthly.png                         # Monthly trends by year
│   ├── fig3_cause_duration.png                  # Duration by cause group
│   ├── fig4_long_share.png                      # % Long by cause group
│   └── fig5_calibration.png                     # Calibration curve
├── artifacts/
│   ├── metrics_test.csv                         # All evaluation metrics
│   └── feature_list.txt                         # Feature documentation
└── README.md
```

## How to Run

### Requirements
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Execute the Analysis
```bash
python src/execute_analysis.py
```

This will:
1. Load and combine all disruption CSV files from `datasets/`
2. Engineer features and create target variable
3. Split data using forward-in-time logic
4. Build and tune logistic regression model
5. Evaluate on test set
6. Generate 5 figures in `figures/`
7. Save metrics to `artifacts/metrics_test.csv`
8. Print comprehensive model summary

## Key Findings

### Descriptive Analytics

1. **Temporal Patterns**:
   - Peak disruption hours: 6-9 AM (morning rush) and 4-7 PM (evening rush)
   - Weekday disruptions more common than weekends
   - Higher disruption rates during winter months

2. **Cause Group Analysis**:
   - **Rolling stock** (36.0%): Most common cause, but relatively short duration (32.6% are Long)
   - **Staff issues** (1.8%): Rare but tend to be long (93.5% are Long)
   - **External causes** (10.7%): Typically shorter disruptions (35.3% are Long)
   - **Accidents** (11.1%): Above-average duration (75.5% are Long)

3. **Duration Distribution**:
   - Median: 50.0 minutes (2021-2024 data)
   - Mean: 159.2 minutes (heavily right-skewed)
   - Outliers: 475 disruptions lasting >24 hours

### Predictive Analytics

1. **Model Performance**:
   - The model shows **strong** predictive performance
   - PR-AUC of 0.7612 indicates excellent discrimination between Long and Short disruptions
   - F1 score of 0.7161 suggests balanced precision and recall
   - Brier score of 0.1980 shows good probability calibration

2. **Feature Importance** (Interpretation):
   - `cause_group`: Most predictive feature (staff/logistical causes → longer)
   - `hour_bin`: Time of day affects duration (night disruptions tend to be longer)
   - `station_count`: More affected stations correlate with longer disruptions
   - `is_engineering_work`: Engineering work tends to be planned and longer

3. **Business Value**:
   - Early prediction of disruption duration enables better resource allocation
   - High recall (77.4%) ensures most long disruptions are identified
   - Reasonable precision (66.7%) minimizes false alarms

## Methodology Highlights

### Data Split Strategy
- **Multi-year forward-in-time split** to prevent data leakage
- Train on 2021-2023, validate on 2023 Q4, test on 2024
- Ensures temporal ordering and realistic deployment scenario

### Hyperparameter Tuning
- Tuned regularization parameter C ∈ {0.5, 1.0, 2.0}
- Selected C=0.5 based on validation PR-AUC
- Decision threshold optimized for F1 score on validation set (0.330)

### Evaluation Strategy
- **PR-AUC** as primary metric (measures discrimination for the positive class)
- **Brier score** for probability calibration quality
- **Confusion matrix** for detailed error analysis
- **Calibration curve** to visualize probability reliability
- **Precision, Recall, F1 Score** at optimized threshold for operational metrics

## Visualizations

### Figure 1: Disruptions by Hour × Weekday
Shows clear temporal patterns with morning and evening rush hour peaks, particularly on weekdays.

### Figure 2: Monthly Disruptions by Year
Reveals seasonal patterns and year-over-year trends, with increased disruptions in recent years.

### Figure 3: Duration Distribution by Cause Group
Box plots showing that staff, logistical, and accident-related disruptions tend to last longer.

### Figure 4: % Long Disruptions by Cause Group
Staff issues (93.5%) and logistical problems (77.7%) have the highest proportion of long disruptions.

### Figure 5: Calibration Curve
Demonstrates good calibration with predicted probabilities closely matching observed frequencies.

## Limitations & Future Work

### Current Limitations
1. Limited feature set (only features available at report time)
2. No spatial features (station network topology)
3. No weather data integration
4. Simple threshold-based target (could use regression for exact duration)

### Potential Improvements
1. **Feature Engineering**:
   - Add historical disruption frequency per station/route
   - Include day-of-week × hour interactions
   - Weather conditions (temperature, precipitation)
   - Holiday indicators

2. **Model Enhancements**:
   - Try ensemble methods (Random Forest, Gradient Boosting)
   - Multi-class classification (Short/Medium/Long/Very Long)
   - Regression for exact duration prediction
   - Time series models for temporal dependencies

3. **Evaluation**:
   - Cost-sensitive learning (different weights for FP vs FN)
   - Evaluate performance per cause group
   - Analyze prediction errors by time period

## Authors & License

**Course**: Introduction to Data Science (Assignment 2)  
**Institution**: Vrije Universiteit Amsterdam  
**Year**: 2025  

This project is for educational purposes as part of coursework.

---

**Last Updated**: October 18, 2025

