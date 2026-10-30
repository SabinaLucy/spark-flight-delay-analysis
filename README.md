
# Spark Flight Delay Analysis

## Project Overview

This project uses **Apache Spark and PySpark** to process and analyze the **2015 U.S. Flight Delays and Cancellations** dataset.

We used two different methods to explore flight delays and compare how execution performance changes with different Spark executor configurations.

- **Sabina Bimbi:** Flight delay classification using Spark MLlib Logistic Regression
- **Samantha Griffith:** Descriptive statistics and correlation analysis using Spark DataFrames

The project was developed in **Google Colab** using a Spark standalone cluster.

## Dataset

**Source:** [2015 Flight Delays and Cancellations on Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)

The dataset contains **5,819,079 flight records** and information about airlines, airports, departure times, arrival times, delays, cancellations, and flight distances.

The dataset files include:

- `flights.csv`
- `airlines.csv`
- `airports.csv`
- `cancellation_codes.csv`

The raw dataset is not included in this repository because of its size. It can be downloaded from Kaggle.

## Project Methods

### 1. Flight Delay Classification

**Contributor: Sabina Bimbi**

I used **Spark MLlib Logistic Regression** to classify whether a flight would arrive at least 15 minutes late.

The workflow included data cleaning, feature preparation, model training, and evaluation.

After preprocessing, the classification dataset contained **5,231,130 records**.

The model was evaluated using accuracy, precision, recall, F1 score, ROC AUC, and a confusion matrix.

I also tested different classification thresholds to improve detection of delayed flights.

#### Model Results

| Metric | Threshold 0.50 | Threshold 0.20 |
|---|---:|---:|
| Accuracy | 0.8082 | 0.6216 |
| Precision | 0.4428 | 0.2693 |
| Recall | 0.0006 | 0.5682 |
| F1 score | 0.0012 | 0.3654 |
| ROC AUC | 0.6384 | 0.6384 |

The lower threshold increased recall, allowing the model to identify more delayed flights, although it also increased false positive predictions.

### 2. Statistical and Correlation Analysis

**Contributor: Samantha Griffith**

Samantha used Spark DataFrames to calculate descriptive statistics and examine relationships among arrival delays, departure delays, and flight distance.

#### Statistical Results

| Statistic | Value |
|---|---:|
| Mean arrival delay | 4.89 minutes |
| Standard deviation of arrival delay | 39.78 minutes |
| Mean departure delay | 9.70 minutes |
| Standard deviation of departure delay | 37.39 minutes |
| Mean flight distance | 825.23 miles |
| Standard deviation of flight distance | 608.79 miles |

Correlation calculations are included in the statistical analysis notebook.

## Technologies and Tools

- Python
- Apache Spark
- PySpark
- Spark MLlib
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Performance Experiment

Both methods were benchmarked using configurations requesting **1, 2, and 4 Spark executors**, with one core per executor.

Each configuration was tested **three times**.

The recorded wall clock times were used to calculate:

- Mean execution time
- Standard deviation
- Speedup relative to one executor

### Performance Comparison

| Executors | Sabina: Mean ± SD (s) | Samantha: Mean ± SD (s) |
|---|---:|---:|
| 1 | 822.62 ± 6.35 | 236.62 ± 9.11 |
| 2 | 787.90 ± 13.10 | 221.39 ± 2.36 |
| 4 | 907.60 ± 9.84 | 220.69 ± 5.75 |

### Speedup

| Executors | Sabina | Samantha |
|---|---:|---:|
| 1 | 1.000 | 1.000 |
| 2 | 1.044 | 1.069 |
| 4 | 0.906 | 1.072 |

Sabina's Logistic Regression workflow performed best with **2 executors**.

Samantha's statistical analysis recorded its lowest average execution time with **4 executors**, although the difference between 2 and 4 executors was small.

Both methods showed limited scaling. The experiments were performed in a Google Colab environment with **2 logical CPUs and approximately 12.67 GB of RAM**.

The results demonstrate how available hardware, task scheduling, data movement, and workload complexity can affect Spark performance.

## Repository Structure

The project contains separate notebooks for both team members and benchmark files for performance evaluation.

```text
spark-flight-delay-analysis/
├── README.md
├── notebooks/
│   ├── Sabina_Flight_Delay_Logistic_Regression.ipynb
│   └── samantha_flight_delay_statistical_analysis.ipynb
└── benchmark/
    ├── flight_delay_benchmark.py
    └── results/
        ├── benchmark_1_executors_run_1.json
        ├── benchmark_1_executors_run_2.json
        ├── benchmark_1_executors_run_3.json
        ├── benchmark_2_executors_run_1.json
        ├── benchmark_2_executors_run_2.json
        ├── benchmark_2_executors_run_3.json
        ├── benchmark_4_executors_run_1.json
        ├── benchmark_4_executors_run_2.json
        └── benchmark_4_executors_run_3.json
```

The structure above is the intended submission layout. File names should match the actual files committed to GitHub.

## Project Findings

The project highlighted two important findings.

First, **model accuracy alone is not enough** when the dataset contains an uneven distribution of classes. Adjusting the classification threshold significantly improved recall for delayed flights.

Second, **adding more Spark executors does not always reduce execution time**. The available hardware and the type of processing performed have a major effect on performance.

## Project Status

**Implementation and benchmarking completed.**

Final notebooks, benchmark results, and individual reports are prepared for submission.

## Contributors

**Sabina Bimbi**  
Spark MLlib Logistic Regression, model evaluation, and performance benchmarking.

**Samantha Griffith**  
Spark statistical analysis, correlation analysis, and performance benchmarking.

**Course:** SAT5165 Big Data Analytics  
**Institution:** Michigan Technological University
