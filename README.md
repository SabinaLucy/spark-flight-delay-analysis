# Spark Flight Delay Analysis

## Project Overview

This project uses **Apache Spark with PySpark** to process and analyze the 2015 U.S. flight delays and cancellations dataset. The project examines flight delays using Spark and evaluates how performance changes with different numbers of Spark executors.

The project is implemented in **Google Colab** using a Spark standalone cluster.

## Dataset

The project uses the **2015 Flight Delays and Cancellations** dataset from the U.S. Department of Transportation.

The dataset contains more than **5 million flight records** and includes information about:

- Flight delays and cancellations
- Airlines
- Airports
- Departure and arrival times
- Flight distance
- Cancellation reasons

### Dataset Source

[Kaggle – 2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays)

The dataset contains four CSV files:

- `flights.csv` – main flight records
- `airlines.csv` – airline information
- `airports.csv` – airport information
- `cancellation_codes.csv` – cancellation reason codes

The large `flights.csv` file is not stored directly in this repository because of its file size. It can be downloaded from the Kaggle source above.

## Project Methods

The same dataset will be used by both team members, with each member applying a different method.

### Logistic Regression Classification

**Team Member: Sabina Bimbi**

Spark MLlib Logistic Regression will be used to classify flights based on whether they experience a significant delay.

### Correlation and Statistical Analysis

**Team Member: Samantha Griffith**

Spark DataFrames will be used to perform correlation and statistical analysis of relationships among flight delays and other numerical flight characteristics.

## Technologies and Tools

- Python
- Apache Spark
- PySpark
- Spark MLlib
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Google Colab

## Performance Experiment

Each method will be tested using:

- **1 Spark executor**
- **2 Spark executors**
- **4 Spark executors**

Each configuration will be repeated at least **three times**. Runtime results will be used to calculate the mean, standard deviation, and speedup relative to the 1-executor configuration.

The project will also examine how available CPU cores, Spark partitions, scheduling overhead, shuffle/data movement, and dataset size affect scalability.

## Repository Structure

```text
spark-flight-delay-analysis/
│
├── data/
│   ├── airlines.csv
│   ├── airports.csv
│   └── cancellation_codes.csv
│
├── notebooks/
│   └── flight_delay_analysis.ipynb
│
├── src/
│   ├── logistic_regression.py
│   └── correlation_analysis.py
│
├── results/
│
├── figures/
│
├── README.md
└── requirements.txt
```

## Project Status

**In Progress**

Final results, performance comparisons, visualizations, and analysis will be added as the project progresses.
