
import sys
import time
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.functions import vector_to_array

DATA_DIR = Path("/content/data")


def prepare_data(spark):

    # Load the four source datasets
    flights = spark.read.csv(
        str(DATA_DIR / "flights.csv"),
        header=True
    )

    airlines = spark.read.csv(
        str(DATA_DIR / "airlines.csv"),
        header=True
    )

    airports = spark.read.csv(
        str(DATA_DIR / "airports.csv"),
        header=True
    )

    cancellation_codes = spark.read.csv(
        str(DATA_DIR / "cancellation_codes.csv"),
        header=True
    )

    # Convert the required flight columns
    cast_types = {
        "MONTH": "int",
        "DAY_OF_WEEK": "int",
        "SCHEDULED_DEPARTURE": "int",
        "DISTANCE": "double",
        "SCHEDULED_TIME": "double",
        "ARRIVAL_DELAY": "double",
        "CANCELLED": "int",
        "DIVERTED": "int"
    }

    for column, data_type in cast_types.items():
        flights = flights.withColumn(
            column,
            F.col(column).cast(data_type)
        )

    # Exclude cancelled, diverted, and October flights
    flights = flights.filter(
        (F.col("CANCELLED") == 0) &
        (F.col("DIVERTED") == 0) &
        (F.col("MONTH") != 10) &
        F.col("ARRIVAL_DELAY").isNotNull()
    )

    # Join cancellation reason descriptions
    cancellation_lookup = cancellation_codes.select(
        F.col("CANCELLATION_REASON"),
        F.col("CANCELLATION_DESCRIPTION")
    )

    flights = flights.join(
        cancellation_lookup,
        on="CANCELLATION_REASON",
        how="left"
    )

    # Join airline names
    airline_lookup = airlines.select(
        F.col("IATA_CODE"),
        F.col("AIRLINE").alias("AIRLINE_NAME")
    )

    flights = flights.join(
        airline_lookup,
        flights["AIRLINE"] == airline_lookup["IATA_CODE"],
        "left"
    )

    # Join origin airport information
    origin_lookup = airports.select(
        F.col("IATA_CODE").alias("ORIGIN_CODE"),
        F.col("AIRPORT").alias("ORIGIN_NAME"),
        F.col("STATE").alias("ORIGIN_STATE")
    )

    flights = flights.join(
        origin_lookup,
        flights["ORIGIN_AIRPORT"] ==
        origin_lookup["ORIGIN_CODE"],
        "left"
    )

    # Join destination airport information
    destination_lookup = airports.select(
        F.col("IATA_CODE").alias("DEST_CODE"),
        F.col("AIRPORT").alias("DEST_NAME"),
        F.col("STATE").alias("DEST_STATE")
    )

    flights = flights.join(
        destination_lookup,
        flights["DESTINATION_AIRPORT"] ==
        destination_lookup["DEST_CODE"],
        "left"
    )

    # Create the binary classification label
    flights = flights.withColumn(
        "label",
        F.when(
            F.col("ARRIVAL_DELAY") >= 15,
            1.0
        ).otherwise(0.0)
    )

    # Extract scheduled departure hour
    flights = flights.withColumn(
        "DEP_HOUR",
        F.floor(
            F.col("SCHEDULED_DEPARTURE") / 100
        ).cast("int")
    )

    # Select the same features used in the original model
    categorical_features = [
        "AIRLINE_NAME",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "ORIGIN_STATE",
        "DEST_STATE"
    ]

    numerical_features = [
        "MONTH",
        "DAY_OF_WEEK",
        "DISTANCE",
        "SCHEDULED_TIME",
        "DEP_HOUR"
    ]

    model_data = flights.select(
        *categorical_features,
        *numerical_features,
        "label"
    )

    return model_data


def train_and_evaluate(model_data):

    # Split data consistently across benchmark runs
    train_data, test_data = model_data.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    categorical_features = [
        "AIRLINE_NAME",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "ORIGIN_STATE",
        "DEST_STATE"
    ]

    numerical_features = [
        "MONTH",
        "DAY_OF_WEEK",
        "DISTANCE",
        "SCHEDULED_TIME",
        "DEP_HOUR"
    ]

    # Index categorical features
    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_INDEX",
            handleInvalid="keep"
        )
        for column in categorical_features
    ]

    # One-hot encode categorical features
    encoders = [
        OneHotEncoder(
            inputCol=f"{column}_INDEX",
            outputCol=f"{column}_OHE"
        )
        for column in categorical_features
    ]

    # Assemble model features
    assembler = VectorAssembler(
        inputCols=numerical_features + [
            f"{column}_OHE"
            for column in categorical_features
        ],
        outputCol="features",
        handleInvalid="error"
    )

    # Configure Logistic Regression
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=20,
        regParam=0.01
    )

    pipeline = Pipeline(
        stages=indexers + encoders + [assembler, lr]
    )

    # Train model
    pipeline_model = pipeline.fit(train_data)

    # Generate predictions
    predictions = pipeline_model.transform(test_data)

    # Evaluate model discrimination
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = auc_evaluator.evaluate(predictions)

    # Apply the previously selected classification threshold
    final_predictions = predictions.withColumn(
        "prediction_adjusted",
        F.when(
            vector_to_array(
                F.col("probability")
            )[1] >= 0.20,
            1.0
        ).otherwise(0.0)
    )

    # Evaluate F1 and accuracy
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction_adjusted",
        metricName="f1"
    )

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction_adjusted",
        metricName="accuracy"
    )

    f1 = f1_evaluator.evaluate(final_predictions)
    accuracy = accuracy_evaluator.evaluate(final_predictions)

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "weighted_f1": float(f1)
    }


def get_executor_count(sc):
    """Return the number of registered Spark executors."""
    executor_ids = sc._jsc.sc().getExecutorIds()
    return executor_ids.size()


def run_benchmark(executor_count, run_number):

    spark = (
        SparkSession.builder
        .appName(
            f"FlightDelay_Benchmark_{executor_count}_"
            f"Executors_Run_{run_number}"
        )
        .master("spark://127.0.0.1:7077")
        .config("spark.executor.instances", str(executor_count))
        .config("spark.executor.cores", "1")
        .config("spark.executor.memory", "1g")
        .config("spark.cores.max", str(executor_count))
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    try:
        # Wait for the requested executors to register
        deadline = time.monotonic() + 90

        while time.monotonic() < deadline:
            actual_executors = get_executor_count(sc)

            if actual_executors == executor_count:
                break

            time.sleep(2)

        actual_executors = get_executor_count(sc)

        print(
            f"Requested executors: {executor_count}",
            flush=True
        )
        print(
            f"Actual executors: {actual_executors}",
            flush=True
        )

        if actual_executors != executor_count:
            raise RuntimeError(
                "Executor allocation does not match "
                "the requested configuration."
            )

        # Start timing the complete data and ML workflow
        start_time = time.perf_counter()

        model_data = prepare_data(spark)

        # Force data preparation to execute
        row_count = model_data.count()

        if row_count != 5231130:
            raise ValueError(
                f"Unexpected dataset size: {row_count}"
            )

        # Train and evaluate the model
        metrics = train_and_evaluate(model_data)

        elapsed_time = time.perf_counter() - start_time

        result = {
            "executors": executor_count,
            "run": run_number,
            "actual_executors": actual_executors,
            "rows": row_count,
            "execution_time_seconds": round(
                elapsed_time, 2
            ),
            **metrics
        }

        # Save each completed run immediately
        results_dir = Path("/content/drive/MyDrive/spark_project/benchmark/results")
        results_dir.mkdir(parents=True, exist_ok=True)

        result_path = results_dir / (
            f"benchmark_{executor_count}_"
            f"executors_run_{run_number}.json"
        )

        result_path.write_text(
            json.dumps(result, indent=4)
        )

        print("BENCHMARK_RESULT:", json.dumps(result),
              flush=True)

        print("RESULT_SAVED:", result_path, flush=True)

        return result

    finally:
        spark.stop()


if __name__ == "__main__":

    executor_count = int(sys.argv[1])
    run_number = int(sys.argv[2])

    run_benchmark(executor_count, run_number)
