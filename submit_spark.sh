#!/bin/bash

# Ensure variables are set
set -e

# Setup Python environment variables for Spark
export py=$(which python3)
export PYSPARK_PYTHON=$py
export PYSPARK_DRIVER_PYTHON=$py
export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}

if [ -z "$py" ]; then
    echo "ERROR: Python3 not found!"
    exit 1
fi

echo "========================================"
echo "Spark Submission Configuration"
echo "========================================"
echo "Python executable: $py"
echo "Java home: $JAVA_HOME"
echo "Master: spark://hadoop1:7077"
echo "Driver memory: 2g"
echo "Executor memory: 2g"
echo "Number of executors: 2"
echo "========================================"

# Submit to Spark cluster with optimized configurations
spark-submit \
  --master spark://hadoop1:7077 \
  --conf spark.pyspark.python=$py \
  --conf spark.pyspark.driver.python=$py \
  --driver-memory 2g \
  --executor-memory 2g \
  --num-executors 2 \
  --conf spark.executor.heartbeatInterval=60s \
  --conf spark.network.timeout=120s \
  --conf spark.executor.maxFailures=5 \
  --conf spark.task.maxFailures=4 \
  --conf spark.python.worker.memory=512m \
  --conf spark.driver.maxResultSize=1g \
  --conf spark.shuffle.memoryFraction=0.3 \
  --conf spark.storage.memoryFraction=0.4 \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  processing.py

exit_code=$?
echo "========================================"
echo "Spark job completed with exit code: $exit_code"
echo "========================================"

exit $exit_code
