#!/bin/bash

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Spark to use this Python environment
export PYSPARK_PYTHON=$(which python3)
export PYSPARK_DRIVER_PYTHON=$(which python3)

# Run the script
python3 processing.py
