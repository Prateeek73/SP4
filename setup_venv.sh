#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables for Spark
export PYSPARK_PYTHON=$(which python3)
export PYSPARK_DRIVER_PYTHON=$(which python3)

echo "Virtual environment setup complete!"
echo "To activate: source venv/bin/activate"
echo ""
echo "Then run:"
echo "  export PYSPARK_PYTHON=\$(which python3)"
echo "  export PYSPARK_DRIVER_PYTHON=\$(which python3)"
echo "  python3 processing.py"
echo ""
echo "Or use spark-submit:"
echo "  spark-submit --conf spark.pyspark.python=\$(which python3) processing.py"
