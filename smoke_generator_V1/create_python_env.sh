#!/bin/bash

# Name of the virtual environment
ENV_NAME="smoke_generator_env"

# Check if python -m venv is available
if ! python -m venv --help &> /dev/null; then
    echo "Error: python -m venv is not available. Make sure you have Python 3 installed."
    exit 1
fi

# Create the virtual environment
echo "Creating virtual environment: $ENV_NAME"
python -m venv $ENV_NAME

# Activate the virtual environment
echo "Activating virtual environment: $ENV_NAME"
if [ "$OS" = "Windows_NT" ]; then
    source $ENV_NAME/Scripts/activate
else
    source $ENV_NAME/bin/activate
fi

# Install required packages
echo "Installing required packages from requirements.txt"
pip install -r requirements.txt

echo "Python environment setup completed."
