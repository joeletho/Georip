#!/usr/bin/env bash

# Get the directory of the current script and then move to its parent directory
CURR_DIR="$(dirname "$(realpath "$0")")"
ROOT_DIR="$(dirname $CURR_DIR)"
VENV_DIR="$ROOT_DIR/.env"

echo "Install path: $ROOT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt if it exists
if [ -f "$CURR_DIR/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$CURR_DIR/requirements.txt"
else
    echo "requirements.txt not found. Please ensure it is in the root directory."
    deactivate
    exit 1
fi

echo "Installation complete."
echo "Run 'source $VENV_DIR/bin/activate' to activate the virtual environment and 'deactivate' to deactivate it."
