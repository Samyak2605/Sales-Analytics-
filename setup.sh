#!/bin/bash
# setup.sh - Turn-key setup script for local evaluation

echo "=========================================================="
echo "  Setting up Phase 3 Sales Analytics Engine Environment"
echo "=========================================================="

# Check if python is installed
if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "[1/4] Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

echo "[3/4] Upgrading pip..."
pip install --upgrade pip

echo "[4/4] Installing requirements..."
pip install -r requirements.txt

echo "=========================================================="
echo "✅ Setup Complete!"
echo "To run the Web UI, use:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
echo "=========================================================="
