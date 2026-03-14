#!/bin/bash
# Quick start script for the Polymarket Implied Rates Dashboard

echo "🚀 Starting Polymarket Implied Rates Dashboard..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Check if database has price history
echo ""
echo "Checking database..."
PRICE_COUNT=$(sqlite3 data/polymarket.db "SELECT COUNT(*) FROM price_history")
echo "Found $PRICE_COUNT price history records"

if [ "$PRICE_COUNT" -lt 100 ]; then
    echo ""
    echo "⚠️  Low price history data detected!"
    echo "Consider running: python3 data/downloaders/get_pricing.py"
    echo "to fetch more historical price data."
    echo ""
fi

# Run dashboard
echo ""
echo "🎉 Launching dashboard..."
echo "The dashboard will open in your browser at http://localhost:3000"
echo ""
streamlit run dashboard.py
