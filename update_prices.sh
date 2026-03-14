#!/bin/bash
# Script to update price history data

echo "📊 Updating Polymarket price history..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

cd data/downloaders

# Run the price fetcher
echo "Fetching price data from Polymarket API..."
echo "(This may take a while depending on how many markets need updating)"
echo ""

python3 get_pricing.py

echo ""
echo "✅ Price update complete!"
echo ""

# Show summary
cd ../..
PRICE_COUNT=$(sqlite3 data/polymarket.db "SELECT COUNT(*) FROM price_history")
MARKETS_WITH_PRICES=$(sqlite3 data/polymarket.db "SELECT COUNT(DISTINCT market_id) FROM price_history")

echo "Summary:"
echo "  Total price points: $PRICE_COUNT"
echo "  Markets with prices: $MARKETS_WITH_PRICES"
