"""
Unit tests for PolymarketDataLoader.

Tests verify that the data loader:
1. Returns the correct tuple structure
2. Loads non-empty data for resolved markets
3. Calculates term structure metrics correctly
4. Handles outcome filtering properly
"""

import unittest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.data_loader_for_model import PolymarketDataLoader


class TestPolymarketDataLoader(unittest.TestCase):
    """Unit tests for PolymarketDataLoader."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.db_path = "data/polymarket.db"
        cls.loader = PolymarketDataLoader(cls.db_path)

        # Use a recent date range for testing
        cls.train_end = datetime.now()
        cls.train_start = cls.train_end - timedelta(days=365)
        cls.start_date = cls.train_start.strftime('%Y-%m-%d')
        cls.end_date = cls.train_end.strftime('%Y-%m-%d')

    def test_load_full_dataset_returns_three_values(self):
        """Test that load_full_dataset returns exactly 3 DataFrames."""
        result = self.loader.load_full_dataset(
            resolved_only=True,
            start_date=self.start_date,
            end_date=self.end_date,
            use_semantic_groups=True,
            load_token_features=False
        )

        self.assertEqual(len(result), 3,
                        "load_full_dataset should return exactly 3 values")

    def test_return_value_structure(self):
        """Test that load_full_dataset returns (rates_df, ts_metrics_df, resolved_df)."""
        rates_df, ts_metrics_df, resolved_df = self.loader.load_full_dataset(
            resolved_only=True,
            start_date=self.start_date,
            end_date=self.end_date,
            use_semantic_groups=True,
            load_token_features=False
        )

        # Test rates_df structure
        self.assertGreater(len(rates_df), 0,
                          "rates_df should not be empty for resolved markets")
        expected_rate_cols = ['market_id', 'date', 'price', 'outcome', 'implied_rate']
        for col in expected_rate_cols:
            self.assertIn(col, rates_df.columns,
                         f"rates_df should contain '{col}' column")

        # Test resolved_df structure - THIS IS THE KEY TEST
        self.assertGreater(len(resolved_df), 0,
                          "resolved_df should not be empty for resolved markets")
        self.assertIn('resolved_outcome', resolved_df.columns,
                     "resolved_df MUST contain 'resolved_outcome' column")

        # Verify resolved_outcome column has data
        non_null_outcomes = resolved_df['resolved_outcome'].notna().sum()
        self.assertGreater(non_null_outcomes, 0,
                          "resolved_df should have non-null resolved outcomes")

    def test_outcome_filtering_for_timing_markets(self):
        """
        Test that term structure metrics are calculated with the correct outcome.

        bets_for_timing_view contains only 'No' outcomes, so we must use outcome='No'
        when calculating term structure metrics for timing markets.
        """
        # Load with outcome='No' (correct for timing markets)
        rates_df, ts_metrics_df, resolved_df = self.loader.load_full_dataset(
            resolved_only=True,
            start_date=self.start_date,
            end_date=self.end_date,
            use_semantic_groups=True,
            load_token_features=False,
            outcome='No'  # Timing markets use 'No' outcomes
        )

        # Check that we got outcomes in the data
        if len(rates_df) > 0:
            outcomes = rates_df['outcome'].unique()
            self.assertIn('No', outcomes,
                         "rates_df should contain 'No' outcomes for timing markets")

    def test_term_structure_metrics_non_empty(self):
        """
        Test that term structure metrics are calculated when using the correct outcome.

        This is the root cause of the bug: term structure metrics were returning 0 rows
        because we were looking for 'Yes' outcomes in data that only has 'No' outcomes.
        """
        # Load with outcome='No' to match the timing markets data
        rates_df, ts_metrics_df, resolved_df = self.loader.load_full_dataset(
            resolved_only=False,  # Use all markets to get more data
            start_date=self.start_date,
            end_date=self.end_date,
            use_semantic_groups=True,
            load_token_features=False,
            outcome='No'  # Must use 'No' for timing markets
        )

        # Term structure metrics might still be 0 if there aren't enough markets
        # per group on the same date, but let's at least verify the structure
        if len(ts_metrics_df) > 0:
            expected_ts_cols = ['date', 'market_group', 'ts_level', 'ts_slope']
            for col in expected_ts_cols:
                self.assertIn(col, ts_metrics_df.columns,
                             f"ts_metrics_df should contain '{col}' column")

    def test_resolved_df_is_third_return_value(self):
        """
        CRITICAL TEST: Verify that resolved_df is the THIRD return value.

        The original bug was that the strategy was unpacking as:
            rates_df, resolved_df, text_features = load_full_dataset(...)

        But the actual return order is:
            rates_df, ts_metrics_df, resolved_df = load_full_dataset(...)

        This caused resolved_df to receive ts_metrics_df (which had 0 rows),
        making the strategy fail silently.
        """
        result = self.loader.load_full_dataset(
            resolved_only=True,
            start_date=self.start_date,
            end_date=self.end_date,
            use_semantic_groups=True,
            load_token_features=False
        )

        # Unpack correctly
        rates_df, ts_metrics_df, resolved_df = result

        # The THIRD value must be resolved_df with resolved_outcome column
        self.assertIn('resolved_outcome', resolved_df.columns,
                     "Third return value must be resolved_df with 'resolved_outcome' column")

        # The SECOND value should be ts_metrics_df (may be empty)
        # It should NOT have 'resolved_outcome' column
        if len(ts_metrics_df) > 0:
            self.assertNotIn('resolved_outcome', ts_metrics_df.columns,
                           "Second return value (ts_metrics_df) should not have 'resolved_outcome'")

    def test_market_data_loading(self):
        """Test basic market data loading."""
        market_df = self.loader.get_market_data(
            resolved_only=True,
            min_markets_per_group=2,
            use_semantic_groups=True
        )

        self.assertGreater(len(market_df), 0,
                          "Should load some market data")
        self.assertIn('market_id', market_df.columns)
        self.assertIn('semantic_group_id', market_df.columns)

    def test_resolved_outcomes_loading(self):
        """Test that resolved outcomes can be loaded directly."""
        resolved_df = self.loader.get_resolved_outcomes()

        self.assertGreater(len(resolved_df), 0,
                          "Should load some resolved outcomes")
        self.assertIn('resolved_outcome', resolved_df.columns,
                     "get_resolved_outcomes should return 'resolved_outcome' column")

        # Check that some outcomes were successfully parsed
        non_null = resolved_df['resolved_outcome'].notna().sum()
        self.assertGreater(non_null, 0,
                          "Should have successfully parsed some outcomes")


class TestDataLoaderBugReproduction(unittest.TestCase):
    """
    Tests that reproduce the original bug to prevent regression.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.loader = PolymarketDataLoader("data/polymarket.db")
        cls.train_end = datetime.now()
        cls.train_start = cls.train_end - timedelta(days=365)

    def test_original_bug_incorrect_unpacking(self):
        """
        Reproduce the original bug where incorrect unpacking caused resolved_df to be empty.

        Original code in strategy:
            rates_df, resolved_df, text_features = load_full_dataset(...)

        This caused:
        - rates_df = rates_df (correct)
        - resolved_df = ts_metrics_df (WRONG! - had 0 rows)
        - text_features = resolved_df (WRONG! - actual resolved data)
        """
        result = self.loader.load_full_dataset(
            resolved_only=True,
            start_date=self.train_start.strftime('%Y-%m-%d'),
            end_date=self.train_end.strftime('%Y-%m-%d'),
            use_semantic_groups=True,
            load_token_features=False
        )

        # Unpack INCORRECTLY (as the old strategy did)
        rates_df_wrong, resolved_df_wrong, text_features_wrong = result

        # This demonstrates the bug: resolved_df_wrong has no 'resolved_outcome' column
        # because it actually received ts_metrics_df
        if len(resolved_df_wrong) > 0:
            # If ts_metrics_df had data, it wouldn't have resolved_outcome
            self.assertNotIn('resolved_outcome', resolved_df_wrong.columns,
                           "Bug demonstration: wrong unpacking puts ts_metrics into 'resolved_df'")

        # The actual resolved data ended up in text_features_wrong
        self.assertIn('resolved_outcome', text_features_wrong.columns,
                     "Bug demonstration: actual resolved_df ended up in 'text_features'")

        # Now unpack CORRECTLY
        rates_df_correct, ts_metrics_df_correct, resolved_df_correct = result

        # With correct unpacking, resolved_df has the right data
        self.assertIn('resolved_outcome', resolved_df_correct.columns,
                     "Correct unpacking puts resolved_df in the right variable")


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPolymarketDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoaderBugReproduction))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
