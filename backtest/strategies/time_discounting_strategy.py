"""
Time Discounting Strategy

Uses the hierarchical Bayesian TimeDiscountingModel to predict event outcomes.
The model learns category-specific and event-specific parameters from historical data
and should exhibit overfitting behavior on sparse semantic groups.

Strategy:
1. Train model on historical resolved markets (expanding window)
2. Generate posterior predictions for active markets
3. Compare model prediction to market price
4. Buy when model thinks market is underpriced (P_model > P_market + threshold)
5. Sell when model thinks market is overpriced (P_model < P_market - threshold)
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backtest.strategy import Strategy, Signal, SignalType
from models.time_discounting_model.model import TimeDiscountingModel
from data.data_loader_for_model import PolymarketDataLoader


class TimeDiscountingStrategy(Strategy):
    @property
    def name(self) -> str:
        return "TimeDiscounting"
    """
    Strategy that uses hierarchical Bayesian model predictions.

    Configuration:
        threshold: Price difference threshold for generating signals (default 0.15)
        min_edge: Minimum edge required to trade (default 0.05)
        min_volume: Minimum market volume to trade (default 100)
        lookback_days: Days of historical data to train on (default 365)
        retrain_interval: Days between model retraining (default 7)
        mcmc_draws: MCMC draws for posterior sampling (default 1000)
        mcmc_tune: MCMC tuning steps (default 500)
        mcmc_chains: Number of MCMC chains (default 2)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.threshold = self.config.get('threshold', 0.15)
        self.min_edge = self.config.get('min_edge', 0.05)
        self.min_volume = self.config.get('min_volume', 100)
        self.lookback_days = self.config.get('lookback_days', 365)
        self.retrain_interval = self.config.get('retrain_interval', 7)
        self.mcmc_draws = self.config.get('mcmc_draws', 1000)
        self.mcmc_tune = self.config.get('mcmc_tune', 500)
        self.mcmc_chains = self.config.get('mcmc_chains', 2)

        # Model state
        self.model = None
        self.last_train_date = None
        self.data_loader = PolymarketDataLoader(self.config.get('db_path', 'data/polymarket.db'))

        print(f"\n=== Time Discounting Strategy Config ===")
        print(f"Threshold: {self.threshold}")
        print(f"Min edge: {self.min_edge}")
        print(f"Min volume: ${self.min_volume}")
        print(f"Lookback days: {self.lookback_days}")
        print(f"Retrain interval: {self.retrain_interval} days")
        print(f"MCMC: {self.mcmc_draws} draws, {self.mcmc_tune} tune, {self.mcmc_chains} chains")

    def train_model(self, train_end_date: str) -> bool:
        """
        Train the TimeDiscountingModel on historical resolved markets.

        Args:
            train_end_date: End date for training data (format: YYYY-MM-DD)

        Returns:
            True if training succeeded, False otherwise
        """
        try:
            # Calculate training window
            train_end = pd.to_datetime(train_end_date)
            train_start = train_end - timedelta(days=self.lookback_days)

            print(f"\n=== Training TimeDiscountingModel ===")
            print(f"Training window: {train_start.date()} to {train_end.date()}")

            # Load training data (resolved markets only)
            # Returns: (rates_df, ts_metrics_df, resolved_df)
            rates_df, _, resolved_df = self.data_loader.load_full_dataset(
                resolved_only=True,
                start_date=train_start.strftime('%Y-%m-%d'),
                end_date=train_end_date,
                use_semantic_groups=True,
                load_token_features=False  # Don't load token cooccurrence
            )

            print(f"Loaded {len(rates_df)} price observations")
            print(f"Loaded {len(resolved_df)} resolved outcomes")

            if len(resolved_df) < 10:
                print("Not enough resolved markets to train (need at least 10)")
                return False

            # Initialize model
            self.model = TimeDiscountingModel(discount_function='hyperbolic')

            # Prepare data
            data = self.model.prepare_data(rates_df, resolved_df)

            print(f"Prepared {data['n_obs']} observations")
            print(f"  Categories: {data['n_categories']}")
            print(f"  Semantic groups: {data['n_events']}")

            if data['n_obs'] < 20:
                print("Not enough observations to train (need at least 20)")
                return False

            # Build and fit model
            self.model.build_model(data)

            print(f"\nFitting model with MCMC...")
            print(f"  Draws: {self.mcmc_draws}, Tune: {self.mcmc_tune}, Chains: {self.mcmc_chains}")

            self.model.fit(
                data,
                draws=self.mcmc_draws,
                tune=self.mcmc_tune,
                chains=self.mcmc_chains,
                target_accept=0.9
            )

            print(f"✓ Model training complete")
            self.last_train_date = train_end_date

            return True

        except Exception as e:
            print(f"✗ Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_predictions(self, predict_date: str) -> pd.DataFrame:
        """
        Generate predictions for active markets on a given date.

        Args:
            predict_date: Date to generate predictions for (format: YYYY-MM-DD)

        Returns:
            DataFrame with predictions
        """
        if self.model is None or self.model.trace is None:
            return pd.DataFrame()

        try:
            # Load active markets for prediction date
            # Returns: (rates_df, ts_metrics_df, resolved_df)
            rates_df, _, resolved_df = self.data_loader.load_full_dataset(
                active_only=True,
                start_date=predict_date,
                end_date=predict_date,
                use_semantic_groups=True,
                load_token_features=False
            )

            if len(rates_df) == 0:
                return pd.DataFrame()

            # For prediction, we need a dummy resolved_df with the same structure
            # but we don't need actual outcomes
            dummy_resolved = pd.DataFrame({
                'market_group': rates_df['event_id'].unique(),
                'market_id': rates_df['market_id'].unique(),
                'resolved_outcome': 'No'  # Dummy value
            })

            # Prepare data
            data = self.model.prepare_data(rates_df, dummy_resolved)

            # Generate predictions using trained model
            predictions = self.model.predict(data, return_samples=False)

            return predictions

        except Exception as e:
            print(f"Prediction generation failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def on_data(self, current_date: str, market_data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on model predictions.

        Args:
            current_date: Current date (format: YYYY-MM-DD)
            market_data: Available market data for this date

        Returns:
            List of trading signals
        """
        signals = []

        # Check if we need to retrain
        should_train = False
        if self.model is None or self.last_train_date is None:
            should_train = True
        else:
            days_since_train = (pd.to_datetime(current_date) - pd.to_datetime(self.last_train_date)).days
            if days_since_train >= self.retrain_interval:
                should_train = True

        if should_train:
            print(f"\n{'='*60}")
            print(f"Date: {current_date} - Training model")
            print(f"{'='*60}")
            success = self.train_model(current_date)
            if not success:
                print("Model training failed, no signals generated")
                return signals

        # Generate predictions for active markets
        predictions = self.generate_predictions(current_date)

        if len(predictions) == 0:
            return signals

        # Merge predictions with current market data
        # Market data has current prices, predictions have model estimates
        merged = market_data.merge(
            predictions[['event_group', 'question', 'predicted_prob_mean',
                        'predicted_prob_lower', 'predicted_prob_upper']],
            left_on=['event_id', 'question'],
            right_on=['event_group', 'question'],
            how='inner'
        )

        if len(merged) == 0:
            return signals

        # Filter by volume
        merged = merged[merged['volume_num'] >= self.min_volume]

        # Generate signals
        for _, row in merged.iterrows():
            # Model predicts P(No)
            model_prob_no = row['predicted_prob_mean']
            market_price_no = row['price']  # Market price for "No" outcome

            # Calculate edge
            edge = model_prob_no - market_price_no

            # Buy signal: model thinks "No" is underpriced
            # (model prob higher than market price)
            if edge > self.threshold and abs(edge) > self.min_edge:
                signals.append(Signal(
                    market_id=row['market_id'],
                    token_id=row['token_id'],
                    outcome='No',
                    signal_type=SignalType.BUY,
                    price=market_price_no,
                    reason=f"Model P(No)={model_prob_no:.3f} > Market={market_price_no:.3f}, edge={edge:.3f}"
                ))

            # Sell signal: model thinks "No" is overpriced
            # (model prob lower than market price)
            elif edge < -self.threshold and abs(edge) > self.min_edge:
                signals.append(Signal(
                    market_id=row['market_id'],
                    token_id=row['token_id'],
                    outcome='No',
                    signal_type=SignalType.SELL,
                    price=market_price_no,
                    reason=f"Model P(No)={model_prob_no:.3f} < Market={market_price_no:.3f}, edge={edge:.3f}"
                ))

        if signals:
            print(f"\nGenerated {len(signals)} signals on {current_date}:")
            for sig in signals[:5]:  # Show first 5
                print(f"  {sig.signal_type.name}: {sig.market_id} @ {sig.price:.3f} ({sig.reason})")
            if len(signals) > 5:
                print(f"  ... and {len(signals) - 5} more")

        return signals
