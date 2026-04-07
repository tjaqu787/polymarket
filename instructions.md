
---

# Strategy Spec: `SpreadDynamicsStrategy`

## Concept

Existing strategies ask: *"Is this rate mispriced?"* This strategy asks: *"Is this rate about to move, and in which direction?"*

The implied hazard rate spread between two contracts in the same term structure (`λ_far - λ_near`) is not static. It has predictable regimes:

- **Widens** during market inactivity (low volume, stale prices, no news)
- **Compresses** after news / information events (volume spike, sharp price move)

The edge is **anticipating the regime transition** — entering *before* the spread moves, not after you've observed the mispricing at its peak.

---

## File Location

```
backtest/strategies/spread_dynamics_strategy.py
```

---

## Core Math

The repo already computes implied rates everywhere. This strategy operates on the **spread between adjacent contracts** in the term structure:

```
λ(t) = -ln(P_yes(t)) / TTE(t)          # implied hazard rate for contract at maturity t

spread(i, j) = λ(t_far_j) - λ(t_near_i)   # rate spread between two tenors

Δspread = spread(t) - spread(t - lookback)  # spread change = the signal
```

The **signal** is not the spread level — it's the *velocity* and *direction* of spread change, conditioned on volume regime.

---

## Class Design

```python
class SpreadDynamicsStrategy(Strategy):
    """
    Trades the anticipated change in implied rate spreads, not the level.

    Enters positions before expected spread compression or widening,
    using volume regime, spread velocity, and rate momentum as signals.

    Two trade types:
        - COMPRESSION trade: short the wide leg, long the tight leg
          (spread expected to narrow after news/volume spike)
        - WIDENING trade: long the wide leg, short the tight leg
          (spread expected to widen during inactivity)
    """
```

---

## Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_days` | 7 | Window for measuring spread change (Δspread) |
| `vol_lookback_days` | 3 | Window for volume regime classification |
| `vol_spike_threshold` | 2.0 | Z-score above which volume = "news event" |
| `vol_drought_threshold` | -0.5 | Z-score below which volume = "inactive" |
| `min_spread_change` | 0.05 | Minimum \|Δspread\| to generate a signal (rate units) |
| `min_spread_level` | 0.02 | Ignore pairs where the spread is near zero (no room to move) |
| `min_tte_days` | 14 | Minimum days to expiry on either leg |
| `min_volume` | 300 | Minimum market volume to be considered |
| `refit_days` | 1 | How often to recompute spread history (daily) |
| `kelly_fraction` | 0.25 | Fractional Kelly |
| `max_position` | 0.10 | Max position size per leg as fraction of portfolio |
| `max_event_exposure` | 0.20 | Max total exposure per semantic group |
| `min_pair_correlation` | 0.60 | Minimum historical price correlation between legs (sanity check) |

---

## Data Requirements

Uses `load_timing_markets()` — same pipeline as other strategies. Needs at least 2 active contracts per semantic group with sufficient price history (`lookback_days` of daily observations) to compute the spread time series.

---

## Spread Pair Construction

For each semantic group on each refit cycle:

1. Take all active contracts, sorted by `time_to_expiration` ascending
2. Form **adjacent pairs**: (near=contract[i], far=contract[i+1])
3. Also consider the **full-span pair**: (shortest TTE, longest TTE) — often the most liquid
4. For each pair, compute implied rates `λ_near`, `λ_far` using existing `calculate_implied_rate()` from `utils/implied_rates.py`
5. Compute the spread time series over `lookback_days`

---

## Signal Logic

### Step 1 — Volume Regime Classification

For each event group, compute a rolling volume z-score over `vol_lookback_days`:

```python
vol_zscore = (current_volume - rolling_mean_volume) / rolling_std_volume
```

- `vol_zscore > vol_spike_threshold` → **News regime** → expect spread compression
- `vol_zscore < vol_drought_threshold` → **Inactive regime** → expect spread widening  
- Otherwise → **Neutral** → no regime-driven signal

### Step 2 — Spread Velocity

```python
Δspread = spread_today - spread_{lookback_days_ago}
spread_velocity = Δspread / lookback_days  # rate per day
```

- Spread has been **widening** (Δspread > 0) + volume is **spiking** → compression trade imminent → enter compression
- Spread has been **compressing** (Δspread < 0) + volume is **dropping** → widening trade imminent → enter widening
- Spread has been **widening** + volume is **inactive** → widening continuation → enter or add widening

### Step 3 — Signal Generation

**Compression trade** (spread expected to narrow):
- SHORT the far leg (sell Yes / buy No on overpriced far contract)
- LONG the near leg (buy Yes on underpriced near contract)
- Signal reason: `"Spread_compression: vol_spike + spread_widening"`

**Widening trade** (spread expected to expand):
- LONG the far leg
- SHORT the near leg
- Signal reason: `"Spread_widening: vol_drought + spread_compressing"`

Each signal carries `metadata` with `spread_level`, `spread_velocity`, `vol_zscore`, and `pair_id` for tracking.

---

## Position Sizing

Edge is defined as the expected spread change magnitude:

```python
edge = abs(spread_velocity) * expected_holding_days
position_size = kelly_fraction * (edge / edge_variance)
position_size = min(position_size, max_position)
```

`edge_variance` is estimated from the rolling standard deviation of `spread_velocity` over the last 30 days of history for that pair. If insufficient history, fall back to `edge / 2` (conservative half-Kelly).

Both legs of a spread trade are sized together — equal and opposite — so the gross exposure per pair is `2 * position_size`.

---

## Spread State Tracking

Maintain a dictionary `self.spread_positions` keyed by `(event_id, near_market_id, far_market_id)`:

```python
{
    'trade_type': 'compression' | 'widening',
    'entry_spread': float,
    'entry_date': pd.Timestamp,
    'entry_vol_zscore': float,
    'near_leg_size': float,
    'far_leg_size': float,
}
```

---

## Exit Rules

| Condition | Action |
|---|---|
| Spread has moved in expected direction by ≥ 50% of `min_spread_change` | Close both legs (take profit) |
| Spread has moved against position by `min_spread_change` | Close both legs (stop loss) |
| Volume regime flips (e.g., was inactive, now spiking) | Close widening trades, re-evaluate |
| Either leg < `min_tte_days` to expiry | Close both legs |
| Either leg resolves | Engine handles via `on_market_close`; close remaining leg manually in `on_market_close` override |

---

## `on_market_close` Override

```python
def on_market_close(self, market_id: str, outcome: str, final_price: float):
    """If one leg of a spread pair resolves, immediately close the other."""
```

Check `self.spread_positions` for any pair containing `market_id`. If found, emit a CLOSE signal on the remaining leg at current market price.

---

## Key Implementation Notes

**Reuse rate infrastructure.** Call `calculate_implied_rate()` from `utils/implied_rates.py` directly. The spread is just the difference between two calls — no new math needed.

**Spread history requires buffering.** Unlike other strategies that only look at today's data, this one needs `lookback_days` of spread history per pair. Compute and cache this from `historical_data` passed into `on_data()`. Use a dict keyed by `pair_id`.

**Volume proxy.** The data has `volume_num` at the market level (not intraday). Use day-over-day change in `volume_num` as the volume signal. On the first `vol_lookback_days` observations, skip signal generation (insufficient history).

**Both legs must be valid.** Before generating any signal, check that both legs pass `min_tte_days` and `min_volume` filters. Asymmetric pairs (one valid, one not) are skipped entirely.

**No model fitting.** Unlike the Gamma/Bayesian strategies, there is no MCMC or CDF fitting here. This is a pure signal-from-data strategy — computationally cheap, runs on every timestep with no refit budget concerns.

---

## Run Script

```
run_spread_dynamics_backtest.py
```

```python
strategy = SpreadDynamicsStrategy(config={
    "db_path": DB_PATH,
    "lookback_days": 7,
    "vol_lookback_days": 3,
    "vol_spike_threshold": 2.0,
    "vol_drought_threshold": -0.5,
    "min_spread_change": 0.05,
    "kelly_fraction": 0.25,
    "max_position": 0.10,
    "max_event_exposure": 0.20,
    "min_tte_days": 14,
    "min_volume": 300,
})
```

---

## What Makes This Different from Existing Strategies

| Strategy | Core signal |
|---|---|
| `CarryStrategy` | Static level: price > 0.90 or < 0.10 |
| `FactoredGammaStrategy` | Rate edge: market rate vs model rate |
| `PoissonTimingStrategy` | CI bounds from CDF fit |
| `SurvivalConditionalStrategy` | Conditional repricing as time passes |
| **`SpreadDynamicsStrategy`** | **Spread velocity + volume regime → anticipate the move** |

The key distinction: every other strategy reacts to a *current* mispricing. This one positions *before* the mispricing corrects, by detecting the regime conditions that historically precede spread movement.