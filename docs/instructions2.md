Strategy Spec: SurvivalConditionalStrategy
Concept
Prediction markets price P(event by T) — a cumulative probability. But as time passes without the event occurring, the correct price to hold is the conditional (survival-adjusted) probability:
P(event in [T1, T2] | not happened yet) = [F(T2) - F(T1)] / [1 - F(T1)]
Markets are systematically slow to update this. Each week that passes without an event should compress the remaining probability mass onto shorter remaining windows — but often doesn't. The edge: sell the stale cumulative price, buy the correctly-repriced conditional.

File Location
backtest/strategies/survival_conditional_strategy.py

Core Math
Given the existing Poisson/Gamma CDF framework already in the repo:
F(t) = P(event by t)          # cumulative — what markets quote
S(t) = 1 - F(t)               # survival function
h(t) = f(t) / S(t)            # hazard rate at time t

Conditional prob for window [T1, T2]:
  P(event in [T1, T2] | survived to T1) = [F(T2) - F(T1)] / S(T1)
The mispricing signal is the gap between what the market quotes and the survival-adjusted fair value.

Class Design
pythonclass SurvivalConditionalStrategy(Strategy):
    """
    Trades conditional (survival-adjusted) probability mispricing in
    time-distributed prediction markets.

    Markets quote P(event by T). As time passes without resolution,
    correct pricing requires conditioning on survival:
        P(event in [T1, T2] | not yet) = [F(T2) - F(T1)] / [1 - F(T1)]

    Markets are slow to reprice this. We exploit the gap.
    """

Configuration Parameters
ParameterDefaultDescriptionmin_tte_days7Ignore contracts expiring in < 7 days (noise-dominated)max_tte_days365Ignore very long-dated contractsrefit_days7Days between model refits per event groupmin_survival_edge0.04Minimum gap between market price and conditional fair value to trademin_volume500Minimum cumulative volume to consider a marketkelly_fraction0.25Fractional Kelly sizingmax_position0.10Max position as fraction of portfoliodistribution'gamma'CDF model: 'exponential', 'gamma', 'weibull'roll_forward_days7How often to re-evaluate conditional repricing ("it didn't happen this week → now what?")

Data Requirements
Uses load_timing_markets() — same as existing timing strategies. Requires markets grouped by event_id (semantic group) with at least 2 active contracts at different maturities to fit a CDF.

Signal Logic
Step 1 — Fit the CDF
For each event group on each refit_days interval, fit a Gamma/Weibull CDF to the term structure of prices (reusing PoissonTimingModel / GammaCDFFitter already in the repo).
Step 2 — Compute Conditional Fair Value
For each active contract with maturity T2, given that the current date is T1 (i.e., the event hasn't happened yet):
pythonconditional_prob = (F(T2) - F(T1)) / (1 - F(T1))
where T1 = 0 effectively at trade entry but advances with each roll_forward_days check.
Step 3 — Compute the Edge
edge = market_price - conditional_fair_value

edge > +min_survival_edge → market overprices the window → SHORT (sell Yes / buy No)
edge < -min_survival_edge → market underprices the window → LONG (buy Yes)

Step 4 — Rolling Forward ("it didn't happen this week → now what?")
Every roll_forward_days, for all open positions:

Recompute conditional_fair_value with updated T1
If the edge has collapsed (market has repriced correctly) → exit
If the edge has widened → optionally add (up to max_position)
If the event still hasn't happened but the contract is now < min_tte_days → close (too close to expiry, noise dominates)


Position Sizing
Use fractional Kelly, same pattern as BayesianCarryStrategy:
pythonkelly_f = edge / variance_of_edge   # from model posterior if Bayesian, else use edge^2
position_size = kelly_fraction * kelly_f
position_size = min(position_size, max_position)

Exit Rules
ConditionActionEdge collapses below min_survival_edge / 2Close positionContract < min_tte_days to expiryClose positionMarket resolves (Yes/No)Engine handles via on_market_closeConditional fair value crosses market price (sign flip)Close and optionally reverse

Key Implementation Notes
Reuse existing model infrastructure. Don't refit from scratch — call PoissonTimingModel.extract_cdf_from_prices() and GammaCDFFitter that already exist. The survival math is a thin layer on top.
T1 is today, not entry date. The conditioning point advances in real time. Each new day without resolution shifts T1 forward and compresses S(T1).
Watch for near-zero survival. If S(T1) < 0.05 (event is 95%+ likely already happened / failed), skip — the conditional probability becomes numerically unstable and the market is probably in resolution limbo.
Group-level exposure cap. Same pattern as other strategies: cap total exposure per event_id semantic group (e.g., 15% of portfolio), not just per contract.

Run Script
run_survival_conditional_backtest.py
Mirrors run_time_discounting_backtest.py. Config:
pythonstrategy = SurvivalConditionalStrategy(config={
    "db_path": DB_PATH,
    "min_survival_edge": 0.04,
    "kelly_fraction": 0.25,
    "max_position": 0.10,
    "refit_days": 7,
    "roll_forward_days": 7,
    "distribution": "gamma",
    "min_volume": 500,
})

What Makes This Different from Existing Strategies
StrategyCore signalCarryStrategyTime decay on extreme probabilities (>90%, <10%)PoissonTimingStrategyCI bounds from fitted CDF vs market priceFactoredGammaStrategyRate edge: implied rate differenceSurvivalConditionalStrategyConditional repricing gap as event non-occurrence accumulates
The carry strategies exploit static mispricing at extremes. This strategy exploits dynamic mispricing that compounds over time — every passing week without resolution is new information the market is slow to incorporate.