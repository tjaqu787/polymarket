# Bayesian Kelly Criterion Implementation

## Overview
This implements a **Bayesian extension of the Kelly Criterion** for position sizing in spread trading. Instead of treating the edge as a known quantity, we maintain a probability distribution over the true edge and update our beliefs as we observe trade outcomes.

## Mathematical Foundation

### The Problem
Traditional Kelly Criterion assumes you know the edge `μ` perfectly:
```
f = μ / σ²
```

But in reality, we only have noisy observations of returns. We're uncertain about the true edge!

### Bayesian Solution

**Model the edge as a random variable:**
```
edge ~ Normal(μ, σ²)
```

**Prior (initial belief before any trades):**
```
edge ~ N(μ₀, σ₀²)
```
- μ₀ = 0 (assume no edge initially)
- σ₀ = 0.05 (wide uncertainty)

**Likelihood (observed returns):**
```
observed_return ~ N(edge, σ_obs²)
```
- σ_obs = 0.02 (observation noise)

**Posterior (updated belief after observing returns):**

Using Normal-Normal conjugate prior, we get analytical updates:

```python
# Precision = 1/variance
precision_prior = 1 / σ₀²
precision_obs = 1 / σ_obs²
precision_posterior = precision_prior + precision_obs

# Posterior variance shrinks with more observations
σ_posterior² = 1 / precision_posterior

# Posterior mean is precision-weighted average of prior and data
μ_posterior = σ_posterior² * (μ₀ * precision_prior + r * precision_obs)
```

**Key insight:** Each observation makes us more certain (σ_posterior decreases)!

### Position Sizing with Uncertainty

**Bayesian Kelly Formula:**
```
f = μ_posterior / (σ_obs² + σ_posterior²)
```

This automatically reduces position size when:
1. **Uncertain about edge** (σ_posterior is large) → bet small
2. **Confident about edge** (σ_posterior is small) → bet larger
3. **Edge is near zero** (μ_posterior ≈ 0) → bet nothing

## Implementation Details

### 1. Posterior Tracking
```python
self.edge_posterior[pair_id] = {
    'mu': posterior_mean,
    'sigma': posterior_std,
    'n_obs': number_of_observations
}
```

### 2. Bayesian Update (after each trade closes)
```python
def update_edge_posterior(self, pair_id, observed_return):
    """Update beliefs using Normal-Normal conjugate prior"""
    posterior = self.edge_posterior[pair_id]

    # Get prior
    mu_prior = posterior['mu']
    sigma_prior = posterior['sigma']

    # Bayesian update
    precision_posterior = 1/sigma_prior² + 1/sigma_obs²
    sigma_posterior = sqrt(1 / precision_posterior)

    mu_posterior = sigma_posterior² * (
        mu_prior/sigma_prior² + observed_return/sigma_obs²
    )

    # Store updated posterior
    self.edge_posterior[pair_id] = {
        'mu': mu_posterior,
        'sigma': sigma_posterior,
        'n_obs': posterior['n_obs'] + 1
    }
```

### 3. Position Sizing
```python
def _calculate_position_size(self, pair_id, edge):
    """Use posterior distribution instead of point estimate"""
    posterior = self.edge_posterior[pair_id]
    mu_posterior = posterior['mu']
    sigma_posterior = posterior['sigma']

    # Bayesian Kelly with uncertainty penalty
    total_variance = sigma_obs² + sigma_posterior²
    position_size = kelly_fraction * (mu_posterior / total_variance)

    return min(position_size, max_position)
```

## Advantages Over Classical Kelly

1. **Automatic risk management**: Bets small when uncertain, large when confident
2. **No arbitrary lookback windows**: Uses all historical data optimally
3. **Asymmetric learning**: Bad outcomes increase uncertainty, good outcomes build confidence
4. **Theoretically principled**: Maximizes expected log wealth under uncertainty

## Example Evolution

**Trade 1** (initial):
- Prior: μ = 0.00, σ = 0.05
- Observed return: +0.03
- Posterior: μ = 0.015, σ = 0.018
- Position size: Small (still uncertain)

**Trade 10** (after 9 observations):
- Prior: μ = 0.015, σ = 0.006
- Observed return: +0.02
- Posterior: μ = 0.016, σ = 0.005
- Position size: Larger (more confident)

**Trade 100**:
- Prior: μ = 0.018, σ = 0.002
- Observed return: -0.01 (bad trade!)
- Posterior: μ = 0.017, σ = 0.002
- Position size: Still large (one bad trade doesn't destroy confidence)

## How to Use

1. **Enable in config:**
```python
"use_bayesian_kelly": True,
"prior_edge_mean": 0.0,
"prior_edge_std": 0.05,
"obs_std": 0.02,
```

2. **Run backtest:**
```bash
python3 run_spread_dynamics_backtest.py
```

3. **Analyze posteriors:**
The script outputs:
- Average uncertainty reduction
- Top pairs by certainty
- Top pairs by positive edge
- Full posterior distribution per pair

## For Your Class Presentation

**Key points to emphasize:**

1. **Conjugate priors**: Normal-Normal gives analytical updates (no MCMC needed!)
2. **Sequential learning**: Online Bayesian updating as trades complete
3. **Uncertainty quantification**: Position sizes reflect confidence
4. **Practical impact**: Compare returns with/without Bayesian Kelly

**Cool visualizations:**
- Plot μ_posterior vs n_observations (learning curve)
- Plot σ_posterior vs n_observations (uncertainty reduction)
- Compare position sizes: Classical Kelly vs Bayesian Kelly

## References
- Kelly, J. L. (1956). "A New Interpretation of Information Rate"
- Thorp, E. O. (2008). "The Kelly Criterion in Blackjack Sports Betting"
- Gelman et al. (2013). "Bayesian Data Analysis" (Chapter 2: Conjugate Priors)
