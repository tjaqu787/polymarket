#!/usr/bin/env python3
"""
Comprehensive test suite for Factored Gamma Timing Model.

Tests all components:
- GammaCDFFitter: MLE fitting and bootstrap CI
- EmpiricalBayesFactors: Factor estimation and adjustment
- FactorAdjustment: Log-space parameter adjustments
- FactoredGammaModel: Full model pipeline
- Integration tests: End-to-end validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import gamma

from models.factored_gamma_model import (
    GammaCDFFitter,
    EmpiricalBayesFactors,
    FactorAdjustment,
    FactoredGammaModel,
    FitResult,
    PredictionResult
)

# Test configuration
N_BOOTSTRAP_TEST = 50  # Reduced for speed
TOLERANCE_PCT = 30  # 30% tolerance for parameter recovery

class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def record(self, name, passed, message=""):
        self.tests.append((name, passed, message))
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
            if message:
                print(f"    {message}")
        else:
            self.failed += 1
            print(f"  ✗ {name}")
            if message:
                print(f"    {message}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed, {self.failed}/{total} failed")
        print(f"{'='*70}")
        if self.failed > 0:
            print("\nFailed tests:")
            for name, passed, msg in self.tests:
                if not passed:
                    print(f"  - {name}: {msg}")
        return self.failed == 0

results = TestResults()

print("="*70)
print("FACTORED GAMMA MODEL - COMPREHENSIVE TEST SUITE")
print("="*70)

# =============================================================================
# TEST SUITE 1: GammaCDFFitter
# =============================================================================
print("\n[SUITE 1] GammaCDFFitter Tests")
print("-"*70)

# Test 1.1: Basic fitting with synthetic data
print("\n[1.1] Synthetic data fit (known parameters)")
np.random.seed(42)
true_alpha, true_beta = 2.0, 0.5
times = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
true_cdf = gamma.cdf(times, true_alpha, 0, 1/true_beta)
noisy_cdf = np.clip(true_cdf + np.random.normal(0, 0.02, len(times)), 0, 1)

fitter = GammaCDFFitter()
try:
    fit_result = fitter.fit(times, noisy_cdf)
    alpha_fit, beta_fit = fit_result['alpha'], fit_result['beta']
    alpha_error = abs(alpha_fit - true_alpha) / true_alpha * 100
    beta_error = abs(beta_fit - true_beta) / true_beta * 100

    passed = alpha_error < TOLERANCE_PCT and beta_error < TOLERANCE_PCT
    results.record(
        "Parameter recovery",
        passed,
        f"α error={alpha_error:.1f}%, β error={beta_error:.1f}% (tolerance={TOLERANCE_PCT}%)"
    )
except Exception as e:
    results.record("Parameter recovery", False, str(e))

# Test 1.2: CDF prediction
print("\n[1.2] CDF prediction accuracy")
try:
    pred_cdf = fitter.predict_cdf(times, alpha_fit, beta_fit)
    pred_rmse = np.sqrt(np.mean((pred_cdf - noisy_cdf)**2))
    passed = pred_rmse < 0.05
    results.record(
        "CDF prediction",
        passed,
        f"RMSE={pred_rmse:.4f}"
    )
except Exception as e:
    results.record("CDF prediction", False, str(e))

# Test 1.3: Bootstrap CI computation
print("\n[1.3] Bootstrap credible intervals")
try:
    ci_result = fitter.bootstrap_credible_interval(
        times=times,
        cdf_values=noisy_cdf,
        eval_times=times,
        ci_level=0.70,
        n_bootstrap=N_BOOTSTRAP_TEST
    )

    interval_widths = ci_result['upper'] - ci_result['lower']
    widths_nonzero = np.all(interval_widths > 0)
    widths_vary = np.std(interval_widths) > 0

    results.record(
        "Bootstrap CI - intervals non-zero",
        widths_nonzero,
        f"min width={interval_widths.min():.4f}"
    )
    results.record(
        "Bootstrap CI - intervals vary",
        widths_vary,
        f"std={np.std(interval_widths):.4f}"
    )
except Exception as e:
    results.record("Bootstrap CI", False, str(e))

# Test 1.4: Edge case - too few points
print("\n[1.4] Edge case: too few points")
try:
    short_times = np.array([1.0, 2.0])
    short_cdf = np.array([0.3, 0.6])
    fitter.fit(short_times, short_cdf)
    results.record("Too few points", False, "Should have raised error")
except ValueError:
    results.record("Too few points", True, "Correctly rejected")
except Exception as e:
    results.record("Too few points", False, f"Wrong error: {e}")

# Test 1.5: Edge case - non-monotonic CDF
print("\n[1.5] Edge case: non-monotonic times")
try:
    bad_times = np.array([1.0, 3.0, 2.0, 4.0])
    bad_cdf = np.array([0.2, 0.4, 0.3, 0.6])
    fitter.fit(bad_times, bad_cdf)
    results.record("Non-monotonic times", False, "Should have raised error")
except ValueError:
    results.record("Non-monotonic times", True, "Correctly rejected")
except Exception as e:
    results.record("Non-monotonic times", False, f"Wrong error: {e}")

# Test 1.6: CDF to PDF conversion
print("\n[1.6] CDF to PDF conversion")
try:
    time_mids, pdf_vals = fitter.cdf_to_pdf(times, true_cdf)

    # PDF should be non-negative
    pdf_nonneg = np.all(pdf_vals >= 0)

    # PDF should integrate to approximately 1
    integral = np.sum(pdf_vals * np.diff(times))
    integral_ok = abs(integral - 1.0) < 0.1

    results.record("PDF non-negative", pdf_nonneg)
    results.record("PDF integrates to 1", integral_ok, f"integral={integral:.3f}")
except Exception as e:
    results.record("CDF to PDF", False, str(e))

# =============================================================================
# TEST SUITE 2: EmpiricalBayesFactors
# =============================================================================
print("\n[SUITE 2] EmpiricalBayesFactors Tests")
print("-"*70)

# Test 2.1: Basic factor estimation
print("\n[2.1] Factor estimation from mock data")
try:
    # Create mock resolved events data
    n_events = 30
    mock_data = []

    for i in range(n_events):
        cat = ['politics', 'crypto', 'sports'][i % 3]
        event_times = np.array([0.5, 1.0, 2.0, 3.0])
        event_cdf = gamma.cdf(event_times, 2.0 + np.random.uniform(-0.5, 0.5), 0, 2.0)

        mock_data.append({
            'event_id': f'event_{i}',
            'category': cat,
            'times': event_times,
            'cdf_values': event_cdf,
            'ts_slope': np.random.uniform(-1, 1),
            'ts_curvature': np.random.uniform(-1, 1),
            'implied_rate': np.random.uniform(0, 0.5)
        })

    mock_df = pd.DataFrame(mock_data)

    eb_factors = EmpiricalBayesFactors()
    eb_factors.fit(mock_df, fitter, min_events_per_category=5)

    # Check that factors were fitted
    factors_exist = eb_factors.fitted and eb_factors.factors is not None
    results.record("Factors fitted", factors_exist)

    # Check that categories have different adjustments
    if factors_exist:
        theta_vals = list(eb_factors.factors['theta_shape'].values())
        phi_vals = list(eb_factors.factors['phi_rate'].values())

        theta_vary = len(set(theta_vals)) > 1
        phi_vary = len(set(phi_vals)) > 1

        results.record(
            "Category adjustments differ",
            theta_vary and phi_vary,
            f"theta range: {min(theta_vals):.3f} to {max(theta_vals):.3f}"
        )

except Exception as e:
    results.record("Factor estimation", False, str(e))
    import traceback
    traceback.print_exc()

# Test 2.2: Get factor adjustments
print("\n[2.2] Get factor adjustments")
try:
    if eb_factors.fitted:
        adjustments = eb_factors.get_factor_adjustments(
            category='politics',
            ts_slope=0.5,
            ts_curvature=0.3,
            implied_rate=0.2
        )

        has_keys = 'log_alpha_adjustment' in adjustments and 'log_beta_adjustment' in adjustments
        results.record("Factor adjustment keys", has_keys)

        # Adjustments should be non-zero for some inputs
        nonzero = adjustments['log_alpha_adjustment'] != 0 or adjustments['log_beta_adjustment'] != 0
        results.record(
            "Factor adjustments non-zero",
            nonzero,
            f"α_adj={adjustments['log_alpha_adjustment']:.3f}, β_adj={adjustments['log_beta_adjustment']:.3f}"
        )
except Exception as e:
    results.record("Get factor adjustments", False, str(e))

# Test 2.3: Unknown category handling
print("\n[2.3] Unknown category handling")
try:
    if eb_factors.fitted:
        adjustments_unknown = eb_factors.get_factor_adjustments(
            category='unknown_category',
            ts_slope=0.0,
            ts_curvature=0.0,
            implied_rate=0.0
        )

        # Should use reference category (0 adjustment)
        uses_reference = adjustments_unknown['log_alpha_adjustment'] == 0.0 and adjustments_unknown['log_beta_adjustment'] == 0.0
        results.record("Unknown category fallback", uses_reference)
except Exception as e:
    results.record("Unknown category", False, str(e))

# =============================================================================
# TEST SUITE 3: FactorAdjustment
# =============================================================================
print("\n[SUITE 3] FactorAdjustment Tests")
print("-"*70)

# Test 3.1: Basic adjustment
print("\n[3.1] Basic parameter adjustment")
try:
    if eb_factors.fitted:
        alpha_base, beta_base = 2.0, 0.5
        alpha_adj, beta_adj = FactorAdjustment.adjust(
            alpha_base=alpha_base,
            beta_base=beta_base,
            category='politics',
            ts_slope=0.5,
            ts_curvature=0.3,
            implied_rate=0.2,
            eb_factors=eb_factors
        )

        # Adjusted should differ from base
        params_changed = alpha_adj != alpha_base or beta_adj != beta_base
        results.record(
            "Parameters adjusted",
            params_changed,
            f"α: {alpha_base:.3f}→{alpha_adj:.3f}, β: {beta_base:.3f}→{beta_adj:.3f}"
        )

        # Adjusted should remain positive
        params_positive = alpha_adj > 0 and beta_adj > 0
        results.record("Adjusted params positive", params_positive)
except Exception as e:
    results.record("Basic adjustment", False, str(e))

# Test 3.2: Edge case - zero base parameter
print("\n[3.2] Edge case: invalid base parameters")
try:
    if eb_factors.fitted:
        FactorAdjustment.adjust(
            alpha_base=0.0,
            beta_base=0.5,
            category='politics',
            ts_slope=0.0,
            ts_curvature=0.0,
            implied_rate=0.0,
            eb_factors=eb_factors
        )
        results.record("Zero alpha_base", False, "Should have raised error")
except ValueError:
    results.record("Zero alpha_base", True, "Correctly rejected")
except Exception as e:
    results.record("Zero alpha_base", False, f"Wrong error: {e}")

# Test 3.3: NaN handling
print("\n[3.3] NaN feature handling")
try:
    if eb_factors.fitted:
        alpha_adj, beta_adj = FactorAdjustment.adjust(
            alpha_base=2.0,
            beta_base=0.5,
            category='politics',
            ts_slope=np.nan,
            ts_curvature=np.nan,
            implied_rate=np.nan,
            eb_factors=eb_factors
        )

        # Should treat NaN as 0 and still work
        nan_handled = not np.isnan(alpha_adj) and not np.isnan(beta_adj)
        results.record("NaN features handled", nan_handled)
except Exception as e:
    results.record("NaN handling", False, str(e))

# =============================================================================
# TEST SUITE 4: FactoredGammaModel
# =============================================================================
print("\n[SUITE 4] FactoredGammaModel Tests")
print("-"*70)

# Test 4.1: Model instantiation
print("\n[4.1] Model instantiation")
try:
    model = FactoredGammaModel(
        min_buckets=3,
        max_rmse=0.3,
        ci_level=0.70,
        n_bootstrap=N_BOOTSTRAP_TEST
    )

    config_correct = (
        model.min_buckets == 3 and
        model.max_rmse == 0.3 and
        model.ci_level == 0.70 and
        model.n_bootstrap == N_BOOTSTRAP_TEST
    )
    results.record("Model instantiation", config_correct)
except Exception as e:
    results.record("Model instantiation", False, str(e))

# Test 4.2: Date parsing
print("\n[4.2] Target date extraction")
test_questions = [
    ("Will X happen by March 15, 2026?", datetime(2026, 3, 15)),
    ("Will Y occur before April 1, 2026?", datetime(2026, 4, 1)),
    ("Will Z be done no later than June 30, 2026?", datetime(2026, 6, 30)),
]

for question, expected_date in test_questions:
    try:
        parsed_date = model._extract_target_date(question)
        correct = parsed_date == expected_date
        results.record(
            f"Parse: '{question[:30]}...'",
            correct,
            f"Got {parsed_date}, expected {expected_date}"
        )
    except Exception as e:
        results.record(f"Parse: '{question[:30]}...'", False, str(e))

# Test 4.3: fit_event with mock data
print("\n[4.3] fit_event with mock data")
try:
    # Create mock event data
    current_date = pd.Timestamp('2025-11-05')
    target_dates = [
        current_date + timedelta(days=30),
        current_date + timedelta(days=90),
        current_date + timedelta(days=180),
        current_date + timedelta(days=365),
    ]

    mock_event_data = []
    for i, target in enumerate(target_dates):
        no_price = 0.9 - (i * 0.15)  # Declining prices
        question = f"Will event happen by {target.strftime('%B %d, %Y')}?"

        mock_event_data.append({
            'market_id': f'market_{i}',
            'token_id': f'token_{i}',
            'outcome': 'No',
            'price': no_price,
            'question': question,
            'category': 'politics',
            'ts_slope': 0.5,
            'ts_curvature': 0.2,
            'implied_rate': 0.3
        })

    event_df = pd.DataFrame(mock_event_data)

    fit_result = model.fit_event(event_df, current_date, 'test_event')

    if fit_result is not None:
        results.record("fit_event succeeded", True)

        # Check FitResult structure
        has_base_params = fit_result.alpha_base > 0 and fit_result.beta_base > 0
        results.record("Base parameters positive", has_base_params)

        # Check adjusted params differ (if factors fitted)
        if model.factors_fitted:
            params_differ = (
                fit_result.alpha_adjusted != fit_result.alpha_base or
                fit_result.beta_adjusted != fit_result.beta_base
            )
            results.record("Adjusted params differ", params_differ)

        # Check CI bounds exist
        has_ci = len(fit_result.credible_intervals) > 0
        results.record("Credible intervals computed", has_ci)

    else:
        results.record("fit_event succeeded", False, "Returned None")

except Exception as e:
    results.record("fit_event", False, str(e))
    import traceback
    traceback.print_exc()

# Test 4.4: predict()
print("\n[4.4] predict() for specific market")
try:
    if fit_result is not None:
        market_id = list(fit_result.credible_intervals.keys())[0]
        prediction = model.predict(market_id, fit_result, current_date)

        if prediction is not None:
            results.record("predict() succeeded", True)

            # Check prediction structure
            has_bounds = prediction.lower_bound < prediction.upper_bound
            results.record("CI bounds ordered correctly", has_bounds)

            # Check interval width
            width_positive = prediction.interval_width > 0
            results.record(
                "Interval width positive",
                width_positive,
                f"width={prediction.interval_width:.4f}"
            )
        else:
            results.record("predict()", False, "Returned None")
except Exception as e:
    results.record("predict()", False, str(e))

# =============================================================================
# TEST SUITE 5: Integration Tests
# =============================================================================
print("\n[SUITE 5] Integration Tests")
print("-"*70)

# Test 5.1: Full pipeline without EB factors
print("\n[5.1] Full pipeline: base model (no factors)")
try:
    model_no_factors = FactoredGammaModel(
        min_buckets=3,
        max_rmse=0.5,
        ci_level=0.70,
        n_bootstrap=N_BOOTSTRAP_TEST
    )

    fit_result_base = model_no_factors.fit_event(event_df, current_date, 'test_event')

    if fit_result_base is not None:
        # Without factors, adjusted should equal base
        no_adjustment = (
            fit_result_base.alpha_adjusted == fit_result_base.alpha_base and
            fit_result_base.beta_adjusted == fit_result_base.beta_base
        )
        results.record(
            "No factors: adjusted = base",
            no_adjustment,
            f"α: {fit_result_base.alpha_adjusted:.3f} vs {fit_result_base.alpha_base:.3f}"
        )
    else:
        results.record("Base model pipeline", False, "fit_event returned None")

except Exception as e:
    results.record("Base model pipeline", False, str(e))

# Test 5.2: CI width comparison
print("\n[5.2] CI widths reflect uncertainty")
try:
    if fit_result is not None:
        widths = [ci['upper'] - ci['lower']
                 for ci in fit_result.credible_intervals.values()]

        # Widths should vary (not all identical)
        widths_vary = len(set([round(w, 6) for w in widths])) > 1
        results.record(
            "CI widths vary",
            widths_vary,
            f"range: {min(widths):.4f} to {max(widths):.4f}"
        )

        # Mean width should be reasonable (not too tight or too wide)
        mean_width = np.mean(widths)
        width_reasonable = 0.001 < mean_width < 0.5
        results.record(
            "CI widths reasonable",
            width_reasonable,
            f"mean width={mean_width:.4f}"
        )
except Exception as e:
    results.record("CI width analysis", False, str(e))

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
success = results.summary()
print("="*70)

if success:
    print("\n🎉 ALL TESTS PASSED 🎉")
    print("\nThe Factored Gamma Model is ready for backtesting!")
    sys.exit(0)
else:
    print("\n⚠️  SOME TESTS FAILED ⚠️")
    print("\nPlease review failures above before running backtest.")
    sys.exit(1)
