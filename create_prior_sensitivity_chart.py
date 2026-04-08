#!/usr/bin/env python3
"""
Create prior sensitivity visualization for presentation.

Shows how total PnL varies with different Bayesian prior assumptions,
demonstrating robustness (or lack thereof) to prior choice.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
results_file = Path("backtest_results/prior_sensitivity_results.csv")

if not results_file.exists():
    print(f"❌ Error: {results_file} not found!")
    print("Run prior_sensitivity_analysis.py first to generate results.")
    exit(1)

print(f"Loading results from: {results_file}")
df = pd.read_csv(results_file)
df = df.sort_values('prior_std')

print(f"Loaded {len(df)} results")
print(f"Prior std range: [{df['prior_std'].min():.3f}, {df['prior_std'].max():.3f}]")
print(f"PnL range: [${df['total_pnl'].min():.4f}, ${df['total_pnl'].max():.4f}]")

# Create output directory
output_dir = Path("presentation_charts")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# CHART: Prior Sensitivity - Total PnL vs Prior Std
# ============================================================================
print("\nGenerating Prior Sensitivity Chart...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# Chart 1: Prior Std vs Total PnL (THE KEY CHART)
ax1.plot(df['prior_std'], df['total_pnl'], marker='o', linewidth=2.5,
         markersize=6, color='#2E86AB', alpha=0.8)
ax1.fill_between(df['prior_std'], df['total_pnl'], 0, alpha=0.15, color='#2E86AB')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

# Highlight min and max
max_idx = df['total_pnl'].idxmax()
min_idx = df['total_pnl'].idxmin()
ax1.scatter(df.loc[max_idx, 'prior_std'], df.loc[max_idx, 'total_pnl'],
            color='green', s=150, zorder=5, label=f'Max: ${df.loc[max_idx, "total_pnl"]:.2f}')
ax1.scatter(df.loc[min_idx, 'prior_std'], df.loc[min_idx, 'total_pnl'],
            color='red', s=150, zorder=5, label=f'Min: ${df.loc[min_idx, "total_pnl"]:.2f}')

ax1.set_xlabel('Prior Standard Deviation', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total PnL ($)', fontsize=12, fontweight='bold')
ax1.set_title('Total PnL vs Prior Choice', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add range annotation
pnl_range = df['total_pnl'].max() - df['total_pnl'].min()
ax1.text(0.02, 0.98, f'PnL Range: ${pnl_range:.4f}',
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontweight='bold')

# Chart 2: Prior Std vs Number of Trades
ax2.plot(df['prior_std'], df['n_trades'], marker='s', linewidth=2.5,
         markersize=6, color='#A23B72', alpha=0.8)
ax2.axhline(y=df['n_trades'].mean(), color='orange', linestyle='--',
            linewidth=1.5, alpha=0.7, label=f'Mean: {df["n_trades"].mean():.0f}')

ax2.set_xlabel('Prior Standard Deviation', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
ax2.set_title('Trade Count vs Prior Choice', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Chart 3: Prior Std vs Uncertainty Reduction
ax3.plot(df['prior_std'], df['avg_uncertainty_reduction'], marker='D', linewidth=2.5,
         markersize=6, color='#06A77D', alpha=0.8)
ax3.axhline(y=df['avg_uncertainty_reduction'].mean(), color='orange', linestyle='--',
            linewidth=1.5, alpha=0.7, label=f'Mean: {df["avg_uncertainty_reduction"].mean():.1f}%')

ax3.set_xlabel('Prior Standard Deviation', fontsize=12, fontweight='bold')
ax3.set_ylabel('Avg Uncertainty Reduction (%)', fontsize=12, fontweight='bold')
ax3.set_title('Learning Efficiency vs Prior Choice', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Add explanation
uncertainty_text = "Looser priors → more room to learn from data"
ax3.text(0.98, 0.02, uncertainty_text,
         transform=ax3.transAxes, fontsize=9, verticalalignment='bottom',
         horizontalalignment='right', style='italic', color='#555')

# Chart 4: Prior Std vs Win Rate
ax4.plot(df['prior_std'], df['win_rate'], marker='^', linewidth=2.5,
         markersize=6, color='#F18F01', alpha=0.8)
ax4.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
            label='50% (coin flip)')
ax4.axhline(y=df['win_rate'].mean(), color='orange', linestyle='--',
            linewidth=1.5, alpha=0.7, label=f'Mean: {df["win_rate"].mean():.1f}%')

ax4.set_xlabel('Prior Standard Deviation', fontsize=12, fontweight='bold')
ax4.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax4.set_title('Win Rate vs Prior Choice', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Overall title
fig.suptitle('Bayesian Prior Sensitivity Analysis: Robustness Check',
             fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig(output_dir / '07_prior_sensitivity.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '07_prior_sensitivity.png'}")
plt.close()

# ============================================================================
# SIMPLIFIED VERSION: Just the key chart for presentation
# ============================================================================
print("\nGenerating Simplified Prior Sensitivity Chart...")

fig, ax = plt.subplots(figsize=(12, 7))

# Main line plot
ax.plot(df['prior_std'], df['total_pnl'], marker='o', linewidth=3,
        markersize=8, color='#2E86AB', alpha=0.9, label='Total PnL')
ax.fill_between(df['prior_std'], df['total_pnl'], 0, alpha=0.2, color='#2E86AB')

# Zero line
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Highlight extremes
max_idx = df['total_pnl'].idxmax()
min_idx = df['total_pnl'].idxmin()
ax.scatter(df.loc[max_idx, 'prior_std'], df.loc[max_idx, 'total_pnl'],
           color='#06A77D', s=200, zorder=5, edgecolor='black', linewidth=2)
ax.scatter(df.loc[min_idx, 'prior_std'], df.loc[min_idx, 'total_pnl'],
           color='#D62828', s=200, zorder=5, edgecolor='black', linewidth=2)

# Annotations
ax.annotate(f'Max: ${df.loc[max_idx, "total_pnl"]:.2f}\n(prior σ={df.loc[max_idx, "prior_std"]:.3f})',
            xy=(df.loc[max_idx, 'prior_std'], df.loc[max_idx, 'total_pnl']),
            xytext=(10, 20), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#06A77D',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#06A77D', linewidth=2),
            arrowprops=dict(arrowstyle='->', color='#06A77D', linewidth=2))

ax.annotate(f'Min: ${df.loc[min_idx, "total_pnl"]:.2f}\n(prior σ={df.loc[min_idx, "prior_std"]:.3f})',
            xy=(df.loc[min_idx, 'prior_std'], df.loc[min_idx, 'total_pnl']),
            xytext=(10, -30), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#D62828',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#D62828', linewidth=2),
            arrowprops=dict(arrowstyle='->', color='#D62828', linewidth=2))

ax.set_xlabel('Prior Standard Deviation (σ)', fontsize=13, fontweight='bold')
ax.set_ylabel('Total PnL ($)', fontsize=13, fontweight='bold')
ax.set_title('Sensitivity to Bayesian Prior: Is the Result Robust?',
             fontsize=16, fontweight='bold', pad=20)

# Add stats box
pnl_range = df['total_pnl'].max() - df['total_pnl'].min()
pnl_mean = df['total_pnl'].mean()
pnl_std = df['total_pnl'].std()

stats_text = f"""Prior Range: [{df['prior_std'].min():.3f}, {df['prior_std'].max():.3f}]
PnL Range: ${pnl_range:.4f}
PnL Mean: ${pnl_mean:.4f}
PnL Std: ${pnl_std:.4f}
N = {len(df)} runs"""

ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
        family='monospace')

# Interpretation
if pnl_range < 5:
    interpretation = "Result is ROBUST to prior choice\n→ 'Financially irrelevant' holds regardless"
    color = '#06A77D'
elif pnl_range < 20:
    interpretation = "MODERATE sensitivity to prior\n→ Still financially irrelevant overall"
    color = '#F18F01'
else:
    interpretation = "HIGH sensitivity to prior\n→ Prior choice matters significantly"
    color = '#D62828'

ax.text(0.98, 0.02, interpretation,
        transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
        horizontalalignment='right', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=3, alpha=0.9),
        color=color)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '07_prior_sensitivity_simple.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '07_prior_sensitivity_simple.png'}")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("PRIOR SENSITIVITY VISUALIZATION COMPLETE")
print("="*70)
print(f"\nGenerated 2 charts:")
print(f"  1. 07_prior_sensitivity.png         (4-panel detailed view)")
print(f"  2. 07_prior_sensitivity_simple.png  (single clean chart for slides)")
print(f"\nKey Stats:")
print(f"  Prior std range: [{df['prior_std'].min():.3f}, {df['prior_std'].max():.3f}]")
print(f"  PnL range:       ${pnl_range:.4f}")
print(f"  PnL mean:        ${pnl_mean:.4f}")
print(f"  PnL std:         ${pnl_std:.4f}")
print(f"\n💡 Punchline:")
if pnl_range < 5:
    print("  'The result is robust—we're consistently broke across all priors.'")
elif pnl_range < 20:
    print("  'Moderate sensitivity, but we're still making pocket change regardless.'")
else:
    print("  'Prior choice matters—but even the best prior can't save us.'")
print("="*70)
