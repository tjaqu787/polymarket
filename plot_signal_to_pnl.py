#!/usr/bin/env python3
"""
Plot signal strength vs P&L for spread dynamics trades.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trades
trades = pd.read_csv('backtest_results/spread_dynamics_trades.csv')

# Filter to entry trades only (exit trades have empty spread_velocity)
entry_trades = trades[trades['trade_type'].notna()].copy()

print(f"Total trades: {len(trades)}")
print(f"Entry trades: {len(entry_trades)}")
print(f"Compression: {len(entry_trades[entry_trades['trade_type'] == 'compression'])}")
print(f"Widening: {len(entry_trades[entry_trades['trade_type'] == 'widening'])}")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Signal Strength vs P&L Analysis', fontsize=16, fontweight='bold')

# 1. Spread Velocity vs P&L (All trades)
ax = axes[0, 0]
scatter = ax.scatter(entry_trades['spread_velocity'], entry_trades['pnl'],
                    c=entry_trades['pnl'], cmap='RdYlGn', alpha=0.6, s=50)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Spread Velocity (entry signal)')
ax.set_ylabel('P&L ($)')
ax.set_title('Spread Velocity vs P&L (All Trades)')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='P&L')

# 2. Spread Velocity vs P&L (By Regime)
ax = axes[0, 1]
for trade_type, color in [('compression', 'blue'), ('widening', 'red')]:
    data = entry_trades[entry_trades['trade_type'] == trade_type]
    ax.scatter(data['spread_velocity'], data['pnl'],
              label=trade_type.capitalize(), alpha=0.6, s=50, color=color)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Spread Velocity')
ax.set_ylabel('P&L ($)')
ax.set_title('Spread Velocity vs P&L by Regime')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Volume Z-score vs P&L
ax = axes[0, 2]
scatter = ax.scatter(entry_trades['vol_zscore'], entry_trades['pnl'],
                    c=entry_trades['pnl'], cmap='RdYlGn', alpha=0.6, s=50)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Volume Z-score')
ax.set_ylabel('P&L ($)')
ax.set_title('Volume Z-score vs P&L')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='P&L')

# 4. Binned Average P&L by Velocity
ax = axes[1, 0]
velocity_bins = pd.cut(entry_trades['spread_velocity'], bins=10)
binned_pnl = entry_trades.groupby(velocity_bins)['pnl'].agg(['mean', 'std', 'count'])
bin_centers = [interval.mid for interval in binned_pnl.index]
ax.bar(bin_centers, binned_pnl['mean'], width=0.05, alpha=0.7,
       color=['red' if x < 0 else 'green' for x in binned_pnl['mean']])
ax.errorbar(bin_centers, binned_pnl['mean'], yerr=binned_pnl['std'],
           fmt='none', color='black', alpha=0.5, capsize=3)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Spread Velocity (binned)')
ax.set_ylabel('Average P&L ($)')
ax.set_title('Average P&L by Velocity Bin')
ax.grid(True, alpha=0.3, axis='y')

# Add sample counts on bars
for i, (center, count) in enumerate(zip(bin_centers, binned_pnl['count'])):
    ax.text(center, binned_pnl['mean'].iloc[i], f'n={int(count)}',
           ha='center', va='bottom' if binned_pnl['mean'].iloc[i] > 0 else 'top',
           fontsize=8)

# 5. Velocity Magnitude vs P&L
ax = axes[1, 1]
entry_trades['velocity_abs'] = entry_trades['spread_velocity'].abs()
scatter = ax.scatter(entry_trades['velocity_abs'], entry_trades['pnl'],
                    c=entry_trades['pnl'], cmap='RdYlGn', alpha=0.6, s=50)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('|Spread Velocity| (signal strength)')
ax.set_ylabel('P&L ($)')
ax.set_title('Signal Strength vs P&L')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='P&L')

# Add trend line
z = np.polyfit(entry_trades['velocity_abs'], entry_trades['pnl'], 1)
p = np.poly1d(z)
x_trend = np.linspace(entry_trades['velocity_abs'].min(), entry_trades['velocity_abs'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
ax.legend()

# 6. Win Rate by Velocity Bins
ax = axes[1, 2]
entry_trades['is_win'] = entry_trades['pnl'] > 0
velocity_bins_winrate = pd.cut(entry_trades['velocity_abs'], bins=8)
binned_winrate = entry_trades.groupby(velocity_bins_winrate)['is_win'].agg(['mean', 'count'])
bin_centers_wr = [interval.mid for interval in binned_winrate.index]
ax.bar(bin_centers_wr, binned_winrate['mean'] * 100, width=0.05, alpha=0.7, color='steelblue')
ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% (random)')
ax.set_xlabel('|Spread Velocity|')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate by Signal Strength')
ax.set_ylim([0, 100])
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

# Add sample counts
for center, count in zip(bin_centers_wr, binned_winrate['count']):
    idx = list(bin_centers_wr).index(center)
    ax.text(center, binned_winrate['mean'].iloc[idx] * 100 + 3, f'n={int(count)}',
           ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('backtest_results/signal_to_pnl.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to: backtest_results/signal_to_pnl.png")

# Summary Statistics
print("\n" + "="*60)
print("SIGNAL-TO-PNL SUMMARY STATISTICS")
print("="*60)

print("\nBy Trade Type:")
for trade_type in ['compression', 'widening']:
    data = entry_trades[entry_trades['trade_type'] == trade_type]
    print(f"\n{trade_type.upper()}:")
    print(f"  Avg velocity:     {data['spread_velocity'].abs().mean():.4f}")
    print(f"  Avg P&L:          ${data['pnl'].mean():.4f}")
    print(f"  Win rate:         {(data['pnl'] > 0).mean()*100:.1f}%")
    print(f"  Total trades:     {len(data)}")

print("\nCorrelation Analysis:")
print(f"  Velocity vs P&L:     {entry_trades['spread_velocity'].corr(entry_trades['pnl']):.3f}")
print(f"  |Velocity| vs P&L:   {entry_trades['velocity_abs'].corr(entry_trades['pnl']):.3f}")
print(f"  Vol Z-score vs P&L:  {entry_trades['vol_zscore'].corr(entry_trades['pnl']):.3f}")
print(f"  Volume vs P&L:       {entry_trades['volume'].corr(entry_trades['pnl']):.3f}")

# Velocity threshold analysis
print("\nPerformance by Velocity Threshold:")
for threshold in [0.05, 0.10, 0.15, 0.20]:
    filtered = entry_trades[entry_trades['velocity_abs'] >= threshold]
    if len(filtered) > 0:
        print(f"\n  |Velocity| >= {threshold:.2f}:")
        print(f"    Trades:     {len(filtered)}")
        print(f"    Avg P&L:    ${filtered['pnl'].mean():.4f}")
        print(f"    Win rate:   {(filtered['pnl'] > 0).mean()*100:.1f}%")
    else:
        print(f"\n  |Velocity| >= {threshold:.2f}: No trades")

plt.show()
