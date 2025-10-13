"""Enhanced visualization and interpretation aids for PE vs Visible Subspace experiments.

This module provides additional plotting functions and interpretation tools to make
the theoretical hypotheses clearer:

1. **Main Hypothesis**: "Visible subspace dimension is the ceiling, PE order is the floor"
2. **Theoretical Predictions**: Markov parameters E_k should improve when r > k
3. **Ceiling Effect**: Errors on V(x0)⊥ should remain high regardless of PE order

The visualizations focus on:
- Clear before/after threshold comparisons
- Effect size quantification
- Statistical significance testing
- Intuitive hypothesis-to-evidence mapping
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..config import ExperimentConfig


def create_hypothesis_summary_table(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Create a summary table directly testing each theoretical prediction.
    
    Returns a table with:
    - Hypothesis statements
    - Quantitative evidence  
    - Statistical tests
    - Pass/Fail status
    """
    
    results = []
    
    # 1. TEST: PE order acts as "floor" - errors should drop when r exceeds thresholds
    for scenario in ['partial', 'full']:
        scenario_df = df[df['scenario'] == scenario]
        visible_dim = scenario_df['dim_visible'].iloc[0]
        
        # Test if errors are lower when PE > visible_dim
        low_pe = scenario_df[scenario_df['pe_order_actual'] <= visible_dim]['errA_V_rel']
        high_pe = scenario_df[scenario_df['pe_order_actual'] > visible_dim]['errA_V_rel']
        
        if len(low_pe) > 0 and len(high_pe) > 0:
            # Statistical test
            stat, p_value = stats.mannwhitneyu(high_pe, low_pe, alternative='less')
            effect_size = (low_pe.median() - high_pe.median()) / low_pe.median()
            
            results.append({
                'hypothesis': f'{scenario}: V(x0) errors drop when r > {visible_dim}',
                'low_pe_median': f'{low_pe.median():.2e}',
                'high_pe_median': f'{high_pe.median():.2e}',
                'effect_size_pct': f'{effect_size*100:.1f}%',
                'p_value': f'{p_value:.3f}',
                'significant': 'YES' if p_value < 0.05 else 'NO'
            })
    
    # 2. TEST: Markov parameter floor effects
    for k in range(min(cfg.n, 4)):  # Test first few Markov parameters
        markov_col = f'markov_err_{k}'
        if markov_col not in df.columns:
            continue
            
        # Test if E_k drops when r > k 
        low_pe = df[df['pe_order_moment'] <= k][markov_col]
        high_pe = df[df['pe_order_moment'] > k][markov_col]
        
        if len(low_pe) > 0 and len(high_pe) > 0:
            stat, p_value = stats.mannwhitneyu(high_pe, low_pe, alternative='less')
            effect_size = (low_pe.median() - high_pe.median()) / low_pe.median()
            
            results.append({
                'hypothesis': f'Markov E_{k}: errors drop when r > {k}',
                'low_pe_median': f'{low_pe.median():.2e}',
                'high_pe_median': f'{high_pe.median():.2e}',
                'effect_size_pct': f'{effect_size*100:.1f}%',
                'p_value': f'{p_value:.3f}',
                'significant': 'YES' if p_value < 0.05 else 'NO'
            })
    
    # 3. TEST: Ceiling effect - V⊥ errors should stay high
    for scenario in ['partial', 'full']:
        scenario_df = df[df['scenario'] == scenario]
        if scenario == 'partial':  # Only meaningful for partial visibility
            v_errors = scenario_df['errA_V_subspace_rel']
            vperp_errors = scenario_df['errA_Vperp_subspace_rel']
            
            # Test if V⊥ errors are significantly higher
            if len(v_errors) > 0 and len(vperp_errors) > 0:
                stat, p_value = stats.mannwhitneyu(vperp_errors, v_errors, alternative='greater')
                ratio = vperp_errors.median() / v_errors.median() if v_errors.median() > 0 else np.inf
                
                results.append({
                    'hypothesis': f'{scenario}: V⊥ errors >> V errors (ceiling)',
                    'low_pe_median': f'{v_errors.median():.2e} (V)',
                    'high_pe_median': f'{vperp_errors.median():.2e} (V⊥)', 
                    'effect_size_pct': f'{ratio:.0f}x ratio',
                    'p_value': f'{p_value:.3f}',
                    'significant': 'YES' if p_value < 0.05 else 'NO'
                })
    
    return pd.DataFrame(results)


def plot_hypothesis_dashboard(df: pd.DataFrame, cfg, outfile: pathlib.Path) -> None:
    """Create a comprehensive dashboard showing all key theoretical predictions."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # 1. MAIN EFFECT: Error reduction at thresholds
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_threshold_effects(df, cfg, ax1)
    
    # 2. MARKOV PARAMETER VALIDATION
    ax2 = fig.add_subplot(gs[0, 2:])  
    _plot_markov_validation_compact(df, cfg, ax2)
    
    # 3. CEILING vs FLOOR comparison
    ax3 = fig.add_subplot(gs[1, :2])
    _plot_ceiling_floor_comparison(df, cfg, ax3)
    
    # 4. EFFECT SIZE HEATMAP
    ax4 = fig.add_subplot(gs[1, 2:])
    _plot_effect_size_heatmap(df, cfg, ax4)
    
    # 5. STATISTICAL SIGNIFICANCE PANEL
    ax5 = fig.add_subplot(gs[2, :])
    _plot_significance_summary(df, cfg, ax5)
    
    fig.suptitle('PE vs Visible Subspace: Theoretical Hypothesis Validation Dashboard', 
                 fontsize=16, fontweight='bold')
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_threshold_effects(df: pd.DataFrame, cfg, ax) -> None:
    """Show clear before/after threshold comparisons."""
    
    scenarios = ['partial', 'full'] 
    colors = ['C0', 'C1']
    
    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        scenario_df = df[df['scenario'] == scenario]
        visible_dim = scenario_df['dim_visible'].iloc[0]
        
        # Split into before/after threshold
        before = scenario_df[scenario_df['pe_order_actual'] <= visible_dim]['errA_V_rel']
        after = scenario_df[scenario_df['pe_order_actual'] > visible_dim]['errA_V_rel']
        
        positions = [i*3, i*3 + 1]
        box_data = [before, after] 
        labels = [f'{scenario}\nr≤{visible_dim}', f'{scenario}\nr>{visible_dim}']
        
        bp = ax.boxplot(box_data, positions=positions, labels=labels, 
                       patch_artist=True, widths=0.6)
        
        # Color the boxes
        for patch, is_after in zip(bp['boxes'], [False, True]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7 if is_after else 0.4)
        
        # Add significance indicator
        if len(before) > 0 and len(after) > 0:
            _, p_val = stats.mannwhitneyu(after, before, alternative='less')
            if p_val < 0.05:
                y_max = max(before.max(), after.max())
                ax.plot([positions[0], positions[1]], [y_max*1.1, y_max*1.1], 'k-', linewidth=1)
                ax.text((positions[0] + positions[1])/2, y_max*1.15, 
                       f'p={p_val:.3f}*', ha='center', fontweight='bold')
    
    ax.set_yscale('log')
    ax.set_ylabel('Error (V-projected, relative)')
    ax.set_title('Threshold Effect: Errors Before vs After r > dim(V)')
    ax.grid(True, alpha=0.3)


def _plot_markov_validation_compact(df: pd.DataFrame, cfg, ax) -> None:
    """Compact validation of Markov parameter theory."""
    
    # Test each Markov parameter E_k
    k_values = []
    effect_sizes = []
    p_values = []
    
    for k in range(min(cfg.n, 4)):
        markov_col = f'markov_err_{k}'
        if markov_col not in df.columns:
            continue
            
        before = df[df['pe_order_moment'] <= k][markov_col]
        after = df[df['pe_order_moment'] > k][markov_col]
        
        if len(before) > 0 and len(after) > 0:
            _, p_val = stats.mannwhitneyu(after, before, alternative='less')
            effect = (before.median() - after.median()) / before.median()
            
            k_values.append(k)
            effect_sizes.append(effect * 100)  # Convert to percentage
            p_values.append(p_val)
    
    # Create bar plot of effect sizes
    bars = ax.bar(k_values, effect_sizes, alpha=0.7, color='darkgreen')
    
    # Color bars by significance
    for bar, p_val in zip(bars, p_values):
        if p_val < 0.05:
            bar.set_color('darkgreen')
        else:
            bar.set_color('lightgray')
            
    # Add significance indicators
    for k, effect, p_val in zip(k_values, effect_sizes, p_values):
        marker = '*' if p_val < 0.05 else ''
        ax.text(k, effect + 2, f'{effect:.0f}%{marker}', 
               ha='center', fontweight='bold' if p_val < 0.05 else 'normal')
    
    ax.set_xlabel('Markov Parameter Order k')
    ax.set_ylabel('Improvement % when r > k')
    ax.set_title('Theory Validation: E_k drops when r > k')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)


def _plot_ceiling_floor_comparison(df: pd.DataFrame, cfg, ax) -> None:
    """Show ceiling vs floor effects side by side."""
    
    # Only use partial scenario for meaningful ceiling effect
    partial_df = df[df['scenario'] == 'partial']
    
    if partial_df.empty:
        ax.text(0.5, 0.5, 'No partial scenario data', ha='center', va='center')
        return
    
    # Group by PE order and compute medians
    grouped = partial_df.groupby('pe_order_actual').agg({
        'errA_V_subspace_rel': 'median',      # Floor effect (should improve)
        'errA_Vperp_subspace_rel': 'median',  # Ceiling effect (should stay high)
        'dim_visible': 'first'
    }).reset_index()
    
    pe_orders = grouped['pe_order_actual']
    v_errors = grouped['errA_V_subspace_rel'] 
    vperp_errors = grouped['errA_Vperp_subspace_rel']
    visible_dim = grouped['dim_visible'].iloc[0]
    
    # Plot both effects
    ax.semilogy(pe_orders, v_errors, 'o-', color='blue', linewidth=2, 
               markersize=6, label='V(x₀): Floor Effect')
    ax.semilogy(pe_orders, vperp_errors, 's--', color='red', linewidth=2,
               markersize=6, label='V(x₀)⊥: Ceiling Effect')
    
    # Mark theoretical threshold
    ax.axvline(visible_dim, color='black', linestyle=':', linewidth=2, alpha=0.8)
    ax.text(visible_dim + 0.1, ax.get_ylim()[1]*0.8, f'dim(V)={visible_dim}', 
           rotation=90, va='top', fontweight='bold')
    
    ax.set_xlabel('PE Order')
    ax.set_ylabel('Error (relative)')
    ax.set_title('Ceiling vs Floor Effects')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_effect_size_heatmap(df: pd.DataFrame, cfg, ax) -> None:
    """Heatmap of effect sizes for different PE orders and error types."""
    
    # Compute effect sizes for different comparisons
    scenarios = ['partial', 'full']
    error_types = ['errA_V_rel', 'errB_V_rel']
    
    effect_matrix = np.zeros((len(scenarios), len(error_types)))
    
    for i, scenario in enumerate(scenarios):
        scenario_df = df[df['scenario'] == scenario]
        visible_dim = scenario_df['dim_visible'].iloc[0]
        
        for j, error_col in enumerate(error_types):
            before = scenario_df[scenario_df['pe_order_actual'] <= visible_dim][error_col]
            after = scenario_df[scenario_df['pe_order_actual'] > visible_dim][error_col]
            
            if len(before) > 0 and len(after) > 0:
                effect = (before.median() - after.median()) / before.median() * 100
                effect_matrix[i, j] = effect
    
    im = ax.imshow(effect_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(error_types)):
            text = f'{effect_matrix[i, j]:.1f}%'
            ax.text(j, i, text, ha='center', va='center', fontweight='bold')
    
    ax.set_xticks(range(len(error_types)))
    ax.set_xticklabels(['A errors', 'B errors'])
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title('Effect Sizes: % Improvement\nwhen r > dim(V)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Improvement %')


def _plot_significance_summary(df: pd.DataFrame, cfg, ax) -> None:
    """Summary of all statistical tests."""
    
    # Create the hypothesis summary table
    summary_table = create_hypothesis_summary_table(df, cfg)
    
    # Display as table
    ax.axis('off')
    
    if not summary_table.empty:
        table_data = []
        for _, row in summary_table.iterrows():
            table_data.append([
                row['hypothesis'],
                row['effect_size_pct'], 
                row['p_value'],
                '✓' if row['significant'] == 'YES' else '✗'
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Hypothesis', 'Effect Size', 'p-value', 'Significant'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.5, 0.15, 0.15, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color code significance
        for i in range(1, len(table_data) + 1):
            if table_data[i-1][3] == '✓':
                table[(i, 3)].set_facecolor('lightgreen')
            else:
                table[(i, 3)].set_facecolor('lightcoral')
    
    ax.set_title('Statistical Hypothesis Tests Summary', fontweight='bold', pad=20)


# Export additional plotting function
def create_theory_validation_plots(df: pd.DataFrame, cfg, output_dir: pathlib.Path) -> None:
    """Create all enhanced interpretation plots."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main dashboard
    plot_hypothesis_dashboard(df, cfg, output_dir / "hypothesis_dashboard.png")
    
    # Summary table as CSV
    summary_table = create_hypothesis_summary_table(df, cfg)
    summary_table.to_csv(output_dir / "hypothesis_tests.csv", index=False)
    
    print("Enhanced interpretation plots created:")
    print(f"  • Dashboard: {output_dir / 'hypothesis_dashboard.png'}")
    print(f"  • Test summary: {output_dir / 'hypothesis_tests.csv'}")