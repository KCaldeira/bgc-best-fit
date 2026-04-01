#!/usr/bin/env python
"""
Run paired difference analysis from the command line.

Usage:
    python run_analysis.py ACCESS-ESM1-5 historical
    python run_analysis.py CNRM-ESM2-1 ssp585
    python run_analysis.py --help
"""

import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
from pathlib import Path

from convolution_fit import (
    load_model_data,
    analyze_paired_difference,
    plot_paired_diff_fit
)


def main():
    parser = argparse.ArgumentParser(
        description='Run paired difference convolution analysis (full - bgc)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py ACCESS-ESM1-5 historical
    python run_analysis.py CNRM-ESM2-1 ssp585
    python run_analysis.py MIROC-ES2L historical --output-dir results

Available models: ACCESS-ESM1-5, CNRM-ESM2-1, MIROC-ES2L
Available scenarios: historical, ssp585
        """
    )

    parser.add_argument('model', type=str,
                        help='Climate model name (e.g., ACCESS-ESM1-5)')
    parser.add_argument('scenario', type=str,
                        help='Scenario name (e.g., historical or ssp585)')
    parser.add_argument('--x-col', type=str, default='tas',
                        help='Column to use as x(t) (default: tas)')
    parser.add_argument('--y-col', type=str, default='gpp',
                        help='Column to use as y(t) (default: gpp)')
    parser.add_argument('--tau-min', type=float, default=0.1,
                        help='Minimum tau (decay time) in years (default: 0.1)')
    parser.add_argument('--tau-max', type=float, default=100.0,
                        help='Maximum tau (decay time) in years (default: 100.0)')
    parser.add_argument('--n-tau', type=int, default=100,
                        help='Number of tau values in log-spaced grid (default: 100)')
    parser.add_argument('--output-dir', type=str, default='data/output',
                        help='Output directory for figures (default: data/output)')
    parser.add_argument('--data-dir', type=str, default='data/input',
                        help='Input data directory (default: data/input)')

    args = parser.parse_args()

    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_dir}...")
    df = load_model_data(args.data_dir)

    # Run analysis
    print(f"\nRunning paired difference analysis:")
    print(f"  Model: {args.model}")
    print(f"  Scenario: {args.scenario}")
    print(f"  x(t): {args.x_col}")
    print(f"  y(t): {args.y_col}")
    print()

    result = analyze_paired_difference(
        df,
        model=args.model,
        scenario=args.scenario,
        x_col=args.x_col,
        y_col=args.y_col,
        tau_range=(args.tau_min, args.tau_max),
        n_tau=args.n_tau,
        show_plots=False
    )

    # Generate and save figure
    fig = plot_paired_diff_fit(df, result, args.model, args.x_col, args.y_col)

    filename = f"paired_diff_{args.model}_{args.scenario}.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {filepath}")

    plt.close(fig)

    # Prepare summary data
    params = result.best_function.params
    summary_data = {
        'model': args.model,
        'scenario_full': result.scenario_full,
        'scenario_bgc': result.scenario_bgc,
        'n_regions': result.n_regions,
        'n_data_points': result.n_total_points,
        'tau_years': result.tau,
        'phi': result.phi,
        'function': result.best_function.name,
        'r_squared': result.best_function.r_squared,
        'bic': result.best_function.bic,
        'param_b': params[0] if len(params) > 0 else None,
        'param_c': params[1] if len(params) > 1 else None,
        'param_d': params[2] if len(params) > 2 else None,
    }

    # Save summary to CSV
    import pandas as pd
    summary_df = pd.DataFrame([summary_data])
    csv_filename = f"paired_diff_{args.model}_{args.scenario}_summary.csv"
    csv_filepath = output_dir / csv_filename
    summary_df.to_csv(csv_filepath, index=False)
    print(f"Summary saved: {csv_filepath}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model:      {args.model}")
    print(f"Scenario:   {result.scenario_full} - {result.scenario_bgc}")
    print(f"Regions:    {result.n_regions}")
    print(f"Data points: {result.n_total_points}")
    print(f"tau:        {result.tau:.2f} years")
    print(f"phi:        {result.phi:.4f}")
    print(f"Function:   {result.best_function.name}")
    print(f"Parameters: {result.best_function.params}")
    print(f"R-squared:  {result.best_function.r_squared:.4f}")
    print(f"BIC:        {result.best_function.bic:.2f}")
    print("="*60)


if __name__ == '__main__':
    main()
