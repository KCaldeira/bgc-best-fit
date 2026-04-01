"""
Convolution Fitting with Exponential Decay Kernel

Given two time series x(t) and y(t), this module fits models of the form:

    y(t) ≈ ∫ K_τ(t-s) f(x(s)) ds

where K_τ(u) = (1/τ) * exp(-u/τ) is an exponential decay kernel.

Key insight: For an exponential kernel, this convolution is equivalent to a
first-order ODE:

    τ * dy/dt + y = f(x(t))

In discrete time with step Δt=1:

    y_t = φ * y_{t-1} + (1-φ) * f(x_t)

where φ = exp(-Δt/τ).

This means for any candidate φ:

    z_{φ,t} = (y_t - φ * y_{t-1}) / (1-φ) ≈ f(x_t)

So we grid over φ and check which value makes z_{φ,t} a simple function of x_t.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import warnings


# =============================================================================
# Data Loading
# =============================================================================

def load_model_data(data_dir: str = "data/input") -> pd.DataFrame:
    """
    Load all CSV files from the data directory into a single DataFrame.

    Returns:
        DataFrame with columns: model, scenario, region, year, tas, pr, gpp, etc.
    """
    data_path = Path(data_dir)
    all_data = []

    for csv_file in data_path.glob("*.csv"):
        # Parse filename: MODEL_SCENARIO.csv
        parts = csv_file.stem.split('_')
        if len(parts) >= 2:
            model = parts[0]
            scenario = '_'.join(parts[1:])
        else:
            model = csv_file.stem
            scenario = 'unknown'

        df = pd.read_csv(csv_file)
        df['scenario'] = scenario
        all_data.append(df)

    if not all_data:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def get_time_series(df: pd.DataFrame,
                    model: str,
                    scenario: str,
                    region: str,
                    x_col: str = 'tas',
                    y_col: str = 'gpp',
                    log_y: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract x(t) and y(t) time series for a specific model/scenario/region.

    Args:
        df: Combined DataFrame from load_model_data()
        model: Climate model name (e.g., 'ACCESS-ESM1-5')
        scenario: Scenario name (e.g., 'historical', 'ssp585')
        region: Country/region name
        x_col: Column to use as x(t) (default: 'tas')
        y_col: Column to use as y(t) (default: 'gpp')
        log_y: If True, take log of y values (default: True)

    Returns:
        Tuple of (years, x_values, y_values) as numpy arrays
    """
    mask = (df['model'] == model) & (df['scenario'] == scenario) & (df['region'] == region)
    subset = df[mask].sort_values('year').copy()

    if len(subset) == 0:
        raise ValueError(f"No data found for model={model}, scenario={scenario}, region={region}")

    years = subset['year'].values
    x = subset[x_col].values
    y = subset[y_col].values

    if log_y:
        # Handle non-positive values
        if np.any(y <= 0):
            warnings.warn(f"Found {np.sum(y <= 0)} non-positive y values, replacing with NaN")
            y = np.where(y > 0, y, np.nan)
        y = np.log(y)

    return years, x, y


def list_available_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """List all available model/scenario/region combinations."""
    return df.groupby(['model', 'scenario', 'region']).size().reset_index(name='n_years')


# =============================================================================
# Core Fitting Algorithm
# =============================================================================

def compute_z_phi(y: np.ndarray, phi: float) -> np.ndarray:
    """
    Compute z_{φ,t} = (y_t - φ * y_{t-1}) / (1-φ)

    For the correct φ, z should be ≈ f(x).

    Args:
        y: Time series y(t)
        phi: Decay parameter in (0, 1), where φ = exp(-Δt/τ)

    Returns:
        z array (length n-1, since we lose first point)
    """
    if phi <= 0 or phi >= 1:
        raise ValueError("phi must be in (0, 1)")

    z = (y[1:] - phi * y[:-1]) / (1 - phi)
    return z


def phi_to_tau(phi: float, dt: float = 1.0) -> float:
    """Convert φ to τ: φ = exp(-dt/τ) => τ = -dt/ln(φ)"""
    return -dt / np.log(phi)


def tau_to_phi(tau: float, dt: float = 1.0) -> float:
    """Convert τ to φ: φ = exp(-dt/τ)"""
    return np.exp(-dt / tau)


# =============================================================================
# Function Candidates for f(x)
# =============================================================================

@dataclass
class FunctionFit:
    """Result of fitting a candidate function."""
    name: str
    n_params: int
    params: np.ndarray
    residuals: np.ndarray
    mse: float
    r_squared: float
    bic: float
    predict: Callable[[np.ndarray], np.ndarray]


def fit_linear(x: np.ndarray, z: np.ndarray) -> FunctionFit:
    """Fit f(x) = a + b*x"""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(z))
    x_clean, z_clean = x[mask], z[mask]
    n = len(x_clean)

    # Fit using least squares
    A = np.column_stack([np.ones(n), x_clean])
    params, residuals_sum, rank, s = np.linalg.lstsq(A, z_clean, rcond=None)

    predictions = params[0] + params[1] * x_clean
    residuals = z_clean - predictions
    mse = np.mean(residuals**2)
    ss_tot = np.var(z_clean) * n
    r_squared = 1 - np.sum(residuals**2) / ss_tot if ss_tot > 0 else 0

    # BIC = n*ln(MSE) + k*ln(n)
    k = 2  # number of parameters
    bic = n * np.log(mse) + k * np.log(n) if mse > 0 else np.inf

    def predict(x_new):
        return params[0] + params[1] * x_new

    return FunctionFit(
        name="linear: a + b*x",
        n_params=2,
        params=params,
        residuals=residuals,
        mse=mse,
        r_squared=r_squared,
        bic=bic,
        predict=predict
    )


def fit_quadratic(x: np.ndarray, z: np.ndarray) -> FunctionFit:
    """Fit f(x) = a + b*x + c*x^2"""
    mask = ~(np.isnan(x) | np.isnan(z))
    x_clean, z_clean = x[mask], z[mask]
    n = len(x_clean)

    A = np.column_stack([np.ones(n), x_clean, x_clean**2])
    params, _, _, _ = np.linalg.lstsq(A, z_clean, rcond=None)

    predictions = params[0] + params[1] * x_clean + params[2] * x_clean**2
    residuals = z_clean - predictions
    mse = np.mean(residuals**2)
    ss_tot = np.var(z_clean) * n
    r_squared = 1 - np.sum(residuals**2) / ss_tot if ss_tot > 0 else 0

    k = 3
    bic = n * np.log(mse) + k * np.log(n) if mse > 0 else np.inf

    def predict(x_new):
        return params[0] + params[1] * x_new + params[2] * x_new**2

    return FunctionFit(
        name="quadratic: a + b*x + c*x^2",
        n_params=3,
        params=params,
        residuals=residuals,
        mse=mse,
        r_squared=r_squared,
        bic=bic,
        predict=predict
    )


def fit_cubic(x: np.ndarray, z: np.ndarray) -> FunctionFit:
    """Fit f(x) = a + b*x + c*x^2 + d*x^3"""
    mask = ~(np.isnan(x) | np.isnan(z))
    x_clean, z_clean = x[mask], z[mask]
    n = len(x_clean)

    A = np.column_stack([np.ones(n), x_clean, x_clean**2, x_clean**3])
    params, _, _, _ = np.linalg.lstsq(A, z_clean, rcond=None)

    predictions = params[0] + params[1] * x_clean + params[2] * x_clean**2 + params[3] * x_clean**3
    residuals = z_clean - predictions
    mse = np.mean(residuals**2)
    ss_tot = np.var(z_clean) * n
    r_squared = 1 - np.sum(residuals**2) / ss_tot if ss_tot > 0 else 0

    k = 4
    bic = n * np.log(mse) + k * np.log(n) if mse > 0 else np.inf

    def predict(x_new):
        return params[0] + params[1] * x_new + params[2] * x_new**2 + params[3] * x_new**3

    return FunctionFit(
        name="cubic: a + b*x + c*x^2 + d*x^3",
        n_params=4,
        params=params,
        residuals=residuals,
        mse=mse,
        r_squared=r_squared,
        bic=bic,
        predict=predict
    )


def fit_saturating(x: np.ndarray, z: np.ndarray) -> FunctionFit:
    """Fit f(x) = a + b*(1 - exp(-c*x)) - saturating form"""
    mask = ~(np.isnan(x) | np.isnan(z))
    x_clean, z_clean = x[mask], z[mask]
    n = len(x_clean)

    def model(x, a, b, c):
        return a + b * (1 - np.exp(-c * x))

    try:
        # Initial guess
        p0 = [np.mean(z_clean), np.std(z_clean), 0.1]
        params, _ = curve_fit(model, x_clean, z_clean, p0=p0, maxfev=5000)

        predictions = model(x_clean, *params)
        residuals = z_clean - predictions
        mse = np.mean(residuals**2)
        ss_tot = np.var(z_clean) * n
        r_squared = 1 - np.sum(residuals**2) / ss_tot if ss_tot > 0 else 0

        k = 3
        bic = n * np.log(mse) + k * np.log(n) if mse > 0 else np.inf

        def predict(x_new):
            return model(x_new, *params)

        return FunctionFit(
            name="saturating: a + b*(1 - exp(-c*x))",
            n_params=3,
            params=params,
            residuals=residuals,
            mse=mse,
            r_squared=r_squared,
            bic=bic,
            predict=predict
        )
    except Exception as e:
        # Return a failed fit
        return FunctionFit(
            name="saturating: a + b*(1 - exp(-c*x))",
            n_params=3,
            params=np.array([np.nan, np.nan, np.nan]),
            residuals=np.full(n, np.nan),
            mse=np.inf,
            r_squared=0,
            bic=np.inf,
            predict=lambda x: np.full_like(x, np.nan)
        )


# All available function fitters
FUNCTION_FITTERS = {
    'linear': fit_linear,
    'quadratic': fit_quadratic,
    'cubic': fit_cubic,
    'saturating': fit_saturating,
}


# =============================================================================
# Joint Optimization over (φ, f)
# =============================================================================

@dataclass
class ConvolutionFitResult:
    """Result of the full convolution fitting procedure."""
    phi: float
    tau: float
    best_function: FunctionFit
    all_functions: Dict[str, FunctionFit]
    phi_grid: np.ndarray
    bic_by_phi: Dict[str, np.ndarray]


def fit_convolution_model(x: np.ndarray,
                          y: np.ndarray,
                          phi_range: Tuple[float, float] = (0.1, 0.99),
                          n_phi: int = 50,
                          functions: Optional[List[str]] = None) -> ConvolutionFitResult:
    """
    Fit the convolution model by grid search over φ.

    For each φ, compute z_{φ,t} and fit candidate functions f(x).
    Select the (φ, f) pair that minimizes BIC.

    Args:
        x: Input time series x(t)
        y: Output time series y(t)
        phi_range: Range of φ values to search (default: 0.1 to 0.99)
        n_phi: Number of φ values to try
        functions: List of function names to try (default: all)

    Returns:
        ConvolutionFitResult with best φ, τ, and function fit
    """
    if functions is None:
        functions = list(FUNCTION_FITTERS.keys())

    # Grid of φ values (log-spaced for better coverage of long decay times)
    phi_grid = np.linspace(phi_range[0], phi_range[1], n_phi)

    # Track BIC for each (φ, function) combination
    bic_by_phi = {func: np.zeros(n_phi) for func in functions}

    best_bic = np.inf
    best_phi = None
    best_func_name = None
    best_fits_at_best_phi = None

    for i, phi in enumerate(phi_grid):
        # Compute z for this φ
        z = compute_z_phi(y, phi)
        x_shifted = x[1:]  # x values corresponding to z

        fits = {}
        for func_name in functions:
            fitter = FUNCTION_FITTERS[func_name]
            fit = fitter(x_shifted, z)
            fits[func_name] = fit
            bic_by_phi[func_name][i] = fit.bic

            if fit.bic < best_bic:
                best_bic = fit.bic
                best_phi = phi
                best_func_name = func_name
                best_fits_at_best_phi = fits.copy()

    # Get the best function fit
    best_function = best_fits_at_best_phi[best_func_name]
    best_tau = phi_to_tau(best_phi)

    return ConvolutionFitResult(
        phi=best_phi,
        tau=best_tau,
        best_function=best_function,
        all_functions=best_fits_at_best_phi,
        phi_grid=phi_grid,
        bic_by_phi=bic_by_phi
    )


# =============================================================================
# Pooled Multi-Region Fitting
# =============================================================================

@dataclass
class PooledFitResult:
    """Result of fitting a single model across all regions."""
    phi: float
    tau: float
    best_function: FunctionFit
    all_functions: Dict[str, FunctionFit]
    phi_grid: np.ndarray
    bic_by_phi: Dict[str, np.ndarray]
    n_regions: int
    n_total_points: int
    regions: List[str]


def fit_convolution_model_pooled(
    df: pd.DataFrame,
    model: str,
    scenario: str,
    regions: Optional[List[str]] = None,
    x_col: str = 'tas',
    y_col: str = 'gpp',
    log_y: bool = False,
    phi_range: Tuple[float, float] = (0.01, 0.99),
    n_phi: int = 100,
    functions: Optional[List[str]] = None
) -> PooledFitResult:
    """
    Fit a single convolution model (τ, f) across ALL regions pooled together.

    For each candidate φ:
    1. Compute z_{φ,t} for each region's time series
    2. Pool all (x, z) pairs from all regions
    3. Fit candidate functions to the pooled data
    4. Select the (φ, f) pair that minimizes pooled BIC

    Args:
        df: Data from load_model_data()
        model: Climate model name
        scenario: Scenario name
        regions: List of regions to include (default: all available)
        x_col, y_col: Column names for x(t) and y(t)
        log_y: Whether to log-transform y
        phi_range, n_phi: Grid search parameters
        functions: List of function names to try

    Returns:
        PooledFitResult with best φ, τ, and function fit
    """
    if functions is None:
        functions = list(FUNCTION_FITTERS.keys())

    # Get available regions for this model/scenario
    mask = (df['model'] == model) & (df['scenario'] == scenario)
    available_regions = df[mask]['region'].unique().tolist()

    if regions is None:
        regions = available_regions
    else:
        regions = [r for r in regions if r in available_regions]

    if len(regions) == 0:
        raise ValueError(f"No regions found for model={model}, scenario={scenario}")

    print(f"Pooling data from {len(regions)} regions...")

    # Extract time series for all regions
    all_series = []
    for region in regions:
        try:
            years, x, y = get_time_series(df, model, scenario, region, x_col, y_col, log_y)
            all_series.append((region, x, y))
        except ValueError as e:
            print(f"  Skipping {region}: {e}")

    if len(all_series) == 0:
        raise ValueError("No valid time series found")

    print(f"  Successfully loaded {len(all_series)} regions")

    # Grid of φ values
    phi_grid = np.linspace(phi_range[0], phi_range[1], n_phi)

    # Track BIC for each (φ, function) combination
    bic_by_phi = {func: np.zeros(n_phi) for func in functions}

    best_bic = np.inf
    best_phi = None
    best_func_name = None
    best_fits_at_best_phi = None

    for i, phi in enumerate(phi_grid):
        # Pool z and x values from all regions
        all_x = []
        all_z = []

        for region, x, y in all_series:
            z = compute_z_phi(y, phi)
            x_shifted = x[1:]

            # Remove NaN values
            valid = ~(np.isnan(x_shifted) | np.isnan(z))
            all_x.append(x_shifted[valid])
            all_z.append(z[valid])

        # Concatenate all data
        pooled_x = np.concatenate(all_x)
        pooled_z = np.concatenate(all_z)

        # Fit each function to pooled data
        fits = {}
        for func_name in functions:
            fitter = FUNCTION_FITTERS[func_name]
            fit = fitter(pooled_x, pooled_z)
            fits[func_name] = fit
            bic_by_phi[func_name][i] = fit.bic

            if fit.bic < best_bic:
                best_bic = fit.bic
                best_phi = phi
                best_func_name = func_name
                best_fits_at_best_phi = fits.copy()

    # Get the best function fit
    best_function = best_fits_at_best_phi[best_func_name]
    best_tau = phi_to_tau(best_phi)

    # Count total points
    n_total = sum(len(x) - 1 for _, x, _ in all_series)

    return PooledFitResult(
        phi=best_phi,
        tau=best_tau,
        best_function=best_function,
        all_functions=best_fits_at_best_phi,
        phi_grid=phi_grid,
        bic_by_phi=bic_by_phi,
        n_regions=len(all_series),
        n_total_points=n_total,
        regions=[r for r, _, _ in all_series]
    )


def plot_pooled_fit(
    df: pd.DataFrame,
    result: PooledFitResult,
    model: str,
    scenario: str,
    x_col: str = 'tas',
    y_col: str = 'gpp',
    log_y: bool = False,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot the pooled fit result showing all regions.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    phi = result.phi

    # Collect pooled data
    all_x = []
    all_z = []

    for region in result.regions:
        try:
            years, x, y = get_time_series(df, model, scenario, region, x_col, y_col, log_y)
            z = compute_z_phi(y, phi)
            x_shifted = x[1:]
            valid = ~(np.isnan(x_shifted) | np.isnan(z))
            all_x.append(x_shifted[valid])
            all_z.append(z[valid])
        except:
            pass

    pooled_x = np.concatenate(all_x)
    pooled_z = np.concatenate(all_z)

    # Panel 1: z vs x with fit (all regions)
    ax1 = axes[0]
    ax1.scatter(pooled_x, pooled_z, alpha=0.1, s=5, label='All regions')
    x_line = np.linspace(np.nanmin(pooled_x), np.nanmax(pooled_x), 100)
    ax1.plot(x_line, result.best_function.predict(x_line), 'r-', lw=2,
             label=f'{result.best_function.name.split(":")[0]}')
    ax1.set_xlabel(f'x ({x_col})')
    ax1.set_ylabel('z_φ ≈ f(x)')
    ax1.set_title(f'Pooled fit (R²={result.best_function.r_squared:.3f})')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel 2: BIC vs tau
    ax2 = axes[1]
    tau_grid = np.array([phi_to_tau(phi) for phi in result.phi_grid])
    for func_name, bic_values in result.bic_by_phi.items():
        ax2.plot(tau_grid, bic_values, label=func_name, lw=2)
    ax2.axvline(result.tau, color='k', linestyle='--', alpha=0.5)
    ax2.scatter([result.tau], [result.best_function.bic], color='red', s=100, zorder=5)
    ax2.set_xlabel('τ (decay time in years)')
    ax2.set_ylabel('BIC (lower is better)')
    ax2.set_title('Model Selection')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Histogram of residuals
    ax3 = axes[2]
    residuals = pooled_z - result.best_function.predict(pooled_x)
    ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Residuals (RMSE={np.sqrt(np.mean(residuals**2)):.4f})')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'Pooled Convolution Fit: {result.n_regions} regions, {result.n_total_points} points\n'
                 f'τ={result.tau:.2f} years, f={result.best_function.name.split(":")[0]}',
                 fontsize=11)
    plt.tight_layout()
    return fig


def analyze_pooled(
    df: pd.DataFrame,
    model: str,
    scenario: str,
    regions: Optional[List[str]] = None,
    x_col: str = 'tas',
    y_col: str = 'gpp',
    log_y: bool = False,
    phi_range: Tuple[float, float] = (0.01, 0.99),
    n_phi: int = 100,
    show_plots: bool = True
) -> PooledFitResult:
    """
    Complete pooled analysis pipeline.

    Fits a single (τ, f) model across all specified regions.
    """
    print(f"Analyzing pooled data: {model} / {scenario}")

    result = fit_convolution_model_pooled(
        df, model, scenario, regions, x_col, y_col, log_y, phi_range, n_phi
    )

    print(f"\nPooled fit results:")
    print(f"  Regions: {result.n_regions}")
    print(f"  Total data points: {result.n_total_points}")
    print(f"  τ = {result.tau:.2f} years (φ = {result.phi:.4f})")
    print(f"  f(x) = {result.best_function.name}")
    print(f"  Parameters: {result.best_function.params}")
    print(f"  R² = {result.best_function.r_squared:.4f}")
    print(f"  BIC = {result.best_function.bic:.2f}")

    if show_plots:
        fig = plot_pooled_fit(df, result, model, scenario, x_col, y_col, log_y)
        plt.show()

    return result


# =============================================================================
# Diagnostic Plotting
# =============================================================================

def plot_z_vs_x_grid(x: np.ndarray,
                     y: np.ndarray,
                     phi_values: List[float],
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create diagnostic scatter plots of z_φ vs x for different φ values.

    The correct φ should make the scatter plot collapse onto a simple curve.
    """
    n_plots = len(phi_values)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, phi in enumerate(phi_values):
        ax = axes[i]
        z = compute_z_phi(y, phi)
        x_shifted = x[1:]

        tau = phi_to_tau(phi)
        ax.scatter(x_shifted, z, alpha=0.5, s=20)
        ax.set_xlabel('x (temperature)')
        ax.set_ylabel('z_φ')
        ax.set_title(f'φ={phi:.3f}, τ={tau:.1f} years')

        # Fit linear for reference
        fit = fit_linear(x_shifted, z)
        x_line = np.linspace(np.nanmin(x_shifted), np.nanmax(x_shifted), 100)
        ax.plot(x_line, fit.predict(x_line), 'r-', lw=2, label=f'R²={fit.r_squared:.3f}')
        ax.legend(loc='best', fontsize=8)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Diagnostic: z_φ vs x for different decay times', fontsize=12)
    plt.tight_layout()
    return fig


def plot_bic_vs_phi(result: ConvolutionFitResult,
                    figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot BIC vs φ for each function candidate."""
    fig, ax = plt.subplots(figsize=figsize)

    tau_grid = np.array([phi_to_tau(phi) for phi in result.phi_grid])

    for func_name, bic_values in result.bic_by_phi.items():
        ax.plot(tau_grid, bic_values, label=func_name, lw=2)

    # Mark the best point
    best_tau = result.tau
    best_bic = result.best_function.bic
    ax.axvline(best_tau, color='k', linestyle='--', alpha=0.5)
    ax.scatter([best_tau], [best_bic], color='red', s=100, zorder=5,
               label=f'Best: τ={best_tau:.1f}, {result.best_function.name.split(":")[0]}')

    ax.set_xlabel('τ (decay time in years)')
    ax.set_ylabel('BIC (lower is better)')
    ax.set_title('Model Selection: BIC vs Decay Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fit_result(x: np.ndarray,
                    y: np.ndarray,
                    result: ConvolutionFitResult,
                    figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Plot the final fit result with three panels:
    1. z_φ vs x with fitted function
    2. Time series comparison (actual y vs reconstructed)
    3. Residuals over time
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    phi = result.phi
    z = compute_z_phi(y, phi)
    x_shifted = x[1:]

    # Panel 1: z vs x with fit
    ax1 = axes[0]
    ax1.scatter(x_shifted, z, alpha=0.5, s=20, label='Data')
    x_line = np.linspace(np.nanmin(x_shifted), np.nanmax(x_shifted), 100)
    ax1.plot(x_line, result.best_function.predict(x_line), 'r-', lw=2,
             label=f'{result.best_function.name.split(":")[0]}')
    ax1.set_xlabel('x (temperature)')
    ax1.set_ylabel('z_φ ≈ f(x)')
    ax1.set_title(f'Function fit (R²={result.best_function.r_squared:.3f})')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Reconstruct y using the model
    ax2 = axes[1]
    y_reconstructed = np.zeros_like(y)
    y_reconstructed[0] = y[0]  # Start from actual initial value
    f_values = result.best_function.predict(x)
    for t in range(1, len(y)):
        y_reconstructed[t] = phi * y_reconstructed[t-1] + (1 - phi) * f_values[t]

    t_axis = np.arange(len(y))
    ax2.plot(t_axis, y, 'b-', lw=1.5, label='Actual y', alpha=0.7)
    ax2.plot(t_axis, y_reconstructed, 'r--', lw=1.5, label='Reconstructed', alpha=0.7)
    ax2.set_xlabel('Time index')
    ax2.set_ylabel('y (log GPP)')
    ax2.set_title(f'Time series (τ={result.tau:.1f} years)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Residuals
    ax3 = axes[2]
    residuals = y - y_reconstructed
    ax3.plot(t_axis, residuals, 'g-', lw=1)
    ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time index')
    ax3.set_ylabel('Residual')
    ax3.set_title(f'Residuals (RMSE={np.sqrt(np.mean(residuals**2)):.4f})')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'Convolution Fit: φ={phi:.3f}, τ={result.tau:.1f} years, f={result.best_function.name.split(":")[0]}',
                 fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_region(df: pd.DataFrame,
                   model: str,
                   scenario: str,
                   region: str,
                   x_col: str = 'tas',
                   y_col: str = 'gpp',
                   log_y: bool = True,
                   phi_range: Tuple[float, float] = (0.1, 0.99),
                   n_phi: int = 50,
                   show_plots: bool = True) -> ConvolutionFitResult:
    """
    Complete analysis pipeline for a single region.

    Args:
        df: Data from load_model_data()
        model, scenario, region: Identifiers for the time series
        x_col, y_col: Column names for x(t) and y(t)
        log_y: Whether to log-transform y
        phi_range, n_phi: Grid search parameters
        show_plots: Whether to display diagnostic plots

    Returns:
        ConvolutionFitResult
    """
    # Extract time series
    years, x, y = get_time_series(df, model, scenario, region, x_col, y_col, log_y)

    print(f"Analyzing: {model} / {scenario} / {region}")
    print(f"  Time span: {years[0]} - {years[-1]} ({len(years)} points)")
    print(f"  x ({x_col}): range [{x.min():.2f}, {x.max():.2f}]")
    print(f"  y ({'log ' if log_y else ''}{y_col}): range [{np.nanmin(y):.2f}, {np.nanmax(y):.2f}]")

    # Fit the model
    result = fit_convolution_model(x, y, phi_range, n_phi)

    print(f"\nBest fit:")
    print(f"  τ = {result.tau:.2f} years (φ = {result.phi:.4f})")
    print(f"  f(x) = {result.best_function.name}")
    print(f"  Parameters: {result.best_function.params}")
    print(f"  R² = {result.best_function.r_squared:.4f}")
    print(f"  BIC = {result.best_function.bic:.2f}")

    if show_plots:
        # Diagnostic grid
        phi_samples = [0.3, 0.5, 0.7, 0.85, 0.95, result.phi]
        fig1 = plot_z_vs_x_grid(x, y, phi_samples)
        plt.show()

        # BIC vs tau
        fig2 = plot_bic_vs_phi(result)
        plt.show()

        # Final fit
        fig3 = plot_fit_result(x, y, result)
        plt.show()

    return result


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    df = load_model_data()

    # List available combinations
    combos = list_available_combinations(df)
    print(f"\nAvailable data: {len(combos)} combinations")
    print(combos.head(20))

    # Analyze one example
    # Get first available combination
    first = combos.iloc[0]
    result = analyze_region(
        df,
        model=first['model'],
        scenario=first['scenario'],
        region=first['region'],
        show_plots=True
    )
