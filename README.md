# BGC Best Fit: Convolution Model Fitting

This project fits time series data to a convolution model with an exponential decay kernel.

## Problem Statement

Given two time series x(t) and y(t), we assume y(t) arises from a convolution:

```
y(t) ≈ ∫ K_τ(t-s) f(x(s)) ds
```

where K_τ(u) = (1/τ) exp(-u/τ) is an exponential decay kernel.

The goal is to estimate:
1. **τ (tau)**: The decay time constant
2. **f(x)**: The unknown function relating x to the "instantaneous" effect

## Key Insight

For an exponential kernel, the convolution is equivalent to a first-order ODE:

```
τ * dy/dt + y = f(x(t))
```

In discrete time with step Δt=1:

```
y_t = φ * y_{t-1} + (1-φ) * f(x_t)
```

where φ = exp(-Δt/τ).

Rearranging:

```
z_{φ,t} = (y_t - φ * y_{t-1}) / (1-φ) ≈ f(x_t)
```

So for each candidate φ (or equivalently τ), we can compute z and check whether it's a simple function of x.

## Method

1. **Grid over φ** (or τ)
2. For each φ, compute z_{φ,t}
3. Fit candidate functions f(x) to the (x, z) data:
   - Linear: a + b*x
   - Quadratic: a + b*x + c*x²
   - Cubic: a + b*x + c*x² + d*x³
   - Saturating: a + b*(1 - exp(-c*x))
4. Select the (φ, f) pair that minimizes BIC (Bayesian Information Criterion)

## Application: Climate/Carbon Cycle

In this project:
- **x(t)** = Temperature (tas)
- **y(t)** = GPP (already log-transformed in the input data)

The hypothesis is that GPP responds to temperature with some ecosystem inertia/memory, captured by the decay time τ.

## Usage

```python
from convolution_fit import load_model_data, analyze_region

# Load all data
df = load_model_data("data/input")

# Analyze a specific region
result = analyze_region(
    df,
    model="ACCESS-ESM1-5",
    scenario="historical",
    region="Afghanistan",
    x_col='tas',
    y_col='gpp',
    log_y=False,  # gpp is already log-transformed in the data
    show_plots=True
)

print(f"Best decay time: τ = {result.tau:.2f} years")
print(f"Best function: {result.best_function.name}")
```

## Data Format

CSV files with columns:
- `model`: Climate model name (e.g., ACCESS-ESM1-5)
- `region`: Country/region name
- `year`: Year
- `tas`: Surface air temperature
- `pr`: Precipitation (already log-transformed)
- `gpp`: Gross Primary Productivity (already log-transformed)
- Additional columns as available

**Note:** The `gpp` and `pr` columns in the input data are already log-transformed, so use `log_y=False` when analyzing.

## Installation

```bash
pip install -r requirements.txt
```

## Files

- `convolution_fit.py`: Main module with fitting functions
- `data/input/`: Input CSV files by model and scenario
- `requirements.txt`: Python dependencies

## References

This approach is based on the equivalence between exponential-kernel convolutions and first-order linear ODEs, which allows transforming the joint estimation of (τ, f) into a more tractable grid-search problem.
