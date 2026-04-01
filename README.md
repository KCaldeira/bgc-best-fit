# BGC Best Fit: Convolution Model Fitting

This project fits time series data to a convolution model with an exponential decay kernel, specifically designed to isolate temperature effects on gross primary productivity (GPP) from climate model output.

## Problem Statement

Given two time series T(t) (temperature) and y(t) (GPP), we assume y(t) arises from a convolution:

```
y(t) ≈ ∫ K_τ(t-s) h(T(s)) ds
```

where K_τ(u) = (1/τ) exp(-u/τ) is an exponential decay kernel.

The goal is to estimate:
1. **τ (tau)**: The decay time constant (in years)
2. **h(T)**: The unknown function relating temperature to the "instantaneous" effect on GPP

## Key Insight

For an exponential kernel, the convolution is equivalent to a first-order ODE:

```
τ * dy/dt + y = h(T(t))
```

In discrete time with step Δt=1 year:

```
y_t = φ * y_{t-1} + (1-φ) * h(T_t)
```

where φ = exp(-Δt/τ) = exp(-1/τ).

Rearranging:

```
z_{φ,t} = (y_t - φ * y_{t-1}) / (1-φ) ≈ h(T_t)
```

So for each candidate τ, we can compute z and check whether it's a simple function of T.

## Paired Difference Analysis

### Why Use Paired Differences?

Climate models run two types of experiments:
- **Full scenario** (e.g., `historical`): CO2 affects both plant physiology (fertilization) AND climate (warming)
- **BGC scenario** (e.g., `hist-bgc`): CO2 affects only plant physiology; climate is held constant

By taking the difference (Full - BGC), we isolate the pure **climate/temperature effect** on GPP, removing the confounding CO2 fertilization signal.

### Mathematical Formulation

We assume both scenarios follow the same convolution model with the same function h():

```
For Full scenario:   τ * d(GPP_full)/dt + GPP_full = h(T_full)
For BGC scenario:    τ * d(GPP_bgc)/dt  + GPP_bgc  = h(T_bgc)
```

Subtracting:

```
τ * d(GPP_full - GPP_bgc)/dt + (GPP_full - GPP_bgc) = h(T_full) - h(T_bgc)
```

Let Δy = GPP_full - GPP_bgc. In discrete time:

```
z_{φ,t} = (Δy_t - φ * Δy_{t-1}) / (1-φ) = h(T_full,t) - h(T_bgc,t)
```

### Fitting the Function h()

Since we're fitting to **differences** h(T_full) - h(T_bgc), the constant term in h() cancels out. For example:

**Linear:** h(T) = a + b*T
```
h(T_full) - h(T_bgc) = b * (T_full - T_bgc)
```
Only `b` can be determined; `a` cancels.

**Quadratic:** h(T) = a + b*T + c*T²
```
h(T_full) - h(T_bgc) = b * (T_full - T_bgc) + c * (T_full² - T_bgc²)
```
Parameters `b` and `c` are determined; `a` cancels.

**Cubic:** h(T) = a + b*T + c*T² + d*T³
```
h(T_full) - h(T_bgc) = b * ΔT + c * ΔT² + d * ΔT³
where ΔTⁿ = T_fullⁿ - T_bgcⁿ
```
Parameters `b`, `c`, and `d` are determined; `a` cancels.

## The `analyze_paired_difference()` Function

This is the main analysis function. Here's what it does step by step:

### Step 1: Load and Pair Data

For each region (country), the function:
1. Loads the time series from the full scenario (e.g., `historical`)
2. Loads the matching time series from the BGC scenario (e.g., `hist-bgc`)
3. Aligns them by year
4. Computes Δy = GPP_full - GPP_bgc for each year

### Step 2: Grid Search Over τ

The function creates a **log-spaced grid** of τ values:
```python
τ values: [0.1, 0.11, 0.12, ..., 10, 15, 22, ..., 100] years
```

Log-spacing ensures good coverage of both short and long decay times.

### Step 3: For Each τ, Compute z and Fit Functions

For each candidate τ:

1. Convert τ to φ: `φ = exp(-1/τ)`

2. For each region, compute:
   ```
   z_t = (Δy_t - φ * Δy_{t-1}) / (1-φ)
   ```

3. Pool all (T_full, T_bgc, z) data from all regions

4. Fit each candidate function to the pooled data:
   - **Linear:** minimize Σ(z - b*ΔT)²
   - **Quadratic:** minimize Σ(z - b*ΔT - c*ΔT²)²
   - **Cubic:** minimize Σ(z - b*ΔT - c*ΔT² - d*ΔT³)²

5. Compute BIC (Bayesian Information Criterion) for each function:
   ```
   BIC = n * ln(MSE) + k * ln(n)
   ```
   where n = number of data points, k = number of parameters

### Step 4: Select Best (τ, h) Pair

The function selects the (τ, function) combination with the **lowest BIC**. Lower BIC indicates a better trade-off between fit quality and model complexity.

### Step 5: Return Results

The function returns a `PairedDiffFitResult` object containing:

| Attribute | Description |
|-----------|-------------|
| `tau` | Best-fit decay time (years) |
| `phi` | Corresponding φ = exp(-1/τ) |
| `best_function` | The selected function fit object |
| `best_function.name` | Function form (e.g., "quadratic: a + b*T + c*T²") |
| `best_function.params` | Array of fitted parameters [b, c, ...] |
| `best_function.r_squared` | R² of the fit |
| `best_function.bic` | BIC value |
| `best_function.predict_single(T)` | Function to evaluate h(T) |
| `best_function.predict_diff(T1, T2)` | Function to evaluate h(T1) - h(T2) |
| `n_regions` | Number of regions analyzed |
| `n_total_points` | Total data points pooled |
| `regions` | List of region names |
| `scenario_full` | Full scenario name |
| `scenario_bgc` | BGC scenario name |

### Interpreting the Parameters

For a **quadratic** fit h(T) = b*T + c*T² (constant undetermined):

- **b > 0**: GPP increases with temperature (at low temperatures)
- **b < 0**: GPP decreases with temperature (at low temperatures)
- **c < 0**: Diminishing returns or optimum temperature (concave down)
- **c > 0**: Accelerating response (concave up)

The function h(T) shows how GPP responds to temperature. Since the constant `a` is undetermined, only the **shape** of h() matters, not its absolute level.

## Installation

```bash
cd /path/to/bgc-best-fit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Available Models and Scenarios

### Climate Models

| Model | Description |
|-------|-------------|
| `ACCESS-ESM1-5` | Australian Community Climate and Earth System Simulator |
| `CNRM-ESM2-1` | Centre National de Recherches Météorologiques Earth System Model |
| `MIROC-ES2L` | Model for Interdisciplinary Research on Climate (low resolution) |

### Scenarios

| Scenario | Paired BGC Scenario | Description |
|----------|---------------------|-------------|
| `historical` | `hist-bgc` | Historical period (1850-2014) |
| `ssp585` | `ssp585-bgc` | Future projection SSP5-8.5 (2015-2100) |

- **Full scenarios** (`historical`, `ssp585`): CO2 affects both climate and plant physiology
- **BGC scenarios** (`hist-bgc`, `ssp585-bgc`): CO2 affects only plant physiology (climate held constant)

### Available Data Files

```
data/input/
├── ACCESS-ESM1-5_historical.csv
├── ACCESS-ESM1-5_hist-bgc.csv
├── ACCESS-ESM1-5_ssp585.csv
├── ACCESS-ESM1-5_ssp585-bgc.csv
├── CNRM-ESM2-1_historical.csv
├── CNRM-ESM2-1_hist-bgc.csv
├── CNRM-ESM2-1_ssp585.csv
├── CNRM-ESM2-1_ssp585-bgc.csv
├── MIROC-ES2L_historical.csv
├── MIROC-ES2L_hist-bgc.csv
├── MIROC-ES2L_ssp585.csv
└── MIROC-ES2L_ssp585-bgc.csv
```

## Command-Line Usage

The easiest way to run the analysis is with the `run_analysis.py` script:

```bash
source .venv/bin/activate
python run_analysis.py MODEL SCENARIO
```

**Examples:**
```bash
python run_analysis.py ACCESS-ESM1-5 historical
python run_analysis.py CNRM-ESM2-1 ssp585
python run_analysis.py MIROC-ES2L historical --tau-min 0.01 --tau-max 50
```

**Output files:**
- `data/output/paired_diff_{model}_{scenario}.png` - diagnostic figure
- `data/output/paired_diff_{model}_{scenario}_summary.csv` - fit results

**All options:**
```bash
python run_analysis.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--x-col` | `tas` | Column for T(t) (temperature) |
| `--y-col` | `gpp` | Column for y(t) (GPP) |
| `--tau-min` | `0.1` | Min τ (decay time) in years |
| `--tau-max` | `100.0` | Max τ (decay time) in years |
| `--n-tau` | `100` | Number of τ values (log-spaced grid) |
| `--output-dir` | `data/output` | Output directory for figures and CSV |
| `--data-dir` | `data/input` | Input data directory |

## Output

### Diagnostic Figure

The output figure has three panels:

1. **Left: h(T) vs temperature** - The fitted function showing how GPP responds to temperature. Parameters are shown in the corner.

2. **Middle: BIC vs τ** - Model selection plot showing BIC for each function type across τ values. Lower BIC is better. The best point is marked.

3. **Right: Residuals** - Histogram of residuals (observed z minus predicted). Should be centered at zero.

### Summary CSV

| Column | Description |
|--------|-------------|
| `model` | Climate model name |
| `scenario_full` | Full scenario (e.g., historical) |
| `scenario_bgc` | BGC scenario (e.g., hist-bgc) |
| `n_regions` | Number of regions analyzed |
| `n_data_points` | Total data points |
| `tau_years` | Best-fit decay time (years) |
| `phi` | Corresponding φ = exp(-1/τ) |
| `function` | Selected function form |
| `r_squared` | R² of the fit |
| `bic` | Bayesian Information Criterion |
| `param_b` | Linear coefficient |
| `param_c` | Quadratic coefficient (if applicable) |
| `param_d` | Cubic coefficient (if applicable) |

## Python API Usage

```python
from convolution_fit import load_model_data, analyze_paired_difference

# Load all data
df = load_model_data("data/input")

# Run paired difference analysis
result = analyze_paired_difference(
    df,
    model="ACCESS-ESM1-5",
    scenario="historical",
    x_col='tas',
    y_col='gpp',
    tau_range=(0.1, 100.0),
    n_tau=100,
    show_plots=True
)

# Access results
print(f"Best decay time: τ = {result.tau:.2f} years")
print(f"Best function: {result.best_function.name}")
print(f"Parameters: {result.best_function.params}")
print(f"R²: {result.best_function.r_squared:.4f}")

# Evaluate h(T) at specific temperatures
import numpy as np
temps = np.array([10, 15, 20, 25, 30])
h_values = result.best_function.predict_single(temps)
print(f"h(T) at {temps}: {h_values}")

# Evaluate h(T_full) - h(T_bgc)
diff = result.best_function.predict_diff(temps + 2, temps)  # 2°C warming
print(f"Effect of 2°C warming: {diff}")
```

**Available analysis functions:**

| Function | Description |
|----------|-------------|
| `analyze_region()` | Single region, single scenario |
| `analyze_pooled()` | All regions pooled, single scenario |
| `analyze_paired_difference()` | All regions, full - bgc difference (recommended) |

## Data Format

**Input:** CSV files in `data/input/` with columns:
- `model`: Climate model name (e.g., ACCESS-ESM1-5)
- `region`: Country/region name
- `year`: Year
- `tas`: Surface air temperature (T)
- `pr`: Precipitation (already log-transformed)
- `gpp`: Gross Primary Productivity (already log-transformed)

**Note:** The `gpp` and `pr` columns in the input data are already log-transformed.

## Project Structure

```
bgc-best-fit/
├── run_analysis.py      # Command-line script
├── convolution_fit.py   # Main module with fitting functions
├── requirements.txt     # Python dependencies
├── README.md
├── .gitignore
├── data/
│   ├── input/           # Input CSV files by model and scenario
│   └── output/          # Generated figures and summary CSVs
└── .venv/               # Python virtual environment
```

## References

This approach is based on the equivalence between exponential-kernel convolutions and first-order linear ODEs, which allows transforming the joint estimation of (τ, h) into a more tractable grid-search problem.
