# Prophet Time Series

Forecasting airline passenger demand 12 months ahead using [Prophet](https://facebook.github.io/prophet/) — with seasonal decomposition, train/test validation, and confidence intervals.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wsamuelw/prophet-time-series/blob/main/Prophet_by_Meta_Demo.ipynb)

## Problem

Airline passenger volumes follow strong seasonal patterns — peaks in summer, troughs in winter. The goal: build a forecasting model that captures these patterns and predicts demand 12 months into the future, then validate against held-out data.

## Approach

1. **Decompose** the time series into trend, seasonality, and residuals using additive decomposition (period=12 for monthly data)
2. **Train** a Prophet model on historical data (pre-2023)
3. **Forecast** 12 months into the future with confidence intervals
4. **Validate** by comparing predictions against actual test data (2023 onwards)

## Results

The model captures the seasonal cycle well — predictions track the upward trend and annual peaks. Confidence intervals widen as the forecast horizon extends, which is expected.

Key output columns:
- `yhat` — point forecast
- `yhat_lower` / `yhat_upper` — 80% confidence interval

## Setup

### Google Colab

Click the badge above — no setup required.

### Local

```bash
pip install prophet pandas matplotlib statsmodels
git clone https://github.com/wsamuelw/prophet-time-series.git
cd prophet-time-series
jupyter notebook Prophet_by_Meta_Demo.ipynb
```

## Data

**Airline Passengers** — monthly international passenger counts (1949–2023).

| Split | Period | Records | Purpose |
|-------|--------|---------|---------|
| Train | Before 2023-01 | ~888 | Model fitting |
| Test | 2023-01 onwards | 12 | Validation |

## Project Structure

```
prophet-time-series/
├── Prophet_by_Meta_Demo.ipynb   # full walkthrough
├── README.md
└── LICENSE
```

## Tech Stack

- **Prophet** — forecasting engine
- **statsmodels** — seasonal decomposition
- **pandas** — data manipulation
- **matplotlib** — visualisation

## Key Decisions

- **Additive decomposition** — chosen over multiplicative because the seasonal amplitude doesn't grow proportionally with the trend
- **12-month forecast horizon** — matches one full seasonal cycle, enough to validate pattern capture without over-extrapolating
- **Default Prophet parameters** — no custom seasonality or holidays added; the default yearly seasonality was sufficient for this dataset

## License

MIT
