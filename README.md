# Prophet Time Series

End-to-end time series forecasting with [Prophet](https://facebook.github.io/prophet/) — Meta's open-source library that handles seasonality, holidays, and missing data out of the box.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wsamuelw/prophet-time-series/blob/main/Prophet_by_Meta_Demo.ipynb)

## What You'll Learn

1. **Seasonal decomposition** — break a time series into trend, seasonality, and residual
2. **Data prep for Prophet** — rename columns to `ds` (date) and `y` (target)
3. **Model fitting** — train a Prophet model on historical data
4. **Forecasting** — predict 12 months into the future with confidence intervals
5. **Visualisation** — plot forecast vs actuals with train/test separation

## What's Inside

| Step | What It Does |
|------|-------------|
| Data loading | Pulls airline passengers CSV from GitHub |
| Train/test split | Training data before 2023-01, test data after |
| Seasonal decomposition | Additive decomposition with `statsmodels` (period=12) |
| Prophet fit | Trains on training data with default parameters |
| Forecast | Generates 12-month future predictions with `yhat`, `yhat_lower`, `yhat_upper` |
| Visualisation | Plots forecast with a red vertical line separating train/test, overlays actual test data |

## Quick Start

### Google Colab (no setup)

Click the badge above — runs entirely in the browser.

### Local

```bash
pip install prophet pandas matplotlib statsmodels
git clone https://github.com/wsamuelw/prophet-time-series.git
cd prophet-time-series
jupyter notebook Prophet_by_Meta_Demo.ipynb
```

## Dataset

**Airline Passengers** — monthly international airline passenger counts (1949–2023). Sourced from [this Prophet tutorial](https://github.com/jonasdieckmann/prophet_tutorial).

| Split | Period | Purpose |
|-------|--------|---------|
| Train | Pre-2023-01 | Fit the model |
| Test | 2023-01 onwards | Evaluate forecast accuracy |

## Key Prophet Concepts

**`ds` and `y`** — Prophet requires these exact column names. `ds` is the datetime, `y` is the value to forecast.

```python
df = df.rename(columns={"Month": "ds", "Passengers": "y"})
```

**`make_future_dataframe`** — extends the date range into the future:

```python
future = model.make_future_dataframe(periods=12, freq='MS')  # 12 months, month-start
```

**`predict`** — returns a DataFrame with `yhat` (forecast), `yhat_lower`, and `yhat_upper` (confidence interval):

```python
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

## Why Prophet?

- **Handles missing data** — no need to fill gaps manually
- **Built-in seasonality** — daily, weekly, yearly patterns detected automatically
- **Holiday effects** — include custom holidays with a simple DataFrame
- **Confidence intervals** — uncertainty estimates come free
- **Fast** — fits in seconds on most datasets

## References

- [Prophet documentation](https://facebook.github.io/prophet/)
- [Prophet quick start](https://facebook.github.io/prophet/docs/quick_start.html)
- [Original paper](https://peerj.com/articles/cs-119/)

## License

MIT
